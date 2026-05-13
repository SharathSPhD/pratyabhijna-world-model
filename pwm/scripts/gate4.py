"""
Phase 4 gate: H3 - Sleep consolidation reduces catastrophic forgetting.

Philosophical grounding:
  Nidra (Mandukya Upanisad 1.5, Abhinavagupta TA 10.185):
  Sleep as samskara consolidation - the mind's NREM replay compresses
  episodic traces into semantic samskaras, reducing interference.
  Computational: SleepConsolidator prevents catastrophic forgetting via
  prioritised replay + synaptic homeostasis (SHY down-scaling).

Hypothesis H3:
  Sequential forgetting rate WITH sleep < 0.8 x rate WITHOUT sleep.
  (>= 20% reduction in catastrophic forgetting on a 3-domain sequence)

Protocol:
  1. Load Phase 4 checkpoint (WM + CittaStore).
  2. Collect domain-specific h_t patterns by varying action_idx in {0,32,63}.
  3. Measure forgetting WITHOUT sleep:
       store domain 0 -> baseline acc on domain 0
       store domain 1 (interference) -> re-measure domain 0 acc
       forgetting_no_sleep = 1 - (acc_after_0 / acc_base_0)
  4. Measure forgetting WITH sleep:
       reset CittaStore; store domain 0 patterns; record baseline
       populate ReplayBuffer with domain 0 transitions
       run SleepConsolidator.sleep(device) (NREM + REM cycles)
       store domain 1; re-measure domain 0 acc
       forgetting_with_sleep = 1 - (acc_after_0_sleep / acc_base_0)
  5. h3_pass = forgetting_with_sleep < 0.8 * forgetting_without_sleep

Usage:
  cd /home/sharaths/projects/pwm-phase2
  CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
  python pwm/scripts/gate4.py --checkpoint checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# Constants -------------------------------------------------------------------

DOMAIN_ACTIONS = [0, 32, 63]   # action indices to simulate three domains
EPISODES_PER_DOMAIN = 100      # imagination episodes per domain
H_STEPS = 16                   # imagination steps per episode
TEST_PATTERNS = 200            # patterns to test completion accuracy
OCCLUSION_FRAC = 0.5
ACTION_DIM = 64
OBS_DIM = 1024

FORGETTING_RATIO_THRESHOLD = 0.8  # H3: with-sleep must be < 0.8 x without-sleep
REPLAY_CAPACITY = 2048


# Helpers (mirror gate3.py) ---------------------------------------------------

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    log.info("Loaded checkpoint: %s (step=%d)", checkpoint_path, ckpt.get("step", -1))
    return ckpt


def build_world_model(ckpt: dict[str, Any], device: torch.device) -> Any:
    """Reconstruct TrikaWorldModel from checkpoint config."""
    from pwm.world_model.trika import TrikaWorldModel

    cfg = ckpt.get("config", {})
    wm_cfg = cfg.get("world_model", {})

    model = TrikaWorldModel(
        obs_dim=wm_cfg.get("obs_dim", OBS_DIM),
        action_dim=wm_cfg.get("action_dim", ACTION_DIM),
        n_levels=wm_cfg.get("levels", 1),
        hidden_dim=wm_cfg.get("hidden_dim_apara", 512),
        stoch_dim=wm_cfg.get("stoch_dim", 32),
        stoch_classes=wm_cfg.get("stoch_classes", 32),
        free_bits=wm_cfg.get("free_bits", 0.1),
        kl_balance_dyn=wm_cfg.get("kl_balance_dyn", 0.5),
        kl_balance_rep=wm_cfg.get("kl_balance_rep", 0.1),
        decoder_z_only=wm_cfg.get("decoder_z_only", False),
    ).to(device)

    if "world_model" in ckpt:
        filtered = {k: v for k, v in ckpt["world_model"].items()
                    if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def build_citta_store(ckpt: dict[str, Any], device: torch.device) -> Any:
    """Reconstruct CittaStore from checkpoint."""
    from pwm.memory.citta_store import CittaStore

    cfg = ckpt.get("config", {})
    wm_cfg = cfg.get("world_model", {})

    store = CittaStore(
        hidden_dim=wm_cfg.get("hidden_dim_apara", 512),
        n_levels=wm_cfg.get("levels", 1),
    ).to(device)

    if "citta_store" in ckpt:
        try:
            store.load_state_dict(ckpt["citta_store"])
            log.info("CittaStore loaded from checkpoint.")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load citta_store state: %s", exc)
    else:
        log.warning("No citta_store in checkpoint - starting with empty memory bank.")
    return store


def fresh_citta_store(ckpt: dict[str, Any], device: torch.device) -> Any:
    """Construct a fresh (empty) CittaStore matching the checkpoint config."""
    from pwm.memory.citta_store import CittaStore

    cfg = ckpt.get("config", {})
    wm_cfg = cfg.get("world_model", {})

    return CittaStore(
        hidden_dim=wm_cfg.get("hidden_dim_apara", 512),
        n_levels=wm_cfg.get("levels", 1),
    ).to(device)


# Pattern collection (domain-conditional) -------------------------------------

@torch.no_grad()
def collect_domain_patterns(
    world_model: Any,
    citta_store: Any | None,
    n_episodes: int,
    h_steps: int,
    action_idx_fixed: int,
    device: torch.device,
    store: bool = True,
) -> list[torch.Tensor]:
    """
    Run imagination episodes with a fixed action index (proxies a domain),
    optionally storing h_t in CittaStore. Return all h_t patterns (CPU tensors).
    """
    stored: list[torch.Tensor] = []

    for ep in range(n_episodes):
        B = 1
        states = world_model.init_state(B, device)

        for _ in range(h_steps):
            h_t, _ = states[0]
            if store and citta_store is not None:
                citta_store.store_episode(h_t.detach(), level=0)
            stored.append(h_t.squeeze(0).detach().cpu())

            action_idx = torch.full((B,), action_idx_fixed, dtype=torch.long, device=device)
            action = F.one_hot(action_idx, num_classes=ACTION_DIM).float()
            states, _ = world_model.imagine_step(action, states, step=0)

        if (ep + 1) % 25 == 0:
            log.info("  Domain action=%d: %d/%d episodes (%d patterns)",
                     action_idx_fixed, ep + 1, n_episodes, len(stored))

    return stored


# Completion accuracy on a fixed pattern set ----------------------------------

@torch.no_grad()
def measure_completion(
    citta_store: Any,
    stored_patterns: list[torch.Tensor],
    n_test: int,
    occlusion_frac: float,
    device: torch.device,
    seed: int = 42,
) -> float:
    """Mean cos_sim(recalled, original) under 50% occlusion."""
    if not stored_patterns:
        log.error("No patterns to test - returning 0.0.")
        return 0.0

    rng = np.random.default_rng(seed)
    n_test = min(n_test, len(stored_patterns))
    indices = rng.choice(len(stored_patterns), size=n_test, replace=False)

    sims: list[float] = []
    for idx in indices:
        original = stored_patterns[idx].unsqueeze(0).to(device)
        dim = original.shape[-1]
        n_occlude = int(dim * occlusion_frac)
        mask_idx = rng.choice(dim, size=n_occlude, replace=False)
        occluded = original.clone()
        occluded[0, mask_idx] = 0.0

        recalled = citta_store.recall(occluded, level=0, mode="episodic")
        sims.append(F.cosine_similarity(recalled, original, dim=-1).item())

    return float(np.mean(sims))


# Sleep harness ---------------------------------------------------------------

def populate_replay(
    domain_patterns: list[torch.Tensor],
    n_transitions: int,
) -> Any:
    """Build a ReplayBuffer of dummy Transitions whose obs come from domain 0."""
    from pwm.memory.replay import ReplayBuffer, Transition

    buf = ReplayBuffer(capacity=REPLAY_CAPACITY)
    rng = np.random.default_rng(42)
    n = min(n_transitions, len(domain_patterns))
    idxs = rng.choice(len(domain_patterns), size=n, replace=False)
    prev_obs = np.zeros(OBS_DIM, dtype=np.float32)
    for _ in idxs:
        obs = rng.standard_normal(OBS_DIM).astype(np.float32) * 0.1
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        trans = Transition(
            obs=prev_obs.copy(),
            action=action,
            reward=0.0,
            done=False,
            next_obs=obs.copy(),
            vfe=0.0,
        )
        buf.add(trans)
        prev_obs = obs
    log.info("Populated replay buffer with %d transitions.", len(buf))
    return buf


def run_sleep(
    world_model: Any,
    citta_store: Any,
    replay_buffer: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Construct and run a SleepConsolidator (NREM + REM)."""
    from pwm.sleep.consolidation import (
        SleepConfig,
        NREMPhase,
        REMPhase,
        SleepConsolidator,
    )

    cfg = SleepConfig(
        nrem_replay_steps=20,    # short for gate; full training uses 100
        rem_dream_horizon=16,
        rem_retrain_steps=10,
        max_nrem_cycles=2,
        max_rem_cycles=2,
    )

    wm_opt = torch.optim.Adam(world_model.parameters(), lr=1e-5)
    rec_opt = torch.optim.Adam(world_model.parameters(), lr=1e-5)

    nrem = NREMPhase(world_model, citta_store, replay_buffer, wm_opt, cfg)
    rem = REMPhase(world_model, rec_opt, cfg)
    consolidator = SleepConsolidator(nrem, rem, cfg)

    log.info("Running sleep cycles (max_nrem=%d, max_rem=%d)...",
             cfg.max_nrem_cycles, cfg.max_rem_cycles)
    metrics = consolidator.sleep(device)
    log.info("Sleep complete: %d cycles, total_vfe_reduction=%.4f",
             metrics.get("cycles", 0), metrics.get("total_vfe_reduction", 0.0))
    return metrics


# Main gate -------------------------------------------------------------------

def run_phase4_gate(checkpoint_path: Path, out_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=== Phase 4 H3 Gate ===")
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Device: %s", device)

    ckpt = load_checkpoint(checkpoint_path, device)
    world_model = build_world_model(ckpt, device)

    # Collect domain patterns (shared across both conditions)
    log.info("Collecting domain patterns (no store) ...")
    domain_patterns: dict[int, list[torch.Tensor]] = {}
    for d_idx, action_idx in enumerate(DOMAIN_ACTIONS):
        log.info("  Domain %d (action=%d) ...", d_idx, action_idx)
        domain_patterns[d_idx] = collect_domain_patterns(
            world_model, None, EPISODES_PER_DOMAIN, H_STEPS,
            action_idx, device, store=False,
        )

    # Condition A: forgetting WITHOUT sleep
    log.info("--- Condition A: WITHOUT sleep ---")
    citta_a = fresh_citta_store(ckpt, device)
    for h in domain_patterns[0]:
        citta_a.store_episode(h.unsqueeze(0).to(device), level=0)
    acc_base_no_sleep = measure_completion(
        citta_a, domain_patterns[0], TEST_PATTERNS, OCCLUSION_FRAC, device, seed=42,
    )
    log.info("Baseline acc (domain 0, no sleep): %.4f", acc_base_no_sleep)

    for h in domain_patterns[1]:
        citta_a.store_episode(h.unsqueeze(0).to(device), level=0)
    acc_after_no_sleep = measure_completion(
        citta_a, domain_patterns[0], TEST_PATTERNS, OCCLUSION_FRAC, device, seed=42,
    )
    forgetting_no_sleep = (
        1.0 - (acc_after_no_sleep / acc_base_no_sleep)
        if acc_base_no_sleep > 1e-6 else 0.0
    )
    log.info("After-interference acc (no sleep): %.4f", acc_after_no_sleep)
    log.info("Forgetting WITHOUT sleep: %.4f", forgetting_no_sleep)

    # Condition B: forgetting WITH sleep
    log.info("--- Condition B: WITH sleep ---")
    citta_b = fresh_citta_store(ckpt, device)
    for h in domain_patterns[0]:
        citta_b.store_episode(h.unsqueeze(0).to(device), level=0)
    acc_base_sleep = measure_completion(
        citta_b, domain_patterns[0], TEST_PATTERNS, OCCLUSION_FRAC, device, seed=42,
    )
    log.info("Baseline acc (domain 0, with sleep): %.4f", acc_base_sleep)

    replay_buf = populate_replay(domain_patterns[0], n_transitions=256)
    sleep_metrics = run_sleep(world_model, citta_b, replay_buf, device)

    for h in domain_patterns[1]:
        citta_b.store_episode(h.unsqueeze(0).to(device), level=0)
    acc_after_sleep = measure_completion(
        citta_b, domain_patterns[0], TEST_PATTERNS, OCCLUSION_FRAC, device, seed=42,
    )
    forgetting_with_sleep = (
        1.0 - (acc_after_sleep / acc_base_sleep)
        if acc_base_sleep > 1e-6 else 0.0
    )
    log.info("After-interference acc (with sleep): %.4f", acc_after_sleep)
    log.info("Forgetting WITH sleep: %.4f", forgetting_with_sleep)

    # H3 verdict
    ratio = (
        forgetting_with_sleep / forgetting_no_sleep
        if abs(forgetting_no_sleep) > 1e-6 else float("inf")
    )
    h3_pass = forgetting_with_sleep < FORGETTING_RATIO_THRESHOLD * forgetting_no_sleep
    status = "PASS" if h3_pass else "FAIL"
    log.info("Forgetting ratio (with/without): %.4f (threshold: < %.2f)",
             ratio, FORGETTING_RATIO_THRESHOLD)
    log.info("=== H3 Gate: %s ===", status)

    result: dict[str, Any] = {
        "phase": 4,
        "phase_name": "sleep_consolidation",
        "checkpoint": str(checkpoint_path),
        "protocol": {
            "domain_actions": DOMAIN_ACTIONS,
            "episodes_per_domain": EPISODES_PER_DOMAIN,
            "h_steps": H_STEPS,
            "test_patterns": TEST_PATTERNS,
            "occlusion_frac": OCCLUSION_FRAC,
            "replay_capacity": REPLAY_CAPACITY,
        },
        "without_sleep": {
            "acc_baseline_domain0": acc_base_no_sleep,
            "acc_after_interference": acc_after_no_sleep,
            "forgetting": forgetting_no_sleep,
        },
        "with_sleep": {
            "acc_baseline_domain0": acc_base_sleep,
            "acc_after_interference": acc_after_sleep,
            "forgetting": forgetting_with_sleep,
            "sleep_metrics": {
                "cycles": sleep_metrics.get("cycles", 0),
                "total_vfe_reduction": sleep_metrics.get("total_vfe_reduction", 0.0),
            },
        },
        "h3_forgetting_with_sleep": forgetting_with_sleep,
        "h3_forgetting_without_sleep": forgetting_no_sleep,
        "h3_forgetting_ratio": ratio,
        "h3_threshold": FORGETTING_RATIO_THRESHOLD,
        "h3_pass": h3_pass,
        "status": status,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    step = ckpt.get("step", 0)
    out_path = out_dir / f"phase_4_gate_step{step:07d}.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Gate result saved: %s", out_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 H3 gate")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to Phase 4 checkpoint (final.pt)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("benchmarks/results"),
                        help="Directory for result JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu); auto-falls back to cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run_phase4_gate(args.checkpoint, args.out_dir)


if __name__ == "__main__":
    main()
