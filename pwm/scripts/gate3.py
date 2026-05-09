"""
Phase 3 gate: H2 — Hopfield CittaStore pattern completion.

Philosophical grounding:
  Smṛti (YS 1.11): pattern recognition through memory — the mind completing
  a partially-presented object by retrieving its stored impression. The Hopfield
  network realises this as attractor convergence from an occluded query.

Hypothesis H2:
  Pattern completion accuracy WITH Hopfield >= accuracy WITHOUT Hopfield x 1.10.
  (>= 10% improvement in occlusion completion accuracy)

Protocol:
  1. Load Phase 3 checkpoint (WM + CittaStore).
  2. Run STORE_EPS imagination episodes, storing h_t in CittaStore episodic bank.
  3. For each stored h_t: create an occluded query by zeroing 50% of dimensions.
  4. Recall: q_hop = citta_store.recall(occluded_h, mode='episodic')
  5. No-recall baseline: q_base = occluded_h (no Hopfield retrieval)
  6. Metric: cos_sim(q, h_original) averaged over all test patterns.
  7. Pass if ratio = acc_hop / acc_base >= 1.10.

Secondary check (sphuratta rate):
  Run SPHURATTA_EPS imagination episodes with Phase 3 EFE actor.
  Count sphuratta events (R_camatk > 95th percentile).
  Pass if 0.5-2.0 events per 100 steps.

Usage:
  cd /home/sharaths/projects/pwm-phase2
  CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \\
  python pwm/scripts/gate3.py --checkpoint checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

STORE_EPS = 200          # episodes to populate the Hopfield bank
TEST_PATTERNS = 500      # patterns to test completion accuracy
H_STEPS = 16             # imagination steps per episode (pattern collection)
OCCLUSION_FRAC = 0.5     # fraction of dims to zero in occluded query
SPHURATTA_EPS = 300      # episodes for sphuratta rate check
SPHURATTA_STEPS = 100    # horizon length for sphuratta rate check
SPHURATTA_PCTILE = 95    # threshold percentile for sphuratta event

COMPLETION_THRESHOLD = 1.10   # H2: Hopfield must improve completion by >=10%
SPHURATTA_RATE_MIN = 0.5      # per 100 steps
SPHURATTA_RATE_MAX = 2.0      # per 100 steps


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    log.info("Loaded checkpoint: %s (step=%d)", checkpoint_path, ckpt.get("step", -1))
    return ckpt


def build_world_model(ckpt: dict[str, Any], device: torch.device) -> Any:
    """Reconstruct TrikaWorldModel from checkpoint config."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pwm.world_model.trika import TrikaWorldModel

    cfg = ckpt.get("config", {})
    wm_cfg = cfg.get("world_model", {})

    model = TrikaWorldModel(
        obs_dim=wm_cfg.get("obs_dim", 1024),
        hidden_dim=wm_cfg.get("hidden_dim_apara", 512),
        latent_dim=wm_cfg.get("latent_dim", 1024),
        stoch_dim=wm_cfg.get("stoch_dim", 32),
        stoch_classes=wm_cfg.get("stoch_classes", 32),
        action_dim=wm_cfg.get("action_dim", 64),
        free_bits=wm_cfg.get("free_bits", 0.1),
        kl_balance_dyn=wm_cfg.get("kl_balance_dyn", 0.5),
        kl_balance_rep=wm_cfg.get("kl_balance_rep", 0.1),
        decoder_z_only=wm_cfg.get("decoder_z_only", False),
    ).to(device)

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
        store.load_state_dict(ckpt["citta_store"])
        log.info("CittaStore loaded from checkpoint.")
    else:
        log.warning("No citta_store in checkpoint — running with empty memory bank.")
    return store


# ── Pattern collection ────────────────────────────────────────────────────────

@torch.no_grad()
def collect_patterns(
    world_model: Any,
    citta_store: Any,
    n_episodes: int,
    h_steps: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Run imagination episodes, store h_t in CittaStore, return all h_t tensors."""
    stored: list[torch.Tensor] = []
    action_dim = 64

    for ep in range(n_episodes):
        B = 1
        states = world_model.init_state(B, device)

        for _ in range(h_steps):
            h_t, _ = states[0]
            citta_store.store_episode(h_t.detach(), level=0)
            stored.append(h_t.squeeze(0).detach().cpu())

            action_idx = torch.randint(0, action_dim, (B,), device=device)
            action = F.one_hot(action_idx, num_classes=action_dim).float()
            states, _ = world_model.imagine_step(action, states, step=0)

        if (ep + 1) % 50 == 0:
            log.info("  Collected %d/%d episodes (%d patterns)", ep + 1, n_episodes, len(stored))

    return stored


# ── Completion accuracy ───────────────────────────────────────────────────────

@torch.no_grad()
def measure_completion(
    citta_store: Any,
    stored_patterns: list[torch.Tensor],
    n_test: int,
    occlusion_frac: float,
    device: torch.device,
) -> tuple[float, float]:
    """
    Measure completion accuracy with and without Hopfield recall.

    Returns:
        acc_hop: mean cos_sim(recalled, original)
        acc_base: mean cos_sim(occluded, original)
    """
    if not stored_patterns:
        log.error("No patterns stored — cannot measure completion.")
        return 0.0, 0.0

    rng = np.random.default_rng(42)
    n_test = min(n_test, len(stored_patterns))
    indices = rng.choice(len(stored_patterns), size=n_test, replace=False)

    hop_sims: list[float] = []
    base_sims: list[float] = []

    for idx in indices:
        original = stored_patterns[idx].unsqueeze(0).to(device)  # (1, dim)

        dim = original.shape[-1]
        n_occlude = int(dim * occlusion_frac)
        mask_idx = rng.choice(dim, size=n_occlude, replace=False)
        occluded = original.clone()
        occluded[0, mask_idx] = 0.0

        cos_base = F.cosine_similarity(occluded, original, dim=-1).item()
        base_sims.append(cos_base)

        recalled = citta_store.recall(occluded, level=0, mode="episodic")
        cos_hop = F.cosine_similarity(recalled, original, dim=-1).item()
        hop_sims.append(cos_hop)

    return float(np.mean(hop_sims)), float(np.mean(base_sims))


# ── Sphuratta rate ────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_sphuratta_rate(
    world_model: Any,
    n_episodes: int,
    h_steps: int,
    device: torch.device,
) -> float:
    """
    Estimate sphuratta events per 100 steps via h_t activation-norm proxy.
    """
    action_dim = 64
    all_camatk: list[float] = []
    sphuratta_count = 0

    for _ in range(n_episodes):
        B = 1
        states = world_model.init_state(B, device)
        prev_norm: float | None = None
        ep_camatk: list[float] = []

        for _ in range(h_steps):
            h_t, _ = states[0]
            curr_norm = h_t.norm().item()
            delta = max(curr_norm - (prev_norm if prev_norm is not None else curr_norm), 0.0)
            ep_camatk.append(delta)
            all_camatk.append(delta)
            prev_norm = curr_norm

            action_idx = torch.randint(0, action_dim, (B,), device=device)
            action = F.one_hot(action_idx, num_classes=action_dim).float()
            states, _ = world_model.imagine_step(action, states, step=0)

        if len(all_camatk) >= 20:
            threshold = float(np.percentile(all_camatk, SPHURATTA_PCTILE))
            for v in ep_camatk:
                if v > threshold:
                    sphuratta_count += 1

    total_steps = n_episodes * h_steps
    rate_per_100 = 100.0 * sphuratta_count / total_steps if total_steps > 0 else 0.0
    return rate_per_100


# ── Main ─────────────────────────────────────────────────────────────────────

def run_phase3_gate(checkpoint_path: Path, out_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=== Phase 3 H2 Gate ===")
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Device: %s", device)

    ckpt = load_checkpoint(checkpoint_path, device)
    world_model = build_world_model(ckpt, device)
    citta_store = build_citta_store(ckpt, device)

    # Step 1: Collect patterns
    log.info("Collecting %d patterns (%d episodes x %d steps)...",
             STORE_EPS * H_STEPS, STORE_EPS, H_STEPS)
    stored_patterns = collect_patterns(world_model, citta_store, STORE_EPS, H_STEPS, device)
    log.info("Stored %d patterns.", len(stored_patterns))

    # Step 2: Pattern completion accuracy
    log.info("Measuring completion on %d test patterns (occlusion=%.0f%%)...",
             TEST_PATTERNS, OCCLUSION_FRAC * 100)
    acc_hop, acc_base = measure_completion(
        citta_store, stored_patterns, TEST_PATTERNS, OCCLUSION_FRAC, device
    )
    completion_ratio = (acc_hop / acc_base) if acc_base > 1e-6 else 0.0
    h2_completion_pass = completion_ratio >= COMPLETION_THRESHOLD

    log.info("Completion WITH Hopfield:    %.4f", acc_hop)
    log.info("Completion WITHOUT Hopfield: %.4f (baseline)", acc_base)
    log.info("Ratio (hop/base):            %.4f (threshold: %.2f)",
             completion_ratio, COMPLETION_THRESHOLD)
    log.info("H2 completion PASS: %s", h2_completion_pass)

    # Step 3: Sphuratta rate
    log.info("Measuring sphuratta rate over %d episodes x %d steps...",
             SPHURATTA_EPS, SPHURATTA_STEPS)
    sphuratta_rate = measure_sphuratta_rate(world_model, SPHURATTA_EPS, SPHURATTA_STEPS, device)
    h2_sphuratta_pass = SPHURATTA_RATE_MIN <= sphuratta_rate <= SPHURATTA_RATE_MAX

    log.info("Sphuratta rate: %.3f events/100 steps (target: %.1f-%.1f)",
             sphuratta_rate, SPHURATTA_RATE_MIN, SPHURATTA_RATE_MAX)
    log.info("H2 sphuratta PASS: %s", h2_sphuratta_pass)

    h2_pass = h2_completion_pass and h2_sphuratta_pass
    status = "PASS" if h2_pass else "FAIL"
    log.info("=== H2 Gate: %s ===", status)

    result: dict[str, Any] = {
        "phase": 3,
        "phase_name": "hopfield_citta_store",
        "checkpoint": str(checkpoint_path),
        "protocol": {
            "store_episodes": STORE_EPS,
            "test_patterns": TEST_PATTERNS,
            "h_steps": H_STEPS,
            "occlusion_frac": OCCLUSION_FRAC,
            "completion_threshold": COMPLETION_THRESHOLD,
            "sphuratta_episodes": SPHURATTA_EPS,
            "sphuratta_steps": SPHURATTA_STEPS,
            "sphuratta_rate_min": SPHURATTA_RATE_MIN,
            "sphuratta_rate_max": SPHURATTA_RATE_MAX,
        },
        "completion": {
            "acc_hopfield": acc_hop,
            "acc_baseline": acc_base,
            "ratio": completion_ratio,
            "threshold": COMPLETION_THRESHOLD,
            "pass": h2_completion_pass,
        },
        "sphuratta": {
            "rate_per_100_steps": sphuratta_rate,
            "target_min": SPHURATTA_RATE_MIN,
            "target_max": SPHURATTA_RATE_MAX,
            "pass": h2_sphuratta_pass,
        },
        "h2_pass": h2_pass,
        "status": status,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    step = ckpt.get("step", 0)
    out_path = out_dir / f"phase_3_gate_step{step:07d}.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Gate result saved: %s", out_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 H2 gate")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to Phase 3 checkpoint (final.pt)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("benchmarks/results"),
                        help="Directory for result JSON")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run_phase3_gate(args.checkpoint, args.out_dir)


if __name__ == "__main__":
    main()
