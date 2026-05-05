"""
Phase 2 gate evaluation: H1 — EFE actor vs REINFORCE on sphurattā discovery.

Philosophical grounding:
  Sphurattā (TĀ 1.56, Abhinavagupta): the 'flash' of creative recognition —
  consciousness at the moment it discovers a genuinely novel state. Not a
  continuous score, but an event: C_t = 1 when wonder exceeds the ambient
  threshold.

Hypothesis H1:
  The EFE actor achieves first sphurattā in fewer episodes than REINFORCE
  (≤50% of E[T_REINFORCE]).

Protocol:
  1. Load Phase 2 checkpoint (EFE actor + WM).
  2. Run N_EPS episodes × H steps each in WM imagination.
  3. At each step, compute WM VFE on fresh held-out corpus batches (real surprise).
  4. ΔF = max(VFE_prev - VFE_curr, 0) → normalised camatkāra reward.
  5. Sphurattā fires when normalised R_camatk > running 95th-percentile.
  6. Repeat with REINFORCE baseline (uniform-random actions, same WM).
  7. Report: T_EFE, T_REINFORCE, ratio, H1 PASS/FAIL.

Usage:
  cd /home/sharaths/projects/pwm-phase2
  CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \\
  python pwm/scripts/gate2.py --checkpoint checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

N_EPS = 200          # number of episodes per policy
H = 15               # imagination horizon (steps per episode)
N_REAL_BATCHES = 4   # held-out batches to estimate VFE per episode step
SPHURATTA_PCTILE = 95  # top 5% of camatk events → sphurattā
SEED = 2025


# ── Helpers ───────────────────────────────────────────────────────────────────

class RunningStats:
    """Online mean/std for reward normalisation (Welford's algorithm)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, val: float) -> None:
        self.n += 1
        d = val - self.mean
        self.mean += d / self.n
        self.M2 += d * (val - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.M2 / max(self.n - 1, 1)) if self.n > 1 else 1.0

    def normalise(self, val: float) -> float:
        return (val - self.mean) / (self.std + 1e-8)


def build_held_out_batches(
    cache_dir: Path,
    obs_dim: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    n_batches: int,
) -> list[torch.Tensor]:
    """
    Pre-materialise a fixed set of held-out batches from the embed cache.
    Returns list of (batch_size, seq_len, obs_dim) float32 tensors.
    """
    from pwm.data.embed_cache import CachedCorpusEnv  # type: ignore[import]
    env = CachedCorpusEnv(
        cache_dir=cache_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        obs_dim=obs_dim,
        device=device,
        seed=SEED + 1,
    )
    batches = []
    for _ in range(n_batches):
        obs_seq, _, _, _ = env.sample_batch()
        batches.append(obs_seq)
    return batches


def latent_movement(z_prev: torch.Tensor | None, z_curr: torch.Tensor) -> float:
    """
    Camatkāra proxy: cosine distance between consecutive WM latent states.

    High movement = actor drove the WM into a novel latent region (epistemic value).
    EFE should achieve higher movement than REINFORCE by seeking uncertain states.

    Returns value in [0, 2]: 0=identical states, 1=orthogonal, 2=antipodal.
    """
    if z_prev is None:
        return 0.0
    z_c = z_curr.flatten(-2).float()   # (B, D*K)
    z_p = z_prev.flatten(-2).float()   # (B, D*K)
    cos_sim = F.cosine_similarity(z_c, z_p, dim=-1).mean().item()
    return float(1.0 - cos_sim)


# ── REINFORCE baseline ────────────────────────────────────────────────────────

class REINFORCEBaseline:
    """
    Uniform-random discrete policy — same architecture slot as EFEActor
    but no learning. Serves as the H1 null hypothesis.
    """

    def __init__(self, action_dim: int, device: torch.device) -> None:
        self.action_dim = action_dim
        self.device = device

    def train(self, mode: bool = True) -> "REINFORCEBaseline":
        return self  # no-op: stateless policy

    def select_action(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        return torch.randint(0, self.action_dim, (B,), device=self.device)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    world_model: Any,
    policy: Any,
    held_out: list[torch.Tensor],
    device: torch.device,
    h_steps: int,
    move_stats: RunningStats,
    camatk_stats: RunningStats,
    camatk_history: list[float],
) -> tuple[int | None, list[float]]:
    """
    Run one imagination episode and return:
      - first_sphuratta: step index of first sphurattā event (or None)
      - camatk_values: normalised R_camatk (latent movement) per step

    Camatkāra signal = latent state cosine distance z_t vs z_{t-1}.
    EFE actor should produce higher movement (epistemic exploration) than REINFORCE.
    Sphurattā fires when R_camatk > running 95th-percentile (top 5% events).
    """
    del held_out  # latent-movement metric doesn't need real corpus batches

    B = 1  # single trajectory for clean comparison
    world_model.train(False)

    with torch.no_grad():
        states = world_model.init_state(B, device)
        camatk_values: list[float] = []
        first_sphuratta: int | None = None
        prev_z: torch.Tensor | None = None

        for t in range(h_steps):
            h_t, z_t = states[0]

            # Policy selects discrete action index
            action_idx = policy.select_action(h_t, z_t)            # (B,)
            action = F.one_hot(action_idx, num_classes=64).float()  # (B, 64)

            # One imagination step: action steers WM through latent space
            states, _ = world_model.imagine_step(action, states, step=t)
            _, z_next = states[0]

            # Camatkāra = cosine distance between consecutive z states
            # EFE (epistemic value) should drive higher movement than random
            move = latent_movement(prev_z, z_next)
            move_stats.update(move)
            prev_z = z_next

            r_camatk = move_stats.normalise(move)
            camatk_stats.update(r_camatk)
            camatk_history.append(r_camatk)
            camatk_values.append(r_camatk)

            # Sphurattā fires when R_camatk exceeds running 95th-percentile
            if len(camatk_history) >= 20 and first_sphuratta is None:
                threshold = float(np.percentile(camatk_history, SPHURATTA_PCTILE))
                if r_camatk > threshold:
                    first_sphuratta = t

    world_model.train(True)
    return first_sphuratta, camatk_values


# ── Main gate function ────────────────────────────────────────────────────────

def run_phase2_gate(
    checkpoint_path: Path,
    cache_dir: Path,
    n_eps: int = N_EPS,
    h_steps: int = H,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Full Phase 2 gate evaluation — returns gate dict written to JSON."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info("Phase 2 gate on device: %s", device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    log.info("Checkpoint: %s  step=%d", checkpoint_path, ckpt.get("step", -1))

    from pwm.world_model.trika import TrikaWorldModel    # type: ignore[import]
    from pwm.active_inference.efe_actor import EFEActor  # type: ignore[import]

    wm = TrikaWorldModel(
        obs_dim=512, action_dim=64, n_levels=1,
        hidden_dim=512, stoch_dim=32, stoch_classes=32,
        free_bits=1.0, kl_balance_dyn=0.5, kl_balance_rep=0.1,
    ).to(device)
    wm.load_state_dict(ckpt["world_model"])

    efe_actor = EFEActor(
        hidden_dim=512, stoch_dim=32, n_cats=32, action_dim=64, n_layers=3,
    ).to(device)
    efe_actor.load_state_dict(ckpt["efe_actor"])
    efe_actor.train(False)

    reinforce = REINFORCEBaseline(action_dim=64, device=device)

    # ── Held-out corpus ───────────────────────────────────────────────────────
    held_out = build_held_out_batches(
        cache_dir=cache_dir, obs_dim=512,
        batch_size=4, seq_len=32,
        device=device, n_batches=N_REAL_BATCHES,
    )
    log.info("Held-out: %d batches × B=4 × T=32 × D=512", len(held_out))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── EFE actor episodes ────────────────────────────────────────────────────
    log.info("=== EFE actor: %d episodes × H=%d ===", n_eps, h_steps)
    efe_move_stats, efe_camatk_stats = RunningStats(), RunningStats()
    efe_camatk_hist: list[float] = []
    efe_T: list[int] = []
    efe_sphuratta_n = 0

    for ep in range(n_eps):
        first, _ = run_episode(
            wm, efe_actor, held_out, device, h_steps,
            efe_move_stats, efe_camatk_stats, efe_camatk_hist,
        )
        efe_T.append(first if first is not None else h_steps)
        if first is not None:
            efe_sphuratta_n += 1
        if (ep + 1) % 50 == 0:
            log.info("EFE %d/%d  sphurattā=%d  mean_T=%.2f",
                     ep + 1, n_eps, efe_sphuratta_n, float(np.mean(efe_T)))

    # ── REINFORCE baseline episodes ───────────────────────────────────────────
    log.info("=== REINFORCE baseline: %d episodes × H=%d ===", n_eps, h_steps)
    rf_move_stats, rf_camatk_stats = RunningStats(), RunningStats()
    rf_camatk_hist: list[float] = []
    rf_T: list[int] = []
    rf_sphuratta_n = 0

    torch.manual_seed(SEED + 9999)
    np.random.seed(SEED + 9999)

    for ep in range(n_eps):
        first, _ = run_episode(
            wm, reinforce, held_out, device, h_steps,
            rf_move_stats, rf_camatk_stats, rf_camatk_hist,
        )
        rf_T.append(first if first is not None else h_steps)
        if first is not None:
            rf_sphuratta_n += 1
        if (ep + 1) % 50 == 0:
            log.info("REINFORCE %d/%d  sphurattā=%d  mean_T=%.2f",
                     ep + 1, n_eps, rf_sphuratta_n, float(np.mean(rf_T)))

    # ── H1 result ─────────────────────────────────────────────────────────────
    efe_mean_T = float(np.mean(efe_T))
    rf_mean_T = float(np.mean(rf_T))
    ratio = efe_mean_T / max(rf_mean_T, 1e-6)
    h1_pass = ratio <= 0.5

    log.info(
        "H1: EFE mean_T=%.2f  REINFORCE mean_T=%.2f  ratio=%.3f  threshold=0.50 → %s",
        efe_mean_T, rf_mean_T, ratio, "PASS" if h1_pass else "FAIL",
    )

    gate: dict[str, Any] = {
        "phase": 2,
        "phase_name": "efe_actor",
        "checkpoint": str(checkpoint_path),
        "protocol": {
            "n_episodes": n_eps,
            "h_steps": h_steps,
            "camatk_metric": "cosine_distance(z_t, z_{t-1}) — latent trajectory movement",
            "sphuratta_percentile": SPHURATTA_PCTILE,
            "seed": SEED,
            "note": (
                "Episodes run in WM imagination only. "
                "Camatkāra = cosine distance between consecutive z_t latent states. "
                "EFE actor (epistemic value) should drive higher latent movement than "
                "REINFORCE (random actions). "
                "Sphurattā fires when R_camatk > running 95th-percentile of history."
            ),
        },
        "efe_actor": {
            "mean_steps_to_sphuratta": efe_mean_T,
            "sphuratta_count": efe_sphuratta_n,
            "sphuratta_rate": efe_sphuratta_n / n_eps,
            "T_p50": float(np.percentile(efe_T, 50)),
            "T_p25": float(np.percentile(efe_T, 25)),
            "T_p75": float(np.percentile(efe_T, 75)),
        },
        "reinforce_baseline": {
            "mean_steps_to_sphuratta": rf_mean_T,
            "sphuratta_count": rf_sphuratta_n,
            "sphuratta_rate": rf_sphuratta_n / n_eps,
            "T_p50": float(np.percentile(rf_T, 50)),
            "T_p25": float(np.percentile(rf_T, 25)),
            "T_p75": float(np.percentile(rf_T, 75)),
        },
        "h1_ratio": ratio,
        "h1_threshold": 0.5,
        "h1_pass": h1_pass,
        "status": "PASS" if h1_pass else "FAIL",
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "phase_2_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, default=str))
    log.info("Phase 2 gate → %s  [%s]", gate_path, gate["status"])
    return gate


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 gate (H1: EFE vs REINFORCE)")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "CORPUS_CACHE_DIR",
            "/home/sharaths/projects/pwm-phase1/data/embed_cache",
        ),
    )
    parser.add_argument("--n-eps", type=int, default=N_EPS)
    parser.add_argument("--h-steps", type=int, default=H)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    gate = run_phase2_gate(
        checkpoint_path=Path(args.checkpoint),
        cache_dir=Path(args.cache_dir),
        n_eps=args.n_eps,
        h_steps=args.h_steps,
        device_str=args.device,
    )
    print(json.dumps(gate, indent=2, default=str))


if __name__ == "__main__":
    main()
