"""
Phase 5 gate: H4 (agama-narration proxy) + H5 (reward scaling vs Phase 2).

Philosophical grounding:
  Agama-pramana (IPK 1.1.3, Utpaladeva): testimony as a valid means of knowledge.
  The LLM, when bridged to the world-model, plays the role of agama -- supplying
  linguistic articulation (vimarsa) to states that are already self-luminous in
  the latent. PHr sutra 4 names jnana-sakti as the power of recognitive
  articulation; the proxy here is the entropy of the stochastic latent at
  sphuratta events: a high-entropy latent is one in which multiple discrete
  modes coexist, and is therefore narratable -- there is something for the
  agama to articulate.

Hypothesis H4 (narration proxy):
  At least 70% of sphuratta events have entropy(z_t) > 0.5 nats, where
  z_t is the stochastic latent of Level 0 (Apara) sampled from the prior.

Hypothesis H5 (reward scaling):
  Mean episode reward in Phase 5 >= 2.0 x Phase 2 baseline (2.5301936708483845).

Gate logic:
  h_pass = h4_pass OR h5_pass -- either narration quality or reward scaling
  is sufficient to advance.

Usage:
  cd /home/sharaths/projects/pwm-phase2
  CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \\
  python pwm/scripts/gate5.py --checkpoint checkpoints/final.pt
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


# Constants

H4_EPISODES = 200
H4_STEPS = 32
H4_SPHURATTA_PCTILE = 75       # Phase 5 uses 75th percentile (not 95th)
H4_ENTROPY_THRESHOLD = 0.5     # nats
H4_MEANINGFUL_FRAC = 0.70

H5_EPISODES = 200
H5_STEPS = 32
H5_PHASE2_BASELINE = 2.5301936708483845
H5_THRESHOLD = 2.0


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    log.info("Loaded checkpoint: %s (step=%d)", checkpoint_path, ckpt.get("step", -1))
    return ckpt


def build_world_model(ckpt: dict[str, Any], device: torch.device) -> Any:
    from pwm.world_model.trika import TrikaWorldModel

    cfg = ckpt.get("config", {})
    wm_cfg = cfg.get("world_model", {})

    model = TrikaWorldModel(
        obs_dim=wm_cfg.get("obs_dim", 1024),
        action_dim=wm_cfg.get("action_dim", 64),
        n_levels=wm_cfg.get("levels", 1),
        hidden_dim=wm_cfg.get("hidden_dim_apara", 512),
        stoch_dim=wm_cfg.get("stoch_dim", 32),
        stoch_classes=wm_cfg.get("stoch_classes", 32),
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


@torch.no_grad()
def collect_rollouts(
    world_model: Any,
    n_episodes: int,
    h_steps: int,
    device: torch.device,
) -> tuple[list[list[float]], list[list[torch.Tensor]]]:
    """
    Run pure-imagination rollouts.

    Returns:
        deltas_per_episode: list of per-step camatkara deltas (h.norm() delta)
        prior_logits_per_episode: list of per-step prior logits for Level 0
    """
    action_dim = 64
    deltas_eps: list[list[float]] = []
    prior_eps: list[list[torch.Tensor]] = []

    for ep in range(n_episodes):
        B = 1
        states = world_model.init_state(B, device)
        prev_norm: float | None = None
        ep_deltas: list[float] = []
        ep_priors: list[torch.Tensor] = []

        for step in range(h_steps):
            h_t, _ = states[0]
            curr_norm = h_t.norm().item()
            delta = max(curr_norm - (prev_norm if prev_norm is not None else curr_norm), 0.0)
            ep_deltas.append(delta)
            prev_norm = curr_norm

            action_idx = torch.randint(0, action_dim, (B,), device=device)
            action = F.one_hot(action_idx, num_classes=action_dim).float()
            states, all_logits_prior = world_model.imagine_step(action, states, step=step)

            ep_priors.append(all_logits_prior[0].detach().cpu())

        deltas_eps.append(ep_deltas)
        prior_eps.append(ep_priors)

        if (ep + 1) % 50 == 0:
            log.info("  Rollout %d/%d", ep + 1, n_episodes)

    return deltas_eps, prior_eps


def latent_entropy_nats(prior_logits: torch.Tensor) -> float:
    """
    Mean entropy (nats) across stoch_dim categorical groups for one timestep.

    prior_logits shape: (B=1, stoch_dim, stoch_classes) or flat.
    """
    if prior_logits.dim() == 2:
        try:
            prior_logits = prior_logits.reshape(prior_logits.shape[0], 32, 32)
        except RuntimeError:
            return 0.0

    log_probs = F.log_softmax(prior_logits, dim=-1)
    probs = torch.softmax(prior_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, stoch_dim)
    return float(entropy.mean().item())


def evaluate_h4(
    deltas_eps: list[list[float]],
    prior_eps: list[list[torch.Tensor]],
) -> tuple[float, int, int]:
    all_deltas = [d for ep in deltas_eps for d in ep]
    if not all_deltas:
        return 0.0, 0, 0

    threshold = float(np.percentile(all_deltas, H4_SPHURATTA_PCTILE))

    n_sphuratta = 0
    n_meaningful = 0

    for ep_d, ep_p in zip(deltas_eps, prior_eps):
        for delta, prior in zip(ep_d, ep_p):
            if delta > threshold:
                n_sphuratta += 1
                ent = latent_entropy_nats(prior)
                if ent > H4_ENTROPY_THRESHOLD:
                    n_meaningful += 1

    rate = n_meaningful / n_sphuratta if n_sphuratta > 0 else 0.0
    return rate, n_sphuratta, n_meaningful


def evaluate_h5(deltas_eps: list[list[float]]) -> float:
    if not deltas_eps:
        return 0.0
    ep_rewards = [float(sum(ep)) for ep in deltas_eps]
    return float(np.mean(ep_rewards))


def run_phase5_gate(checkpoint_path: Path, out_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=== Phase 5 H4/H5 Gate ===")
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Device: %s", device)

    ckpt = load_checkpoint(checkpoint_path, device)
    world_model = build_world_model(ckpt, device)

    log.info("H4: rolling out %d episodes x %d steps...", H4_EPISODES, H4_STEPS)
    deltas_h4, priors_h4 = collect_rollouts(world_model, H4_EPISODES, H4_STEPS, device)
    meaningful_rate, n_sphur, n_meaning = evaluate_h4(deltas_h4, priors_h4)
    h4_pass = meaningful_rate >= H4_MEANINGFUL_FRAC
    log.info("H4: %d sphuratta events; %d meaningful (entropy>%.2f); rate=%.3f (threshold %.2f)",
             n_sphur, n_meaning, H4_ENTROPY_THRESHOLD, meaningful_rate, H4_MEANINGFUL_FRAC)
    log.info("H4 PASS: %s", h4_pass)

    log.info("H5: rolling out %d episodes x %d steps...", H5_EPISODES, H5_STEPS)
    deltas_h5, _ = collect_rollouts(world_model, H5_EPISODES, H5_STEPS, device)
    mean_reward = evaluate_h5(deltas_h5)
    reward_ratio = mean_reward / H5_PHASE2_BASELINE if H5_PHASE2_BASELINE > 0 else 0.0
    h5_pass = reward_ratio >= H5_THRESHOLD
    log.info("H5: mean reward=%.4f; Phase 2 baseline=%.4f; ratio=%.3f (threshold %.2f)",
             mean_reward, H5_PHASE2_BASELINE, reward_ratio, H5_THRESHOLD)
    log.info("H5 PASS: %s", h5_pass)

    h_pass = h4_pass or h5_pass
    status = "PASS" if h_pass else "FAIL"
    log.info("=== Phase 5 Gate: %s (H4 OR H5) ===", status)

    result: dict[str, Any] = {
        "phase": 5,
        "phase_name": "llm_vimarsa_bridge",
        "checkpoint": str(checkpoint_path),
        "protocol": {
            "h4_episodes": H4_EPISODES,
            "h4_steps": H4_STEPS,
            "h4_sphuratta_pctile": H4_SPHURATTA_PCTILE,
            "h4_entropy_threshold_nats": H4_ENTROPY_THRESHOLD,
            "h4_meaningful_frac_threshold": H4_MEANINGFUL_FRAC,
            "h5_episodes": H5_EPISODES,
            "h5_steps": H5_STEPS,
            "h5_phase2_baseline": H5_PHASE2_BASELINE,
            "h5_threshold": H5_THRESHOLD,
        },
        "h4_meaningful_rate": meaningful_rate,
        "h4_n_sphuratta": n_sphur,
        "h4_n_meaningful": n_meaning,
        "h4_threshold": H4_MEANINGFUL_FRAC,
        "h4_pass": h4_pass,
        "h5_mean_reward_phase5": mean_reward,
        "h5_phase2_baseline": H5_PHASE2_BASELINE,
        "h5_reward_ratio": reward_ratio,
        "h5_threshold": H5_THRESHOLD,
        "h5_pass": h5_pass,
        "h_pass": h_pass,
        "status": status,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    step = ckpt.get("step", 0)
    out_path = out_dir / f"phase_5_gate_step{step:07d}.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Gate result saved: %s", out_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 H4/H5 gate")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to Phase 5 checkpoint (final.pt)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("benchmarks/results"),
                        help="Directory for result JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run_phase5_gate(args.checkpoint, args.out_dir)


if __name__ == "__main__":
    main()
