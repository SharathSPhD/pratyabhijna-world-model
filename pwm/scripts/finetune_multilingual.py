"""
finetune_multilingual.py — Fine-tune TrikaWorldModel on multilingual creative corpus.

Sprint 1 action: break the degenerate English-only fixed-point attractor.

Strategy (TRIZ Principle 35 — Parameter Changes):
  The WM trained on English converged to a single attractor (energy=11.57,
  active_dims=[211,491,306] for ALL inputs). Fine-tuning on multilingual
  data (Bengali, Hindi, Kannada, Tamil, Telugu, Sanskrit, English poetry,
  Carnatic lyrics) forces the WM to discriminate between different input
  distributions, breaking the degenerate attractor.

Sprint 1 gate criterion:
  χ² test on active_dims distribution across language domains → p < 0.05.
  Run with --test-only to verify without training.

Usage:
  python -m pwm.scripts.finetune_multilingual [--steps 100000] [--lr 5e-5]
  python -m pwm.scripts.finetune_multilingual --test-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[2]))

from pwm.world_model.trika import TrikaWorldModel  # type: ignore[import]
from pwm.data.embed_cache import CachedCorpusEnv  # type: ignore[import]

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_IN  = Path("checkpoints/step_1000000.pt")
CHECKPOINT_OUT = Path("checkpoints/step_multilingual.pt")
CACHE_DIR      = Path("data/multilingual_embed_cache")

WM_CFG = dict(
    obs_dim=512,
    action_dim=64,
    hidden_dim=512,
    stoch_dim=32,
    stoch_classes=32,
    n_levels=3,
    decoder_z_only=True,
)


# ── Domain sensitivity test ───────────────────────────────────────────────────

def test_domain_sensitivity(wm: TrikaWorldModel, cache_dir: Path,
                             device: torch.device) -> dict:
    """
    Sprint 1 gate test: χ² on active_dim distribution across domains.
    Returns dict with p-value and per-domain active_dim histograms.
    """
    import json
    meta = json.loads((cache_dir / "meta.json").read_text())
    N = meta["n"]
    obs_dim = meta["obs_dim"]
    domain_offsets = meta["domain_offsets"]
    domain_names = sorted(domain_offsets.keys(), key=lambda k: domain_offsets[k])

    emb = np.memmap(
        cache_dir / "embeddings.npy",
        dtype=np.float16, mode="r", shape=(N, obs_dim),
    )

    wm.eval()
    n_samples = 20   # samples per domain for test
    argmax_by_domain: dict[str, list[int]] = {}

    with torch.no_grad():
        for i, domain in enumerate(domain_names):
            start = domain_offsets[domain]
            end = domain_offsets[domain_names[i + 1]] if i + 1 < len(domain_names) else N
            domain_len = end - start

            argmaxes = []
            for _ in range(n_samples):
                # Sample a seq_len=40 sequence from this domain
                idx = np.random.randint(start, max(start + 1, end - 40))
                seq = torch.tensor(
                    emb[idx : idx + 40].astype(np.float32), device=device
                ).unsqueeze(0)   # (1, 40, 512)

                states = wm.init_state(1, device)
                for t in range(40):
                    obs_t = seq[:, t, :]
                    act_t = torch.zeros(1, 64, device=device)
                    states, _, _ = wm.observe_step(obs_t, act_t, states, t)

                h = states[0][0].squeeze(0)
                argmaxes.append(int(h.abs().argmax().item()))

            argmax_by_domain[domain] = argmaxes
            log.info(
                "Domain %-15s: argmax mode=%d  unique=%d/%d",
                domain, max(set(argmaxes), key=argmaxes.count),
                len(set(argmaxes)), n_samples,
            )

    # χ² test: is active_dim distribution different across domains?
    from scipy import stats as sp_stats  # type: ignore[import]
    all_dims = sorted(set(d for v in argmax_by_domain.values() for d in v))
    if len(all_dims) < 2:
        log.warning("All domains have the same active_dim — attractor still degenerate.")
        return {"p_value": 1.0, "unique_dims": 1, "domains": argmax_by_domain}

    # Build contingency table: domains × dim_buckets
    contingency = []
    for domain in domain_names:
        row = [argmax_by_domain[domain].count(d) for d in all_dims]
        contingency.append(row)
    contingency = np.array(contingency)

    try:
        _, p_value, _, _ = sp_stats.chi2_contingency(contingency)
    except ValueError:
        p_value = 1.0

    n_unique = len(set(d for v in argmax_by_domain.values() for d in v))
    log.info(
        "χ² test p=%.4f | unique active_dims=%d | Gate: %s",
        p_value, n_unique, "PASS (p<0.05)" if p_value < 0.05 else "FAIL"
    )
    return {
        "p_value": float(p_value),
        "unique_dims": n_unique,
        "domains": {k: list(v) for k, v in argmax_by_domain.items()},
    }


# ── Fine-tuning loop ──────────────────────────────────────────────────────────

def finetune(
    steps: int = 100_000,
    lr: float = 5e-5,
    batch_size: int = 32,
    seq_len: int = 64,
    grad_clip: float = 100.0,
    save_every: int = 10_000,
    log_every: int = 500,
    device: torch.device | None = None,
) -> None:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", dev)

    # Load WM with correct checkpoint config
    wm = TrikaWorldModel(**WM_CFG).to(dev)
    ckpt = torch.load(CHECKPOINT_IN, map_location=dev, weights_only=False)
    missing, unexpected = wm.load_state_dict(ckpt["world_model"], strict=False)
    if missing:
        log.warning("Missing keys (%d): %s...", len(missing), missing[:3])
    if unexpected:
        log.warning("Unexpected keys (%d): %s...", len(unexpected), unexpected[:3])
    log.info("Loaded: %s", CHECKPOINT_IN)

    # Corpus env using multilingual embed cache
    env = CachedCorpusEnv(
        cache_dir=CACHE_DIR,
        batch_size=batch_size,
        seq_len=seq_len,
        obs_dim=512,
        action_dim=64,
        device=dev,
        seed=42,
    )
    log.info(
        "Corpus: %d chunks across %d domains",
        env.meta["n"], len(env.meta["domain_offsets"]),
    )

    # Fine-tune WM only (Phase A — world model loss)
    # Lower LR than pre-training to avoid catastrophic forgetting
    opt = torch.optim.Adam(wm.parameters(), lr=lr, eps=1e-8, weight_decay=1e-6)

    wm.train()
    t0 = time.time()
    losses: list[float] = []

    for step in range(steps):
        obs_seq, action_seq, reward_seq, done_seq = env.sample_batch()

        init_states = wm.init_state(batch_size, dev)

        loss_dict = wm.world_model_loss(
            obs_seq, action_seq, reward_seq, done_seq, init_states
        )
        total_loss = loss_dict.get("total", None)
        if total_loss is None:
            # Fallback: sum all tensor values
            total_loss = sum(
                v for v in loss_dict.values()
                if isinstance(v, torch.Tensor) and v.requires_grad
            )
        if not isinstance(total_loss, torch.Tensor):
            continue

        opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), grad_clip)
        opt.step()

        losses.append(float(total_loss.detach()))

        if step % log_every == 0:
            mean_loss = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / max(elapsed, 1)
            eta_s = (steps - step) / steps_per_sec
            log.info(
                "Step %6d/%d | loss=%.4f | %.1f steps/s | ETA %.0fs",
                step, steps, mean_loss, steps_per_sec, eta_s,
            )

        if step > 0 and step % save_every == 0:
            ckpt_path = CHECKPOINT_OUT.parent / f"finetune_step_{step:07d}.pt"
            torch.save({
                "world_model": wm.state_dict(),
                "step": step,
                "loss": losses[-1],
                "corpus": str(CACHE_DIR),
            }, ckpt_path)
            log.info("Saved intermediate: %s", ckpt_path)

    # Final checkpoint
    torch.save({
        "world_model": wm.state_dict(),
        "step": steps,
        "loss": losses[-1] if losses else None,
        "corpus": str(CACHE_DIR),
        "mean_loss_final_1k": float(np.mean(losses[-1000:])) if losses else None,
    }, CHECKPOINT_OUT)
    log.info("Fine-tuning complete: %s", CHECKPOINT_OUT)

    # Sprint 1 gate test
    log.info("Running Sprint 1 gate test...")
    wm.eval()
    gate = test_domain_sensitivity(wm, CACHE_DIR, dev)

    gate_path = Path("benchmarks/results/sprint1_gate.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2))
    log.info("Gate results saved: %s", gate_path)

    if gate["p_value"] < 0.05:
        log.info("✓ Sprint 1 GATE PASSED: active_dims vary by domain (p=%.4f)", gate["p_value"])
    else:
        log.warning("✗ Sprint 1 GATE FAILED: p=%.4f (need p<0.05)", gate["p_value"])


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fine-tune WM on multilingual corpus")
    parser.add_argument("--steps",      type=int,   default=100_000)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--seq-len",    type=int,   default=64)
    parser.add_argument("--save-every", type=int,   default=10_000)
    parser.add_argument("--log-every",  type=int,   default=500)
    parser.add_argument("--test-only",  action="store_true",
                        help="Run domain sensitivity test on existing checkpoint only")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_only:
        log.info("Test-only mode: loading %s", CHECKPOINT_IN)
        wm = TrikaWorldModel(**WM_CFG).to(dev)
        ckpt = torch.load(CHECKPOINT_IN, map_location=dev, weights_only=False)
        wm.load_state_dict(ckpt["world_model"], strict=False)
        wm.eval()
        gate = test_domain_sensitivity(wm, CACHE_DIR, dev)
        print(json.dumps(gate, indent=2))
    else:
        finetune(
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            save_every=args.save_every,
            log_every=args.log_every,
            device=dev,
        )


if __name__ == "__main__":
    main()
