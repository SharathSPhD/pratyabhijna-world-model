#!/usr/bin/env python3
"""Quick checkpoint probe — print key weight norms and training health indicators.

Usage:
  python scripts/probe_checkpoint.py checkpoints/step_0010000.pt
"""

import sys
import torch

def probe(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", -1)
    wm = ckpt["world_model"]
    actor = ckpt["efe_actor"]
    critic = ckpt["critic"]

    print(f"\n{'='*60}")
    print(f"Checkpoint: {path}  step={step}")
    print(f"{'='*60}")

    # World-model obs-processing health (critical: should be non-zero in v6)
    print("\n── WM obs-processing modules (norm>1.0 = healthy) ──")
    for k in wm:
        if "encoder" in k or "prior" in k or "decoder" in k:
            print(f"  {k:50s}  norm={wm[k].norm():.4f}")

    # input_proj W_z vs W_a columns
    print("\n── GRU input_proj (W_z=latent cols, W_a=action cols) ──")
    for k in wm:
        if "input_proj" in k and "weight" in k:
            v = wm[k]
            half = 1024  # latent_dim = 32*32
            W_z = v[:, :half]
            W_a = v[:, half:]
            print(f"  {k}: W_z_norm={W_z.norm():.4f}  W_a_norm={W_a.norm():.4f}")

    # GRU recurrence
    print("\n── GRU recurrence norms ──")
    for k in wm:
        if "sequence_model" in k and "weight" in k:
            print(f"  {k:50s}  norm={wm[k].norm():.4f}")

    # Actor / critic health
    print("\n── EFE Actor (trained from step 0 in v6) ──")
    for k in actor:
        if "weight" in k:
            print(f"  {k:50s}  norm={actor[k].norm():.4f}")

    print(f"\n── Critic ──")
    for k in critic:
        if "weight" in k:
            print(f"  {k:50s}  norm={critic[k].norm():.4f}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/probe_checkpoint.py <path>")
        sys.exit(1)
    probe(sys.argv[1])
