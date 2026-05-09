#!/bin/bash
# Phase 2 v5: Domain-Selective Corpus + Reduced Free-Bits
#
# Root causes fixed vs v4 (IDL only, CachedCorpusEnv passive, free_bits=1.0):
#
#   Layer 3 (passive env) — DomainSelectiveCachedCorpusEnv:
#     Actions 0-31  → gutenberg domain  (11,069 chunks, literary prose)
#     Actions 32-63 → philosophy domain (47,583 chunks, wiki philosophy)
#     obs_{t+1} is NOW conditioned on a_t: the modal action in each batch
#     selects which corpus slice to sample from. VFE gradients through W_a
#     are non-zero at every step, not just from IDL imagination rollouts.
#
#   Layer 4 (free_bits ceiling) — reduced 1.0 → 0.1 nats per variable:
#     Old: total KL floor = 32×1.0 = 32 nats → prior_net grad = 0 whenever KL < 32.
#     New: total KL floor = 32×0.1 = 3.2 nats → prior learns domain distributions.
#     The IDL-trained WM already has action-conditional h_t (cos_sim ≈ -1.000).
#     With free_bits=0.1, p(z|h_t) will diverge per domain → prior entropy becomes
#     action-dependent → EFE epistemic term is non-zero → H1 can fire.
#
# Warm start from v4 final checkpoint (IDL-trained WM, cos_sim=-1.000):
#   - Preserves IDL-trained action-conditional h_t geometry in W_a
#   - Actor/critic reset to random init (domain reward structure is new)
#   - free_bits=0.1 triggers from step 0, prior will adapt immediately
#
# Expected signals:
#   - Step 5K:  KL begins to climb above 3.2 nat floor as prior adapts
#   - Step 50K: Two clusters in prior (gutenberg vs philosophy) visible in UMAP
#   - Step 100K: sphurattā rate > 5% (gate2.py early check)
#   - Step 300K: H1 gate target ≥ sphurattā_efe > sphurattā_reinforce (ratio < 1.0)
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v5.sh

set -e
cd /home/sharaths/projects/pwm-phase2

# Preserve v4 final checkpoint before v5 may overwrite checkpoints/final.pt
V4_CKPT="checkpoints/final.pt"
V4_DEST="checkpoints/final_v4_seed45_idl.pt"
if [ -f "$V4_CKPT" ] && [ ! -f "$V4_DEST" ]; then
    cp "$V4_CKPT" "$V4_DEST"
    echo "Preserved v4 checkpoint → $V4_DEST ($(du -sh "$V4_DEST" | cut -f1))"
elif [ -f "$V4_DEST" ]; then
    echo "v4 checkpoint already preserved: $V4_DEST"
else
    echo "WARNING: $V4_CKPT not found — v5 will start from Phase 1 WM"
fi

source /home/sharaths/vllm-env/bin/activate

# Resolve warm-start checkpoint: prefer v4 IDL checkpoint, fall back to Phase 1
if [ -f "$V4_DEST" ]; then
    RESUME_CKPT="$(pwd)/$V4_DEST"
    echo "Warm-starting from v4 IDL checkpoint: $RESUME_CKPT"
else
    RESUME_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
    echo "WARNING: v4 checkpoint not found — warm-starting from Phase 1: $RESUME_CKPT"
fi

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$RESUME_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v5 \
  training.max_steps=400000 \
  training.seed=46 \
  2>&1 | tee outputs/phase2_v5.log
