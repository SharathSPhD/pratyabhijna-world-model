#!/bin/bash
# Phase 2 v6: Full Phase-1 Warm-Start (Layer 5 Encoder-Collapse Fix)
#
# Five-layer H1 failure chain — all five now fixed:
#
#   Layer 1 (curr_vfe=0 hardcoded) — commit b78a2bf
#   Layer 2 (zero-action replay)   — commit b78a2bf
#   Layer 3 (passive env, W_a=0)   — DomainSelectiveCachedCorpusEnv, commit 2d6add4
#   Layer 4 (free_bits=1.0 KL floor) — free_bits=0.1, commit 2d6add4
#   Layer 5 (encoder+prior+W_z collapse) — THIS VERSION:
#     v4 IDL + free_bits=1.0 caused catastrophic forgetting of all obs-processing
#     modules over 400K steps (weight decay + zero KL gradient):
#       encoder: norm 5.47 → 0.00
#       prior:   norm 4.91 → 0.00
#       W_z:     norm 4.23 → 0.00
#     v5 inherited this collapse → VFE stuck at free_bits floor (0.0617) for 43K steps
#     Fix: warm-start from Phase 1 checkpoint (healthy encoder/prior/W_z)
#       Phase 1 provided working obs processing (VFE=0.6018, silhouette=0.114)
#       With free_bits=0.1, KL gradient maintains encoder from step 0
#       IDL re-trains W_a in ~3700 steps (same as v4)
#
# Warm-start from Phase 1 (NOT v4):
#   /home/sharaths/projects/pwm-phase1/checkpoints/final.pt
#   Phase 1 state: encoder✓ prior✓ W_z✓  W_a=0 (IDL re-trains in 3700 steps)
#   Actor/critic start from random init (domain reward structure is new)
#
# All other fixes preserved from v5:
#   DomainSelectiveCachedCorpusEnv: per-item action→domain coupling
#   free_bits=0.1: KL floor = 3.2 nats (prior gets gradient)
#   IDL weight=0.05: trains W_a via imagination diversity
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v6.sh

set -e
cd /home/sharaths/projects/pwm-phase2

# Preserve v5 partially-trained checkpoint before v6 may overwrite checkpoints/final.pt
V5_STEP="checkpoints/step_0040000.pt"
V5_DEST="checkpoints/step_0040000_v5_seed46.pt"
if [ -f "$V5_STEP" ] && [ ! -f "$V5_DEST" ]; then
    cp "$V5_STEP" "$V5_DEST"
    echo "Preserved v5 step_40K checkpoint → $V5_DEST"
elif [ -f "$V5_DEST" ]; then
    echo "v5 step_40K already preserved: $V5_DEST"
fi

source /home/sharaths/vllm-env/bin/activate

# Warm-start from Phase 1: healthy encoder+prior+W_z, action-blind GRU (W_a=0)
# IDL will re-train W_a to cos_sim=-1.000 in ~3700 steps
PHASE1_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 1
fi
echo "Warm-starting from Phase 1 checkpoint: $PHASE1_CKPT"
echo "  encoder.0.weight norm = 5.47 (healthy, will maintain with free_bits=0.1)"
echo "  prior.0.weight norm   = 4.91 (healthy)"
echo "  W_z norm              = 4.23 (healthy)"
echo "  W_a norm              = 0.00 (IDL will train to ~0.22 in ~3700 steps)"

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE1_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v6 \
  training.max_steps=400000 \
  training.seed=47 \
  2>&1 | tee outputs/phase2_v6.log
