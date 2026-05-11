#!/bin/bash
# Phase 6: Full System — all ablations, H1–H9, paper data
#
# Prerequisites:
#   - Phase 5 gate PASS (H4 narration rate ≥70% OR H5 reward ≥2×Phase2)
#   - checkpoints/final.pt = Phase 5 final (seed=54)
#
# What changes from Phase 5:
#   - mala_regularisers.enabled=true: āṇava+māyīya+kārma impurity penalties
#   - Full ablation suite A1-A6 (config overrides)
#   - 1M steps, 3 seeds (42, 123, 456)
#
# Exit criteria (H6-H9):
#   H6: Reward entropy > 0.5 nats (non-trivial reward distribution)
#   H7: Phase 6 WM VFE < Phase 3 VFE × 0.85 (15% improvement)
#   H8: Encoder weight norm in [1.0, 50.0] (latent not collapsed/exploded)
#   H9: Mean action entropy > 1.0 nats (diverse policy)
#
# Usage:
#   bash scripts/launch_phase6.sh

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE5_CKPT="checkpoints/final.pt"
if [ ! -f "$PHASE5_CKPT" ]; then
    echo "ERROR: Phase 5 checkpoint not found. Run Phase 5 first."
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

PHASE5_DEST="checkpoints/final_phase5_seed54.pt"
if [ ! -f "$PHASE5_DEST" ]; then
    cp "$PHASE5_CKPT" "$PHASE5_DEST"
    echo "Preserved Phase 5 checkpoint -> $PHASE5_DEST"
fi

echo "=== Phase 6: Full System + Ablations ==="
echo "Warm-starting from Phase 5 final: $PHASE5_CKPT"
echo "mala_regularisers.enabled=true, seed=55, max_steps=1000000"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE5_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase6_full \
  training.max_steps=1000000 \
  training.seed=55 \
  2>&1 | tee outputs/phase6.log
