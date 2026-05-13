#!/bin/bash
# Phase 4: Sleep Consolidation (NREM + REM) + H3 sequential forgetting
#
# Prerequisites:
#   - Phase 3 gate PASS (Hopfield +10% completion accuracy)
#   - checkpoints/final.pt = Phase 3 final (seed=49)
#
# What changes from Phase 3:
#   - sleep.enabled=true: NREM replay every 10K steps + REM dreaming
#   - sleep.nrem_steps=200: replay steps per NREM cycle
#   - sleep.rem_steps=50: dreaming steps per REM cycle
#   - ThermSleepBudget: stops sleep when efficiency < threshold
#
# Exit criteria (H3):
#   Sequential forgetting rate WITH sleep < 0.8 * rate WITHOUT sleep
#   (>= 20% reduction in catastrophic forgetting across 3 sequential domains)
#
# Usage:
#   bash scripts/launch_phase4.sh

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE3_CKPT="checkpoints/final.pt"
if [ ! -f "$PHASE3_CKPT" ]; then
    echo "ERROR: Phase 3 checkpoint not found. Run Phase 3 first."
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

PHASE3_DEST="checkpoints/final_phase3_seed53.pt"
if [ ! -f "$PHASE3_DEST" ]; then
    cp "$PHASE3_CKPT" "$PHASE3_DEST"
    echo "Preserved Phase 3 checkpoint -> $PHASE3_DEST"
fi

echo "=== Phase 4: Sleep Consolidation ==="
echo "Warm-starting from Phase 3 final: $PHASE3_CKPT"
echo "sleep.enabled=true: NREM every 10K steps + REM dreaming"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE3_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase4_sleep \
  training.max_steps=300000 \
  training.seed=50 \
  2>&1 | tee outputs/phase4.log
