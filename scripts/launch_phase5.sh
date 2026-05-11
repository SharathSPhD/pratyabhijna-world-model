#!/bin/bash
# Phase 5: LLM Āgama + Vimarśa Bridge
#
# Prerequisites:
#   - Phase 4 gate PASS (sleep consolidation ≥20% forgetting reduction)
#   - checkpoints/final.pt = Phase 4 final (seed=50)
#
# What changes from Phase 4:
#   - llm.enabled=true: LLM narrates sphurattā events
#   - world_model.levels=3: full Trika hierarchy (Aparā+Parāparā+Parā)
#   - VimarsaBridge: WM ↔ LLM cross-attention active
#   - AWM proposals every 500 steps
#
# Exit criteria (H4+H5):
#   H4: ≥70% of sphurattā events have high-entropy latents (narration proxy)
#   H5: Phase 5 mean reward ≥ 2× Phase 2 baseline (2.530)
#
# Usage:
#   bash scripts/launch_phase5.sh

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE4_CKPT="checkpoints/final.pt"
if [ ! -f "$PHASE4_CKPT" ]; then
    echo "ERROR: Phase 4 checkpoint not found. Run Phase 4 first."
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

PHASE4_DEST="checkpoints/final_phase4_seed50.pt"
if [ ! -f "$PHASE4_DEST" ]; then
    cp "$PHASE4_CKPT" "$PHASE4_DEST"
    echo "Preserved Phase 4 checkpoint -> $PHASE4_DEST"
fi

echo "=== Phase 5: LLM Āgama + Vimarśa Bridge ==="
echo "Warm-starting from Phase 4 final: $PHASE4_CKPT"
echo "llm.enabled=true, 3-level Trika, seed=54"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE4_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase5_llm \
  training.max_steps=500000 \
  training.seed=54 \
  2>&1 | tee outputs/phase5.log
