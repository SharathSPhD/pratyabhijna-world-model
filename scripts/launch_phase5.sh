#!/bin/bash
# Phase 5: LLM vimarsa-bridge + agama-pramana
#
# Prerequisites:
#   - Phase 4 gate PASS
#   - checkpoints/final_phase4_seed50.pt (preserved before Phase 5 start)
#
# What changes from Phase 4:
#   - vimarsa.enabled=true: WM<->LLM cross-attention bridge active
#   - camatkara_narrator wired to sphuratta events
#   - LLM provides narration (jnana-sakti) at high-entropy latent states
#
# Exit criteria (H4 OR H5):
#   H4: >=70% sphuratta events with entropy(z_t) > 0.5 nats
#   H5: mean episode reward >= 2.0 x Phase 2 baseline (2.5302)
#
# Usage:
#   bash scripts/launch_phase5.sh
#   nohup bash scripts/launch_phase5.sh > outputs/phase5_nohup.log 2>&1 &

set -e
cd /home/sharaths/projects/pwm-phase2

# Phase 4 final checkpoint (preserved at gate time)
P4_CKPT="checkpoints/final_phase4_seed50.pt"
if [ ! -f "$P4_CKPT" ]; then
    FALLBACK="checkpoints/final.pt"
    if [ -f "$FALLBACK" ]; then
        cp "$FALLBACK" "$P4_CKPT"
        echo "Preserved Phase 4 checkpoint -> $P4_CKPT"
    else
        echo "ERROR: Phase 4 checkpoint not found: $P4_CKPT"
        echo "Run Phase 4 first (bash scripts/launch_phase4.sh)"
        exit 1
    fi
fi

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 5: LLM Vimarsa Bridge ==="
echo "Warm-starting from Phase 4 final: $P4_CKPT"
echo "vimarsa.enabled=true: WM<->LLM cross-attention bridge active"
echo "camatkara_narrator wired to sphuratta events"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$P4_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase5_llm \
  training.max_steps=500000 \
  training.seed=54 \
  2>&1 | tee outputs/phase5.log
