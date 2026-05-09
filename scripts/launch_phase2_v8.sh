#!/bin/bash
# Phase 2 v8: Fix Layer 7 — WM Encoder Collapse via Gradient Starvation
#
# Seven-layer H1 failure chain — all seven now fixed:
#   Layer 1 (curr_vfe=0 hardcoded)           — commit b78a2bf
#   Layer 2 (zero-action replay)             — commit b78a2bf
#   Layer 3 (passive env, W_a blind)         — DomainSelectiveCachedCorpusEnv
#   Layer 4 (free_bits=1.0 KL floor)         — free_bits=0.1
#   Layer 5 (encoder+prior+W_z collapse)     — Phase 1 warm-start
#   Layer 6 (GRU posterior bypass)           — decoder_z_only=True
#   Layer 7 (gradient starvation collapse)   — THIS VERSION:
#     After IDL converges (cos_sim=-1.0, ~step 300), IDL gradient -> 0.
#     Decoder decays (norm 0.46->0.008 by step 100K) -> reconstruction gradient -> 0.
#     KL stays at free_bits floor -> KL gradient -> 0.
#     All WM gradient sources dry up; weight decay drives encoder to 0 by step 100K.
#     Observed: encoder.0.weight 2.13 (step 50K) -> 0.000 (step 100K).
#     Fix: freeze WM at step 10K. Encoder stays at 2.49 (healthy) forever.
#     Phase 2a (0-10K): train WM+IDL -> W_a=0.52, cos_sim=-1.0
#     Phase 2b (10K-400K): WM frozen; actor/critic learn on frozen imagination.
#
# All v7 fixes preserved:
#   DomainSelectiveCachedCorpusEnv + free_bits=0.1 + Phase 1 warm-start + decoder_z_only
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v8.sh
#   # or background:
#   nohup bash scripts/launch_phase2_v8.sh 2>&1 | tee outputs/phase2_v8.log &

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE1_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 2 v8 (Layer 7 fix: WM freeze at step 10K) ==="
echo "Warm-starting from Phase 1 checkpoint: $PHASE1_CKPT"
echo "  encoder.0.weight norm = 5.47 (healthy)"
echo "  W_a norm              = 0.00 (IDL will train to ~0.52 by step 10K)"
echo ""
echo "Training phases:"
echo "  Steps    0-10K: WM + IDL training (IDL converges at step ~300)"
echo "  Steps 10K-400K: WM FROZEN (encoder stays at 2.49); actor/critic learn"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE1_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v8 \
  training.max_steps=400000 \
  training.seed=49 \
  2>&1 | tee outputs/phase2_v8.log
