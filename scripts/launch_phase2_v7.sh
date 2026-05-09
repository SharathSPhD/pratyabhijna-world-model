#!/bin/bash
# Phase 2 v7: Fix Layer 6 — decoder uses only z_t (GRU posterior bypass fix)
#
# Six-layer H1 failure chain — all six now fixed:
#   Layer 1 (curr_vfe=0 hardcoded)           — commit b78a2bf
#   Layer 2 (zero-action replay)             — commit b78a2bf
#   Layer 3 (passive env, W_a blind)         — DomainSelectiveCachedCorpusEnv
#   Layer 4 (free_bits=1.0 KL floor)         — free_bits=0.1
#   Layer 5 (encoder+prior+W_z collapse)     — Phase 1 warm-start
#   Layer 6 (GRU posterior bypass)           — THIS VERSION:
#     GRU (norm~2.9) reconstructs o_t from h_{t-1} alone.
#     Decoder learned w_z=0 -> encoder gets zero reconstruction gradient.
#     Encoder decays under weight decay despite healthy Phase 1 init (v6 failure).
#     Fix: decoder receives ONLY z_t (latent_dim=1024 dims).
#     Architecturally impossible for decoder to bypass z_t.
#     Encoder MUST participate in reconstruction -> gradient from step 0.
#
# All v6 fixes preserved:
#   DomainSelectiveCachedCorpusEnv + free_bits=0.1 + Phase 1 warm-start
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v7.sh
#   # or background:
#   nohup bash scripts/launch_phase2_v7.sh > outputs/phase2_v7_nohup.log 2>&1 &

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE1_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 2 v7 (Layer 6 fix: decoder_z_only=True) ==="
echo "Warm-starting from Phase 1 checkpoint: $PHASE1_CKPT"
echo "  encoder.0.weight norm = 5.47 (healthy)"
echo "  prior.0.weight norm   = 4.91 (healthy)"
echo "  W_z norm              = 4.23 (healthy)"
echo "  W_a norm              = 0.00 (IDL will train in ~3700 steps)"
echo ""
echo "Key change from v6:"
echo "  decoder input: (hidden_dim+latent_dim)=1536 -> latent_dim=1024 (z only)"
echo "  Encoder must carry o_t info -> non-zero reconstruction gradient from step 0"
echo ""
echo "Monitoring signals:"
echo "  Step   100: VFE > 0.20 (encoder forced)"
echo "  Step  3700: cos_sim -> -1.000 (IDL re-converges)"
echo "  Step 50000: VFE floor breakout above 0.062"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE1_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v7 \
  training.max_steps=400000 \
  training.seed=48 \
  2>&1 | tee outputs/phase2_v7.log
