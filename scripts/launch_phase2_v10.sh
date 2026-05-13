#!/bin/bash
# Phase 2 v10: Action-Consistency Bonus + Percentile Clamping — Fix Layer 9
#
# Nine-layer H1 failure chain — all nine now fixed:
#   Layer 1 (curr_vfe=0 hardcoded)           — commit b78a2bf
#   Layer 2 (zero-action replay)             — commit b78a2bf
#   Layer 3 (passive env, W_a blind)         — DomainSelectiveCachedCorpusEnv
#   Layer 4 (free_bits=1.0 KL floor)         — free_bits=0.1
#   Layer 5 (encoder+prior+W_z collapse)     — Phase 1 warm-start
#   Layer 6 (GRU posterior bypass)           — decoder_z_only=True
#   Layer 7 (gradient starvation collapse)   — WM freeze at step 10K
#   Layer 8 (action-independent proxy)       — domain-affinity reward (v9)
#   Layer 9 (cold-start deadlock)            — THIS VERSION:
#     P(committed|uniform, H=13, 64 actions) = (1/2)^13 ≈ 0.01% → batch always mixed
#     → domain-affinity advantage ≈ 0 → actor PG gradient ≈ 0 → stays uniform.
#     Fix 1: consistency bonus r_t += ±2 for same/different domain consecutive actions
#             (step-level Markov reward; discriminative from step 1)
#     Fix 2: DreamerV3 percentile clamping replaces std normalization
#             (IQR 5–95th ÷ scale.clamp(min=1.0) → advantage.clamp(-1,1))
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v10.sh
#   # or background:
#   nohup bash scripts/launch_phase2_v10.sh 2>&1 | tee outputs/phase2_v10.log &

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE1_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 2 v10 (Layer 9 fix: consistency bonus + percentile clamping) ==="
echo "Warm-starting from Phase 1 checkpoint: $PHASE1_CKPT"
echo ""
echo "Training phases:"
echo "  Steps    0-10K: WM + IDL training (IDL converges at step ~300)"
echo "  Steps 10K-     : WM FROZEN; domain_axis computed from IDL geometry"
echo "  Steps 10K-400K : Actor/critic learn with domain-affinity + consistency rewards"
echo ""
echo "Rewards:"
echo "  r_affinity_t = domain_sign × (h_t · v̂)     (trajectory-level, v9)"
echo "  r_consistency_t = +2 if domain(a_t)==domain(a_{t-1}) else -2  (step-level, v10)"
echo ""
echo "Advantage normalisation (v10):"
echo "  scale = IQR(5–95th percentile).clamp(min=1.0)"
echo "  advantage = (returns - mean) / scale, clamped to [-1, 1]"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE1_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v10 \
  training.max_steps=400000 \
  training.seed=51 \
  2>&1 | tee outputs/phase2_v10.log
