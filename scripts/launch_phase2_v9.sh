#!/bin/bash
# Phase 2 v9: Domain-Affinity Reward — Fix Layer 8 (action-independent proxy)
#
# Eight-layer H1 failure chain — all eight now fixed:
#   Layer 1 (curr_vfe=0 hardcoded)           — commit b78a2bf
#   Layer 2 (zero-action replay)             — commit b78a2bf
#   Layer 3 (passive env, W_a blind)         — DomainSelectiveCachedCorpusEnv
#   Layer 4 (free_bits=1.0 KL floor)         — free_bits=0.1
#   Layer 5 (encoder+prior+W_z collapse)     — Phase 1 warm-start
#   Layer 6 (GRU posterior bypass)           — decoder_z_only=True
#   Layer 7 (gradient starvation collapse)   — WM freeze at step 10K
#   Layer 8 (action-independent proxy)       — THIS VERSION:
#     ||h_t||₂ grows from 4.5→18.6 identically for ANY action (GRU init dynamics).
#     ΔActivation positive 94.7% of steps regardless of actor decision.
#     Advantage normalization zeros out signal → EFE entropy = 5.99/6.00 bits.
#     Gate: EFE=0/200, RF=1/200, ratio=1.004 → FAIL.
#     Fix: domain-affinity reward r_t = domain_sign × (h_t · v̂)
#          v̂ = IDL axis normalize(h_guten − h_philo), pre-computed at WM freeze.
#          Committed trajectories → monotonically growing positive reward.
#          Mixed trajectories (REINFORCE 50/50) → mean reward ≈ 0.
#     Gate v9: REINFORCE-calibrated fixed threshold; H=15→32; pass criterion 0.75.
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v9.sh
#   # or background:
#   nohup bash scripts/launch_phase2_v9.sh 2>&1 | tee outputs/phase2_v9.log &

set -e
cd /home/sharaths/projects/pwm-phase2

PHASE1_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 2 v9 (Layer 8 fix: domain-affinity reward) ==="
echo "Warm-starting from Phase 1 checkpoint: $PHASE1_CKPT"
echo ""
echo "Training phases:"
echo "  Steps    0-10K: WM + IDL training (IDL converges at step ~300)"
echo "  Steps 10K-     : WM FROZEN; domain_axis computed from IDL geometry"
echo "  Steps 10K-400K : Actor/critic learn with domain-affinity reward"
echo ""
echo "Domain-affinity reward:"
echo "  r_t = domain_sign × (h_t · v̂)"
echo "  v̂   = normalize(h_proto_guten − h_proto_philo)"
echo "  Committed trajectories → r_t grows monotonically"
echo "  REINFORCE (50/50 mix) → mean r_t ≈ 0"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$PHASE1_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v9 \
  training.max_steps=400000 \
  training.seed=50 \
  2>&1 | tee outputs/phase2_v9.log
