#!/bin/bash
# Phase 2 v11: Actor PG Fix — Actual Imagination Actions (Layer 10)
#
# Ten-layer H1 failure chain — all ten now fixed:
#   Layer 1 (curr_vfe=0 hardcoded)           — commit b78a2bf
#   Layer 2 (zero-action replay)             — commit b78a2bf
#   Layer 3 (passive env, W_a blind)         — DomainSelectiveCachedCorpusEnv
#   Layer 4 (free_bits=1.0 KL floor)         — free_bits=0.1
#   Layer 5 (encoder+prior+W_z collapse)     — Phase 1 warm-start
#   Layer 6 (GRU posterior bypass)           — decoder_z_only=True
#   Layer 7 (gradient starvation collapse)   — WM freeze at step 10K
#   Layer 8 (action-independent proxy)       — domain-affinity reward (v9)
#   Layer 9 (cold-start deadlock)            — consistency bonus + percentile clamping (v10)
#   Layer 10 (fresh-sample PG bug)           — THIS VERSION:
#     EFEActor.actor_loss computed log_prob = dist.log_prob(dist.sample()) using a fresh
#     sample independent of the imagination actions that generated advantages.
#     With E[A] = 0 (zero-mean normalization): E[∇log π(a_fresh) × A(a_actual)] = 0.
#     Fix: collect act_idx during imagination loop, pass to actor_loss as `actions`,
#     compute log_prob = dist.log_prob(actions) — actual PG gradient restored.
#
# Warm-start: checkpoints/step_0010000.pt (v10 WM-freeze checkpoint, step=10K)
#   - WM is frozen (cosine separation = -1.0); domain_axis recomputed at first step
#   - Actor weights mildly corrupted by 29K steps of noise gradient — not critical,
#     will self-correct once real gradient flows
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v11.sh
#   # or background:
#   nohup bash scripts/launch_phase2_v11.sh 2>&1 | tee outputs/phase2_v11.log &

set -e
cd /home/sharaths/projects/pwm-phase2

# Load WM-only from v10's WM-freeze checkpoint.
# Using PWM_RESUME_WM_ONLY (not RESUME_CHECKPOINT) to get fresh actor+critic Adam state.
# The v10 actor checkpoint had ~zero Adam second moments from 10K steps of near-zero EFE
# gradient — restoring that stale state causes effective lr ≈ lr/eps ≈ 100K× nominal,
# which is why v11 run 1 exploded (actor_loss → -78K) despite the log_prob clamp.
# Fresh Adam state means real PG gradient uses nominal lr=3e-5 from the start.
V10_WM_CKPT="checkpoints/step_0010000.pt"
if [ ! -f "$V10_WM_CKPT" ]; then
    echo "WARN: v10 WM-freeze checkpoint not found: $V10_WM_CKPT"
    echo "Fallback: Phase 1 checkpoint (equivalent WM quality)"
    V10_WM_CKPT="/home/sharaths/projects/pwm-phase1/checkpoints/final.pt"
    if [ ! -f "$V10_WM_CKPT" ]; then
        echo "ERROR: Phase 1 checkpoint not found either"; exit 1
    fi
fi
RESUME_ENV="PWM_RESUME_WM_ONLY=$V10_WM_CKPT"

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 2 v11 (Layer 10 fix: actual imagination actions for log_prob) ==="
echo "WM warm-start from: $V10_WM_CKPT (WM-only; actor+critic fresh for clean Adam)"
echo ""
echo "Key fix:"
echo "  OLD: log_prob = dist.log_prob(dist.sample())  ← fresh sample ⊥ advantage → zero PG"
echo "  NEW: log_prob = dist.log_prob(actions)         ← actual actions → real PG gradient"
echo ""
echo "Training resumes from step 10K → 400K (390K actor/critic steps)"
echo "Domain axis recomputed at first train_step (frozen WM in checkpoint)"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
eval "$RESUME_ENV" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe_v11 \
  training.max_steps=400000 \
  training.seed=52 \
  2>&1 | tee outputs/phase2_v11.log
