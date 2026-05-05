#!/bin/bash
# Phase 2 re-run: 100K more steps with fixed camatkāra reward signal.
#
# Context:
#   The initial 300K-step run trained the actor with zero reward (bug: curr_vfe=0.0
#   in imagination → ΔF=0 always). The actor learned a max-entropy policy (useful
#   exploration prior). This re-run continues from that checkpoint with the fix applied:
#   _last_real_vfe is now cached in Phase A and passed to Phase B/C imagination.
#
# With real ΔF rewards, the EFE actor should learn to navigate toward novel WM states
#   within ~10K–30K steps (reward signal is now non-zero and correlated with action).
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_rerun.sh

set -e
cd /home/sharaths/projects/pwm-phase2

source /home/sharaths/vllm-env/bin/activate

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_CHECKPOINT=/home/sharaths/projects/pwm-phase2/checkpoints/final.pt \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe \
  training.max_steps=400000 \
  training.seed=43 \
  2>&1 | tee -a outputs/phase2_rerun.log
