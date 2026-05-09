#!/bin/bash
# Phase 2 re-run v3: clean restart from Phase 1 WM with all fixes applied.
#
# Context / root-cause analysis:
#   Run 1 (300K steps, seed=42): zero-reward bug — curr_vfe=0.0 hardcoded in
#     _phase_b/_phase_c → ΔF=0 always → actor trained as max-entropy policy only.
#   Run 2 (305K steps, seed=43): prior-entropy fix applied but action collection
#     still stored np.zeros(action_dim) → WM GRU action-conditioning never trained
#     → prior entropy constant regardless of action → reward still ~0.
#
# Fixes applied before this run:
#   Fix 1: Action collection → random one-hot (np.eye(D)[randint(D)]) in both
#     the warm-up loop and the per-step collection in Trainer.train().
#   Fix 2: Entropy computation → per-dimension Cat(32) entropy, NOT single Cat(1024).
#     log_p = log_softmax(logits, dim=-1)  # (B, D, K) per-dim
#     H_total = -(log_p.exp() * log_p).sum(-1).sum(-1).mean()
#   Fix 3: Clean restart using PWM_RESUME_WM_ONLY (WM weights only from Phase 1)
#     instead of PWM_RESUME_CHECKPOINT (full Phase 2 checkpoint with poisoned
#     optimizer state from zero-reward training).
#
# Expected result:
#   With action-conditioned WM and correct entropy metric, prior entropy varies
#   across actions from step 0. Actor should discover first sphurattā within
#   ~20K–50K steps and show clear advantage over REINFORCE by step 100K.
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_rerun.sh

set -e
cd /home/sharaths/projects/pwm-phase2

source /home/sharaths/vllm-env/bin/activate

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY=/home/sharaths/projects/pwm-phase1/checkpoints/final.pt \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe \
  training.max_steps=400000 \
  training.seed=44 \
  2>&1 | tee -a outputs/phase2_rerun3.log
