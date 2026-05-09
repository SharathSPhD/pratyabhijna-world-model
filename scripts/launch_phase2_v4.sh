#!/bin/bash
# Phase 2 re-run v4: clean restart with Imagination Diversity Loss (IDL).
#
# Root cause of H1 FAIL in v1-v3:
#   The CachedCorpusEnv is passive — obs_{t+1} is independent of action_t.
#   Phase A VFE loss therefore drives W_a (action columns of input_proj) → 0.
#   Consequence: different actor actions produce near-identical h_t trajectories,
#   so prior entropy H(p(z|h_t)) is constant regardless of action choice.
#   Result: ΔH = 0, R_camatk = 0, sphurattā = 0 for both EFE and REINFORCE.
#
# Fix: Imagination Diversity Loss (IDL, weight=0.05)
#   For two random distinct one-hot actions (a1, a2), minimise cosine similarity
#   between their imagined h_t outputs from the same init state:
#       L_div = cos_sim(h_t(a1), h_t(a2))
#   This trains W_a to produce orthogonal h_t directions for different actions,
#   making prior entropy action-dependent and giving EFE a signal to exploit.
#
# Strategy:
#   - Start fresh from Phase 1 WM (PWM_RESUME_WM_ONLY)
#   - 400K steps, seed=45
#   - IDL fires from step 0 alongside VFE (weight 0.05 keeps VFE dominant)
#   - Monitor: train/action_cos_sim should fall from ~1.0 → < 0.5 by step 50K
#   - Run gate2.py at step 50K, 100K, 200K checkpoints
#
# Usage:
#   cd /home/sharaths/projects/pwm-phase2
#   bash scripts/launch_phase2_v4.sh

set -e
cd /home/sharaths/projects/pwm-phase2

source /home/sharaths/vllm-env/bin/activate

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY=/home/sharaths/projects/pwm-phase1/checkpoints/final.pt \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase2_efe \
  training.max_steps=400000 \
  training.seed=45 \
  2>&1 | tee outputs/phase2_v4.log
