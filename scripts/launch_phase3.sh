#!/bin/bash
# Phase 3: Hopfield CittaStore + full camatkāra reward (ΔF + ΔI_Hopfield)
#
# Prerequisites:
#   - Phase 2 v7 gate PASS (ratio < 0.5)
#   - checkpoints/final.pt = Phase 2 v7 final (seed=48)
#
# What changes from Phase 2:
#   - memory.enabled=true: CittaStore wired to pancakrtya_loop
#   - reward.alpha_2=0.3: ΔI_Hopfield term active in R_camatk
#   - sphuratta detection now uses BOTH VFE drop AND Hopfield entropy drop
#   - 300K steps (shorter: WM already trained, only memory + camatk updating)
#
# Exit criteria (H2):
#   Pattern completion with Hopfield >= pattern completion without * 1.10
#   Sphuratta rate: 0.5-2 events per 100 steps
#
# Usage:
#   bash scripts/launch_phase3.sh
#   nohup bash scripts/launch_phase3.sh > outputs/phase3_nohup.log 2>&1 &

set -e
cd /home/sharaths/projects/pwm-phase2

V7_CKPT="checkpoints/final.pt"
if [ ! -f "$V7_CKPT" ]; then
    echo "ERROR: Phase 2 v7 checkpoint not found: $V7_CKPT"
    echo "Run Phase 2 v7 first (bash scripts/launch_phase2_v7.sh)"
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

# Preserve Phase 2 final before Phase 3 may overwrite
PHASE2_DEST="checkpoints/final_phase2_v7_seed48.pt"
if [ ! -f "$PHASE2_DEST" ]; then
    cp "$V7_CKPT" "$PHASE2_DEST"
    echo "Preserved Phase 2 v7 checkpoint -> $PHASE2_DEST"
fi

echo "=== Phase 3: Hopfield CittaStore ==="
echo "Warm-starting from Phase 2 v7 final: $V7_CKPT"
echo "memory.enabled=true: CittaStore episodic (beta=4.0) + semantic (beta=0.25)"
echo "reward.alpha_2=0.3: Delta_I_Hopfield term active in R_camatk"
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$V7_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase3_hopfield \
  training.max_steps=300000 \
  training.seed=49 \
  2>&1 | tee outputs/phase3.log
