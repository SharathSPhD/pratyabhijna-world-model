#!/bin/bash
# Phase 6: Full pancakrtya system + all ablations
#
# Prerequisites:
#   - Phase 5 gate PASS (H4 OR H5)
#   - checkpoints/final_phase5_seed54.pt (preserved before Phase 6 start)
#
# What changes from Phase 5:
#   - All five sakti acts integrated: srsti, sthiti, samhara, tirodhana, anugraha
#   - Full reward stack: alpha_1*delta_F + alpha_2*delta_I_Hopfield + alpha_3*empowerment
#   - Mala regularisers active (anava, mayiya, karma)
#   - Sleep consolidation, vimarsa bridge, skill library all online
#
# Exit criteria (H6 AND H7 AND H8 AND H9):
#   H6: reward entropy > 0.5 nats (non-trivial distribution)
#   H7: imagination VFE < Phase 3 VFE x 0.85
#   H8: encoder weight norm in [1.0, 50.0]
#   H9: action entropy > 1.0 nats (diverse policy)
#
# Usage:
#   bash scripts/launch_phase6.sh
#   nohup bash scripts/launch_phase6.sh > outputs/phase6_nohup.log 2>&1 &

set -e
cd /home/sharaths/projects/pwm-phase2

# Phase 5 final checkpoint (preserved at gate time)
P5_CKPT="checkpoints/final_phase5_seed54.pt"
if [ ! -f "$P5_CKPT" ]; then
    FALLBACK="checkpoints/final.pt"
    if [ -f "$FALLBACK" ]; then
        cp "$FALLBACK" "$P5_CKPT"
        echo "Preserved Phase 5 checkpoint -> $P5_CKPT"
    else
        echo "ERROR: Phase 5 checkpoint not found: $P5_CKPT"
        echo "Run Phase 5 first (bash scripts/launch_phase5.sh)"
        exit 1
    fi
fi

source /home/sharaths/vllm-env/bin/activate

echo "=== Phase 6: Full Pancakrtya System ==="
echo "Warm-starting from Phase 5 final: $P5_CKPT"
echo "All five sakti acts integrated; full reward stack active."
echo ""

mkdir -p outputs

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
WANDB_PROJECT=pratyabhijna-world-model \
PWM_RESUME_WM_ONLY="$P5_CKPT" \
/home/sharaths/vllm-env/bin/python pwm/scripts/train.py \
  --config-name phase6_full \
  training.max_steps=1000000 \
  training.seed=55 \
  2>&1 | tee outputs/phase6.log
