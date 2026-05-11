#!/bin/bash
# Run Phase 6 gate evaluation (H6-H9) on the final checkpoint.
#
# H6-H9 pass criteria:
#   H6: Reward entropy > 0.5 nats (non-trivial reward distribution)
#   H7: Phase 6 VFE < Phase 3 VFE × 0.85 (15% 3-level improvement)
#   H8: Encoder weight norm in [1.0, 50.0] (no latent collapse/explosion)
#   H9: Action entropy > 1.0 nats (diverse policy)
#
# Usage:
#   bash scripts/run_gate6.sh                      # uses checkpoints/final.pt
#   bash scripts/run_gate6.sh checkpoints/step_0800000.pt

set -e
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

CKPT="${1:-checkpoints/final.pt}"
PHASE3_CKPT="checkpoints/final_phase3_seed53.pt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "Available checkpoints:"
    ls checkpoints/*.pt 2>/dev/null | sort
    exit 1
fi

echo "=== Phase 6 H6-H9 Gate ==="
echo "Checkpoint: $CKPT"
echo "Phase 3 ref: $PHASE3_CKPT"
echo ""

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate6.py \
    --checkpoint "$CKPT" \
    --phase3-checkpoint "$PHASE3_CKPT"

echo ""
echo "Gate complete. Check benchmarks/results/ for JSON output."
