#!/bin/bash
# Run Phase 4 gate evaluation (H3) on the final checkpoint.
#
# H3 pass criterion:
#   Sequential forgetting rate WITH sleep < 0.8 x rate WITHOUT sleep
#   (>= 20% reduction in catastrophic forgetting across a 3-domain sequence).
#
# Usage:
#   bash scripts/run_gate4.sh                      # uses checkpoints/final.pt
#   bash scripts/run_gate4.sh checkpoints/step_0200000.pt

set -e
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

CKPT="${1:-checkpoints/final.pt}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "Available checkpoints:"
    ls checkpoints/*.pt 2>/dev/null | sort
    exit 1
fi

echo "=== Phase 4 H3 Gate ==="
echo "Checkpoint: $CKPT"
echo ""

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate4.py \
    --checkpoint "$CKPT" \
    --device cuda

echo ""
echo "Gate complete. Check benchmarks/results/ for JSON output."
