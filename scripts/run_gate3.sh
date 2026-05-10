#!/bin/bash
# Run Phase 3 gate evaluation (H2) on the final checkpoint.
#
# H2 pass criterion:
#   Hopfield pattern completion accuracy >= baseline * 1.10 (10% improvement).
#   Secondary: sphurattā rate 0.5–2.0 events per 100 steps over 300 episodes.
#
# Usage:
#   bash scripts/run_gate3.sh                      # uses checkpoints/final.pt
#   bash scripts/run_gate3.sh checkpoints/step_0200000.pt

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

echo "=== Phase 3 H2 Gate ==="
echo "Checkpoint: $CKPT"
echo ""

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate3.py \
    --checkpoint "$CKPT" \
    --device cuda

echo ""
echo "Gate complete. Check benchmarks/results/ for JSON output."
