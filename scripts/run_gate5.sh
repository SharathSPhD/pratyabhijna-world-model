#!/bin/bash
# Run Phase 5 gate evaluation (H4 + H5) on the final checkpoint.
#
# H4 pass criterion:
#   >=70% of sphuratta events (delta > p75) have stochastic-latent
#   entropy > 0.5 nats -- a proxy for "would generate meaningful narration".
#
# H5 pass criterion:
#   Mean episode reward in Phase 5 >= 2.0 x Phase 2 baseline (2.5302).
#
# Gate passes if H4 OR H5 passes.
#
# Usage:
#   bash scripts/run_gate5.sh                      # uses checkpoints/final.pt
#   bash scripts/run_gate5.sh checkpoints/step_0500000.pt

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

echo "=== Phase 5 H4/H5 Gate ==="
echo "Checkpoint: $CKPT"
echo ""

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate5.py \
    --checkpoint "$CKPT" \
    --device cuda

echo ""
echo "Gate complete. Check benchmarks/results/ for JSON output."
