#!/bin/bash
# Run Phase 2 gate evaluation (H1) on the v7 final checkpoint.
# Call at step 200K (interim check) or step 400K (final gate).
#
# Usage:
#   bash scripts/run_gate2_v7.sh                       # uses checkpoints/final.pt
#   bash scripts/run_gate2_v7.sh checkpoints/step_0200000.pt
#
# H1 pass criterion:
#   EFE actor achieves first sphurattā in ≤50% episodes vs REINFORCE baseline.
#   Metric: T_EFE / T_REINFORCE < 1.0 (ratio < 1.0 = EFE faster).

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

echo "=== Phase 2 H1 Gate (v7) ==="
echo "Checkpoint: $CKPT"
echo "Architecture: decoder_z_only=True, free_bits=0.1"
echo ""

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate2.py \
    --checkpoint "$CKPT" \
    --n-eps 200 \
    --device cuda

echo ""
echo "Gate complete. Check benchmarks/results/ for JSON output."
