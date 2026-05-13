#!/bin/bash
# Run Phase 6 gate evaluation (H6-H9) on the final checkpoint.
#
# H6 pass: reward entropy > 0.5 nats over 500 episodes.
# H7 pass: Phase 6 imagination VFE < Phase 3 VFE x 0.85 (skipped if P3 ckpt missing).
# H8 pass: encoder weight norm in [1.0, 50.0].
# H9 pass: mean action entropy > 1.0 nats over 200 episodes.
#
# Gate passes only if ALL four hypotheses pass.
#
# Usage:
#   bash scripts/run_gate6.sh                                    # uses checkpoints/final.pt
#   bash scripts/run_gate6.sh checkpoints/final.pt checkpoints/final_phase3_seed53.pt

set -e
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

CKPT="${1:-checkpoints/final.pt}"
P3_CKPT="${2:-checkpoints/final_phase3_seed53.pt}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "Available checkpoints:"
    ls checkpoints/*.pt 2>/dev/null | sort
    exit 1
fi

echo "=== Phase 6 H6-H9 Gate ==="
echo "Checkpoint:        $CKPT"
echo "Phase 3 reference: $P3_CKPT"
echo ""

P3_ARG=""
if [ -f "$P3_CKPT" ]; then
    P3_ARG="--phase3-checkpoint $P3_CKPT"
else
    echo "Note: Phase 3 reference checkpoint not found; H7 will be skipped."
fi

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate6.py \
    --checkpoint "$CKPT" \
    $P3_ARG \
    --device cuda

echo ""
echo "Gate complete. Check benchmarks/results/ for JSON output."
