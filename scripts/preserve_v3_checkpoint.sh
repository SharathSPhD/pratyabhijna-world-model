#!/bin/bash
# Preserve v3 (seed=44) final checkpoint before v4 (seed=45) overwrites it.
# Run after v3 completes (checkpoints/final.pt appears) and before v4 finishes.
#
# Usage: bash scripts/preserve_v3_checkpoint.sh
set -e

CKPT_DIR=/home/sharaths/projects/pwm-phase2/checkpoints

if [ ! -f "$CKPT_DIR/final.pt" ]; then
    echo "ERROR: $CKPT_DIR/final.pt not found. Has v3 finished?"
    exit 1
fi

DEST="$CKPT_DIR/final_v3_seed44.pt"
if [ -f "$DEST" ]; then
    echo "Already preserved: $DEST"
else
    cp "$CKPT_DIR/final.pt" "$DEST"
    echo "Preserved: $DEST ($(du -sh "$DEST" | cut -f1))"
fi
