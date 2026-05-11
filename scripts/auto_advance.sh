#!/bin/bash
# Auto-advance pipeline: waits for current phase training to finish,
# runs the gate script, and launches the next phase if the gate passes.
#
# Usage:
#   nohup bash scripts/auto_advance.sh > outputs/auto_advance.log 2>&1 &
#
# Currently wired for Phase 3 → gate3 → Phase 4.
# Re-run for each phase transition.

set -euo pipefail
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

LOG_FILE="outputs/auto_advance.log"
PHASE3_PID_PATTERN="train.py.*phase3_hopfield"
GATE3_RESULT_PATTERN="benchmarks/results/phase_3_gate_step*.json"
PHASE4_LOG="outputs/phase4_nohup.log"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ── Step 1: Wait for Phase 3 training to finish ───────────────────────────────
log "Waiting for Phase 3 training process to finish..."
while pgrep -f "$PHASE3_PID_PATTERN" > /dev/null; do
    log "  Phase 3 still running ($(pgrep -f "$PHASE3_PID_PATTERN" | wc -l) pids)..."
    sleep 120
done
log "Phase 3 training process exited."

# ── Step 2: Verify checkpoint exists ─────────────────────────────────────────
CKPT="checkpoints/final.pt"
if [ ! -f "$CKPT" ]; then
    log "ERROR: $CKPT not found after Phase 3 completed. Aborting."
    exit 1
fi
log "Phase 3 checkpoint found: $CKPT"

# Preserve as final_phase3_seed53.pt for Phase 4 warm-start
PHASE3_FINAL="checkpoints/final_phase3_seed53.pt"
if [ ! -f "$PHASE3_FINAL" ]; then
    cp "$CKPT" "$PHASE3_FINAL"
    log "Preserved Phase 3 checkpoint -> $PHASE3_FINAL"
fi

# ── Step 3: Run gate3 ─────────────────────────────────────────────────────────
log "Running Phase 3 gate (H2 — Hopfield completion)..."
CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate3.py \
    --checkpoint "$CKPT" \
    --device cuda \
    2>&1 | tee outputs/gate3_auto.log
log "gate3 exit code: $?"

# ── Step 4: Read gate result ──────────────────────────────────────────────────
GATE3_JSON=$(ls -t $GATE3_RESULT_PATTERN 2>/dev/null | head -1)
if [ -z "$GATE3_JSON" ]; then
    log "ERROR: No gate3 result JSON found. Check outputs/gate3_auto.log"
    exit 1
fi

H2_PASS=$(python3 -c "import json; d=json.load(open('$GATE3_JSON')); print(d.get('h2_pass', False))")
log "Gate3 result: h2_pass=$H2_PASS (full result: $GATE3_JSON)"

# ── Step 5: Launch Phase 4 if gate passes ─────────────────────────────────────
if [ "$H2_PASS" = "True" ]; then
    log "H2 gate PASSED. Launching Phase 4..."
    nohup bash scripts/launch_phase4.sh > "$PHASE4_LOG" 2>&1 &
    PHASE4_PID=$!
    log "Phase 4 launched (PID=$PHASE4_PID). Monitor: tail -f $PHASE4_LOG"
else
    log "H2 gate FAILED. Phase 4 NOT launched."
    log "Review gate3 output: $GATE3_JSON"
    cat "$GATE3_JSON"
fi

log "Auto-advance complete."
