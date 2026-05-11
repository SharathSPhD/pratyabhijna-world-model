#!/bin/bash
# Auto-advance: Phase 5 completes → gate5 → launch Phase 6 if H4 or H5 passes.
# Run this AFTER Phase 5 training has started.
#
# Usage:
#   nohup bash scripts/auto_advance_phase5.sh > outputs/auto_advance_phase5.log 2>&1 &

set -euo pipefail
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

PHASE5_PID_PATTERN="train.py.*phase5_llm"
GATE5_RESULT_PATTERN="benchmarks/results/phase_5_gate_step*.json"
PHASE6_LOG="outputs/phase6_nohup.log"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "Waiting for Phase 5 training to finish..."
while pgrep -f "$PHASE5_PID_PATTERN" > /dev/null; do
    log "  Phase 5 still running..."
    sleep 120
done
log "Phase 5 training exited."

CKPT="checkpoints/final.pt"
[ -f "$CKPT" ] || { log "ERROR: $CKPT missing. Aborting."; exit 1; }

PHASE5_FINAL="checkpoints/final_phase5_seed54.pt"
[ -f "$PHASE5_FINAL" ] || { cp "$CKPT" "$PHASE5_FINAL"; log "Preserved -> $PHASE5_FINAL"; }

log "Running Phase 5 gate (H4/H5 — vimarsa bridge)..."
CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate5.py \
    --checkpoint "$CKPT" \
    --device cuda \
    2>&1 | tee outputs/gate5_auto.log
log "gate5 exit code: $?"

GATE5_JSON=$(ls -t $GATE5_RESULT_PATTERN 2>/dev/null | head -1)
[ -n "$GATE5_JSON" ] || { log "ERROR: No gate5 result JSON. Check outputs/gate5_auto.log"; exit 1; }

GATE5_PASS=$(python3 -c "import json; d=json.load(open('$GATE5_JSON')); print(d.get('gate_pass', False))")
log "Gate5 result: gate_pass=$GATE5_PASS ($GATE5_JSON)"

if [ "$GATE5_PASS" = "True" ]; then
    log "Gate5 PASSED. Launching Phase 6..."
    nohup bash scripts/launch_phase6.sh > "$PHASE6_LOG" 2>&1 &
    log "Phase 6 launched (PID=$!). Monitor: tail -f $PHASE6_LOG"
    nohup bash scripts/auto_advance_phase6.sh > outputs/auto_advance_phase6.log 2>&1 &
    log "Phase 6 auto-advance watcher started (PID=$!)."
else
    log "Gate5 FAILED. Phase 6 NOT launched."
    cat "$GATE5_JSON"
fi

log "Phase 5 auto-advance complete."
