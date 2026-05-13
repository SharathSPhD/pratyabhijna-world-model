#!/bin/bash
# Auto-advance: Phase 4 completes → gate4 → launch Phase 5 if H3 passes.
# Run this AFTER Phase 4 training has started.
#
# Usage:
#   nohup bash scripts/auto_advance_phase4.sh > outputs/auto_advance_phase4.log 2>&1 &

set -euo pipefail
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

PHASE4_PID_PATTERN="train.py.*phase4_sleep"
GATE4_RESULT_PATTERN="benchmarks/results/phase_4_gate_step*.json"
PHASE5_LOG="outputs/phase5_nohup.log"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "Waiting for Phase 4 training to finish..."
while pgrep -f "$PHASE4_PID_PATTERN" > /dev/null; do
    log "  Phase 4 still running..."
    sleep 120
done
log "Phase 4 training exited."

CKPT="checkpoints/final.pt"
[ -f "$CKPT" ] || { log "ERROR: $CKPT missing. Aborting."; exit 1; }

PHASE4_FINAL="checkpoints/final_phase4_seed50.pt"
[ -f "$PHASE4_FINAL" ] || { cp "$CKPT" "$PHASE4_FINAL"; log "Preserved -> $PHASE4_FINAL"; }

log "Running Phase 4 gate (H3 — sleep forgetting)..."
CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate4.py \
    --checkpoint "$CKPT" \
    --device cuda \
    2>&1 | tee outputs/gate4_auto.log
log "gate4 exit code: $?"

GATE4_JSON=$(ls -t $GATE4_RESULT_PATTERN 2>/dev/null | head -1)
[ -n "$GATE4_JSON" ] || { log "ERROR: No gate4 result JSON. Check outputs/gate4_auto.log"; exit 1; }

H3_PASS=$(python3 -c "import json; d=json.load(open('$GATE4_JSON')); print(d.get('h3_pass', False))")
log "Gate4 result: h3_pass=$H3_PASS ($GATE4_JSON)"

if [ "$H3_PASS" = "True" ]; then
    log "H3 gate PASSED. Launching Phase 5..."
    nohup bash scripts/launch_phase5.sh > "$PHASE5_LOG" 2>&1 &
    log "Phase 5 launched (PID=$!). Monitor: tail -f $PHASE5_LOG"
    # Immediately start Phase 5 auto-advance watcher
    nohup bash scripts/auto_advance_phase5.sh > outputs/auto_advance_phase5.log 2>&1 &
    log "Phase 5 auto-advance watcher started (PID=$!)."
else
    log "H3 gate FAILED. Phase 5 NOT launched."
    cat "$GATE4_JSON"
fi

log "Phase 4 auto-advance complete."
