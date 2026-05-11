#!/bin/bash
# Auto-advance: Phase 6 completes → gate6 → update paper with final results.
# Run this AFTER Phase 6 training has started.
#
# Usage:
#   nohup bash scripts/auto_advance_phase6.sh > outputs/auto_advance_phase6.log 2>&1 &

set -euo pipefail
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

PHASE6_PID_PATTERN="train.py.*phase6_full"
GATE6_RESULT_PATTERN="benchmarks/results/phase_6_gate_step*.json"
PAPER_DIR="/home/sharaths/projects/PWM/paper"
P3_CKPT="checkpoints/final_phase3_seed53.pt"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "Waiting for Phase 6 training to finish..."
while pgrep -f "$PHASE6_PID_PATTERN" > /dev/null; do
    log "  Phase 6 still running..."
    sleep 120
done
log "Phase 6 training exited."

CKPT="checkpoints/final.pt"
[ -f "$CKPT" ] || { log "ERROR: $CKPT missing. Aborting."; exit 1; }

PHASE6_FINAL="checkpoints/final_phase6_seed55.pt"
[ -f "$PHASE6_FINAL" ] || { cp "$CKPT" "$PHASE6_FINAL"; log "Preserved -> $PHASE6_FINAL"; }

log "Running Phase 6 gate (H6-H9 ablations)..."
P3_ARG=""
[ -f "$P3_CKPT" ] && P3_ARG="--phase3-checkpoint $P3_CKPT"

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate6.py \
    --checkpoint "$CKPT" \
    $P3_ARG \
    --device cuda \
    2>&1 | tee outputs/gate6_auto.log
log "gate6 exit code: $?"

GATE6_JSON=$(ls -t $GATE6_RESULT_PATTERN 2>/dev/null | head -1)
[ -n "$GATE6_JSON" ] || { log "ERROR: No gate6 result JSON. Check outputs/gate6_auto.log"; exit 1; }

ALL_PASS=$(python3 -c "import json; d=json.load(open('$GATE6_JSON')); print(d.get('h_all_pass', False))")
log "Gate6 result: h_all_pass=$ALL_PASS ($GATE6_JSON)"

if [ "$ALL_PASS" = "True" ]; then
    log "=== ALL GATES PASSED (H2-H9). COLLECTING FINAL RESULTS ==="
    # Print summary of all gate results for paper update
    python3 - <<'PYEOF'
import json, glob, os
results_dir = "benchmarks/results"
print("\n=== Final Gate Results Summary ===")
for phase in [3, 4, 5, 6]:
    pattern = f"{results_dir}/phase_{phase}_gate_step*.json"
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        d = json.load(open(latest))
        print(f"\nPhase {phase}: {os.path.basename(latest)}")
        for k, v in d.items():
            if isinstance(v, (bool, int, float, str)):
                print(f"  {k}: {v}")
PYEOF
    log "Results printed above. Update paper §5.3-§5.6 and abstract with these values."
    log "Then rebuild PDF and submit to arXiv."
else
    log "Gate6 FAILED (h_all_pass=False). Check individual H6-H9 criteria."
    python3 -c "import json; d=json.load(open('$GATE6_JSON')); [print(f'  {k}: {v}') for k,v in d.items() if isinstance(v,(bool,float,int))]"
fi

log "Phase 6 auto-advance complete."
