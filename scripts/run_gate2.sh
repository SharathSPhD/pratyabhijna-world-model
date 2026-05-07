#!/bin/bash
# Run Phase 2 gate2.py on a checkpoint — saves result to benchmarks/results/
#
# Usage:
#   bash scripts/run_gate2.sh [checkpoint] [n_eps]
#   bash scripts/run_gate2.sh checkpoints/step_0050000.pt 100
#   bash scripts/run_gate2.sh checkpoints/final.pt 200      (full gate)
#
# If checkpoint not specified, uses checkpoints/final.pt
# If n_eps not specified, uses 100 (faster early reads)

set -e
cd /home/sharaths/projects/pwm-phase2
source /home/sharaths/vllm-env/bin/activate

CKPT="${1:-checkpoints/final.pt}"
N_EPS="${2:-100}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

# Derive step number for output filename
STEP=$(python - <<EOF
import torch, sys
try:
    ckpt = torch.load("$CKPT", map_location="cpu", weights_only=False)
    print(ckpt.get("step", -1))
except:
    print("unknown")
EOF
)

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
OUTFILE="benchmarks/results/phase_2_gate_v6_step${STEP}_${N_EPS}eps_${TIMESTAMP}.json"

echo "=== Phase 2 gate2.py ==="
echo "Checkpoint: $CKPT (step=$STEP)"
echo "N_eps: $N_EPS"
echo "Output: $OUTFILE"
echo ""

CORPUS_CACHE_DIR=/home/sharaths/projects/pwm-phase1/data/embed_cache \
/home/sharaths/vllm-env/bin/python pwm/scripts/gate2.py \
    --checkpoint "$CKPT" \
    --n-eps "$N_EPS" \
    --device cuda \
    2>&1 | tee "$OUTFILE.log"

# Extract JSON from log (last {...} block)
python - <<PYEOF
import sys, json, re

with open("$OUTFILE.log") as f:
    content = f.read()

# Find the last JSON block
matches = re.findall(r'\{[^{}]*"phase"[^{}]*\}', content, re.DOTALL)
if matches:
    result = json.loads(matches[-1])
    with open("$OUTFILE", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Gate result saved: $OUTFILE")
    print(f"H1 status: {result.get('h1', {}).get('status', 'N/A')}")
    print(f"Ratio: {result.get('h1', {}).get('ratio', 'N/A')}")
else:
    # Save full log as fallback
    import shutil
    shutil.copy("$OUTFILE.log", "$OUTFILE")
    print("Could not extract JSON, saved full log instead.")
PYEOF
