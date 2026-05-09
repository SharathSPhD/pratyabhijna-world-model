#!/bin/bash
# Probe encoder/prior/W_z norms in a Phase 2 checkpoint (Layer 6 health check).
# Safe to run while training is live — reads checkpoint file only.
#
# Usage:
#   bash scripts/probe_encoder_norms.sh checkpoints/step_0010000.pt
#   bash scripts/probe_encoder_norms.sh checkpoints/final.pt

set -e
CKPT="${1:-checkpoints/final.pt}"
cd /home/sharaths/projects/pwm-phase2

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

source /home/sharaths/vllm-env/bin/activate

/home/sharaths/vllm-env/bin/python - <<PYEOF
import torch, sys
ckpt = torch.load("$CKPT", map_location="cpu", weights_only=False)
sd = ckpt.get("world_model", {})
step = ckpt.get("step", -1)

keys = [
    ("encoder.0.weight",   "levels.0.encoder.0.weight"),
    ("prior.0.weight",     "levels.0.prior.0.weight"),
    ("W_z (input_proj)",   "levels.0.input_proj.weight"),
    ("decoder.0.weight",   "levels.0.decoder.0.weight"),
    ("GRU weight_ih_l0",   "levels.0.backbone.weight_ih_l0"),
]

print(f"=== Encoder norms at step={step} ===")
all_ok = True
for label, key in keys:
    t = sd.get(key)
    if t is not None:
        norm = t.norm().item()
        # Encoder collapse threshold: norm < 0.05
        status = "OK" if norm > 0.05 else "COLLAPSED!"
        if norm < 0.05:
            all_ok = False
        print(f"  {label:<25} norm={norm:.4f}  [{status}]")
    else:
        # decoder key changes shape in decoder_z_only — may be stored differently
        alt = [k for k in sd if key.split(".")[-2] in k and "decoder" in k]
        if alt:
            t2 = sd[alt[0]]
            print(f"  {label:<25} norm={t2.norm().item():.4f}  (key={alt[0]})")
        else:
            print(f"  {label:<25} NOT FOUND in checkpoint")

W_a = sd.get("levels.0.input_proj.weight")
if W_a is not None:
    # input_proj: (hidden_dim, latent_dim+action_dim) — W_a is the action columns
    latent_dim = 1024  # 32*32 for Cat(32x32)
    W_action = W_a[:, latent_dim:]
    print(f"  {'W_a (action cols)':<25} norm={W_action.norm().item():.4f}")

print()
if all_ok:
    print("Verdict: Encoder HEALTHY — no Layer 6 collapse detected.")
else:
    print("Verdict: COLLAPSE DETECTED — encoder weights near zero!")
PYEOF
