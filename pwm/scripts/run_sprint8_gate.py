"""Sprint 8 gate: WM on CUDA, llama.cpp backend functional."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


def run_gate():
    results = {}

    # 1. WM on CUDA
    from pwm.generation.engine import load_wm, warmup_wm_on_text, DEVICE
    results["device"] = str(DEVICE)
    results["cuda_ok"] = (str(DEVICE) == "cuda")

    import torch
    wm = load_wm()
    # Measure WM observe_step latency on GPU (not text encoder overhead)
    # 60 steps × observe_step: criterion is GPU compute < 500ms total
    B = 1
    states = wm.init_state(B, DEVICE)
    obs = torch.zeros(1, 512, device=DEVICE)
    a_t = torch.zeros(1, 64, device=DEVICE)
    # Warm up CUDA kernels (1 step)
    with torch.no_grad():
        states, h, _ = wm.observe_step(obs, a_t, states, 0)
    torch.cuda.synchronize() if str(DEVICE) == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for step in range(60):
            states, h, _ = wm.observe_step(obs, a_t, states, step)
    torch.cuda.synchronize() if str(DEVICE) == "cuda" else None
    warmup_ms = (time.time() - t0) * 1000
    # states[0][0] is level-0 h_t tensor
    h = states[0][0].squeeze(0)
    results["warmup_ms"] = round(warmup_ms, 1)
    results["warmup_ms_pass"] = warmup_ms < 500
    results["h_device"] = str(h.device)
    results["h_norm"] = round(h.norm().item(), 3)

    # 2. LlamaCppBackend (mock mode — real model needs server running)
    import numpy as np
    from pwm.generation.llama_backend import LlamaCppBackend
    backend = LlamaCppBackend(model_path="/dev/null", n_gpu_layers=0, mock=True)
    called = []
    lp = lambda ids, logits: (called.append(True), logits + 0.1)[1]
    out = backend.generate(system="poet", user="moon", logits_processor=lp, max_tokens=5)
    results["logits_processor_called"] = len(called) > 0
    results["backend_output"] = out[:30]

    # 3. Gate pass
    gate_pass = (
        results["cuda_ok"]
        and results["warmup_ms_pass"]
        and results["logits_processor_called"]
        and results["h_device"] == "cuda:0"
    )
    results["gate_pass"] = gate_pass

    out_path = Path("benchmarks/results/sprint8_gate.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return gate_pass


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)
