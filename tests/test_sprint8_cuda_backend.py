import numpy as np
import pytest
import torch


def test_llama_backend_import():
    from pwm.generation.llama_backend import LlamaCppBackend
    assert LlamaCppBackend is not None


def test_llama_backend_generate_with_bias():
    """logits_processor must shift logit distribution."""
    from pwm.generation.llama_backend import LlamaCppBackend
    backend = LlamaCppBackend(model_path="/dev/null", n_gpu_layers=0, mock=True)
    called = []

    def bias_fn(token_ids, logits):
        called.append(True)
        return logits + 0.1

    result = backend.generate(
        system="You are a poet.",
        user="Write: moon",
        logits_processor=bias_fn,
        max_tokens=5,
    )
    assert isinstance(result, str)
    assert len(called) > 0, "logits_processor must be called"


def test_llama_backend_sse_stream():
    """stream() must yield token strings."""
    from pwm.generation.llama_backend import LlamaCppBackend
    backend = LlamaCppBackend(model_path="/dev/null", n_gpu_layers=0, mock=True)
    tokens = list(backend.stream(
        system="poet",
        user="moon",
        logits_processor=None,
        max_tokens=5,
    ))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_wm_on_cuda():
    """WM must load to CUDA, not CPU."""
    import sys
    sys.path.insert(0, "/home/sharaths/projects/pwm-phase3")
    from pwm.generation.engine import load_wm, DEVICE
    assert str(DEVICE) == "cuda", f"Expected DEVICE=cuda, got {DEVICE}"
    wm = load_wm()
    params = list(wm.parameters())
    assert any(p.device.type == "cuda" for p in params), "No WM params on CUDA"
