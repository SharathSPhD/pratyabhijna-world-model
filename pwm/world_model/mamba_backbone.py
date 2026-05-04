"""
MambaBackbone: Mamba-2 SSM backbone with GRU fallback.

Philosophical grounding:
  Spanda (SpandaK 1.1): The SSM's selective scan is a mathematical formalisation
  of the Spanda doctrine — continuous pulsation with selective attention to what
  matters, ignoring the irrelevant. Mamba's selective state-space model learns
  which historical context to retain and which to discard (Δ, B, C gates).

  The GRU fallback preserves the research prototype invariant: the same model
  runs on CPU (tests, analysis) and GPU (training) without code changes.

Architecture:
  Mamba-2 (Dao & Gu 2024, Structured State-Space Dual form):
    d_model=512, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=256
    rmsnorm=True (layer norm inside Mamba block)

  Training mode: parallel SSD scan — O(T log T) vs O(T²) for attention
  Inference mode: recurrent step with InferenceParams — O(1) per step

  Hardware note: Mamba-2 CUDA kernels require GPU tensors (causal_conv1d).
  When input is on CPU (or mamba_ssm not installed), routes to GRUFallback
  transparently. Validated on DGX Spark GB10 Blackwell, CUDA 13.0.

Adapted from: dreamprice/src/retail_world_model/models/mamba_backbone.py
  — same hardware, same CUDA version, production-validated kernel config.
"""

from __future__ import annotations
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2                                 # type: ignore[import]
    from mamba_ssm.utils.generation import InferenceParams       # type: ignore[import]
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class GRUFallback(nn.Module):
    """GRU fallback — CPU-safe, no CUDA kernel required."""

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self._hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model). Parallel sequence."""
        out, _ = self.gru(x)
        return out

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, d_model) → (B, d_model). Single recurrent step."""
        out, self._hidden = self.gru(x_t.unsqueeze(1), self._hidden)
        return out.squeeze(1)

    def reset_state(self) -> None:
        self._hidden = None


class MambaBackbone(nn.Module):
    """
    Mamba-2 sequence backbone for TrikaCoreLevel.

    Forward (training): parallel SSD scan over full sequence.
    Step (inference): O(1) recurrent mode using InferenceParams.

    Kernel alignment: `.contiguous()` is called before step() to ensure
    memory layout is aligned for causal_conv1d (required on Blackwell).
    """

    # Validated config from DreamPrice on this DGX Spark
    DEFAULT_CFG = dict(
        d_model=512,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        chunk_size=256,
    )

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self._has_mamba = HAS_MAMBA

        if HAS_MAMBA:
            self.mamba = Mamba2(  # type: ignore[possibly-unbound]
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                rmsnorm=True,
                layer_idx=0,
            )

        self.gru_fallback = GRUFallback(d_model=d_model)

    def _use_mamba(self, x: torch.Tensor) -> bool:
        return self._has_mamba and x.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model). Training: parallel SSD scan."""
        if self._use_mamba(x):
            return self.mamba(x)  # type: ignore[no-any-return]
        return self.gru_fallback(x)

    def step(
        self,
        x_t: torch.Tensor,
        inference_params: object | None = None,
    ) -> torch.Tensor:
        """x_t: (B, d_model) → (B, d_model). Inference: single recurrent step."""
        if self._use_mamba(x_t):
            out = self.mamba(  # type: ignore[no-any-return]
                x_t.unsqueeze(1).contiguous(),  # align strides for causal_conv1d
                inference_params=inference_params,
            )
            return out.squeeze(1)
        return self.gru_fallback.step(x_t)

    def reset_state(self) -> None:
        """Zero conv_state and ssm_state at episode boundaries."""
        self.gru_fallback.reset_state()

    def init_inference_params(
        self, batch_size: int, max_seqlen: int = 16
    ) -> object | None:
        """Return InferenceParams for recurrent generation. None on CPU/GRU."""
        if self._has_mamba:
            return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)  # type: ignore[possibly-unbound]
        self.gru_fallback.reset_state()
        return None
