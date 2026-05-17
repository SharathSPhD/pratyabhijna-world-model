"""
VimarsaBridgeV2 — h_t → vocab-size logit bias via trained linear projection.

Sanskrit concept: Vimarśa (ĪPK 1.5.11, Utpaladeva) — the WM's reflexive
self-awareness shapes every generated token. The projection layer encodes
the WM's creative state into a vocabulary-space bias that is injected at
every token during LLM generation via the llama-cpp-python logits_processor hook.

Architecture:
  Linear(hidden_dim → vocab_size) with learnable scale parameter.
  ~66MB parameters for hidden_dim=512, vocab_size=128256.
  No cross-attention or prefix fusion — pure bias addition at the logit level.

Training (pwm/scripts/train_vimarsa_bridge.py):
  Objective: next-token cross-entropy on (h_t, next_token_id) corpus pairs.
  Expected: final_loss < initial_loss; KL-div vs no-bias > 0.05.

Checkpoint: checkpoints/vimarsa_bridge_v2.pt (auto-loaded by load_or_init).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_DEFAULT_CKPT = Path("checkpoints/vimarsa_bridge_v2.pt")


class VimarsaBridgeV2(nn.Module):
    """
    WM hidden state → vocabulary logit bias.

    Applied at every LLM token via llama-cpp-python logits_processor.
    Training target: next-token prediction on WM corpus.
    """

    def __init__(self, hidden_dim: int = 512, vocab_size: int = 128256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Learnable scale — starts at 1.0, annealed during training
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h_t: Tensor) -> Tensor:
        """
        h_t: (batch, hidden_dim) or (hidden_dim,)
        Returns: (batch, vocab_size) logit bias
        """
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        scale = torch.exp(self.log_scale).clamp(0.01, 3.0)
        return scale * self.proj(h_t)

    def as_logits_processor(self, h_t: Tensor) -> Callable:
        """
        Return a logits_processor callback for llama-cpp-python.

        The callback fires on every token during LLM generation.
        Adds the WM-derived logit bias to the raw LLM logits.

        Args:
            h_t: WM hidden state tensor, shape (1, hidden_dim) or (hidden_dim,)
        Returns:
            Callable[[list[int], np.ndarray], np.ndarray]
        """
        with torch.no_grad():
            bias = self.forward(h_t).squeeze(0).cpu().numpy().astype(np.float32)
            # bias: (vocab_size,)

        def _logits_processor(token_ids: list, logits: np.ndarray) -> np.ndarray:
            return logits + bias

        return _logits_processor

    def train_step(
        self,
        h_t: Tensor,
        target_token_ids: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single supervised training step.

        Args:
            h_t: (batch, hidden_dim) WM states
            target_token_ids: (batch,) next token IDs
            optimizer: Adam or AdamW
        Returns:
            Loss value (cross-entropy)
        """
        logits = self.proj(h_t)  # (batch, vocab_size) — skip scale for training
        loss = F.cross_entropy(logits, target_token_ids)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    @classmethod
    def load_or_init(
        cls,
        hidden_dim: int = 512,
        vocab_size: int = 128256,
        ckpt_path: Path = _DEFAULT_CKPT,
        device: str = "cpu",
    ) -> "VimarsaBridgeV2":
        """
        Load from checkpoint if it exists, else return freshly initialised.
        The fresh initialisation has random weights — KL-div will be > 0
        but may be low until training. Run train_vimarsa_bridge.py first.
        """
        bridge = cls(hidden_dim=hidden_dim, vocab_size=vocab_size)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            bridge.load_state_dict(state)
            logger.info(f"[VimarsaBridgeV2] Loaded checkpoint: {ckpt_path}")
        else:
            logger.info("[VimarsaBridgeV2] No checkpoint — using random init weights")
        return bridge.to(torch.device(device))
