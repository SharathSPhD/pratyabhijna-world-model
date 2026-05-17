"""
lora_adapters.py — LoRA r=8 domain adapters for multilingual BPE embedding.

Philosophical grounding (CLAUDE.md §9):
  Mālā regularisers (āṇava, māyīya, kārma, pwm/rewards/mala.py) prevent
  catastrophic forgetting by constraining parameter drift. LoRA is the
  computational analogue: it adds expressiveness (domain-specific A and B
  matrices) while freezing the original weights, preserving the WM's
  hard-won English geometry.

Computational realisation (TRIZ Principle 3 — Local Quality):
  Instead of globally fine-tuning the 512-dim TextEncoder projection, we
  add local rank-8 adapters per domain. Each domain gets its own (A, B)
  pair; inference selects the adapter matching the creative spec's domain.

Architecture:
  LoRALinear:  frozen W + trainable B @ A  (r=8, α=16, scale=α/r=2.0)
  DomainLoRABank: N_domains × LoRALinear adapters, indexed by Domain string.

Sprint 5 entry criteria (from sprint4_gate.json):
  - Sprint 2 gate pass (✓) and Sprint 4 gate pass (✓).

Known limitation (Sprint 1 memory constraint):
  CUDA training is currently blocked by unified memory saturation (120/128GB
  used). Adapters can be trained on CPU with batch_size=1 or when Ollama
  model is unloaded. The module is ready for training; the gate test runs
  on untrained (random) adapters to verify shapes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# Domain strings from domain_metadata.py (kept in sync)
Domain = Literal[
    "sanskrit_classical", "carnatic", "hindustani",
    "western_pop", "western_jazz",
    "kannada_film", "hindi_film", "tamil_classical",
    "telugu_padyam", "bengali_lyric",
    "english_romantic", "english_modernist", "english_beat",
    "world_fusion", "generic",
]

ALL_DOMAINS: list[str] = [
    "sanskrit_classical", "carnatic", "hindustani",
    "western_pop", "western_jazz",
    "kannada_film", "hindi_film", "tamil_classical",
    "telugu_padyam", "bengali_lyric",
    "english_romantic", "english_modernist", "english_beat",
    "world_fusion", "generic",
]


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA decomposition.

    Forward: y = W(x) + (α/r) * B(A(x))

    W is frozen (requires_grad=False).
    A, B are the trainable rank-r adapter matrices.

    Attributes:
        in_features:  Input dimension.
        out_features: Output dimension.
        r:            LoRA rank (default 8).
        alpha:        LoRA scaling factor (default 16 → scale = alpha/r = 2).
        scale:        alpha / r — applied to the LoRA output.
        W:            Frozen pre-trained weight matrix (out_features × in_features).
        A:            Trainable adapter matrix A (r × in_features). Kaiming init.
        B:            Trainable adapter matrix B (out_features × r). Zero init.
    """

    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Frozen pre-trained projection
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.W.weight.requires_grad_(False)
        if self.W.bias is not None:
            self.W.bias.requires_grad_(False)

        # Trainable LoRA matrices: A (r × in) then B (out × r)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        # Initialise: A ~ N(0, 1/sqrt(r)) (Kaiming), B = 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = W(x) + scale * B(A(x))"""
        return self.W(x) + self.scale * self.B(self.A(x))

    def merge_weights(self) -> nn.Linear:
        """
        Return a plain nn.Linear with LoRA merged into W for fast inference.

        Merged weight: W_merged = W + (α/r) * (B.weight @ A.weight)
        This eliminates the adapter overhead at inference time.
        """
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.W.bias is not None)
        merged.weight.data = (
            self.W.weight.data + self.scale * (self.B.weight @ self.A.weight)
        )
        if self.W.bias is not None and merged.bias is not None:
            merged.bias.data = self.W.weight.data.new_zeros(self.out_features)
            merged.bias.data.copy_(self.W.bias.data)
        return merged

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the LoRA parameters (A and B); W is frozen."""
        return list(self.A.parameters()) + list(self.B.parameters())

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, α={self.alpha}, scale={self.scale:.2f}")


class DomainLoRABank(nn.Module):
    """
    A bank of LoRA adapters, one per creative domain.

    For each domain in ALL_DOMAINS, maintains a LoRALinear(obs_dim, obs_dim)
    adapter that learns domain-specific adjustments to the TextEncoder projection.

    Usage:
        bank = DomainLoRABank(obs_dim=512, r=8, alpha=16)
        domain_embedding = bank("carnatic", raw_embedding)  # (B, obs_dim)

    Training:
        Only bank.adapters[domain].A and .B parameters are trainable.
        Freeze W with: bank.freeze_base_weights()

    Inference:
        bank.forward(domain, x) selects the adapter for the given domain.
        For unknown domains, falls back to the "generic" adapter.
    """

    def __init__(self, obs_dim: int = 512, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.r = r
        self.alpha = alpha

        # One LoRA adapter per domain
        self.adapters = nn.ModuleDict({
            domain: LoRALinear(obs_dim, obs_dim, r=r, alpha=alpha, bias=False)
            for domain in ALL_DOMAINS
        })

    def _get(self, domain: str) -> LoRALinear:
        """Return the LoRALinear adapter for domain (falls back to 'generic')."""
        key = domain if domain in self.adapters else "generic"
        return cast(LoRALinear, self.adapters[key])

    def forward(self, domain: str, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the domain-specific LoRA adapter to embedding x.

        Args:
            domain: Creative domain string (must be in ALL_DOMAINS or "generic").
            x:      Input tensor of shape (B, obs_dim) or (obs_dim,).

        Returns:
            Adapted embedding of same shape as x.
        """
        return self._get(domain)(x)

    def freeze_base_weights(self) -> None:
        """Freeze all W matrices; only A and B remain trainable."""
        for adapter in self.adapters.values():
            lora = cast(LoRALinear, adapter)
            lora.W.weight.requires_grad_(False)
            if lora.W.bias is not None:
                lora.W.bias.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return all trainable LoRA parameters (A and B matrices across all domains)."""
        params: list[nn.Parameter] = []
        for adapter in self.adapters.values():
            params.extend(cast(LoRALinear, adapter).trainable_parameters())
        return params

    def n_trainable_params(self) -> int:
        """Count trainable parameters (for logging)."""
        return sum(p.numel() for p in self.trainable_parameters())

    def n_total_params(self) -> int:
        """Count total parameters (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Path) -> None:
        """Save only the trainable (A, B) matrices — not the frozen W weights."""
        lora_state: dict[str, dict[str, torch.Tensor]] = {}
        for domain, adapter in self.adapters.items():
            lora = cast(LoRALinear, adapter)
            lora_state[domain] = {
                "A": lora.A.weight.data.clone(),
                "B": lora.B.weight.data.clone(),
            }
        torch.save({"lora_state": lora_state, "r": self.r,
                    "alpha": self.alpha, "obs_dim": self.obs_dim}, path)

    @classmethod
    def load(cls, path: Path, obs_dim: int = 512) -> "DomainLoRABank":
        """Load LoRA adapter weights from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        lora_state = ckpt["lora_state"]
        r = ckpt.get("r", 8)
        alpha = ckpt.get("alpha", 16.0)
        bank = cls(obs_dim=obs_dim, r=r, alpha=alpha)
        for domain, weights in lora_state.items():
            if domain in bank.adapters:
                lora = cast(LoRALinear, bank.adapters[domain])
                lora.A.weight.data.copy_(weights["A"])
                lora.B.weight.data.copy_(weights["B"])
        return bank


# ─── Convenience factory ─────────────────────────────────────────────────────

def make_lora_bank(obs_dim: int = 512, r: int = 8, alpha: float = 16.0,
                   checkpoint: Path | None = None) -> DomainLoRABank:
    """
    Create a DomainLoRABank, optionally loading trained weights.

    If checkpoint is None, returns a bank with zero-initialised B matrices
    (equivalent to no adaptation — the W-only path). This is the correct
    initial state: before training, LoRA adds nothing.

    Args:
        obs_dim:    TextEncoder output dimension (must match WM obs_dim=512).
        r:          LoRA rank (default 8 per Sprint 5 spec).
        alpha:      LoRA scaling (default 16 → scale 2.0).
        checkpoint: Optional path to a saved DomainLoRABank state.

    Returns:
        DomainLoRABank ready for inference or fine-tuning.
    """
    if checkpoint and checkpoint.exists():
        bank = DomainLoRABank.load(checkpoint, obs_dim=obs_dim)
        print(f"  [LoRA] Loaded adapters from {checkpoint.name} "
              f"({bank.n_trainable_params():,} trainable params)")
    else:
        bank = DomainLoRABank(obs_dim=obs_dim, r=r, alpha=alpha)
        if checkpoint:
            print(f"  [LoRA] Checkpoint {checkpoint} not found — using zero adapters")
        else:
            print(f"  [LoRA] New bank: {bank.n_trainable_params():,} trainable "
                  f"/ {bank.n_total_params():,} total params | r={r}, α={alpha}")
    bank.freeze_base_weights()
    return bank
