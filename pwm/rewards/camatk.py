"""
CamatkaraReward: Intrinsic creative reward signal.

Philosophical grounding:
  Camatkāra (Abhinavagupta, Locana ad Dhvanyāloka 1.1; Gnoli 1968):
  'Aesthetic wonder' — the flash of recognition, *citrasya camatkaraḥ* (the wonder of the
  image). Not a static score but a temporal event: consciousness recognising the unexpected.

  Mathematically: R_camatk(t) = α₁·ΔF_vfe(t) + α₂·ΔI_Hopfield(t) + α₃·Empowerment(t)

  This signal is self-certified: the system decides what constitutes creative discovery
  based on its own generative model's surprise reduction and memory structure.
  No external LLM judge. No circularity. Pure svātantrya.

  Resolves the H9 crisis from PCE v0.4 (ρ=0.0 between proxy and LLM judge).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class RunningStats:
    """Online mean/std for reward normalisation (Welford's algorithm)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, val: float) -> None:
        self.n += 1
        delta = val - self.mean
        self.mean += delta / self.n
        delta2 = val - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        return (self.M2 / max(self.n - 1, 1)) ** 0.5 if self.n > 1 else 1.0


class CamatkaraReward(nn.Module):
    """
    Computes the intrinsic camatkāra reward at each world model step.

    Components:
      1. ΔF_vfe: free energy reduction — the Friston 'Eureka' signal
      2. ΔI_Hopfield: information gain about the Citta-store — recognition signal
      3. Empowerment: mutual information I(A; S_{t+k}) — creative agency signal
    """

    def __init__(
        self,
        alpha_1: float = 0.4,  # VFE reduction weight
        alpha_2: float = 0.3,  # Hopfield information gain weight
        alpha_3: float = 0.3,  # Empowerment weight
    ) -> None:
        super().__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3

        self._vfe_stats = RunningStats()
        self._hopfield_stats = RunningStats()
        self._empowerment_stats = RunningStats()

        self._prev_vfe: float | None = None

    def compute(
        self,
        curr_vfe: float,
        hopfield_entropy_delta: float,
        empowerment: float,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute normalised camatkāra reward.

        Args:
            curr_vfe: current VFE scalar
            hopfield_entropy_delta: |H_pre - H_post| from Hopfield write
            empowerment: ensemble disagreement proxy for I(A; S_{t+k})
        Returns:
            r_camatk: scalar reward tensor
            components: dict of individual components (for logging)
        """
        # Component 1: VFE reduction (only reward drops)
        if self._prev_vfe is not None:
            delta_f = max(self._prev_vfe - curr_vfe, 0.0)
        else:
            delta_f = 0.0
        self._prev_vfe = curr_vfe
        self._vfe_stats.update(delta_f)
        delta_f_norm = (delta_f - self._vfe_stats.mean) / (self._vfe_stats.std + 1e-8)

        # Component 2: Hopfield information gain
        self._hopfield_stats.update(hopfield_entropy_delta)
        delta_i_norm = (
            (hopfield_entropy_delta - self._hopfield_stats.mean)
            / (self._hopfield_stats.std + 1e-8)
        )

        # Component 3: Empowerment
        self._empowerment_stats.update(empowerment)
        emp_norm = (
            (empowerment - self._empowerment_stats.mean)
            / (self._empowerment_stats.std + 1e-8)
        )

        r = (
            self.alpha_1 * delta_f_norm
            + self.alpha_2 * delta_i_norm
            + self.alpha_3 * emp_norm
        )

        components = {
            "delta_f": delta_f,
            "delta_f_norm": delta_f_norm,
            "delta_i": hopfield_entropy_delta,
            "delta_i_norm": delta_i_norm,
            "empowerment": empowerment,
            "empowerment_norm": emp_norm,
            "r_camatk": r,
        }

        return torch.tensor(r, dtype=torch.float32), components

    def sphuratta_score(
        self,
        vfe: float,
        vfe_percentile: float,
        hopfield_entropy: float,
        hopfield_threshold: float,
        last_sphuratta_step: int | None,
        current_step: int,
        min_gap: int = 100,
    ) -> bool:
        """
        Sphurattā (TĀ 1.56, Abhinavagupta) — recognition flash detector.
        Fires when BOTH VFE drops sharply AND Hopfield retrieval entropy drops.
        Requires a minimum gap between events to prevent false clustering.
        """
        if last_sphuratta_step is not None:
            if current_step - last_sphuratta_step < min_gap:
                return False

        vfe_criterion = vfe < vfe_percentile
        hopfield_criterion = hopfield_entropy < hopfield_threshold
        return vfe_criterion and hopfield_criterion

    def svātantrya_score(
        self,
        z_output: Tensor,
        corpus_embeddings: Tensor,
    ) -> Tensor:
        """
        S_svātantrya: compositional novelty — minimum distance from training corpus
        in WM latent space. High = genuinely novel. Low = statistical interpolation.

        S_svātantrya(x) = min_{x' ∈ corpus} d_latent(z(x), z(x'))
        """
        z = z_output.flatten(-2)
        dists = torch.cdist(z.unsqueeze(0), corpus_embeddings.unsqueeze(0))
        return dists.min(-1).values.squeeze(0)
