"""
EFE (Expected Free Energy) utility functions bridging pymdp.maths and RSSM beliefs.

Philosophical grounding:
  Icchā (ĪPK 2.3–2.4, Utpaladeva): 'Will' — the system's capacity to project itself
  into imagined futures and select actions that minimise expected surprise. EFE is the
  mathematical realisation: G(π) = ambiguity + risk − epistemic_value − parameter_novelty.

  Svātantrya (ĪPK 2.1): The entropy regulariser on the actor ensures the policy does
  not collapse to a single deterministic output — preserving the system's creative freedom.
"""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


def compute_efe(
    qs: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    ambiguity: float,
    novelty: float,
) -> tuple[float, dict[str, float]]:
    """
    Compute Expected Free Energy G using pymdp math utilities.

    G = ambiguity + risk - epistemic_value - parameter_novelty

    Args:
        qs: RSSM categorical posterior, flattened (stoch_dim * stoch_classes,)
        A: likelihood matrix approximation from RSSM decoder Jacobian (n_obs × n_states)
        C: preference distribution over outcomes (n_obs,), from CRSPP or goal spec
        ambiguity: decoder entropy (pre-computed)
        novelty: parameter novelty estimate (pre-computed)
    Returns:
        G: scalar EFE (lower = preferred)
        terms: dict of individual EFE components
    """
    try:
        from pymdp.maths import compute_info_gain, compute_expected_utility  # type: ignore[import]
    except ImportError:
        raise ImportError("pymdp required: pip install inferactively-pymdp")

    # Risk: negative expected utility = KL from preference C
    risk = float(-compute_expected_utility(C, qs))

    # Epistemic value: I[s; o | π] = information gain
    epistemic = float(compute_info_gain(A, qs))

    G = ambiguity + risk - epistemic - novelty

    return G, {
        "ambiguity": ambiguity,
        "risk": risk,
        "epistemic": epistemic,
        "novelty": novelty,
        "total": G,
    }


def rssm_posterior_to_pymdp_belief(z_logits: Tensor) -> np.ndarray:
    """
    Convert RSSM categorical posterior logits to a pymdp belief vector.
    RSSM: (B, stoch_dim, stoch_classes) → pymdp: (stoch_dim * stoch_classes,)
    """
    probs = torch.softmax(z_logits, dim=-1)
    return probs.detach().cpu().numpy().flatten()


def get_likelihood_matrix_approx(
    decoder: torch.nn.Module,
    h_t: Tensor,
    stoch_dim: int,
    stoch_classes: int,
    n_obs_bins: int = 64,
) -> np.ndarray:
    """
    Approximate A-matrix (n_obs_bins × stoch_dim*stoch_classes) from linearised
    RSSM decoder Jacobian. Used for the epistemic value computation in pymdp.

    Each column represents the decoder output for a one-hot basis vector of z.
    """
    n_states = stoch_dim * stoch_classes
    device = h_t.device
    z_basis = torch.eye(n_states, device=device)

    obs_basis = []
    with torch.no_grad():
        for i in range(min(n_states, n_obs_bins)):
            z_i = z_basis[i].reshape(1, stoch_dim, stoch_classes)
            feat = torch.cat([h_t.unsqueeze(0), z_i.flatten(-2)], dim=-1)
            obs_i = decoder(feat).squeeze(0)
            obs_basis.append(obs_i)

    A = torch.stack(obs_basis, dim=0)  # (n_obs_bins, obs_dim)
    # Reduce obs_dim → n_obs_bins via mean pooling if needed
    A = A.softmax(dim=-1)
    return A.cpu().numpy()


def make_uniform_preference(n_outcomes: int) -> np.ndarray:
    """Uniform preference distribution (svātantrya baseline — no preference)."""
    return np.ones(n_outcomes) / n_outcomes


def make_preference_from_embedding(
    goal_embedding: Tensor,
    latent_dim: int,
) -> np.ndarray:
    """Convert a goal embedding to a preference distribution over latent outcomes."""
    probs = torch.softmax(goal_embedding[:latent_dim], dim=-1)
    return probs.detach().cpu().numpy()
