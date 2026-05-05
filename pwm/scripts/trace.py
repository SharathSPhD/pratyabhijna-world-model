"""
Latent trace inspector for the Pratyabhijñā World Model.

Philosophical grounding:
  Anugraha (IPK 3.1.8, Utpaladeva): Revelation — making the hidden visible.
  The trace inspector reveals the WM's internal latent trajectory through
  a text sequence, showing how z_t evolves as the model reads.

Usage:
  python pwm/scripts/trace.py --checkpoint checkpoints/final.pt \\
    --text "The ocean of consciousness ripples with every thought." \\
    --output traces/example_trace.json

  Or via entry point: pwm-trace --text "..."

Output JSON:
  {
    "text": "...",
    "steps": [
      {"t": 0, "h_norm": float, "z_argmax": [int,...], "z_entropy": float},
      ...
    ],
    "h_trajectory": [[...], ...],   # (T, hidden_dim) for plotting
    "vfe_per_step": [float, ...]
  }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv  # type: ignore[import]

load_dotenv()

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    _HYDRA = True
except ImportError:
    _HYDRA = False

log = logging.getLogger(__name__)


def trace_text(
    world_model: Any,
    text: str,
    obs_dim: int = 512,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Run WM on a single text string, collect per-token latent trace.

    Splits text into sentences/chunks, embeds each, then runs through WM
    step by step and records the latent state at each step.

    Returns a trace dict for inspection and plotting.
    """
    from pwm.perception.text import TextEncoder  # type: ignore[import]

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = TextEncoder(obs_dim=obs_dim).to(dev)
    world_model.train(False)

    # Split text into sentence-level chunks
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    log.info("Tracing %d sentences through WM...", len(sentences))

    with torch.no_grad():
        embs = enc(sentences, device=dev)  # (T, obs_dim)
        T = embs.shape[0]
        states = world_model.init_state(1, dev)
        action = torch.zeros(1, 64, device=dev)

        steps: list[dict[str, Any]] = []
        h_trajectory: list[list[float]] = []
        vfe_per_step: list[float] = []

        for t in range(T):
            obs_t = embs[t : t + 1]  # (1, obs_dim)
            new_states, logits_post, logits_prior = world_model.observe_step(
                obs_t, action, states, step=t
            )
            h_t, z_t = new_states[0]  # (1, hidden_dim), (1, stoch_dim, stoch_classes)

            # Per-step stats
            h_norm = float(h_t.norm().item())
            z_argmax = z_t[0].argmax(dim=-1).cpu().tolist()     # (stoch_dim,) list of ints
            z_probs = z_t[0].softmax(dim=-1).float()            # (stoch_dim, stoch_classes)
            z_entropy = float(-(z_probs * (z_probs + 1e-9).log()).sum(dim=-1).mean().item())

            steps.append({
                "t": t,
                "text": sentences[t],
                "h_norm": h_norm,
                "z_argmax": z_argmax,
                "z_entropy": z_entropy,
            })
            h_trajectory.append(h_t[0].float().cpu().tolist())
            vfe_per_step.append(0.0)  # VFE per step requires full sequence loss — zero here

            states = new_states

    return {
        "text": text,
        "n_steps": T,
        "steps": steps,
        "h_trajectory_shape": [T, len(h_trajectory[0])] if h_trajectory else [0, 0],
        "z_entropy_mean": float(np.mean([s["z_entropy"] for s in steps])),
        "h_norm_mean": float(np.mean([s["h_norm"] for s in steps])),
    }


def run_trace(cfg: Any, text: str, output_path: Path | None = None) -> dict[str, Any]:
    """Load checkpoint, run trace on text, save and return trace dict."""
    from pwm.world_model.trika import TrikaWorldModel  # type: ignore[import]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(os.environ.get("PWM_CHECKPOINT", "checkpoints/final.pt"))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    wm_cfg = cfg.world_model
    wm = TrikaWorldModel(
        obs_dim=wm_cfg.obs_dim,
        action_dim=wm_cfg.action_dim,
        n_levels=wm_cfg.levels,
        hidden_dim=wm_cfg.hidden_dim_apara,
        stoch_dim=wm_cfg.stoch_dim,
        stoch_classes=wm_cfg.stoch_classes,
        free_bits=wm_cfg.free_bits,
        kl_balance_dyn=wm_cfg.kl_balance_dyn,
        kl_balance_rep=wm_cfg.kl_balance_rep,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"])
    log.info("Loaded WM from %s (step=%d)", checkpoint_path, ckpt.get("step", -1))

    trace = trace_text(wm, text=text, obs_dim=wm_cfg.obs_dim, device=device)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(trace, indent=2))
        log.info("Trace saved: %s", output_path)

    return trace


if _HYDRA:
    @hydra.main(  # type: ignore[misc]
        config_path="../../configs",
        config_name="phase1_apara",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:  # type: ignore[misc,no-redef]
        """Trace inspector entry point."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
        import sys
        # Get text from env or default
        text = os.environ.get(
            "PWM_TRACE_TEXT",
            "Consciousness recognises itself in every creative act. The ocean of awareness ripples."
        )
        output = Path(os.environ.get("PWM_TRACE_OUTPUT", "traces/trace.json"))
        trace = run_trace(cfg, text=text, output_path=output)
        print(json.dumps({k: v for k, v in trace.items() if k != "h_trajectory"}, indent=2))
else:
    def main() -> None:  # type: ignore[misc,no-redef]
        raise RuntimeError("Hydra not installed. pip install hydra-core")


if __name__ == "__main__":
    main()
