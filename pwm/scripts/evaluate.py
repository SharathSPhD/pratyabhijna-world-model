"""
PWM Phase 1 evaluation entry point.

Philosophical grounding:
  Parīkṣā (NBh ad NS 1.1.1, Vātsyāyana): Examination — the rigorous testing
  of knowledge claims. Just as the Nyāya philosopher examines every inference
  before accepting it, we rigorously test the WM before claiming Phase 1 exit.

Phase 1 exit criteria:
  1. Held-out reconstruction MSE < LSTM baseline (wm_vs_lstm_ratio < 1.0).
  2. UMAP of z_t latents shows domain cluster separation (silhouette > 0.1).

Usage:
  python pwm/scripts/evaluate.py --config-name phase1_apara \\
    checkpoint=checkpoints/final.pt corpus_root=/path/to/corpus

  Or via entry point:
    pwm-eval checkpoint=checkpoints/final.pt

Output:
  benchmarks/results/phase_1_gate_final.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv  # type: ignore[import]

load_dotenv()

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    _HYDRA = True
except ImportError:
    _HYDRA = False

from pwm.world_model.trika import TrikaWorldModel  # type: ignore[import]

log = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path, cfg: Any, device: torch.device) -> TrikaWorldModel:
    """Load WM state from checkpoint."""
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
    return wm


def run_phase1_gate(cfg: Any) -> dict[str, Any]:
    """
    Run full Phase 1 exit gate evaluation.

    Returns the final gate metrics dict (written to JSON).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Running Phase 1 gate on device: %s", device)

    # Resolve paths
    checkpoint_path = Path(os.environ.get("PWM_CHECKPOINT", "checkpoints/final.pt"))
    corpus_root = Path(
        os.environ.get("CORPUS_ROOT", getattr(cfg, "corpus", None) and cfg.corpus.data_dir or "data/corpus")
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Corpus: %s", corpus_root)

    world_model = load_checkpoint(checkpoint_path, cfg, device)

    mp_str = getattr(cfg.training, "mixed_precision", "bfloat16")
    mp_dtype = torch.bfloat16 if mp_str == "bfloat16" else torch.float32
    use_amp = mp_str in ("bfloat16", "float16")

    gate: dict[str, Any] = {
        "phase": 1,
        "phase_name": "apara_rssm_text",
        "checkpoint": str(checkpoint_path),
        "corpus": str(corpus_root),
        "device": str(device),
    }

    # ── Criterion 1: Perplexity vs LSTM ──────────────────────────────────────
    log.info("=== Criterion 1: Perplexity gate ===")
    try:
        from pwm.eval.perplexity import run_perplexity_report  # type: ignore[import]
        perf_metrics = run_perplexity_report(
            world_model=world_model,
            corpus_dir=corpus_root,
            obs_dim=cfg.world_model.obs_dim,
            device=device,
            lstm_train_steps=2000,
            n_batches=50,
        )
        gate["perplexity"] = perf_metrics
        gate["criterion_1_pass"] = perf_metrics.get("perplexity_gate_pass", False)
    except Exception as exc:
        log.exception("Perplexity gate failed: %s", exc)
        gate["perplexity"] = {"error": str(exc)}
        gate["criterion_1_pass"] = False

    # ── Criterion 2: UMAP cluster separation ──────────────────────────────────
    log.info("=== Criterion 2: UMAP cluster gate ===")
    try:
        from pwm.eval.umap_viz import collect_latents_with_labels, compute_umap_and_score  # type: ignore[import]
        features, labels = collect_latents_with_labels(
            world_model=world_model,
            corpus_dir=corpus_root,
            obs_dim=cfg.world_model.obs_dim,
            n_samples=2000,
            device=device,
        )
        output_dir = Path("benchmarks/results/figures")
        umap_metrics = compute_umap_and_score(features, labels, output_dir=output_dir)
        gate["umap"] = umap_metrics
        gate["criterion_2_pass"] = umap_metrics.get("umap_gate_pass", False)
    except Exception as exc:
        log.exception("UMAP gate failed: %s", exc)
        gate["umap"] = {"error": str(exc)}
        gate["criterion_2_pass"] = False

    # ── Svātantrya score ───────────────────────────────────────────────────────
    log.info("=== Svātantrya evaluation ===")
    try:
        from pwm.eval.svat import collect_latents, compute_sva_score  # type: ignore[import]
        from pwm.eval.perplexity import build_held_out_loader          # type: ignore[import]

        _, held_loader = build_held_out_loader(
            corpus_dir=corpus_root,
            obs_dim=cfg.world_model.obs_dim,
            device=device,
        )
        h_latents, z_indices = collect_latents(
            world_model=world_model, loader=held_loader, n_batches=50, device=device
        )
        if z_indices.shape[0] > 0:
            sva_metrics = compute_sva_score(z_indices, n_cats=cfg.world_model.stoch_classes)
            gate["svat"] = sva_metrics
        else:
            gate["svat"] = {"note": "latent collection requires observe_step interface"}
    except Exception as exc:
        log.exception("Svātantrya failed: %s", exc)
        gate["svat"] = {"error": str(exc)}

    # ── Camatkāra evaluation ──────────────────────────────────────────────────
    log.info("=== Camatkāra evaluation ===")
    try:
        from pwm.eval.camatk_eval import run_camatk_report              # type: ignore[import]
        from pwm.eval.perplexity import build_held_out_loader            # type: ignore[import]
        from pwm.memory.citta_store import CittaStore                    # type: ignore[import]
        from pwm.rewards.camatk import CamatkaraReward                  # type: ignore[import]

        wm_cfg = cfg.world_model
        rew_cfg = cfg.reward
        citta = CittaStore(hidden_dim=wm_cfg.hidden_dim_apara, n_levels=wm_cfg.levels).to(device)
        camatk = CamatkaraReward(alpha_1=rew_cfg.alpha_1, alpha_2=rew_cfg.alpha_2, alpha_3=rew_cfg.alpha_3)

        _, held_loader = build_held_out_loader(corpus_dir=corpus_root, obs_dim=wm_cfg.obs_dim, device=device)
        camatk_metrics = run_camatk_report(
            world_model=world_model, camatk_fn=camatk, citta_store=citta,
            held_loader=held_loader, device=device,
        )
        gate["camatk"] = camatk_metrics
    except Exception as exc:
        log.exception("Camatkāra failed: %s", exc)
        gate["camatk"] = {"error": str(exc)}

    # ── Overall gate decision ─────────────────────────────────────────────────
    c1 = gate.get("criterion_1_pass", False)
    c2 = gate.get("criterion_2_pass", False)
    gate["status"] = "PASS" if (c1 and c2) else "PARTIAL" if (c1 or c2) else "FAIL"
    gate["criteria_passed"] = [k for k, v in {"perplexity": c1, "umap_cluster": c2}.items() if v]
    gate["criteria_failed"] = [k for k, v in {"perplexity": c1, "umap_cluster": c2}.items() if not v]

    log.info(
        "Phase 1 gate: %s  (criterion_1=%s, criterion_2=%s)",
        gate["status"], c1, c2,
    )

    # Write to benchmarks/results/
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "phase_1_gate_final.json"
    gate_path.write_text(json.dumps(gate, indent=2, default=str))
    log.info("Phase 1 gate written: %s", gate_path)

    return gate


if _HYDRA:
    @hydra.main(  # type: ignore[misc]
        config_path="../../configs",
        config_name="phase1_apara",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        log.info("Phase 1 evaluation config:\n%s", OmegaConf.to_yaml(cfg))
        gate = run_phase1_gate(cfg)
        print(json.dumps(gate, indent=2, default=str))
else:
    def main() -> None:  # type: ignore[misc,no-redef]
        raise RuntimeError("Hydra is not installed. pip install hydra-core")


if __name__ == "__main__":
    main()
