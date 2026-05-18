# Pratyabhijna World Model (PWM)

> "Consciousness recognises itself in every creative act."
> — Utpaladeva, Isvarapratyabhijnakarika 1.3

PWM is a research prototype that asks whether a Dreamer-class textual world model can give creative AI an inner state distinct from its surface text. It is a **composite system**: a trainable world model (Trika RSSM, EFE actor, Hopfield memory, NREM/REM consolidation, camatkara intrinsic reward) coupled to a **frozen 120B LLM** through a single learned channel, the Vimarsa Bridge. The world model holds creative state; the LLM speaks. They are not competitors.

The project is intentionally evidence-first. Internal world-model and active-inference gates are strong (H1 29.7x, H5a 2.142x, A6 23.6x). The H5b live ablation is an honest negative result for the bridge-bias coupling channel under a stripped text proxy — not a head-to-head between the world model and the LLM. The repository, paper, and website should be read as an active research program, not a finished product benchmark.

**Live site:** <https://sharathsphd.github.io/pratyabhijna-world-model/>

**Paper source:** `paper/main.tex`

**Evidence artifacts:** `benchmarks/results/`

## What PWM Achieved

PWM demonstrates that static creative text can be used as a world-model substrate if the passive-corpus failure modes are handled explicitly. The key result is not simply "LLM plus prompt"; the system learns latent state, imagines rollouts, assigns intrinsic reward, retrieves associative memory, and invokes the LLM selectively through a Vimarsha bridge.

The strongest current outcomes are:

| Result | Evidence | Reading |
|---|---|---|
| EFE actor over REINFORCE | `benchmarks/results/phase_2_gate.json` reports 29.72x mean reward ratio | Strong internal control result |
| Hopfield memory completion | `benchmarks/results/phase_3_gate_step0300000.json` reports 1.307x completion ratio | Supports associative CittaStore utility |
| Sleep consolidation | `benchmarks/results/phase_4_gate_step0300000.json` reports near-zero forgetting | Positive gate, but with/without values are not very discriminative |
| Vimarsha bridge training | `benchmarks/results/phase4_gate.json` reports 71.2% bridge loss reduction | The bridge learns a real mapping from WM state to token bias |
| H5a internal imagination reward | `benchmarks/results/phase_5_gate_step0500000.json` reports 2.142x over Phase 2 baseline | Internal PWM process gain, not live text superiority |
| H5b live text ablation | `benchmarks/results/h5_live_ablation.json` reports PWM < bare 120B LLM, g=-0.47 | Honest negative result under text-only scoring |
| Three-level Trika hierarchy | `benchmarks/results/ablation_a6_1level_wm.json` reports 23.6x advantage | Strong but includes two partial timeout seeds |
| TTFT / WM trace | `benchmarks/results/phase7_gate.json` and sprint gates report implementation success; `ttft_live_validation.json` aggregate fails under measurement confound | Promising warm-path mechanism, not a clean aggregate latency win |

## Hypotheses

The journal paper now splits H5 into two claims:

| ID | Claim | Current status |
|---|---|---|
| H1 | EFE actor beats REINFORCE on sparse creative reward | PASS, 29.72x reward ratio |
| H2 | Hopfield CittaStore improves occlusion completion | PASS, 1.307x completion ratio |
| H3 | Sleep reduces forgetting across sequential domains | PASS gate, but weak discriminative evidence |
| H4 | Vimarsha bridge produces meaningful narration proxy | PASS proxy |
| H5a | PWM internal imagination reward exceeds Phase 2/PCE-aligned baseline | PASS, 2.142x |
| H5b | Vimarsa Bridge v2 logit-bias channel lifts text-only camatkara on the same 120B model | FAIL on English-script domains, g=-0.47; near-parity on Kannada |
| H6 | Camatkara reward distribution has non-trivial entropy | PASS, 1.897 nats in `phase_6_gate_step1000000.json` |
| H7 | Three-level Trika world model beats one-level ablation | PASS, 23.6x advantage in A6 |
| H8 | Mala regularisers prevent latent collapse | PASS, encoder norm 13.20 |
| H9 | IDL policy retains committed action diversity | PASS, entropy 0.582 nats |

This is best summarized as **9 of 10 split hypotheses passing**, with H5b documented as a real negative live result.

## Architecture

```text
Static corpora
  -> text embeddings / observations
  -> Trika RSSM world model: Aparā, Parāparā, Parā
  -> EFE actor + camatkara reward
  -> Hopfield CittaStore + sleep consolidation
  -> Vimarsha bridge
  -> gated frozen-LLM narration
```

Key modules:

| Area | Files |
|---|---|
| World model | `pwm/world_model/trika.py`, `pwm/world_model/rssm.py`, `pwm/world_model/losses.py` |
| Active inference | `pwm/active_inference/efe_actor.py`, `pwm/active_inference/efe_utils.py` |
| Reward | `pwm/rewards/camatk.py`, `pwm/rewards/mala.py` |
| Memory | `pwm/memory/citta_store.py`, `pwm/memory/replay.py` |
| Sleep | `pwm/sleep/` |
| Vimarsha bridge | `pwm/vimarsa/bridge.py`, `pwm/vimarsa/bridge_v2.py` |
| Training loop | `pwm/scripts/train.py`, `pwm/pipeline/pancakrtya_loop.py` |
| Generation/API loop | `pwm/pipeline/pancakrtya_loop_v2.py`, `pwm/generation/`, `api/main.py` |

## Important Caveats

- H5b is an ablation of the trained logit-bias coupling channel (`VimarsaBridgeV2.as_logits_processor`), not a competition between two generators. Both conditions use the same 120B model and the same prompts; only the logits processor changes. The negative result bounds the bridge channel under a stripped text proxy, not the world model as a substrate.
- H5a and H5b are different protocols. H5a uses internal world-model/imagination reward; H5b uses a text-only scorer that excludes VFE/world-model terms for fairness to the unconditioned baseline.
- The API exposes two paths in `api/main.py`: the v1 endpoints (`POST /v1/generate`, `WS /v1/ws/generate`) route through `PancakrtyaLoopV2` and are the faithful surface; the legacy endpoints (`POST /generate`, `/refine`, `/batch`) are a parallel orchestration that uses only a WM-derived prompt prefix and are deprecated in spirit. Use the v1 path to reproduce paper claims.
- The TTFT live validation aggregate fails because of cold-start/Ollama measurement confounds, even though sprint tests and warm-path measurements support the architecture.
- No human evaluation study has been completed yet. Automated camatkara proxies are not a substitute for human aesthetic judgment.
- Some runtime paths assume local checkpoints and models. Fresh-clone reproducibility is strongest for source, configs, paper, and JSON artifacts, not for full 120B local inference.
- `docs/`, `.claude/`, and `.env.example` are local-only and intentionally not tracked on the remote.

## Setup

Use the project Python environment for ML work:

```bash
source /home/sharaths/vllm-env/bin/activate
pip install -e ".[dev,paper]"
```

For your own machine, provide local environment variables rather than relying on a checked-in `.env.example`. Typical values include model/checkpoint paths, optional W&B/MLflow settings, and LLM backend configuration.

## Reproduce And Inspect

```bash
# Unit tests
pytest

# Gate/result inspection
ls benchmarks/results

# Paper build
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Website
cd site
npm install
npm run build
```

The most important evidence files are:

- `benchmarks/results/phase_2_gate.json`
- `benchmarks/results/phase_3_gate_step0300000.json`
- `benchmarks/results/phase_4_gate_step0300000.json`
- `benchmarks/results/phase_5_gate_step0500000.json`
- `benchmarks/results/phase_6_gate_step1000000.json`
- `benchmarks/results/h5_live_ablation.json`
- `benchmarks/results/ablation_a6_1level_wm.json`
- `benchmarks/results/ttft_live_validation.json`
- `benchmarks/results/phase8_gate.json`

## Companion Dataset

The Hugging Face dataset card lives in `hf_dataset/README.md` and describes the creative outputs and hypothesis-result splits prepared for publication.

## Citation

```bibtex
@article{subramanian2026pwm,
  title        = {Pratyabhijna World Model: Kashmir Saiva Philosophy as Active Inference Architecture},
  author       = {Subramanian, Sharath S.},
  journal      = {arXiv preprint},
  year         = {2026},
  note         = {Code: https://github.com/SharathSPhD/pratyabhijna-world-model}
}
```
