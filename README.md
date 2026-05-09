# Pratyabhijñā World Model (PWM)

> *"Consciousness recognises itself in every creative act."*
> — Utpaladeva, *Īśvarapratyabhijñākārikā* 1.3

A creative AI research system operationalising Kashmir Śaiva philosophy through active inference on a DreamerV3-class world model with frozen LLM augmentation.

## Overview

PWM realises the **śakti cascade** (MV 1.4; ĪPK 3.1–3.2) — the seven-fold movement of consciousness
from pure awareness to embodied act — as a single coherent Python call-stack sharing one continuous WM state `(h_t, z_t)`:

| Step | Sanskrit | Philosophical act | Computational primitive |
|------|----------|-------------------|------------------------|
| 1 | **Cit** | ābhāsana — the world manifests | RSSM `observe(o_t, h, z, a)` → `h_t, z_t` |
| 2 | **Ānanda** | rakti — the pleasure/surprise signal arises | Camatkāra reward `R = αΔF + βΔI + γEmp` |
| 3 | **Icchā** | will selects the next action | EFE actor `π(a | h_t, z_t)` |
| 4 | **Apohana** | smṛti — episodic context refined | Hopfield CittaStore read/write |
| 5 | **Jñāna** | patterns named (LLM fast-path) | Nemotron call on sphurattā only |
| 6 | **Kriyā** | the act is performed | Action commit + skill-library emit |
| 7 | **Vimarśa** | self-reflexive deliberation (sphurattā gate) | VimarshaAgent (smolagents, rare) |

> **Architecture constraint**: steps 1–6 share a single `(h_t, z_t)` tensor — they are NOT
> separate agents. Only Vimarśa (step 7) is a true smolagents agent, invoked only when
> VFE < 5th-percentile AND Hopfield entropy drops (sphurattā event, ≪ 1% of steps).

## Architecture

```
TIER 3 — Pañcakṛtya Pipeline (smolagents)
  cit → ānanda → icchā → apohana → jñāna → kriyā → vimarśa
  Avacchedaka store (Pratyākṣa PCEH): typed inter-agent messages
  Sākṣī-keeper: ≤500-token witness invariant

TIER 2 — LLM Āgama Layer (Conscious / Knowledge)
  Nemotron-3-Super 120B MoE (FP4, TensorRT-LLM)
  + Nemotron-Super-49B Dense (FP8, vLLM) — fast sub-agents
  LoRA vimarśa bridge ↔ WM latent projection
  Zero paid API calls (svātantrya principle)

TIER 1 — World Model Substrate (Subconscious / Prakāśa)
  Parā level   (Mamba-2, stride 16)  — global, slow
  Parāparā level (GRU, stride 4)     — coupling, mid
  Aparā level  (GRU, stride 1)       — fast, embodied
  RSSM: h_t ∈ ℝ^1024, z_t ~ Cat(32×32)
  Hopfield Citta-store (episodic smṛti + semantic ālayavijñāna)
```

### Key modules

| Module | File | Sanskrit concept |
|--------|------|-----------------|
| World model | `pwm/world_model/trika.py` | Trika (three-tier RSSM) |
| EFE actor | `pwm/active_inference/efe_actor.py` | Svātantrya (free-energy minimisation) |
| Camatkāra reward | `pwm/rewards/camatk.py` | Camatkāra (aesthetic surprise) |
| Hopfield memory | `pwm/memory/citta_store.py` | Citta (episodic + semantic store) |
| Śakti cascade | `pwm/pipeline/pancakrtya_loop.py` | Pañcakṛtya (seven-step loop) |
| Sleep consolidation | `pwm/sleep/nrem.py`, `rem.py` | Tirodhāna / Anugraha |
| LLM bridge | `pwm/vimarsa/bridge.py` | Vimarśa (reflexive cognition) |

## Pre-registered Hypotheses

| ID | Claim | Phase |
|----|-------|-------|
| H1 | EFE actor > REINFORCE on sparse creative reward | Phase 2 |
| H2 | Hopfield improves pattern completion | Phase 3 |
| H3 | Sleep reduces catastrophic forgetting | Phase 4 |
| H4 | Vimarśa bridge improves narration quality | Phase 5 |
| H5 | PWM > PCE v0.4 on creative quality | Phase 6 |

All statistical tests: paired permutation (50K), Hedges' g, BCa 95% CI (10K), Holm-Bonferroni FWE.

## Training Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Corpus pipeline (4.6M tokens) | ✅ PASS |
| 1 | Aparā RSSM text world model | ✅ PASS (ratio 1.011, silhouette 0.114) |
| 2 | EFE actor + camatkāra reward (H1) | 🔄 IN PROGRESS (v6, seed=47) |
| 3 | Hopfield CittaStore (H2) | ⏳ |
| 4 | Sleep consolidation (H3) | ⏳ |
| 5 | LLM āgama + vimarśa bridge (H4) | ⏳ |
| 6 | Full system + ablations (H5–H9) | ⏳ |

## Setup

```bash
source /home/sharaths/vllm-env/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Phase 0: Corpus ingestion
python -m corpus.build --sources all --min-tokens 100000

# Phase 1: Train Aparā RSSM
python pwm/scripts/train.py

# Phase 2: EFE actor (active inference)
bash scripts/launch_phase2_v6.sh

# Gate evaluation
bash scripts/run_gate2.sh checkpoints/final.pt 200

# Evaluate
python scripts/evaluate.py
```

## Citation

```bibtex
@article{pwm2025,
  title={Pratyabhijñā World Model: Creative AI through Recognition,
         Active Inference, and Associative Memory},
  author={SharathSPhD},
  year={2025}
}
```
