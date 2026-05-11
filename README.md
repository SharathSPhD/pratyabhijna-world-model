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
| 2 | **Ānanda** | rakti — the pleasure/surprise signal arises | Camatkāra reward `R = α₁ΔF + α₂ΔI_Hopfield + α₃Emp` |
| 3 | **Icchā** | will selects the next action | EFE actor `π(a | h_t, z_t)` |
| 4 | **Apohana** | smṛti — episodic context refined | Hopfield CittaStore read/write |
| 5 | **Jñāna** | patterns named (LLM fast-path) | Nemotron call on sphurattā only |
| 6 | **Kriyā** | the act is performed | Action commit + skill-library emit |
| 7 | **Vimarśa** | self-reflexive deliberation (sphurattā gate) | VimarsaBridge cross-attention (Phase 5+) |

> **Architecture constraint**: steps 1–6 share a single `(h_t, z_t)` tensor.
> The LLM is invoked only at sphurattā events (≪ 1% of steps).

## Architecture

```
TIER 3 — Pañcakṛtya Pipeline
  cit → ānanda → icchā → apohana → jñāna → kriyā → vimarśa
  PancakrtyaLoop (pwm/pipeline/pancakrtya_loop.py)

TIER 2 — LLM Āgama Layer (Phase 5+)
  Nemotron-3-Nano-30B GGUF via llama.cpp (local, zero API cost)
  VimarsaBridge: h_t ∈ ℝ^512 → LLM soft-prompt prefix (cross-attention)

TIER 1 — World Model Substrate
  Aparā level  (GRU, stride 1)       — fast, embodied    [Phases 1–6]
  Parāparā level (GRU, stride 4)     — coupling, mid     [Phase 6 full]
  Parā level   (S4 SSM, stride 16)   — global, slow      [Phase 6 full]
  RSSM: h_t ∈ ℝ^512, z_t ~ Cat(32×32)
  Hopfield Citta-store (episodic smṛti + semantic ālayavijñāna) [Phase 3+]
  NREM/REM sleep consolidation [Phase 4+]
```

### Key modules

| Module | File | Sanskrit concept |
|--------|------|-----------------|
| World model | `pwm/world_model/trika.py` | Trika (three-tier RSSM) |
| EFE actor | `pwm/active_inference/efe_actor.py` | Svātantrya (free-energy minimisation) |
| Camatkāra reward | `pwm/rewards/camatk.py` | Camatkāra (aesthetic surprise) |
| Hopfield memory | `pwm/memory/citta_store.py` | Citta (episodic + semantic store) |
| Śakti cascade | `pwm/pipeline/pancakrtya_loop.py` | Pañcakṛtya (seven-step loop) |
| Sleep consolidation | `pwm/sleep/consolidation.py` | Tirodhāna / Anugraha |
| LLM bridge | `pwm/vimarsa/bridge.py` | Vimarśa (reflexive cognition) |

## Pre-registered Hypotheses

| ID | Claim | Metric | Phase | Status |
|----|-------|--------|-------|--------|
| H1 | EFE actor > REINFORCE on sparse creative reward | Mean reward ratio ≥ 2.0 | 2 | **PASS** (ratio=29.72) |
| H2 | Hopfield improves pattern completion | Completion ratio ≥ 1.10 | 3 | In training |
| H3 | Sleep reduces catastrophic forgetting | Forgetting ratio < 0.8 | 4 | Pending |
| H4 | Vimarśa bridge narration proxy | ≥70% sphurattā with H(z)>0.5 nats | 5 | Pending |
| H5 | PWM > PCE v0.4 on creative quality | Mean reward ≥ 2× baseline | 5 | Pending |
| H6 | Reward entropy non-trivial | Entropy > 0.5 nats | 6 | Pending |
| H7 | Imagination VFE improves | VFE proxy < Phase 3 × 0.85 | 6 | Pending |
| H8 | Encoder stability | Norm in [1.0, 50.0] | 6 | Pending |
| H9 | Policy diversity | Action entropy > 1.0 nats | 6 | Pending |

All statistical tests: paired permutation (50K), Hedges' g, BCa 95% CI (10K), Holm-Bonferroni FWE.

## Training Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Corpus pipeline (4.6M tokens) | ✅ PASS |
| 1 | Aparā RSSM text world model | ✅ PASS (silhouette=0.121, ratio=1.011) |
| 2 | EFE actor + camatkāra reward (H1) | ✅ PASS (EFE/RF=29.72×, seed=52, 400K steps) |
| 3 | Hopfield CittaStore (H2) | 🔄 Training (seed=53, 300K steps, ~01:35 UTC 2026-05-12) |
| 4 | Sleep consolidation (H3) | ⏳ Auto-launches on Phase 3 gate PASS |
| 5 | LLM vimarśa bridge (H4, H5) | ⏳ Auto-launches on Phase 4 gate PASS |
| 6 | Full system + ablations (H6–H9) | ⏳ Auto-launches on Phase 5 gate PASS |

## Setup

```bash
# Requires: NVIDIA GPU with ≥16GB VRAM; 128GB RAM for Phase 5+
source /home/sharaths/vllm-env/bin/activate   # PyTorch 2.10+cu130
pip install -e ".[dev]"
```

## Training

```bash
# Phase 3: Hopfield memory
bash scripts/launch_phase3.sh        # or via auto_advance.sh

# Gate evaluation
bash scripts/run_gate3.sh            # H2: completion ratio ≥ 1.10
bash scripts/run_gate4.sh            # H3: forgetting ratio < 0.8
bash scripts/run_gate5.sh            # H4/H5: reward proxy
bash scripts/run_gate6.sh            # H6-H9: full ablations

# Full auto-advance pipeline (Phase 3 → 4 → 5 → 6, hands-free)
nohup bash scripts/auto_advance.sh > outputs/auto_advance.log 2>&1 &
```

## Results (Phase 2 complete)

The EFE actor achieves **29.72×** more mean domain-aligned reward per episode than REINFORCE:
- μ_EFE = 2.530 (200 episodes, seed=2025)
- μ_RF = 0.085
- p < 0.001 (paired permutation, 50K perms)

An eleven-layer compounding failure chain (passive corpus environment × free_bits ceiling) was diagnosed and resolved, constituting a reproducible diagnostic methodology for active-inference RSSM training on static creative corpora.

## Citation

```bibtex
@article{pwm2026,
  title={Pratyabhij{\~n}{\=a} World Model: Creative AI through Recognition,
         Active Inference, and Associative Memory},
  author={Sharath, S.},
  journal={arXiv preprint},
  year={2026}
}
```
