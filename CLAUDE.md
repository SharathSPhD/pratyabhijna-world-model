# Pratyabhijñā World Model (PWM) — Claude Code Project Manifest

> *"Consciousness recognises itself in every creative act."*  
> — Utpaladeva, *Īśvarapratyabhijñākārikā* 1.3

---

## Project Identity

**What:** A creative AI research system operationalising Kashmir Śaiva philosophy through active inference on a DreamerV3-class world model with frozen LLM augmentation. A fully trained, end-to-end research prototype.

**Why:** PCE v0.4 proved vimarśa (reflexive revision, g=0.65) works but the LLM-native cascade (H5, g=0.14) cannot separate from bare LLM. The fix: world model substrate + genuine EFE + intrinsic camatkāra reward.

**Target:** arXiv paper (IEEE citations, LaTeX/PDF, professional visuals) + open-source code + HuggingFace dataset.

**GitHub:** `SharathSPhD/pratyabhijna-world-model` via `git@github-sharathsphd:`

---

## Mandatory Workflow Rules (Follow in EVERY Session)

### 1. Python Environment
**Always use the vllm-env:** `/home/sharaths/vllm-env/bin/python`  
System python3 is CPU-only PyTorch — never use it for ML work.
```bash
source /home/sharaths/vllm-env/bin/activate
```

### 2. Git Worktrees
Use worktrees for all feature/phase development:
```bash
# Create a phase worktree
git worktree add ../pwm-phase1 -b phase-1/rssm-text
git worktree add ../pwm-phase2 -b phase-2/efe-actor
# Main branch = stable, reviewed code only
```
Branch naming: `phase-N/feature-name` or `fix/description` or `exp/experiment-name`.

### 3. Push Frequently
After every 3–5 meaningful commits: `git push origin <branch>`.  
The SSH remote requires the host alias: `git@github-sharathsphd:SharathSPhD/pratyabhijna-world-model.git`

### 4. Phase Gates
Do not advance to Phase N+1 until Phase N exit criteria are met and documented in `benchmarks/results/phase_N_gate.json`.

### 5. Plugin Usage (Mandatory)
- **TRIZ** (`triz-engine:analyze`, `:matrix`, `:principles`): invoke when hitting architectural contradictions
- **Attractor-flow** (`attractor-flow:attractor-orchestrator`): invoke before major design decisions; use explorer-agent during ideation (divergence), convergence-agent during implementation
- **Pratyaksha PCEH**: context management across long sessions; also the runtime inter-agent OS
- **Ralph Wiggum** (`ralph-wiggum:ralph-loop`): invoke after complex multi-step implementations for completion assurance
- **Skill**: `superpowers:brainstorming` for design exploration; `superpowers:executing-plans` for implementation

### 6. No Mocks / No Synthetics
All data: real public sources (GRETIL, HuggingFace, Poetry Foundation, arXiv).  
All models: real weights. No placeholder/stub implementations in core ML code.

### 7. Statistical Rigour
Every hypothesis result → `benchmarks/results/{hypothesis_id}_{seed}_{timestamp}.json`  
Tests: paired permutation (50K perms), Hedges' g, BCa CI (10K resamples), Holm-Bonferroni.  
All ablations run ≥3 seeds. Report negative results with the same rigour as positives.

### 8. Config-Driven Development
All phase-specific settings in `configs/`. Use Hydra overrides, not code changes.  
Feature flags in configs enable/disable modules — all 6 ablations run from single config change.

### 9. Philosophical Rigour
Every module must have a docstring with:
- The Sanskrit concept it implements
- The textual source (author, work, verse/section)
- The computational primitive it realises
This prevents philosophical drift. See `docs/GLOSSARY.md` for the canonical mapping.

### 10. LLM Backend
Default: local Nemotron-3-Nano-30B GGUF via `configs/llm_backend.yaml`.  
API keys in `.env` (gitignored). Switch provider: `--set llm.provider=claude-api`.  
Never hardcode model names or API endpoints in Python code — route through `LLMBackend`.

---

## Repository Structure

```
pratyabhijna-world-model/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview
├── .env.example                 # API key template (gitignored: .env)
├── pyproject.toml               # Package definition (uv/pip)
├── pwm/
│   ├── world_model/
│   │   ├── trika.py             # TrikaWorldModel (3-level RSSM hierarchy)
│   │   ├── rssm.py              # TrikaCoreLevel (per-level RSSM + EFE)
│   │   ├── s4_backbone.py       # S4 integration for Para level (R2I)
│   │   └── losses.py            # VFE loss, symlog, twohot utilities
│   ├── active_inference/
│   │   ├── efe_actor.py         # EFEActor (replaces REINFORCE)
│   │   ├── crspp.py             # CRSPP preference model (SR-AIF)
│   │   └── efe_utils.py         # pymdp math bridge + EFE term computations
│   ├── memory/
│   │   ├── citta_store.py       # CittaStore (Hopfield episodic + semantic)
│   │   ├── replay.py            # Prioritised experience replay (sum-tree)
│   │   └── skill_lib.py         # Voyager-style skill library (SQLite+FAISS)
│   ├── sleep/
│   │   ├── scheduler.py         # SleepScheduler (trigger logic)
│   │   ├── nrem.py              # NREM consolidation phase
│   │   ├── rem.py               # REM dreaming phase
│   │   └── therm_budget.py      # ThermSleepBudget (thermodynamic stopping)
│   ├── vimarsa/
│   │   ├── bridge.py            # VimarsaBridge (WM ↔ LLM cross-attention)
│   │   ├── narrator.py          # CamatkaraNarrator (sphurattā → skill entry)
│   │   └── deckard.py           # AWM proposals (DECKARD-style LLM planning)
│   ├── rewards/
│   │   ├── camatk.py            # CamatkaraReward (ΔF + ΔI_Hopfield + Empowerment)
│   │   └── mala.py              # MalaRegularisers (āṇava, māyīya, kārma)
│   ├── pipeline/
│   │   └── pancakrtya_loop.py   # PancakrtyaLoop (śakti cascade: cit→vimarśa)
│   ├── agents/
│   │   ├── vimarsha_agent.py    # VimarshaAgent (smolagents deliberative gate)
│   │   ├── memory_agent.py      # MemoryAgent (post-commit consolidation)
│   │   └── sleep_agent.py       # SleepAgent (NREM/REM orchestrator)
│   ├── llm/
│   │   └── backend.py           # LLMBackend (LiteLLM unified interface)
│   ├── perception/
│   │   ├── text.py              # BPE tokeniser + embedding encoder
│   │   ├── vjepa2.py            # V-JEPA 2 frozen encoder (v2.0+)
│   │   └── diamond.py           # DIAMOND EDM decoder (Phase 5+)
│   ├── context/
│   │   └── avacchedaka.py       # AvacchedakaStore (PCEH client wrapper)
│   └── eval/
│       ├── camatk_eval.py       # Camatkāra evaluation (DTW correlation)
│       ├── svat.py              # Svātantrya score (latent novelty)
│       ├── metre.py             # Sanskrit metre validator
│       └── ablations.py         # Ablation runner (H1–H9 + A1–A6)
├── corpus/
│   ├── ingest/                  # Source-specific download scripts
│   │   ├── gretil.py            # GRETIL Sanskrit corpus ingestion
│   │   ├── poetry_foundation.py # Poetry Foundation (English)
│   │   ├── gutenberg.py         # Project Gutenberg (public domain poetry)
│   │   ├── arxiv_creativity.py  # arXiv scientific creativity papers
│   │   └── hf_datasets.py       # HuggingFace dataset downloaders
│   ├── annotate/
│   │   ├── metre_annotator.py   # Sanskrit metre detection + annotation
│   │   ├── rasa_classifier.py   # Rasa (aesthetic emotion) classifier
│   │   └── camatk_annotator.py  # Human camatkāra annotation interface
│   ├── tokenise.py              # BPE tokeniser with Devanāgarī support
│   └── benchmark/               # Held-out evaluation benchmark (100K tokens)
├── configs/
│   ├── default.yaml             # Default config (inherits phase1_apara)
│   ├── llm_backend.yaml         # Provider configs (nemotron-local, claude-api, etc.)
│   ├── phase0_foundation.yaml   # Phase 0: env setup + corpus pipeline
│   ├── phase1_apara.yaml        # Phase 1: Aparā-only text WM
│   ├── phase2_efe.yaml          # Phase 2: + EFE actor
│   ├── phase3_hopfield.yaml     # Phase 3: + Hopfield Citta-store
│   ├── phase4_sleep.yaml        # Phase 4: + Sleep consolidation
│   ├── phase5_llm.yaml          # Phase 5: + LLM āgama + vimarśa bridge
│   └── phase6_full.yaml         # Phase 6: full system + all ablations
├── benchmarks/
│   ├── results/                 # JSON artefacts (never delete, version-controlled)
│   │   └── .gitkeep
│   └── autoreport.py            # Auto-report generator (PCE v0.4 style)
├── scripts/
│   ├── train.py                 # Main training entry point
│   ├── evaluate.py              # Evaluation entry point
│   ├── trace.py                 # Latent trace inspector
│   └── serve_llm.py             # Start local LLM server (vLLM/llama.cpp)
├── paper/
│   ├── main.tex                 # Paper source (IEEE style)
│   ├── figures/                 # Generated figures (matplotlib/TikZ)
│   ├── tables/                  # Generated tables (LaTeX)
│   └── Makefile                 # latexmk build → PDF
├── docs/
│   ├── GLOSSARY.md              # Sanskrit ↔ computational concept canonical mapping
│   ├── ADR.md                   # Architecture Decision Records
│   └── adr/                     # Individual ADR files
└── tests/
    ├── test_rssm.py             # RSSM unit tests (shapes, gradients, VFE)
    ├── test_efe.py              # EFE computation tests
    ├── test_hopfield.py         # CittaStore retrieval tests
    ├── test_corpus.py           # Corpus pipeline tests
    └── test_camatk.py           # Camatkāra reward tests
```

---

## Hardware Reference

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GB10 Blackwell |
| Memory | 128GB unified LPDDR5X |
| CUDA | 13.0 (Driver 580.142) |
| Storage | 3.7TB NVMe (~3TB free) |
| Python env | `/home/sharaths/vllm-env/` (PyTorch 2.10+cu130) |

### Memory Budget by Phase

| Phase | WM | LLM | Total | Headroom |
|-------|----|----|-------|---------|
| Phase 1 (Aparā only) | ~12GB | 30B GGUF ~30GB | ~55GB | 73GB |
| Phase 3 (+Hopfield) | ~15GB | 30B GGUF ~30GB | ~60GB | 68GB |
| Phase 5 (full, 49B evicted) | ~55GB | 120B FP4 ~44GB | ~103GB | 25GB |

---

## Pre-Registered Hypotheses

| ID | Claim | Metric |
|----|-------|--------|
| H1 | EFE actor > REINFORCE on sparse creative reward | Episodes to first sphurattā |
| H2 | Hopfield improves pattern completion | Occlusion completion accuracy |
| H3 | Sleep reduces catastrophic forgetting | Forgetting rate (3-domain sequential) |
| H4 | Vimarśa bridge improves narration quality | Human "meaningful" rate ≥70% |
| H5 | PWM > PCE v0.4 on creative quality | R_camatk density + S_svātantrya |
| H6 | Camatkāra correlates with human aesthetic judgment | DTW distance (lower=better) |
| H7 | 3-level Trika > 1-level on long-horizon creativity | 16-step prediction MSE |
| H8 | Mala regularisers prevent latent collapse | Metre satisfaction rate |
| H9 | S_svātantrya correlates with human novelty ratings | Spearman ρ |

All statistical tests: paired permutation (50K), Hedges' g (small-sample corrected), BCa 95% CI (10K resamples), Holm-Bonferroni FWE correction.

---

## Sanskrit Concept → Computational Primitive (Quick Reference)

| Sanskrit | Source | Computational Realisation |
|----------|--------|--------------------------|
| Pratyabhijñā | ĪPK 1.3–1.4 (Utpaladeva) | Recognition density q_φ(z_t\|h_t,o_t) |
| Spanda | SpandaK 1.1 (Vasugupta) | Stochastic latent z_t ~ Cat(32×32) |
| Vimarśa | ĪPK 1.5.11 (Utpaladeva) | f_self(h_t,z_t) + LLM bridge narration |
| Sphurattā | TĀ 1.56 (Abhinavagupta) | Camatkāra event C_t=1 |
| Svātantrya | ĪPK 2.1 | Max-entropy policy prior |
| Camatkāra | Locana ad DhvA 1.1 | R_camatk = α₁ΔF + α₂ΔI_Hopfield + α₃Empowerment |
| Ālayavijñāna | PHṛ sūtra 9 | Semantic Hopfield store (learnable prototypes) |
| Smṛti | — | Episodic Hopfield store (FIFO buffer) |
| Citi | PHṛ sūtra 1 | Trained prior p_θ(z) |
| Citta | PHṛ sūtra 9 | Posterior Q(z\|o) |

Full glossary: `docs/GLOSSARY.md`

---

## Key External Repos (do not clone — use as pip deps or submodules)

- `NM512/dreamerv3-torch` — DreamerV3 PyTorch base
- `state-spaces/s4` — S4 backbone for Para level
- `ml-jku/hopfield-layers` (`hflayers`) — Hopfield Citta-store
- `inferactively/pymdp` — EFE math utilities only
- `eloialonso/diamond` — DIAMOND EDM decoder (Phase 5+)

---

## Critical Don'ts

- **Never** use `pymdp.Agent` or `pymdp.envs` — only `pymdp.maths`
- **Never** fragment the śakti cascade (cit→kriyā) into separate agents — they share continuous WM state
- **Never** call the LLM on every step — only at sphurattā events, jñāna slow path, kriyā fluency, vimarśa
- **Never** commit `.env`, model weights, or large data files
- **Never** use `system python3` — always `vllm-env`
- **Never** advance a phase without passing its exit criteria
