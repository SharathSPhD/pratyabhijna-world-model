# PWM Project Status — Handover Document

> Last updated: 2026-05-04 | Git HEAD: `3f5de79` (main)
> Author: Claude Sonnet 4.6 session `f5307e26` → continuation session

---

## 1. Project Identity

**Full name:** Pratyabhijñā World Model (PWM)  
**Repository:** `git@github-sharathsphd:SharathSPhD/pratyabhijna-world-model.git`  
**GitHub:** https://github.com/SharathSPhD/pratyabhijna-world-model  
**Local path:** `/home/sharaths/projects/PWM`  
**Target outputs:** arXiv paper (IEEE citations) + open-source code + HuggingFace dataset

**What it is:** A creative AI research system operationalising Kashmir Śaiva philosophy through active inference on a DreamerV3-class world model with frozen LLM augmentation. The world model learns to recognise (pratyabhijñā) creative patterns in text, then generates narrations at moments of peak surprise (sphurattā events) via a gated LLM layer.

**Why it exists:** PCE v0.4 proved vimarśa (reflexive revision, Hedges' g=0.65) works but the LLM-native cascade (H5, g=0.14) cannot separate from bare LLM. Fix: world model substrate + genuine EFE + intrinsic camatkāra reward.

---

## 2. Environment — Non-Negotiable Setup

```bash
# ALWAYS use this Python env — system python3 is CPU-only PyTorch
source /home/sharaths/vllm-env/bin/activate

# GPU: NVIDIA GB10 Blackwell, CUDA 13.0 (capability 12.1), 128GB unified LPDDR5X
# Note: PyTorch 2.10 reports capability warning (max 12.0) — ops still run correctly

# SSH remote (must use this alias, not github.com):
git remote: git@github-sharathsphd:SharathSPhD/pratyabhijna-world-model.git

# Install:
pip install -e ".[dev]"
# hflayers NOT on PyPI — we have native implementation in pwm/memory/citta_store.py
```

**Key .env variables** (not committed — see `.env.example`):
```
LLM_PROVIDER=nemotron-local          # or claude-api, openai-api, gemini-api
LLM_PRIMARY_API_BASE=http://localhost:8000/v1
LLM_FAST_API_BASE=http://localhost:8001/v1
ANTHROPIC_API_KEY=...                # only needed for claude-api provider
WANDB_PROJECT=pratyabhijna-world-model
CORPUS_ROOT=data/corpus
```

---

## 3. Repository State

### Git log (all commits on main)

```
3f5de79  feat(phase1): CorpusDataset + PhaseOneEnv — real text training for Phase 1
9caa45f  fix: corpus/build.py — use merve/poetry, fix bs4 call API
5281fcc  chore: Phase 0 gate PASS — 4.6M corpus tokens, 17/17 tests
22c6493  feat: Phase 0 completion — corpus ingest, remaining modules, GLOSSARY
3d44672  feat: Phase 0 foundation — world model core, corpus pipeline, training loop
```

### Branches

| Branch | State | Description |
|--------|-------|-------------|
| `main` | HEAD at `3f5de79` | Stable, all tests pass |
| `phase-1/rssm-text` | In progress, at `3f5de79` | Phase 1 Aparā training |

### Worktrees

| Path | Branch | Purpose |
|------|--------|---------|
| `/home/sharaths/projects/PWM` | `main` | Primary |
| `/home/sharaths/projects/pwm-phase1` | `phase-1/rssm-text` | Phase 1 development |

---

## 4. Architecture — Complete Component Inventory

### 4.1 World Model (`pwm/world_model/`)

| File | Class | Status | Key details |
|------|-------|--------|-------------|
| `trika.py` | `TrikaWorldModel` | **Complete** | 3-level RSSM hierarchy (Aparā/Parāparā/Parā). Dual-list pattern: `_level_list: list[TrikaCoreLevel]` + `self.levels = nn.ModuleList(...)`. `world_model_loss()` returns per-level prefixed losses. Tested. |
| `rssm.py` | `TrikaCoreLevel` | **Complete** | Per-level RSSM. DRAMA decoupled posterior (z only from obs, not h). `input_proj = Linear(latent+action → hidden)` before backbone. KL free_bits. Symlog/twohot reward head. |
| `mamba_backbone.py` | `MambaBackbone`, `GRUFallback` | **Complete** | Mamba-2 SSM for Parā level (d_model=512, d_state=64, headdim=64, chunk_size=256). GRUFallback for CPU (tested). Transparent routing via `_use_mamba(x) = has_mamba and x.is_cuda`. Adapted from DreamPrice project (`/home/sharaths/projects/dreamprice`). |
| `losses.py` | `make_twohot_bins`, `symlog`, `twohot_loss` | **Complete** | Scale-invariant reward encoding (DreamerV3 §3). `self.bins: Tensor` declared before `register_buffer()` to prevent Pyright widening. |

**Validated WM run:**
```python
TrikaWorldModel(obs_dim=512, hidden_dim=512, stoch_dim=32, stoch_classes=32, action_dim=64, n_levels=1)
# 5.4M parameters, VFE 7.23→6.38 over 20 steps on random data, no NaN
```

### 4.2 Active Inference (`pwm/active_inference/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `efe_actor.py` | `EFEActor`, `CRSPPPreference` | **Complete** | EFE = pragmatic_value + epistemic_value. Categorical(action_dim) policy. `actor_loss()` for Phase B. Phase 2+ replaces `EFEActorStub` in train.py. |
| `crspp.py` | `SRMatrix`, `CRSPPModel` | **Complete** | SR-AIF (Lefrançois 2024). SR Bellman update with soft target network. `composite_reward()` blends R_camatk + SR value. |
| `efe_utils.py` | — | **Stub** | pymdp.maths bridge (type-ignored). Only `efe_utils` imported by pancakrtya_loop. |

### 4.3 Memory (`pwm/memory/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `citta_store.py` | `HopfieldBank`, `CittaStoreLevel`, `CittaStore` | **Complete** | Native Hopfield (no hflayers dep). High β=4.0 episodic (smṛti), low β=0.25 semantic (ālayavijñāna). Blend gate: `sigmoid(Linear(cat(q, recalled)))`. Tested. |
| `replay.py` | `SumTree`, `ReplayBuffer`, `Transition` | **Complete** | PER sum-tree. α=0.6, β_start=0.4→1.0 over 100K frames. `sample()` returns `(transitions, indices, weights)`. Tested. |
| `skill_lib.py` | `SkillLibrary` | **Missing** | Listed in CLAUDE.md but not yet created. Phase 3+ dependency. SQLite + FAISS. |

### 4.4 Sleep (`pwm/sleep/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `consolidation.py` | `NREMPhase`, `REMPhase`, `SleepConsolidator` | **Complete** | NREM: replay→WM loss→SHY down-scale (cfg.shy_scale=0.95). REM: H=32 imagination→recognition net retrain. ThermSleep: ΔF_vfe/(ΔF_vfe+1e-6) efficiency ratio. Phase 4+. |

**Note:** CLAUDE.md lists `pwm/sleep/scheduler.py`, `nrem.py`, `rem.py`, `therm_budget.py` as separate files. All are **consolidated into `consolidation.py`** — no need to re-split.

### 4.5 Vimarsa (`pwm/vimarsa/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `bridge.py` | `VimarsaBridge` | **Complete** | Multi-head cross-attention (WM h_t → k=4 soft-prompt tokens for LLM). Text fallback `format_prefix_text()` for text-only LLMs. Phase 5+. |
| `narrator.py` | `CamatkaraNarrator`, `NarrationResult` | **Complete** | LLM narration at sphurattā events. Skill commit at camatk>0.7 and quality>0.6. Embedding proxy via WM hidden state. Phase 5+. |
| `deckard.py` | `DECKARDPlanner`, `AWMProposal` | **Complete** | LLM AWM proposals (creative intentions). JSON-parsed from LLM response. 9 rasas as target. Phase 5+. |

### 4.6 Agents (`pwm/agents/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `vimarsha_agent.py` | `VimarshaAgent`, `VimarshaStub` | **Complete** | ONLY true smolagents agent. commit/revise/reject gate. `VimarshaStub` identity fallback when smolagents not installed. |
| `memory_agent.py` | `MemoryAgent` | **Complete** | Post-commit consolidation hook (NOT smolagents). Stores h_t in CittaStore. Semantic threshold=0.75. |
| `sleep_agent.py` | `SleepAgent`, `SleepScheduler` | **Complete** | NREM/REM orchestrator. Plateau detection + interval trigger. `maybe_sleep()` called from training loop. |

### 4.7 Pipeline (`pwm/pipeline/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `pancakrtya_loop.py` | `PancakrtyaLoop`, `LoopState`, `LoopConfig` | **Complete** | Six-stage śakti cascade (cit→ānanda→icchā→apohana→jñāna→kriyā) as **single Python call stack** — not separate processes. VimarshaAgent called on sphurattā. LLM called only at sphurattā events (jñāna). `done_t.any()` resets sequence model state. |

**CRITICAL:** Do not break the śakti cascade into separate agents/processes. The continuous WM state `(h_t, z_t)` must flow unbroken through all steps.

### 4.8 Rewards (`pwm/rewards/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `camatk.py` | `CamatkaraReward` | **Complete** | R_camatk = α₁ΔF_vfe + α₂ΔI_Hopfield + α₃Empowerment. Sphurattā: VFE < 5th-percentile AND Hopfield entropy drop AND min_gap. |
| `mala.py` | `MalaRegulariser`, `AnavaRegulariser`, `MayiyaRegulariser`, `KarmaRegulariser` | **Complete** | Three impurity regularisers. Āṇava: entropy penalty (free_nats=1.0). Māyīya: batch diversity loss. Kārma: action entropy penalty. Combined call: `mala(logits=..., z_sample=..., action_logits=...)`. |

### 4.9 Perception (`pwm/perception/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `text.py` | `TextEncoder`, `TokenSequenceEncoder` | **Complete** | `TextEncoder`: lazy-loads `all-MiniLM-L6-v2` (384-dim) → Linear(384, obs_dim=512). Tested in Phase 1 pipeline. |
| `vjepa2.py` | — | **Missing** | V-JEPA 2 frozen encoder (Phase 5+ for visual observations). |
| `diamond.py` | — | **Missing** | DIAMOND EDM decoder (Phase 5+). |

### 4.10 LLM Backend (`pwm/llm/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `backend.py` | `LLMBackend` | **Complete** | LiteLLM unified interface. Roles: agama, jnana, icccha, vimarsha. Provider switch via `configs/llm_backend.yaml` or `--set llm.provider=...`. Never hardcode model names in Python. |

### 4.11 Context (`pwm/context/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `avacchedaka.py` | `AvacchedakaStore` | **Complete** | PCEH client wrapper. Qualificands: vimarsha, anumana. Used by `CommitNarrationTool` in VimarshaAgent. |

### 4.12 Data (`pwm/data/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `corpus_dataset.py` | `CorpusDataset`, `PhaseOneEnv` | **Complete** | 59,070 chunks from 4.6M token corpus. `PhaseOneEnv` is the drop-in replacement for `TextEnv` stub in train.py. Tested: 5-step training, VFE 6.76→6.39, no NaN. |

### 4.13 Training (`pwm/scripts/`)

| File | Class | Status | Notes |
|------|-------|--------|-------|
| `train.py` | `PWMTrainer` | **Complete (stub env)** | 3-phase DreamerV3 loop. Currently uses `TextEnv` (random obs). **Phase 1 task**: swap `TextEnv` → `PhaseOneEnv`. Phase A: Adam lr=1e-4, clip=1000, B=32, T=64. Phase B: Adam lr=3e-5, clip=100, H=13. Phase C: twohot CE + EMA decay=0.98. |

**To wire Phase 1 training:**
```python
# In train.py PWMTrainer.__init__, replace:
self.env = TextEnv(...)
# with:
from pwm.data.corpus_dataset import PhaseOneEnv
self.env = PhaseOneEnv(corpus_dir=cfg.corpus.data_dir, batch_size=B, seq_len=T, ...)
```

### 4.14 Evaluation (`pwm/eval/`)

| File | Status | Notes |
|------|--------|-------|
| `camatk_eval.py` | **Missing** | DTW correlation for H6 |
| `svat.py` | **Missing** | Svātantrya score (latent novelty) for H9 |
| `metre.py` | **Missing** | Sanskrit metre validator for H8 |
| `ablations.py` | **Missing** | Ablation runner H1–H9 + A1–A6 |

### 4.15 Corpus (`corpus/`)

| File | Status | Notes |
|------|--------|-------|
| `build.py` | **Complete** | 994 lines. CLI: `python -m corpus.build --sources hf_poetry,hf_wiki_philosophy,gutenberg`. Sources: hf_poetry (merve/poetry), hf_wiki_philosophy, hf_c4, hf_pile, hf_sanskrit, gretil, gutenberg. |
| `ingest/gretil.py` | **Complete** | Śaiva texts from GRETIL (network-dependent) |
| `ingest/poetry_foundation.py` | **Complete** | Poetry Foundation (network-dependent) |
| `ingest/gutenberg.py` | **Complete** | Public domain via gutendex API |
| `ingest/arxiv_creativity.py` | **Complete** | arXiv creativity papers |
| `ingest/hf_datasets.py` | **Complete** | HuggingFace: merve/poetry, ai4bharat/sangraha |
| `annotate/` | **Missing** | Metre annotator, rasa classifier, camatkāra annotator |
| `tokenise.py` | **Missing** | BPE tokeniser with Devanāgarī support (32K vocab) |

### 4.16 Benchmarks

| File | Status | Notes |
|------|--------|-------|
| `benchmarks/autoreport.py` | **Complete** | 278 lines. Generates H1–H9 JSON artefacts from results/. |
| `benchmarks/results/phase_0_gate.json` | **PASS** | 4.6M tokens, 17 tests, VFE finite |
| `benchmarks/results/phase_1_gate.json` | **Pending** | Write after Phase 1 convergence |

---

## 5. Phase Status

### Phase 0 — Foundation ✅ COMPLETE

**Gate:** `benchmarks/results/phase_0_gate.json` → status PASS (2026-05-04)

| Criterion | Result |
|-----------|--------|
| ≥100K corpus tokens | 4,622,674 (46× target) |
| All unit tests pass | 17/17 |
| WM forward pass finite | VFE=7.22, no NaN/Inf on GB10 |
| DreamerV3 DMC baseline | **Deferred** — requires DMControl env; not blocking Phase 1 |

**Corpus breakdown:**
- `hf_wiki_philosophy` (wikimedia/wikipedia): 3,668,726 tokens
- `gutenberg` (consciousness, aesthetics, imagination topics): 953,948 tokens
- `hf_poetry` (merve/poetry): 573 poems — field now fixed to `content`

### Phase 1 — Aparā RSSM Text Training 🔄 IN PROGRESS

**Branch:** `phase-1/rssm-text` | **Worktree:** `/home/sharaths/projects/pwm-phase1`

**Completed:**
- [x] `PhaseOneEnv` — real corpus text → sentence-transformers → (B,T,512) obs batches
- [x] `CorpusDataset` — 59,070 text chunks ready
- [x] End-to-end smoke test: VFE 6.76→6.39 over 5 steps on real text, no NaN

**Remaining:**
- [ ] Wire `PhaseOneEnv` into `PWMTrainer` (replace `TextEnv` in train.py `__init__`)
- [ ] Run full Phase A: B=32, T=64, 10,000 steps, lr=1e-4, clip=1000
- [ ] Evaluate held-out perplexity vs LSTM baseline
- [ ] Run UMAP on latent z_t — verify metre-cluster separation
- [ ] Write `benchmarks/results/phase_1_gate.json`

**Phase 1 exit criteria** (from `configs/phase1_apara.yaml`):
1. Held-out perplexity competitive with LSTM baseline
2. UMAP shows metre-cluster separation in z_t latent space

**How to run Phase 1 training:**
```bash
source /home/sharaths/vllm-env/bin/activate
cd /home/sharaths/projects/pwm-phase1

# Full Phase A training (modify train.py to use PhaseOneEnv first)
python pwm/scripts/train.py

# Or via Hydra with phase1 config:
python pwm/scripts/train.py --config-name phase1_apara \
    training.batch_size=32 training.seq_len=64 training.phase_a_steps=10000
```

### Phase 2 — EFE Actor ⬜ NOT STARTED

**Prerequisite:** Phase 1 gate JSON written.  
**Config:** `configs/phase2_efe.yaml` (inherits phase1_apara).  
**Key change:** `actor.type = "efe"` — activate `EFEActor` + `CRSPPModel` from `pwm/active_inference/`.  
**Exit criterion:** EFE achieves first sphurattā in ≤50% episodes vs REINFORCE baseline.

### Phase 3 — Hopfield CittaStore ⬜ NOT STARTED

**Prerequisite:** Phase 2 gate.  
**Config:** `configs/phase3_hopfield.yaml`.  
**Key changes:** `memory.enabled=true`; full R_camatk (α₁ΔF + α₂ΔI + α₃Emp).  
**Missing:** `pwm/memory/skill_lib.py` (SQLite + FAISS).  
**Exit criteria:** Pattern completion +10%; sphurattā 0.5–2 events/100 steps.

### Phase 4 — Sleep Consolidation ⬜ NOT STARTED

**Prerequisite:** Phase 3 gate.  
**Config:** `configs/phase4_sleep.yaml`.  
**Key change:** `sleep.enabled=true` — `SleepAgent.maybe_sleep()` in training loop.  
**Exit criteria:** Forgetting reduced ≥20% vs no-sleep; H6 DTW > random baseline.

### Phase 5 — LLM Āgama + Vimarśa Bridge ⬜ NOT STARTED

**Prerequisite:** Phase 4 gate.  
**Config:** `configs/phase5_llm.yaml`.  
**Key changes:** `llm.enabled=true`; `world_model.levels=3` (upgrade to full Trika).  
**Backbone at Parā level:** Mamba-2 (validated on this GB10 via DreamPrice).  
**Missing:** `pwm/perception/vjepa2.py`, `pwm/perception/diamond.py`.  
**Exit criteria:** End-to-end latency ≤30s; H4 meaningful narration rate ≥70%.

### Phase 6 — Full System + Ablations ⬜ NOT STARTED

**Prerequisite:** Phase 5 gate.  
**Config:** `configs/phase6_full.yaml`.  
**Ablations A1–A6** (one config flag each):
- A1: EFE vs REINFORCE (`actor.type=reinforce`)
- A2: Hopfield on/off (`memory.enabled=false`)
- A3: Sleep on/off (`sleep.enabled=false`)
- A4: Vimarśa on/off (`llm.enabled=false`)
- A5: Mala on/off (`mala_regularisers.enabled=false`)
- A6: Trika 1-level vs 3-level (`world_model.levels=1`)

---

## 6. Pre-Registered Hypotheses

All results must be stored in `benchmarks/results/{hypothesis_id}_{seed}_{timestamp}.json`.  
Statistics: paired permutation (50K perms), Hedges' g, BCa 95% CI (10K resamples), Holm-Bonferroni FWE.

| ID | Claim | Phase | Key metric |
|----|-------|-------|-----------|
| H1 | EFE actor > REINFORCE on sparse creative reward | 2 | Episodes to first sphurattā |
| H2 | Hopfield improves pattern completion | 3 | Occlusion completion accuracy |
| H3 | Sleep reduces catastrophic forgetting | 4 | Forgetting rate (3-domain sequential) |
| H4 | Vimarśa bridge improves narration quality | 5 | Human "meaningful" rate ≥70% |
| H5 | PWM > PCE v0.4 on creative quality | 6 | R_camatk density + S_svātantrya |
| H6 | Camatkāra correlates with human aesthetic judgment | 4+ | DTW distance (lower=better) |
| H7 | 3-level Trika > 1-level on long-horizon creativity | 5 | 16-step prediction MSE |
| H8 | Mala regularisers prevent latent collapse | 6 | Metre satisfaction rate |
| H9 | S_svātantrya correlates with human novelty ratings | 6 | Spearman ρ |

---

## 7. Critical Rules (MUST follow every session)

1. **Python env:** Always `source /home/sharaths/vllm-env/bin/activate`. Never `python3` from system.
2. **Git worktrees:** All feature work in worktrees (`git worktree add`). Main = stable only.
3. **Push frequently:** After every 3–5 commits: `git push origin <branch>`.
4. **Phase gates:** Never advance phase without writing gate JSON and all exit criteria met.
5. **No mocks:** All data from real public sources. All models from real weights.
6. **śakti cascade integrity:** `PancakrtyaLoop` must remain a single Python call stack. Never split into processes.
7. **LLM gating:** LLM called ONLY at sphurattā events. Never on every step.
8. **Philosophical docstrings:** Every module must have Sanskrit concept + textual source + computational primitive. See `docs/GLOSSARY.md`.
9. **Config-driven:** All parameters in `configs/`. Use Hydra overrides. No code changes for hyperparameter sweeps.
10. **Statistical rigour:** Every hypothesis result → JSON artefact. Report negative results with same rigour as positives.

---

## 8. Known Issues and Workarounds

| Issue | Workaround / Status |
|-------|---------------------|
| PyTorch CUDA capability warning (12.1 > 12.0 max) | Harmless — GB10 ops run correctly. Warning logged at startup. |
| `hflayers` not on PyPI | Not needed — `CittaStore` has native Hopfield implementation. Comment in pyproject.toml. |
| `indic-nlp-library` / `aksharamukha` optional | Removed from core deps, commented. Install manually if needed for Sanskrit annotation. |
| `hf_poetry` source returned 0 tokens (first run) | Fixed: `poem_sentiment` had `trust_remote_code` removed by HF. Now uses `merve/poetry` with `content` field. |
| `gutendex.com` timeouts for gutenberg search | 3 of 8 topics timed out. 4 succeeded with 953K tokens. Retry if needed. |
| GRETIL network-dependent | Network not required for Phase 1. GRETIL texts load if online; skip gracefully if not. |
| `train.py` still uses `TextEnv` stub | **Phase 1 task:** swap to `PhaseOneEnv`. See §4.13 above. |
| `EFEActorStub` in train.py | Zero-init placeholder. Replace with `EFEActor` in Phase 2. |
| `pwm/memory/skill_lib.py` missing | Phase 3+ dependency. Create when Hopfield is integrated. |
| `pwm/eval/` entirely missing | Evaluation modules needed for Phase 3+. Create as phases are reached. |

---

## 9. Corpus State

| Source | Tokens | Files | Notes |
|--------|--------|-------|-------|
| hf_wiki_philosophy | 3,668,726 | 10 batches × 500 articles | Philosophy + arts Wikipedia articles |
| gutenberg | 953,948 | 16 books | Consciousness, aesthetics, imagination, literary theory |
| hf_poetry (merve/poetry) | ~12,000 (est.) | 573 poems | Fixed in 9caa45f — field was `content` |
| gretil | 0 (pending) | — | Network-dependent; run when online |
| arxiv | 0 (pending) | — | Run with `--sources arxiv` |
| **TOTAL** | **4,622,674** | — | Manifest: `data/corpus/corpus_manifest.json` (gitignored) |

**Corpus chunks ready for training:** 59,070 (chunk_chars=1024)

**Run additional ingestion:**
```bash
# Add Sanskrit + arXiv:
python -m corpus.build --sources gretil,arxiv --output-dir data/corpus
# Add more poetry:
python -m corpus.build --sources hf_poetry --output-dir data/corpus
```

---

## 10. Test Suite

All 17 tests pass in <1s on CPU:

```
tests/test_citta_store.py   4 tests — HopfieldBank recall, CittaStore blend gate, retrieval
tests/test_losses.py        4 tests — symlog, twohot encode/decode, VFE bound
tests/test_rssm.py          6 tests — shapes, gradients, KL free bits, stoch sampling
tests/test_trika.py         3 tests — 3-level WM, world_model_loss, imagine_step
```

**Run:** `source /home/sharaths/vllm-env/bin/activate && pytest tests/ -v`

**Missing tests** (create as phases progress):
- `test_efe.py` — EFE computation, SR loss, CRSPP composite reward
- `test_hopfield.py` — CittaStore retrieval (covered partially in test_citta_store)
- `test_corpus.py` — Corpus pipeline, CorpusDataset chunk counts
- `test_camatk.py` — Camatkāra reward, sphurattā detection
- `test_mala.py` — Three regulariser forward passes

---

## 11. Key Technical Decisions (with reasoning)

### Mamba-2 over S4 for Parā level
DreamPrice project (`/home/sharaths/projects/dreamprice`) validated Mamba-2 on this exact GPU (GB10 Blackwell) with `d_model=512, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=256, rmsnorm=True`. S4 was originally planned but Mamba-2 is simpler, faster, and already proven here. See `memory/dreamprice_patterns.md`.

### DRAMA decoupled posterior
Posterior `z_t = f(x_t)` only — no `h_t` input to recognition density. This is the DreamPrice pattern. Simplifies the information flow and prevents the posterior from short-circuiting the dynamics model. `input_proj = Linear(latent+action → hidden)` before backbone handles the coupling instead.

### Native Hopfield over hflayers
`hflayers` is not available on PyPI for Python 3.12. Our `CittaStore` implements modern Hopfield attention natively: `attn = softmax(β * q @ K^T)`, `retrieved = attn @ K`. Functionally equivalent, no external dependency.

### Single-file sleep consolidation
CLAUDE.md lists `scheduler.py`, `nrem.py`, `rem.py`, `therm_budget.py` as separate files. All consolidated into `sleep/consolidation.py` — one file is cleaner for the current scope. Do not re-split unless Phase 4 complexity demands it.

### VimarshaAgent as sole smolagents agent
All other śakti cascade steps share the continuous WM state `(h_t, z_t)` in a single Python call stack. Only `VimarshaAgent` uses smolagents (for the commit/revise/reject deliberation). MemoryAgent and SleepAgent are synchronous hooks, not agents.

### LiteLLM for LLM routing
Provider-agnostic: `nemotron-local` (primary endpoint), `claude-api`, `openai-api`, `gemini-api`, `custom`. Switch via `LLM_PROVIDER` env var or Hydra override. Never hardcode model names in Python.

---

## 12. Immediate Next Steps (Phase 1 continuation)

### Step 1: Wire PhaseOneEnv into PWMTrainer

In `pwm/scripts/train.py`, `PWMTrainer.__init__` (around line 295), replace:
```python
self.env = TextEnv(
    batch_size=B, obs_dim=wm_cfg.obs_dim,
    action_dim=wm_cfg.action_dim, seq_len=T, device=self.device,
)
```
with:
```python
from pwm.data.corpus_dataset import PhaseOneEnv
self.env = PhaseOneEnv(
    corpus_dir=cfg.corpus.get("data_dir", "data/corpus"),
    batch_size=B, seq_len=T,
    obs_dim=wm_cfg.obs_dim, action_dim=wm_cfg.action_dim,
    device=self.device, num_workers=2,
)
```

### Step 2: Run Phase A (WM training)

```bash
source /home/sharaths/vllm-env/bin/activate
cd /home/sharaths/projects/pwm-phase1
python pwm/scripts/train.py  # uses phase1_apara.yaml via default config
```

Target: VFE should descend from ~7.0 to <2.0 over 10K steps on corpus text.

### Step 3: Checkpoint and evaluate

```python
# Checkpoints save to: checkpoints/pwm_step_{N}.pt
# To evaluate latent structure:
python scripts/evaluate.py --checkpoint checkpoints/pwm_step_10000.pt
```

### Step 4: Write Phase 1 gate

```bash
# After satisfying exit criteria, write:
# benchmarks/results/phase_1_gate.json
# Then commit and merge phase-1/rssm-text → main
```

---

## 13. File Inventory (key files, line counts)

```
pwm/scripts/train.py                  805 lines   3-phase DreamerV3 trainer
corpus/build.py                       994 lines   corpus ingestion pipeline
pwm/pipeline/pancakrtya_loop.py       259 lines   śakti cascade
pwm/world_model/rssm.py               329 lines   RSSM per-level
pwm/world_model/trika.py              218 lines   3-level WM
pwm/world_model/mamba_backbone.py     145 lines   Mamba-2 + GRU fallback
pwm/world_model/losses.py              94 lines   symlog, twohot, VFE
pwm/memory/citta_store.py             203 lines   Hopfield CittaStore
pwm/memory/replay.py                  171 lines   PER replay buffer
pwm/active_inference/efe_actor.py     160 lines   EFE actor + CRSPPPreference
pwm/active_inference/crspp.py         148 lines   SR-AIF preference model
pwm/rewards/camatk.py                 ~200 lines  camatkāra reward
pwm/rewards/mala.py                   160 lines   three mala regularisers
pwm/vimarsa/bridge.py                 103 lines   WM↔LLM cross-attention
pwm/vimarsa/narrator.py               171 lines   camatkāra narration
pwm/vimarsa/deckard.py                146 lines   DECKARD AWM planner
pwm/agents/vimarsha_agent.py          238 lines   smolagents deliberative gate
pwm/agents/memory_agent.py            104 lines   post-commit consolidation
pwm/agents/sleep_agent.py             136 lines   NREM/REM orchestrator
pwm/sleep/consolidation.py            260 lines   NREM + REM + ThermSleep
pwm/perception/text.py                114 lines   sentence-transformer encoder
pwm/data/corpus_dataset.py            170 lines   CorpusDataset + PhaseOneEnv
pwm/llm/backend.py                    ~200 lines  LiteLLM unified interface
pwm/context/avacchedaka.py            ~100 lines  PCEH context store client
benchmarks/autoreport.py              278 lines   H1-H9 result generator
docs/GLOSSARY.md                       96 lines   Sanskrit↔computational mapping
```

---

## 14. Reference: Sanskrit Concept → Module

| Sanskrit concept | Module | Class/function |
|-----------------|--------|---------------|
| Pratyabhijñā (recognition density) | `rssm.py` | `TrikaCoreLevel.posterior()` |
| Spanda (stochastic latent) | `rssm.py` | `z_t ~ Categorical(32×32)` |
| Vimarśa (self-reflexive evaluation) | `agents/vimarsha_agent.py` | `VimarshaAgent.run()` |
| Sphurattā (camatkāra event) | `rewards/camatk.py` | `CamatkaraReward.detect_sphuratta()` |
| Svātantrya (creative freedom) | `active_inference/efe_actor.py` | `EFEActor` entropy bonus |
| Camatkāra (aesthetic wonder) | `rewards/camatk.py` | `CamatkaraReward.compute()` |
| Ālayavijñāna (semantic memory) | `memory/citta_store.py` | `HopfieldBank(β=0.25)` |
| Smṛti (episodic memory) | `memory/citta_store.py` | `HopfieldBank(β=4.0)` |
| Pañcakṛtya (five divine acts) | `pipeline/pancakrtya_loop.py` | `PancakrtyaLoop.step()` |
| Svapna (dream/REM) | `sleep/consolidation.py` | `REMPhase.run_cycle()` |
| Āṇavamala (latent collapse) | `rewards/mala.py` | `AnavaRegulariser` |
| Māyīyamala (mode collapse) | `rewards/mala.py` | `MayiyaRegulariser` |
| Kārmamala (reward hacking) | `rewards/mala.py` | `KarmaRegulariser` |
| Icchā-śakti (creative will) | `vimarsa/deckard.py` | `DECKARDPlanner.propose()` |
| Āgama (LLM testimony) | `llm/backend.py` | `LLMBackend.call(role="agama")` |

Full glossary: `docs/GLOSSARY.md`
