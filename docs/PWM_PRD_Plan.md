# The Pratyabhijñā World Model (PWM): PRD & Implementation Plan
## Product Requirements Document + Phased Engineering Plan
### v1.0 — May 2026

---

## Part A: Product Requirements Document

### A.1 Vision Statement

Build a creative AI system — the Pratyabhijñā World Model — whose core computational primitive is *recognition* (pratyabhijñā), grounded in a learned generative world model governed by active-inference free-energy minimisation, operating on a bespoke creative corpus, and augmented by a frozen LLM in a clearly subordinated āgama role. The system should be:

1. **Philosophically defensible**: every Sanskrit concept in the system traces to a specific textual locus AND a specific computational primitive.
2. **Measurably creative**: creativity is measured intrinsically by the camatkāra signal (R_camatk = ΔF + ΔI_Hopfield + Empowerment), validated by correlation with human aesthetic judgments.
3. **Reproducible and publishable**: every numerical claim traces to a JSON artefact; all ablations are pre-registered.
4. **Implementable on a single DGX Spark 128GB** by one researcher in 5–7 months.

### A.2 User Stories

**US1: Researcher**
> As a researcher, I want to train the PWM on a creative domain corpus and run controlled ablations of each Pratyabhijñā module, so that I can produce a publishable paper demonstrating the superiority of the world-model substrate over the LLM-native PCE v0.4.

**Acceptance criteria:**
- Training runs on DGX Spark without OOM errors at the recommended configuration.
- All six ablations (EFE vs REINFORCE; Hopfield on/off; Sleep on/off; Vimarśa on/off; Mala regularisers on/off; Trika depth 1/2/3) run from a single config change.
- All metrics are logged to WandB/MLflow with full reproducibility artefacts (seeds, checksums, git SHA).

**US2: Creative User**
> As a user, I want to express a creative intent in natural language and receive a generated text artifact (language-agnostic) accompanied by a textual "sphurattā log" that explains the recognition events that produced it, so that I understand why the system made the creative choices it did.

**Acceptance criteria:**
- CLI: `pwm create --intent "compose something that captures the tension before a storm" --provider nemotron-local`
- CLI with commercial API: `pwm create --intent "..." --set llm.provider=claude-api`
- Output: generated text + sphurattā log with narrated WM recognition events.
- Response time: ≤ 30s for a paragraph-length output on DGX Spark (local); ≤ 20s via commercial API.

**US3: Evaluator**
> As an evaluator, I want to inspect the system's latent state, camatkāra reward signal, and skill library to understand why a given output was generated, so that I can assess whether the system's creativity is genuine rather than statistical interpolation.

**Acceptance criteria:**
- `pwm trace --run_id RUN_ID --step T` shows h_t, z_t, R_camatk, skill library query, and Hopfield retrieval at step T.
- `pwm evaluate --output PATH --domain DOMAIN` computes R_camatk and S_svātantrya for a given output.
- `pwm correlate --camatk_log LOG --human_annotations ANN` computes DTW distance between camatkāra event sequence and human annotation timing.

**US4: Benchmark User**
> As a researcher in computational creativity, I want to use the PWM Creative Corpus and Camatkāra Benchmark as a standard evaluation resource, so that I can compare my own systems against a well-defined creativity baseline.

**Acceptance criteria:**
- Corpus is released on HuggingFace/Zenodo under CC-BY licence.
- Benchmark evaluation script runs in < 1 hour on standard hardware.
- Baseline numbers for GPT-4, Claude, DreamerV3-bare, and PWM are reported in the paper.

### A.3 Functional Requirements

**FR1: World Model Training**
- Train a 3-level Trika RSSM (Aparā/Parāparā/Para) on the creative corpus.
- Minimise the variational free energy loss (DreamerV3-style, extended).
- Support text and multimodal (text+image) observations.
- Pass the Aparā-DMC baseline within ±5% of published DreamerV3 scores.

**FR2: Active Inference Actor**
- Replace REINFORCE with EFE minimisation.
- Implement ambiguity, risk, epistemic value, and parameter novelty terms.
- Use CRSPP online preference learning (SR-AIF recipe).
- On sparse-reward creative tasks, achieve faster convergence than REINFORCE baseline.

**FR3: Hopfield Citta-Store**
- Two-mode Hopfield (episodic + semantic) per Trika level.
- Online episodic write at every step; offline semantic update during sleep.
- Sphurattā detection: fires on coincident VFE drop + Hopfield entropy drop.
- Pattern completion under occlusion improves vs no-Hopfield baseline.

**FR4: Sleep Consolidation**
- NREM phase: prioritised replay, VFE descent, Hopfield consolidation, SHY down-scaling.
- REM phase: generative dreaming, EFE actor-critic update, recognition-net retraining, semantic prototype clustering.
- ThermSleep budget: stop when learning efficiency < θ_efficiency or FLOP budget exhausted.
- On sequential task benchmark: catastrophic forgetting reduced ≥20% vs no-sleep baseline.

**FR5: LLM Āgama Layer (Vimarśa Bridge)**
- Default backend: Nemotron 3 Super 120B A12B MoE (~44 GB FP4, TRT-LLM on DGX Spark) for deliberative calls; Nemotron-Super-49B (~28 GB FP8, vLLM) for fast knowledge calls.
- Provider-agnostic: `llm.provider` config switches to claude-api / openai-api / gemini-api / custom via LiteLLM; commercial path requires no local inference stack.
- LoRA-scale vimarśa bridge operates on WM latent projections — independent of LLM provider.
- LLM queried ONLY at: jñāna deep path (needs_jnana=True), kriyā fluency pass, vimarśa deliberation, sphurattā narration.
- LLM narration quality: human evaluators rate ≥70% of narrations as "meaningful" for the context.

**FR5b: Multi-Provider LLM Configurability**
- Single config key `llm.provider` selects: nemotron-local | claude-api | openai-api | gemini-api | custom.
- All LLM calls route through `LLMBackend` (LiteLLM) — no provider-specific code in WM or agent logic.
- CLI: `python -m pwm.main "..." --set llm.provider=claude-api` with env var API keys.
- System runs identically across providers; performance degrades gracefully with smaller models.

**FR6: Camatkāra Evaluation**
- R_camatk computed at every step; logged to WandB.
- Camatkāra density and timing exported per creative episode.
- Svātantrya score computed for all generated outputs.
- Human evaluation protocol implemented: mark-while-reading protocol, DTW correlation.

**FR7: Creative Corpus and Benchmark**
- ≥500K tokens Sanskrit poetry (annotated).
- ≥800K tokens Western poetry (annotated).
- ≥400K tokens scientific creativity (annotated).
- ≥200K tokens cross-domain bridges (annotated).
- ≥100K tokens held-out evaluation benchmark (human-annotated with camatkāra timing).

### A.4 Non-Functional Requirements

- **Peak GPU memory:** ≤ 128 GB on DGX Spark at full Phase 5 configuration.
- **Training stability:** loss curves should not diverge for ≥ 48 hours on the primary domain.
- **Reproducibility:** 3 seeds per ablation; variance across seeds reported.
- **Documentation:** every module has a docstring linking to the Sanskrit concept and the textual source.
- **Evaluation latency:** all evaluation scripts run on a single DGX Spark in ≤ 2 hours.

### A.5 Out of Scope (v1.0)

- Real-time deployment API.
- More than 3 Trika levels (full 36-tattva hierarchy).
- Online learning from user feedback during production use.
- Distributed multi-GPU training.
- Audio generation (deferred to v2.0).
- Visual/multimodal inputs (V-JEPA 2, DIAMOND decoder deferred to v2.0 — v1.0 is **text-only**).
- Language-specific specialisation (Sanskrit metres, English prosody) — core creativity first; language-specific derivatives are future child projects.

---

## Part B: Pre-registered Hypotheses (Research Programme)

Following PCE v0.4's rigorous pre-registration discipline, all claims are pre-registered before data collection.

| ID | Claim | Contrast | Domain | Primary metric |
|---|---|---|---|---|
| **H1** | EFE actor > REINFORCE on sparse creative reward | EFE-PWM vs REINFORCE-PWM | Sanskrit poetry | Episodes to first sphurattā event |
| **H2** | Hopfield Citta-store improves pattern completion | PWM+Hopfield vs PWM−Hopfield | All | Occlusion completion accuracy |
| **H3** | Sleep consolidation reduces forgetting | PWM+Sleep vs PWM−Sleep | Sequential 3-domain | Catastrophic forgetting rate |
| **H4** | Vimarśa bridge improves camatkāra narration quality | PWM+Vimarśa vs PWM−Vimarśa | All | Human "meaningful" rate |
| **H5** | PWM > PCE v0.4 on creative quality | PWM vs PCE-v0.4-cascade | Matched domains | R_camatk density + S_svātantrya |
| **H6** | Camatkāra signal correlates with human aesthetic judgment | PWM intrinsic vs human marks | Held-out evaluation set | DTW distance (lower = better) |
| **H7** | Trika 3-level > 1-level on long-horizon creativity | 3-level vs 1-level PWM | Cross-domain bridge task | 16-step prediction MSE + creativity |
| **H8** | Mala regularisers prevent latent collapse | PWM+Mala vs PWM−Mala | Sanskrit poetry (metres) | Metre constraint satisfaction rate |
| **H9** | S_svātantrya (latent distance) correlates with human novelty ratings | S_svātantrya vs human novelty | Held-out set | Spearman ρ |

**Statistical protocol:** identical to PCE v0.4 — paired permutation (50K permutations), Hedges' g with small-sample correction, BCa bootstrap 95% CI (10K resamples), Holm-Bonferroni correction, fixed-effects pooling for aggregate hypotheses, strict JSON output, negative-result obligation.

---

## Part C: Phased Implementation Plan

### Phase 0: Foundation (Weeks 1–3)

**Goal:** Set up the development environment, baseline DreamerV3, and the creative corpus pipeline.

**Tasks:**

0.1 **Environment setup on DGX Spark**
```bash
# PyTorch 2.x with CUDA Blackwell support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Active inference (EFE module only — no full POMDP rewrite)
pip install inferactively-pymdp
python -c "from pymdp.maths import compute_info_gain, compute_expected_utility; print('pymdp OK')"

# LLM backend (LiteLLM — provider-agnostic)
pip install litellm smolagents
# Local path (DGX Spark): download Nemotron models and build TRT-LLM engines
# API path: export ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY

# Pratyākṣa context infrastructure (PCEH plugin)
git clone https://github.com/SharathSPhD/pratyaksha-context-eng-harness.git \
  ~/.claude/plugins/pratyaksha-context-eng-harness
git clone https://github.com/SharathSPhD/context-engineering-harness.git
git clone https://github.com/SharathSPhD/pramana.git  # Phase 4+ fine-tuning

# DreamerV3 baseline
git clone https://github.com/NM512/dreamerv3-torch
cd dreamerv3-torch && pip install -e .
python train.py --configs dmc_vision --task dmc_walker_walk  # baseline test
```

0.2 **Corpus curation pipeline**
- Build corpus ingestion scripts for Sanskrit poetry (GRETIL sources), Western poetry (Poetry Foundation, Project Gutenberg), scientific creativity (arXiv, specific curated papers).
- Implement BPE tokeniser with Devanāgarī support (add Sanskrit script to tiktoken/sentencepiece).
- Implement annotation pipeline: metre validator (for Sanskrit), rasa classifier, form detector.
- Build the held-out evaluation benchmark: select 50 examples per domain; submit for human annotation.

0.3 **Baseline metrics**
- Run `NM512/dreamerv3-torch` on DMC Walker-Walk: confirm HNS within ±5% of published.
- Run PCE v0.4 on the held-out benchmark (for comparison baseline).
- Record wall-clock and memory usage for each component.

**Exit criteria:** DreamerV3 baseline matches published; corpus pipeline produces ≥100K tokens; DGX Spark runs without OOM at size50m config.

---

### Phase 1: Core World Model on Creative Domain (Weeks 4–8)

**Goal:** Train a DreamerV3-class RSSM on the creative corpus (text domain); establish the spanda/recognition dynamics on creative sequences.

**Tasks:**

1.1 **Adapt DreamerV3 for text sequences**
- Replace the CNN encoder with a text embedding encoder (BPE → linear projection).
- Adapt the decoder to generate text token distributions (softmax over vocabulary).
- Replace image reconstruction loss with token cross-entropy.
- Keep the 32×32 categorical latent (spanda layer).

1.2 **Aparā-level RSSM training**
- Train on the Sanskrit poetry corpus (primary domain for smallest first).
- Monitor: L_pred (token CE), L_dyn (KL balancing), L_rep (representation KL), free bits.
- Validate: held-out perplexity; sample quality (generate continuations of test stanzas).
- Ablation: free bits on vs off (confirm posterior collapse prevention).

1.3 **Spanda validation**
- Visualise the latent trajectory z_t across a generated stanza: does it pulse in synchrony with the metre? (Anuṣṭubh has 8 syllables per pāda — does the latent have corresponding periodicity?)
- Visualise the UMAP embedding of the 32×32 categorical latent across different metres: do different metres cluster?

1.4 **VFE monitoring and sphurattā calibration**
- Plot VFE over training; identify natural threshold for "surprise" events.
- Set sphurattā threshold to 5th percentile of VFE distribution (bottom 5% = most surprising).
- Manual inspection: do low-VFE steps correspond to metrically correct syllables? Do high-VFE steps correspond to prosodic errors?

**Deliverable:** A trained Aparā RSSM on Sanskrit poetry with confirmed spanda dynamics. VFE-based surprise detection operational. Latent space visualisations showing metre clustering.

**Exit criteria:** Held-out perplexity competitive with a simple LSTM on the same corpus; metre-cluster separation visible in UMAP.

---

### Phase 2: EFE Actor (Weeks 9–12)

**Goal:** Replace REINFORCE with EFE minimisation; validate H1 (EFE > REINFORCE on sparse creative reward).

**Tasks:**

2.1 **EFE implementation**
- Implement `EFEActor` as specified in the Architecture Spec §3.
- Implement all four EFE terms: ambiguity (decoder variance), risk (CRSPP preference), epistemic (pseudo-observation KL), novelty (ensemble disagreement).
- Add entropy regulariser on actor for svātantrya preservation.

2.2 **CRSPP preference model**
- Train CRSPP on positive examples from the corpus (human-selected "excellent" stanzas) and negative examples (random or low-rated).
- Validate: do the learned preferences correspond to high-camatkāra states?

2.3 **Sparse reward creative task design**
- Design a sparse-reward creative task: generate a metrically valid anuṣṭubh stanza. Reward = 1 only if the full stanza passes the metre validator; 0 otherwise.
- Compare EFE actor vs REINFORCE actor on this task (pre-registered H1).

2.4 **Information gain validation**
- On a held-out "novel" stanza (new topic, new imagery), confirm that the epistemic value term is high at the novel tokens and low at the formulaic tokens.

**Deliverable:** EFE actor running; H1 ablation data collected; CRSPP preference model trained on Sanskrit poetry quality.

**Exit criteria:** EFE actor achieves first metre-valid stanza in ≤ 50% of the episodes needed by REINFORCE; or the comparison is run to sufficient n for H1 assessment.

---

### Phase 3: Hopfield Citta-Store (Weeks 13–16)

**Goal:** Integrate the Hopfield Citta-store; validate H2 (pattern completion under occlusion) and sphurattā detection.

**Tasks:**

3.1 **CittaStore integration**
- Implement `CittaStore` as specified in Architecture Spec §4.
- Wire episodic write at every step; semantic update during consolidation.
- Integrate retrieval output `c_t` into WM's next-step prediction: `concat([h_t, z_t, c_t])` → decoder.

3.2 **β-schedule design**
- Implement β-schedule: start at intermediate β (metastable = concept-blending regime) during generation; shift to high β (episodic recall) at sphurattā events.
- Monitor retrieval entropy: confirm metastable regime is achieving multi-pattern mixing.

3.3 **Pattern completion benchmark**
- Design occlusion task: present a partially masked stanza (30% of tokens masked) and measure how well the WM reconstructs the masked tokens with vs without Hopfield.
- Expected finding (H2): Hopfield significantly improves completion accuracy on mask-rate ≥ 30%.

3.4 **Sphurattā detector calibration**
- Run 1000 creative episodes; log all sphurattā events.
- Manual inspection of 50 random events: are these genuine "recognition moments" in the generated text?
- Adjust threshold percentile until ≈1 sphurattā event per 100–200 steps.

**Deliverable:** Hopfield Citta-store operational; sphurattā events firing at calibrated rate; H2 ablation data collected.

**Exit criteria:** Pattern completion accuracy improves by ≥10% with Hopfield; sphurattā rate is 0.5–2 events/100 steps.

---

### Phase 4: Sleep Consolidation (Weeks 17–22)

**Goal:** Implement NREM/REM sleep loop; validate H3 (catastrophic forgetting reduction) and H6 (camatkāra correlation with human judgment).

**Tasks:**

4.1 **Prioritised replay buffer**
- Implement sum-tree prioritised buffer (Schaul 2016 recipe).
- Priority = |VFE| × recency_bonus × domain_bonus.
- Validate: samples from the buffer are indeed higher-VFE (more surprising) than uniform sampling.

4.2 **NREM phase implementation**
- Implement `SleepScheduler.run_nrem()` as specified.
- Implement synaptic homeostasis (SHY) down-scaling for un-accessed Hopfield patterns.
- Implement semantic distillation: prune well-modelled episodic patterns → add to semantic store.

4.3 **REM dreaming phase implementation**
- Implement `run_rem()`: prior-only imagination rollout; actor-critic EFE update; recognition-net retraining; K-means prototype clustering.
- Monitor dream latent diversity: do REM dreams explore novel regions of latent space not seen in training?

4.4 **ThermSleep budget implementation**
- Implement `ThermSleepBudget` with FLOP counter and VFE-gain monitor.
- Calibrate stopping criterion on DGX Spark.

4.5 **Sequential task benchmark (H3)**
- Design three sequential creative tasks: (a) Sanskrit poetry, (b) English imagist poetry, (c) scientific analogy generation.
- Train on task A, evaluate on A; train on B, evaluate on A and B; train on C, evaluate on A, B, C.
- Compare PWM+Sleep vs PWM−Sleep: measure forgetting on A after training on B and C.

4.6 **Human evaluation (H6)**
- Recruit 10 human evaluators (literary background preferred).
- Mark-while-reading protocol: evaluators read 20 generated stanzas and mark moments of "aesthetic surprise" or "recognition" (sphurattā equivalent).
- Compute DTW distance between human marks and WM camatkāra event sequence.

**Deliverable:** Sleep loop operational with ThermSleep budget; H3 sequential forgetting data; H6 human evaluation data.

**Exit criteria:** H3 forgetting reduced ≥20% vs no-sleep; H6 DTW distance significantly better than random baseline.

---

### Phase 5: LLM Āgama + Vimarśa Bridge (Weeks 23–26)

**Goal:** Integrate frozen LLM as āgama knowledge layer; implement vimarśa bridge; validate H4 and H5.

**Tasks:**

5.1 **LLM loading and quantisation**
- Load Llama-3-70B in NF4 4-bit via bitsandbytes.
- Confirm 38 GB memory footprint and ≤2s per narration call.
- Alternative: Llama-3-8B at full precision (~16 GB) for faster iteration during development.

5.2 **Vimarśa bridge training**
- Implement `VimarsaBridge` (WM latent → LLM token dim projection; LoRA adapters).
- Training: given a sphurattā event (h_t, z_t), the bridge should project to an LLM prefix that causes the LLM to generate a narration accurately describing the creative context.
- Supervised pre-training: use PCE v0.4 cascade traces as supervision (the vimarśa narrations from v0.4 are gold-standard narrations for matched contexts).

5.3 **Goal specification pipeline**
- Implement `encode_goal()`: translate natural-language creative intent to CRSPP preference vector.
- Validate: does the system generate in the intended style when the preference is set from a goal description?

5.4 **DECKARD AWM proposals**
- Implement long-horizon planning checkpoint: every 500 steps, request an AWM proposal from the LLM.
- Implement AWM parser: extract subgoal sequence from LLM output.
- Implement AWM verification: use EFE planner to evaluate each subgoal's reachability in imagination.

5.5 **Skill library**
- Implement Voyager-style skill library: SQLite + FAISS index.
- At every sphurattā event: write narration + latent embedding.
- At planning time: retrieve top-k similar skills from library.
- Measure library size growth and retrieval quality over a 10-hour creative session.

5.6 **End-to-end pipeline validation (H4, H5)**
- H4: human evaluators rate sphurattā narrations as "meaningful" — target ≥70%.
- H5: compare PWM creative quality against PCE v0.4 cascade on matched prompts using R_camatk and S_svātantrya.

**Deliverable:** Full two-tier system running; vimarśa narrations; DECKARD AWM proposals; skill library populated; H4 and H5 data collected.

**Exit criteria:** End-to-end pipeline produces a generated stanza + sphurattā log in ≤30 seconds; H4 meaningful rate ≥70%.

---

### Phase 6: Creative Pipeline, Evaluation, and Paper (Weeks 27–32)

**Goal:** Full pañcakṛtya control loop; all ablations; human evaluation study; paper and dataset release.

**Tasks:**

6.1 **Pañcakṛtya control loop**
- Implement the full `PancakrtyaLoop` as specified.
- Wire mala regularisers into the WM training objective.
- Run the full 3-domain sequential evaluation with sleep.

6.2 **Creative domain expansion**
- Extend to English poetry (Tier B corpus) and scientific analogy (Tier C corpus).
- Implement domain-specific decoders and preference models.
- Validate cross-domain transfer: does the system use Sanskrit metre knowledge to generate English formal verse?

6.3 **DIAMOND decoder (optional)**
- If multi-modal outputs are prioritised: add the DIAMOND EDM decoder.
- Generate text-conditioned image outputs for the poetry domain.
- This adds ~8 GB GPU memory; confirm DGX Spark remains in budget.

6.4 **All ablations (H1–H9)**
- Run all pre-registered ablations with ≥3 seeds each.
- Statistical analysis: paired permutation test, Hedges' g, BCa CI, Holm-Bonferroni.
- Generate autoreport (following PCE v0.4 autoreport pattern) for all nine hypotheses.

6.5 **Svātantrya benchmark**
- For all generated outputs: compute S_svātantrya (nearest-neighbour distance in WM latent space from training corpus).
- Human novelty ratings: 10 evaluators rate 100 outputs on 1–5 novelty scale.
- Compute Spearman ρ between S_svātantrya and human novelty (H9).

6.6 **Camatkāra benchmark finalisation**
- Publish camatkāra event timing data alongside human aesthetic mark data.
- Release as `pwm-creativity-benchmark` on HuggingFace.

6.7 **Paper and release**
- Write paper covering: philosophical framework, system architecture, experimental results, camatkāra as evaluation, comparison with PCE v0.4 and LLM baselines.
- Release corpus on Zenodo (following PCE v0.4's companion Pratyākṣa release pattern).
- Release code on GitHub under MIT licence.
- Publish live site (Astro, following PCE v0.4 pattern) with showcase outputs and cascade traces.

**Exit criteria:** All nine hypotheses assessed; paper draft complete; code released; benchmark released.

---

## Part D: Ablation Design (All Six Core Ablations)

| Ablation | Component | ON state | OFF state | Primary metric |
|---|---|---|---|---|
| A1 | EFE actor | EFE with all four terms | REINFORCE (DreamerV3 original) | Episodes to first sphurattā; creative quality |
| A2 | Hopfield Citta-store | Full two-mode Hopfield | Cosine-similarity vector store (v0.4 style) | Pattern completion accuracy; sphurattā rate |
| A3 | Sleep consolidation | NREM + REM phases | No sleep phases (online-only) | Forgetting rate on sequential tasks |
| A4 | Vimarśa bridge | Full LLM + bridge | No LLM (rule-based narration) | Narration quality; H4 |
| A5 | Mala regularisers | All three active | No regularisers | Metre satisfaction; latent collapse diagnosis |
| A6 | Trika hierarchy | 3 levels | 1 level (Aparā only) | Long-horizon coherence; 16-step prediction error |

---

## Part E: Benchmarks

### E.1 Standard RL Benchmarks (World Model Competence)
- **DMC Walker Walk** (visual control): confirms RSSM training stability.
- **Crafter** (open-world): confirms EFE actor vs REINFORCE on sparse rewards.
- **Memory Maze** (long-horizon memory): confirms S4 Para level advantage over GRU-only.

### E.2 Creative Benchmarks (Primary Contribution)
- **Sanskrit Metre Generation**: generate metrically valid stanzas; measure metre satisfaction rate and R_camatk.
- **POEMetric** (from PCE v0.4): poetry generation quality score on English poems.
- **AUT (Alternate Uses Task)**: divergent thinking; measure DivScore.
- **BBH Scientific Creativity** (from PCE v0.4): scientific hypothesis generation.

### E.3 New PWM-Specific Benchmarks
- **Camatkāra Correlation** (H6): DTW distance between WM camatkāra sequence and human aesthetic marks.
- **Svātantrya Score** (H9): Spearman ρ between WM novelty metric and human novelty ratings.
- **Sequential Forgetting** (H3): mean accuracy across 3 domains after sequential training, with and without sleep.
- **Cross-Domain Transfer**: train on Sanskrit, test on English formal verse — does metre knowledge transfer?

---

## Part F: Risk Register and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| EFE actor diverges (high-variance epistemic term) | Medium | High | Anneal epistemic/novelty coefficients; start with ambiguity+risk only; use SR-AIF actor-critic recipe |
| Hopfield retrieval collapses (β too high, one attractor dominates) | Medium | Medium | Schedule β; monitor retrieval entropy daily; start at intermediate β; expose β as ablation |
| Sequential forgetting despite sleep | Low | High | Add EWC or SI regulariser as backstop; increase sleep frequency; increase REM dream diversity |
| LLM compute dominance (70B too slow) | High | Medium | Use Llama-3-8B during development; cache LLM outputs by skill description; query only at sphurattā; batched narration |
| Sphurattā false positives (every update fires) | Medium | Medium | Require both VFE drop AND Hopfield entropy drop; use rolling percentile, not fixed threshold |
| Camatkāra gaming (agent inflates ΔI by writing to Hopfield repeatedly) | Medium | Medium | Cap each component; use empowerment as primary signal; penalise high-frequency writes |
| Mala regularisers conflict with capacity (entropy penalty washes out self-prior) | Low | Low | Small λ coefficients (0.01); add in Phase 6, not Phase 1; treat as polish |
| Memory OOM on full Phase 5 system | Medium | High | Progressive feature flags; gradient checkpointing on S4 level; offload V-JEPA 2 encoder to separate CUDA stream |
| Sanskrit corpus quality (noisy OCR from GRETIL) | Medium | Medium | Pre-processing pipeline with metre validator as noise filter; start with well-known texts only |
| H9 (svātantrya ρ) is low because WM latent ≠ semantic novelty | Medium | Medium | Try multiple distance metrics; validate with interpolation experiments in latent space |
| LLM narration quality is poor with frozen 4-bit model | Medium | Medium | Try full-precision 8B as alternative; use structured prompting with cascade trace as context |
| Philosophical drift (concepts become vague) | Low | Medium | Maintain glossary in GLOSSARY.md with textual citations; every module docstring cites Sanskrit source |

---

## Part G: Development Environment and Tooling

### G.1 Repository Structure
```
pwm/
├── CLAUDE.md              # Claude Code plugin manifest
├── README.md              # Project overview (modelled on PCE v0.4 README)
├── pyproject.toml         # Package definition
├── pwm/
│   ├── world_model/
│   │   ├── trika.py       # TrikaWorldModel (3-level RSSM)
│   │   ├── rssm.py        # TrikaCoreLevel (per-level RSSM)
│   │   ├── s4_backbone.py # S4 integration for Para level
│   │   └── losses.py      # VFE loss, symlog, twohot
│   ├── active_inference/
│   │   ├── efe_actor.py   # EFEActor (EFE minimisation)
│   │   ├── crspp.py       # CRSPP preference model
│   │   └── efe_utils.py   # EFE term computations
│   ├── memory/
│   │   ├── citta_store.py # CittaStore (Hopfield Episodic + Semantic)
│   │   ├── replay.py      # Prioritised replay buffer
│   │   └── skill_lib.py   # Voyager-style skill library
│   ├── sleep/
│   │   ├── scheduler.py   # SleepScheduler
│   │   ├── nrem.py        # NREM consolidation phase
│   │   ├── rem.py         # REM dreaming phase
│   │   └── therm_budget.py # ThermSleepBudget
│   ├── vimarsa/
│   │   ├── bridge.py      # VimarsaBridge (WM ↔ LLM)
│   │   ├── narrator.py    # CamatkaraNarrator
│   │   └── deckard.py     # AWM proposal (DECKARD-style)
│   ├── rewards/
│   │   ├── camatk.py      # CamatkaraReward
│   │   └── mala.py        # MalaRegularisers
│   ├── loop/
│   │   └── pancakrtya.py  # PancakrtyaLoop (outer control cycle)
│   ├── perception/
│   │   ├── vjepa2.py      # V-JEPA 2 frozen encoder wrapper
│   │   ├── text.py        # BPE tokeniser + embedding
│   │   └── diamond.py     # DIAMOND EDM decoder (Phase 5+)
│   └── eval/
│       ├── camatk_eval.py # Camatkāra evaluation (DTW correlation)
│       ├── svat.py        # Svātantrya score
│       ├── metre.py       # Sanskrit metre validator
│       └── ablations.py   # Ablation runner
├── corpus/
│   ├── ingest/            # Corpus ingestion scripts
│   ├── annotate/          # Annotation pipelines
│   └── benchmark/         # Held-out evaluation benchmark
├── configs/
│   ├── phase1_aparA.yaml  # Phase 1: Aparā-only text
│   ├── phase2_efe.yaml    # Phase 2: + EFE actor
│   ├── phase3_hopfield.yaml # Phase 3: + Hopfield
│   ├── phase4_sleep.yaml  # Phase 4: + Sleep
│   ├── phase5_llm.yaml    # Phase 5: + LLM āgama
│   └── phase6_full.yaml   # Phase 6: full system
├── benchmarks/
│   ├── results/           # JSON artefacts (reproducible)
│   └── autoreport.py      # Autoreport generator (PCE v0.4 style)
├── docs/
│   ├── SPEC_v1.0.md       # Technical spec
│   ├── PRD_v1.0.md        # This document
│   ├── GLOSSARY.md        # Sanskrit ↔ computational concept glossary
│   └── adr/               # Architecture Decision Records
└── paper/
    └── main.tex           # Paper source (tectonic/LaTeX)
```

### G.2 Key Configuration (Hydra)

```yaml
# configs/phase5_llm.yaml (reference full config)
defaults:
  - world_model: trika_3level
  - actor: efe
  - memory: hopfield_full
  - sleep: nrem_rem
  - llm: llama3_70b_nf4
  - corpus: multi_domain

world_model:
  levels: 3
  stoch_dim: 32
  stoch_classes: 32
  hidden_dim_aparA: 512
  hidden_dim_parAparA: 1024
  hidden_dim_para: 1024
  backbone_para: s4
  kl_balance_dyn: 0.5
  kl_balance_rep: 0.1
  free_bits: 1.0

actor:
  type: efe
  horizon: 15
  gamma: 0.997
  lam: 0.95
  entropy_coef: 3e-4
  efe_alpha_ambiguity: 0.3
  efe_alpha_risk: 0.3
  efe_alpha_epistemic: 0.2
  efe_alpha_novelty: 0.2
  crspp_lr: 1e-4

memory:
  n_episodic_aparA: 1000
  n_episodic_parAparA: 500
  n_episodic_para: 100
  n_semantic_aparA: 64
  beta_episodic_scale: 4.0
  beta_semantic_scale: 0.25
  sphuratta_percentile: 5
  sphuratta_min_gap: 100

sleep:
  interval: 5000           # sleep every 5K steps
  nrem_batches: 100
  rem_episodes: 20
  dream_horizon: 30
  therm_budget_petaflops: 1.0
  shsy_rate: 0.01

llm:
  model: meta-llama/Meta-Llama-3-70B-Instruct
  quantisation: nf4
  max_narration_tokens: 128
  sphuratta_cache_size: 1000  # cache LLM responses by skill description

reward:
  alpha_1: 0.4  # VFE reduction
  alpha_2: 0.3  # Hopfield information gain
  alpha_3: 0.3  # Empowerment
  lambda_ext: 0.1  # external task reward scaling

training:
  batch_size: 32
  lr_wm: 1e-4
  lr_actor: 3e-5
  lr_bridge: 3e-4
  gradient_clip: 100.0
  replay_capacity: 1_000_000
  min_buffer_steps: 5000
  mixed_precision: bfloat16
  torch_compile: true
```

---

## Part H: The Research Paper Outline

**Title:** *Pratyabhijñā World Model: Creative AI through Recognition, Active Inference, and Associative Memory*

**Abstract:** We present the Pratyabhijñā World Model (PWM), a creative AI system that operationalises Kashmir Śaiva philosophy — specifically the doctrine of *pratyabhijñā* (recognition) — through active inference, instantiated on a world-model substrate with a frozen LLM augmentation layer. We show that the key computational primitives demanded by pratyabhijñā — a persistent recognition density, stochastic latent transitions (*spanda*), reflexive self-awareness (*vimarśa*), and associative memory (*ālayavijñāna*) — map directly onto the technical components of a DreamerV3-class RSSM with EFE actor, Hopfield Citta-store, and NREM/REM sleep consolidation. We introduce *camatkāra* — aesthetic wonder — as an intrinsic, information-theoretic creative reward (R_camatk = ΔF + ΔI_Hopfield + Empowerment) that resolves the evaluation crisis in PCE v0.4 (ρ=0.0 between proxy and LLM judge). We release the PWM Creative Corpus (2M tokens, four domains, camatkāra annotations) and the Camatkāra Benchmark. Our main empirical finding is that the camatkāra signal correlates significantly with human aesthetic judgments (DTW correlation, H6), and that the two-tier WM/LLM architecture outperforms the LLM-native PCE v0.4 cascade on compositional novelty (S_svātantrya) and camatkāra density.

**Sections:**
1. Introduction: The pratyabhijñā programme; PCE v0.4's positive finding (H8a) and evaluation crisis (H9)
2. Background: Kashmir Śaiva philosophy; active inference; world models; Hopfield networks; sleep consolidation
3. The PWM Architecture: two-tier design; Trika RSSM; EFE actor; Hopfield Citta-store; sleep loop; vimarśa bridge
4. Camatkāra: definition, operationalisation, validation protocol
5. The Creative Corpus and Camatkāra Benchmark
6. Experiments: ablations H1–H9; comparison with PCE v0.4 and LLM baselines
7. Analysis: latent space structure; sphurattā temporal analysis; sleep effect on forgetting; LLM narration quality
8. Discussion: philosophical implications; limitations; future work (H-JEPA, 36-tattva hierarchy, audio)
9. Conclusion

---

*Document status: v1.0. Created May 2026.*
*Companion documents: PWM_Master_Research.md (research rationale), PWM_Architecture_Spec.md (technical spec)*
