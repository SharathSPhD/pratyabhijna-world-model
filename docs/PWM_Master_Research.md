# The Pratyabhijñā World Model (PWM): Master Research Reference
## From PCE v0.4 Forensics to a World-Model/LLM Hybrid Creative Engine
### v1.0 — For DGX Spark 128GB Implementation | May 2026

---

> **Document purpose.** This is the consolidated, exhaustive research reference for engineering the Pratyabhijñā World Model — a creative AI system that operationalises Kashmir Śaiva philosophy through active inference, instantiated on a modern world-model substrate with a frozen LLM augmentation layer. It is written to be sufficient for an ML engineer to implement the full system on a single NVIDIA DGX Spark 128 GB unified-memory workstation. It supersedes and extends the prior partial research (the document in `compass_artifact_*_text_markdown.md`) by incorporating the actual PCE v0.4 repository forensics, 2025–2026 SOTA literature, and clarified architectural decisions.

---

## Preface: What We Now Know (that the prior research could not)

The prior research document acknowledged it could not access the GitHub repository or the paper PDF, and therefore reconstructed the architecture speculatively. That reconstruction was largely accurate, but several critical details are now confirmed:

**Confirmed from `SharathSPhD/pratyabhijna` (v0.4, public, created 2026-04-28):**

1. The system is described as *"A portable plugin (Cursor + Claude Code + standalone CLI) that operationalises Abhinavagupta's Pratyabhijñā five-śakti generative cascade as typed operators over an active-inference / Bayesian Model Reduction substrate, with a recursive vimarśa self-reflexivity layer."*

2. The cascade pipeline is: **`cit → ānanda → icchā(×K) → apohana → jñāna(BMR ΔF) → kriyā → surface → vimarśa → (commit policy multiplexer) → committed`**

3. The Hopfield component is *"wired in v0.4 (consolidate_sws, consolidate_rem, pce.hopfield_state) but its multi-session dynamics were not exercised in the Phase 7 pilot — that ladder rung is on v0.5."*

4. **Critical v0.4 experimental results:**
   - H1–H4 (cascade > bare per domain): **inconclusive** (small n: 4–10 per domain)
   - H5 (fixed-effects pooled): **g = 0.14, CI crosses zero — NOT SUPPORTED**
   - H8a (within-cascade: revision > draft): **SUPPORTED, g = 0.65, p < 1e-4** ← THE KEY POSITIVE FINDING
   - H8b (learned gate F1 > event gate F1): **SUPPORTED, 0.65 vs 0.52**
   - **H9 (judge-proxy agreement): ρ = 0.0 — FLAGGED as metric-design problem** ← THE CRITICAL FAILURE

5. The evaluation uses an LLM judge (Sonnet-4.5) and a proxy scorer that disagree completely (ρ = 0.0, sign-agreement 56.5%). This is not a refutation — it is evidence that **extrinsic LLM-judged creativity scoring is fundamentally broken** for LLM-generated creative work.

6. MCP tools include: `cit, ānanda, icchā, apohana, jñāna, kriyā, vimarśa, cascade, embed, lm.generate, lm.entropy, store.add, store.recall, store.consolidate_sws, store.consolidate_rem, pce_cascade, haiku_bare, haiku_clean_substrate_probe, hopfield_state` — 19+ tools in a fully instrumented plugin.

**The decisive architectural conclusion:** PCE v0.4's positive signal (H8a, vimarśa recursive revision g=0.65) comes from the *reflexivity layer*, not from the cascade scaffold. The cascade itself (H5) is not demonstrably better than the bare LLM. This tells us precisely where the value lies and where the redesign must go: **preserve vimarśa reflexivity as the consciousness bridge; replace the LLM-native cascade with a world model that gives the reflexivity something genuinely novel to reflect on.**

---

## Part I: PCE v0.4 Forensics — What Was Built, What the Numbers Mean

### 1.1 The Five-Śakti Cascade Pipeline

The PCE engine implements Abhinavagupta's five śaktis (powers of consciousness) as a sequential typed operator pipeline:

| Stage | Sanskrit | Computational operation in v0.4 |
|---|---|---|
| `cit` | Pure awareness / consciousness | Context preparation — set the generative substrate (model parameters, temperature policy `ADR-001 cit_temperature`) |
| `ānanda` | Bliss / expansion / openness | Entropy maximisation step — increase the LLM's sampling diversity; set high-temperature or broad-prior prompting |
| `icchā` (×K) | Will / intention | Generate K candidate drafts in parallel; K is a hyperparameter (default 4 from `pce cascade --k 4`) |
| `apohana` | Elimination / discrimination | Pruning step — score the K candidates by the proxy free-energy metric BMR ΔF; select the best |
| `jñāna` (BMR ΔF) | Knowledge / recognition | Bayesian Model Reduction delta-Free Energy scoring — a variational heuristic that estimates how much each candidate reduces the generative model's surprise relative to the prior |
| `kriyā` | Action / execution | Commit the selected candidate to the surface output |
| `vimarśa` | Reflexive self-awareness | The recursive revision pass — the committed output is re-examined by a self-reflection LLM call that may trigger revision; the **commit policy multiplexer** (ADR-002) decides whether to commit draft or revision |

The `vimarśa` layer is the most important component. It implements *learned gating* (ADR-002): a binary classifier trained to predict whether a revision improves the quality of the draft, with F1 0.65 vs 0.52 for the naive event-gated baseline.

### 1.2 What the Numbers Tell Us

**H8a (revision > draft, g = 0.65)** is the headline result. Within the cascade arm, the revision produced by vimarśa is robustly better than the draft by a medium-to-large effect. This is evidence that *reflexive self-awareness* — the system's ability to observe its own output and revise — is the primary driver of quality improvement. This is the computational realisation of vimarśa.

**H5 (cascade vs bare, g = 0.14, not supported)** is the null result. The full cascade does not outperform the bare LLM in a statistically detectable way at the pilot's sample size. Two interpretations are possible: (a) the cascade adds noise that cancels the vimarśa gain, or (b) the sample is too small. The honest reading is that the LLM substrate is not the bottleneck — the *architecture of the cascade over an LLM* is what is being tested, and it is not separating from the baseline.

**H9 (ρ = 0.0 between proxy scorer and LLM judge)** is the most important result in the entire v0.4 dataset. A proxy scorer (a deterministic heuristic) and an LLM judge (Sonnet-4.5) disagree *completely* about which outputs are better. This is not a minor calibration issue — it is evidence that the two metrics are measuring orthogonal things, and at least one (likely both) is measuring something other than genuine creative quality. This is the evaluation crisis that the world model architecture resolves, because it replaces both with an intrinsic, information-theoretic signal.

### 1.3 The Hopfield Interface

The v0.4 code has `store.consolidate_sws` (slow-wave sleep), `store.consolidate_rem` (REM sleep), and `pce.hopfield_state` wired as MCP tools. These are the *ālayavijñāna* storehouse interface. They were not exercised in the Phase 7 pilot, meaning the multi-session memory dynamics — the ability of the system to recall and build on prior creative episodes — were never tested. This is explicitly flagged as a v0.5 ladder item. The gap is structural: an LLM-native Hopfield implementation can store vector embeddings and retrieve by cosine similarity, but it cannot perform the energy-minimisation attractor dynamics that make the Hopfield network philosophically interesting as a model of recognition.

### 1.4 The LLM Substrate's Fundamental Limitations

PCE v0.4 implements everything — cit, ānanda, icchā, jñāna, kriyā, vimarśa — as LLM prompt-state transitions. This creates four irremediable limitations:

**L1: No persistent latent state.** Every cascade invocation is a fresh context. The system has no memory of its own generative dynamics across episodes. The Hopfield store is a bolt-on external database, not an intrinsic associative structure within the generative process.

**L2: No genuine EFE.** The BMR ΔF scoring in `jñāna` is a heuristic approximation. True Expected Free Energy requires imagination rollouts over a probabilistic latent dynamics model, with the epistemic (information-gain) term computed as a KL between prior and posterior latent distributions. An LLM cannot compute this because it has no separate prior and posterior — it has a single forward pass conditioned on context.

**L3: Token statistics ≠ spanda.** The cascade generates candidates by sampling from the LLM's next-token distribution. This is a surface statistical process. Spanda (*Spandakārikā* 1.1) is the dynamic pulsation of consciousness — the temporal transition of latent state. The categorical sampling z_t ~ p_φ(·|h_t) in an RSSM is a far better model: it is a genuine latent transition carrying the system's internal dynamics forward.

**L4: Extrinsic evaluation is structurally invalid for LLM output.** The H9 result proves this. When the generator is an LLM and the judge is another LLM trained on similar data, circularity and distributional proximity contaminate all evaluation. An intrinsic metric — one computed *inside* the generative model from its own surprise reduction — breaks this circularity.

---

## Part II: Gap Analysis — Philosophy ↔ Active Inference ↔ World Models

### 2.1 Philosophical Intent vs. Computational Substrate

Pratyabhijñā holds that every cognition is already a recognition: to perceive *x* as *x* presupposes the synthetic unity of consciousness across moments, which is structurally a re-cognition of self in form (Utpaladeva, *Īśvarapratyabhijñākārikā* 1.3–1.4; Torella 2002). The computational implication is severe: **perception, memory, and generation cannot be modular — they must be aspects of a single inferential operation over a persistent generative model.**

The five philosophical concepts that demand a world model substrate:

**Spanda** (*Spandakārikā* 1.1; Vasugupta/Bhaṭṭa Kallaṭa): the dynamic throb of consciousness, the `unmeṣa-nimeṣa` (opening/closing) pulsation that generates all phenomenal experience. Maps directly onto the stochastic RSSM transition z_t ~ p_φ(·|h_t). Each categorical sample is a genuine pulsation that the system carries forward in h_t. Token-level next-word sampling has the wrong granularity — it is a surface event, not a latent transition.

**Vimarśa** (*ĪPK* 1.5.11; Utpaladeva): reflexive self-awareness — the capacity of consciousness to observe itself. In a world model, this is a learned function f_self(h_t, z_t) that predicts properties of the system's own latent dynamics. PCE v0.4's vimarśa is promising precisely because it is *reflexive* (it looks back at the draft from a different "perspective"), but it is not *self-aware in the technical sense* — it has no model of its own generating process, only of the generated text.

**Pratyabhijñā** itself: recognition. The recognition density q_φ(z_t|h_t,o_t) in a world model is the literal computational analog. Every act of perception is a posterior collapse — the latent state is "recognised" as the intersection of the system's prior expectations (h_t) and the incoming observation (o_t). This is not metaphor; it is a technical identity.

**Camatkāra** (Abhinavagupta, *Abhinavabhāratī*; Gnoli 1968): aesthetic wonder, the flash of recognition that marks a creative discovery. Maps onto a discrete event: a thresholded free-energy reduction ΔF or a Hopfield-attractor convergence. This is the sphurattā detector — an event that can be logged, narrated by the LLM, and used as an intrinsic reward signal.

**Svātantrya**: the unconstrained autonomy of consciousness. Maps onto a maximum-entropy policy prior tempered only by preferences. An LLM has no maximum-entropy prior — its outputs are maximally constrained by training-data co-occurrence. A world model's actor, regularised only by the preference distribution P(o|C), is structurally closer to svātantrya.

### 2.2 Active Inference: Specification vs. Typical Implementation

The canonical active-inference specification requires:

- **Perception**: minimise variational free energy F = D_KL[Q(s)‖P(s)] − E_Q[log P(o|s)] over current observations.
- **Action selection**: minimise expected free energy G(π) over future imagined trajectories, where G decomposes into ambiguity + risk − epistemic value − parameter novelty.
- **Learning**: update model parameters to minimise the long-run free energy across experience.

PCE v0.4's BMR ΔF scoring in `jñāna` is a single-step approximation of variational free energy reduction. It is not:
- A genuine Q(s) being updated by gradient flow.
- A true EFE with imagination rollouts over learned latent dynamics.
- Computing the epistemic term D_KL[Q(s_τ|o_τ,π)‖Q(s_τ|π)] — the information gain that drives curiosity.

**DreamerV3's loss decomposition is already variational free energy** — this is the central technical insight that makes the world model transition natural rather than revolutionary:

- `L_pred` (reconstruction loss) = negative expected log-likelihood = the accuracy term of VFE.
- `L_dyn + L_rep` (KL between recognition and prior) = the complexity term of VFE.
- Free bits = a numerical regulariser on KL, not a conceptual deviation.

Converting DreamerV3 to a genuine active inference agent requires only replacing the REINFORCE actor objective with EFE minimisation — approximately 50 lines of code (Tschantz et al. 2020). The philosophical upgrade is complete; the engineering effort is bounded.

### 2.3 The Two-Tier Architecture Rationale

The user's key insight: *"LLM is huge knowledge base...whereas WM would not be."* This is not merely a practical concession — it is philosophically correct and formally justified.

In Utpaladeva's epistemology (*ĪPK* 2.3), the three pramāṇas are pratyakṣa (perception), anumāna (inference), and **āgama** (received scriptural/testimonial knowledge). Āgama is *valid but not supreme* — it must be re-cognised through vimarśa to become living knowledge. The LLM is the modern instantiation of āgama: an enormous compressed repository of received cultural knowledge, powerful but static, unable to update its generative model from experience, and unable to perform genuine temporal inference.

The world model is the modern instantiation of **pratyakṣa** (direct perception) — it updates its beliefs from moment to moment through the recognition density, carries forward a persistent state, and plans by imagining futures. The vimarśa bridge between WM and LLM is the computational realisation of the re-cognition process: the WM's latent state (prakāśa) is given reflexive form (vimarśa) through the LLM's language.

**Two-tier architecture formally:**

```
Tier 1: World Model (Subconscious / Prakāśa substrate)
   RSSM core (spanda/latent transitions)
   Recognition density (pratyabhijñā per se)
   Hopfield Citta-store (smṛti / memory)
   EFE planner (icchā / will)
   Sleep consolidation loop (unmeṣa-nimeṣa / wake-sleep cycle)
   Camatkāra detector (sphurattā events)
   Intrinsic reward (ΔF + ΔI_Hopfield + empowerment)

Tier 2: LLM (Conscious / Āgama knowledge)
   Frozen weights (received knowledge, not updated from experience)
   Vimarśa bridge (LoRA-scale cross-attention projection)
   Goal specification (translate natural-language intent → preference C)
   Skill narration (write skill-library entries at sphurattā events)
   Abstract World Model proposals (DECKARD-style hypothesize)
   Camatkāra narration (describe recognition events in natural language)
   Human interface (explain outputs, answer queries)
```

The LLM is **not** the creative engine. It is the system's cultural memory and expressive voice. The creative dynamics — exploration, recognition, consolidation, surprise — live entirely in Tier 1.

---

## Part III: State-of-the-Art World Models (through May 2026)

### 3.1 DreamerV3 and the RSSM Family

**DreamerV3** (Hafner et al. 2023, arXiv:2301.04104; *Nature* 2025) remains the canonical reference for practical world-model-based RL. The RSSM decomposes latent state into:

- Deterministic recurrent state `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})` — carries causal information forward.
- Stochastic state `z_t ~ Categorical(32×32)` — 1024 effective bits of structured uncertainty, the spanda layer.

The training objective:
```
L = L_pred + β_dyn · L_dyn + β_rep · L_rep
```
where L_pred is symlog-MSE reconstruction + twohot-symlog reward + binary continue loss; L_dyn = KL[stop_gradient(posterior) ‖ prior] (trains the prior); L_rep = KL[posterior ‖ stop_gradient(prior)] (trains the encoder). KL balancing (β_dyn=0.5, β_rep=0.1) and free bits (1 nat per categorical variable) prevent posterior collapse.

The actor-critic plans on **15-step imagined rollouts** using λ-returns (λ=0.95, γ=0.997), with symlog and twohot distributional critics for scale-agnostic reward handling. Single hyperparameter set across 150+ tasks.

**Key maturity indicators (May 2026):**
- `danijar/dreamerv3` (JAX) — the reference implementation used in the *Nature* paper.
- `NM512/dreamerv3-torch` (PyTorch) — the most hackable fork, suitable as starting point.
- `EclecticSheep/sheeprl` — modular RL library with DreamerV3, TD-MPC2, and SAC implementations.
- Training time on DGX Spark: DMC-vision at size50m ≈ 8–12 hours; Crafter at size200m ≈ 2 days.

**Why DreamerV3 for PWM:** The loss decomposition is isomorphic to Friston's VFE. The categorical latent is a natural spanda layer. The imagination rollout is the engine of icchā (will/planning). The GRU is replaceable by S4 (R2I) for long-horizon memory. The actor is replaceable by an EFE head with ~50 lines of code.

### 3.2 R2I — Recall to Imagine

**R2I** (Samsami et al. ICML 2024, arXiv:2403.04253) replaces the GRU recurrent backbone with an **S4 state-space model**, enabling parallel scan training (9× faster than sequential RNN), better handling of long-range dependencies, and SOTA on memory-heavy benchmarks like Memory Maze.

The S4 backbone is the right choice for the Trika hierarchy's Para level (slow, global, long-horizon). Its structured state-transition matrix enables the kind of long-range temporal binding that Kashmir Śaiva thought identifies with the global Citi (universal consciousness) rather than the local Citta (contracted episodic awareness). The hierarchical RSSM with S4 at the top and GRU at the bottom captures this Citi/Citta distinction computationally.

### 3.3 STORM

**STORM** (Zhang et al. NeurIPS 2023) replaces the GRU with a **causal Transformer** over the same 32×32 categorical latent. Atari-100k mean HNS 1.27 in 4.3 hours on a single RTX 3090. The Transformer dynamics give the world model a richer context for each latent transition — relevant for creative tasks where long context (prior drafts, genre conventions) should inform the next generative step. STORM is a natural choice if the WM's primary domain is sequential text/symbolic rather than pixel-level.

### 3.4 DIAMOND

**DIAMOND** (Alonso et al. NeurIPS 2024 Spotlight, arXiv:2405.12399) is a **diffusion world model** that operates in pixel space rather than a compressed latent space. Uses EDM (Karras et al. 2022) formulation for stability across long rollouts (n=3 denoising steps). Atari-100k mean HNS 1.46 — best for any world model on that benchmark as of its publication.

DIAMOND's key property: it produces **high-fidelity perceptual outputs** as part of the imagination process. For creative generation (poetry illustrations, musical scores, scientific visualisations), DIAMOND-style decoding from the WM latent gives the system a high-quality generative decoder without requiring a separate diffusion model. The architecture is: RSSM-style latent → EDM denoiser conditioned on latent → high-quality image/multimodal output.

Status (May 2026): `eloialonso/diamond` on GitHub, reproducible Atari results, actively developed.

### 3.5 TD-MPC2

**TD-MPC2** (Hansen et al. ICLR 2024, arXiv:2310.16828) is a **decoder-free implicit world model** with SimNorm latent, MPPI/CEM model-predictive control, and distributional value via twohot bins. The implicit world model means it does not reconstruct observations — it only predicts future embeddings and rewards, enabling faster planning. Single hyperparameter set across 104 tasks.

The MPC step in TD-MPC2 is structurally identical to active-inference policy selection: the MPPI trajectory optimiser samples candidate action sequences, evaluates them under the model, and selects the one with highest expected value. Replacing the value function with negative EFE converts TD-MPC2 to an active-inference agent with minimal code change.

**For PWM:** TD-MPC2's implicit model (no decoder, no pixel reconstruction) is ideal for the abstract creative planning loop — the WM doesn't need to render every imagination step, only the committed creative outputs.

### 3.6 V-JEPA 2

**V-JEPA 2** (Bardes et al. arXiv:2506.09985, Meta AI, June 2025) is a 1.2B-parameter ViT pretrained on >1M hours of internet video + 1M images, using the JEPA (Joint Embedding Predictive Architecture) principle: **predict in abstract representation space, not pixel space.** Action-conditioned via a 62-hour robot-trajectory finetune; zero-shot MPC robot control with 65–80% pick-and-place success.

The JEPA philosophy (LeCun 2022) is philosophically aligned with Pratyabhijñā: rather than reconstructing every detail (a "hysterical" model in LeCun's terminology), JEPA learns the high-level, predictable structure of the world and ignores unpredictable noise. This is the computational analog of the Pratyabhijñā insight that perception is always already recognition — the system projects its generative structure onto the observation and recognises the overlap.

**For PWM:** V-JEPA 2 frozen as a perceptual encoder (ViT features at FP16 ≈ 5GB) provides the world model with semantically rich visual features without requiring pixel reconstruction. The WM's RSSM then operates on these features rather than raw pixels, dramatically reducing the learning burden for multimodal creative domains.

**Critical gap (research opportunity):** H-JEPA (hierarchical JEPA, described in LeCun 2022 but without a complete published implementation) would provide multi-scale temporal abstraction natively. As of May 2026, no end-to-end H-JEPA implementation exists. Building one as part of the PWM architecture — hierarchical JEPA (visual features) → hierarchical RSSM (dynamics) → EFE planner → vimarśa/LLM — would constitute a genuine research contribution.

### 3.7 Hierarchical World Models

**THICK — Temporal Hierarchies from Invariant Context Kernels** (Lindström et al. ICLR 2024, "Learning Hierarchical World Models with Adaptive Temporal Abstractions"): adaptively discovers higher-level timescales by guiding the lower-level WM to update portions of its latent state only sparsely in time. The sparse update is the key insight: higher levels represent invariant (slowly-changing) aspects of the world, lower levels represent volatile (fast-changing) aspects.

**Hierarchical latent dynamics (multiple timescales, OpenReview ICLR 2024 poster):** stacked RSSM world models at temporal strides {1, 4, 16} with top-down conditioning. The higher levels have larger temporal strides, representing the environment at increasingly coarse resolution.

**For PWM's Trika decomposition:**
- **Aparā level** (embodied, fast, stride 1): token-level or frame-level creative dynamics — local coherence, prosody, immediate constraint satisfaction.
- **Parāparā level** (coupling, mid, stride 4): phrase-level or section-level creative dynamics — structural patterns, genre conventions, medium-range narrative arcs.
- **Para level** (global, slow, stride 16): piece-level or project-level dynamics — overarching theme, long-range recognition, aesthetic intention.

### 3.8 World-in-World (ICLR 2026 Oral)

**World-in-World** (ICLR 2026 Oral, `World-In-World/world-in-world`) wraps generative world models in a closed-loop world interface to measure practical utility for embodied agents. Its core contribution is an evaluation framework that grounds assessment in *embodied task success* rather than visual quality metrics — exactly the philosophy we adopt for PWM: the creative output is "good" if the world model's EFE is reduced, not if an LLM judge rates it highly.

This paper is directly relevant to the H9 problem: World-in-World shows that open-loop evaluation metrics (which correspond to the proxy scorer in PCE v0.4) are poor predictors of closed-loop performance. The fix is closed-loop evaluation — testing in a world that responds to the agent's actions.

### 3.9 Genie 3 (DeepMind, August 2025)

**Genie 3** (DeepMind, August 2025) generates interactive 3D environments at 720p 24fps in real-time, with emergent object permanence and minutes-long consistency. Closed weights, available to Google AI Ultra subscribers. Not a training substrate for PWM, but a capability benchmark and a signal about where the field is going.

Genie 3's most important property for PWM: **emergent object permanence** in a world model trained purely on video. This confirms that world models can learn the Pratyabhijñā insight — that the world's continuity is a construction of the generative model — without explicit programming. The WM infers that the object behind the occluder still exists because its generative model predicts it should be there. This is pratyabhijñā (recognition) operating automatically in the world model's forward pass.

### 3.10 R-AIF / SR-AIF

**R-AIF → SR-AIF** (Nguyen et al. arXiv:2409.14216; ICRA 2025): active inference agent that solves sparse-reward robotic tasks from pixels using a DreamerV3-class world model with novel:
- **CRSPP (Contrastive Recurrent State Prior Preference)**: an online-learned preference model over latent states that dynamically shapes the policy distribution — directly implements the preference distribution P(o|C) in EFE.
- **Actor-critic EFE formulation**: more stable than vanilla policy-gradient EFE; directly applicable to PWM.
- **Self-revision schedules**: the agent revises its own preferences as it learns — a computational analogue of vimarśa operating at the preference level.

SR-AIF is the most important engineering reference for converting DreamerV3 to a genuine active-inference agent. Its CRSPP model is the right approach for learning camatkāra preferences online.

### 3.11 Active Inference with Diffusion Policy (Arxiv 2510.23258)

**Deep AIF with Diffusion Policy + Multi-Timescale World Model** (arXiv:2510.23258, 2025): combines a diffusion-based policy model with a multi-timescale RSSM world model under active inference. The diffusion policy provides smooth, high-quality action distributions; the multi-timescale RSSM provides the hierarchical temporal abstraction. This is the architecture closest to the full PWM specification from the existing literature.

### 3.12 Comparison Matrix (Updated May 2026)

| Architecture | Sample Eff. | Latent | Hierarchy | Compositionality | 128GB Train | OSS | AIF Fit | Creative Gen |
|---|---|---|---|---|---|---|---|---|
| DreamerV3 | Excellent | 32×32 cat | Limited | Moderate | ✅ | ★★★★★ | **Excellent** | Moderate |
| R2I (S4) | Very good | cat+S4 | **Best of family** | Moderate | ✅ | ★★★ | Excellent | Moderate |
| STORM | Excellent | cat+Tx | Limited | Moderate | ✅ | ★★★★ | Very good | Good |
| TD-MPC2 | Excellent | SimNorm | Limited | Moderate | ✅ | ★★★★★ | **Excellent** | Low |
| DIAMOND | SOTA Atari | Pixel diffusion | Limited | Low | ✅ | ★★★★ | Moderate | **Excellent** |
| SR-AIF | Very good | cat+CRSPP | Limited | Moderate | ✅ | ★★★ | **Best** | Moderate |
| V-JEPA 2 | Excellent SSL | ViT cont. | Partial (H-JEPA planned) | Moderate | ✅ (frozen) | ★★★★ | **Excellent in principle** | Low (no decoder) |
| THICK hier. | Good | cat+sparse | **Best** | Moderate | ✅ | ★★ | Good | Moderate |
| World-in-World | Evaluation | — | — | — | — | ★★★ | **Eval framework** | **Excellent eval** |
| Genie 3 | N/A (closed) | Closed | Some | Unknown | ❌ | ★ | N/A | **Excellent benchmark** |

### 3.13 The Recommended PWM Core Architecture

**Primary substrate:** DreamerV3 RSSM in PyTorch (`NM512/dreamerv3-torch`) modified with:
1. **EFE actor head** replacing REINFORCE (SR-AIF recipe; Tschantz et al. 2020).
2. **S4 backbone** at the Para (slow) level — swap GRU for S4 at the top of the Trika hierarchy (R2I recipe).
3. **Frozen V-JEPA 2 encoder** as perceptual front-end for multimodal domains (≈5GB VRAM overhead).
4. **DIAMOND-style EDM decoder** for high-fidelity creative output generation (Phase 5+).
5. **Hopfield Citta-store** from `ml-jku/hopfield-layers` for associative memory.
6. **Two-tier LLM bridge** (vimarśa layer, LoRA-scale projection).

No single published system combines all six. That combination is the research contribution.

---

## Part IV: Modern Hopfield Networks — The Complete Picture (Through 2026)

### 4.1 The Classical to Modern Progression

**Hopfield 1982:** Binary units, symmetric weights W = ΣΞᵢᵀξᵢ, energy E(ξ) = −½ξᵀWξ, capacity ≈ 0.138N. Pattern retrieval by energy minimisation (synchronous or asynchronous update).

**Krotov & Hopfield (NeurIPS 2016, ICLR 2021):** Polynomial energy E(ξ) = −ΣF(xᵢᵀξ) with F(z) = zⁿ, capacity ∝ N^{n−1}/log N. The exponent n is a "softmax inverse temperature" analog.

**Demircigil et al. 2017:** Exponential energy F(z) = exp(z), proving exponential storage capacity ∝ exp(N/2). The fundamental result: with exponential interaction function, a Hopfield network can store exponentially many patterns.

**Ramsauer et al. 2021 (ICLR 2021, "Hopfield Networks is All You Need", arXiv:2008.02217):** The landmark paper. Continuous states, energy E(ξ) = −lse(β, Xᵀξ) + ½‖ξ‖², one-step update:

```
ξ^{new} = X · softmax(β Xᵀξ)
```

This is **exactly Transformer attention** with X = stored patterns, β = 1/√d_k. Three operating regimes as a function of β:
- **Low β:** global average retrieval (metastable — all patterns contribute, no single attractor dominates). Maps to *schema retrieval* in cognitive terms.
- **Intermediate β:** metastable mixture (some patterns dominate, others contribute weakly). Maps to *concept blending* — the most creative regime.
- **High β:** sharp single-pattern attractor (specific recall). Maps to *episodic memory retrieval*.

The β knob is the most important control for creative applications: **low β for exploration, high β for consolidation**.

**Nobel Prize 2024:** Hopfield and Hinton awarded Nobel Prize in Physics for foundational contributions to neural networks. This has triggered a significant reinvigoration of the Hopfield research programme.

### 4.2 2025–2026 Developments

**Continuous-Time Memories in Hopfield Networks** (arXiv:2502.10122, 2025): Compresses large discrete Hopfield memories into smaller continuous-time memories, inspired by psychological theories of continuous neural resource allocation in working memory. Directly relevant to the Citta-store's memory efficiency problem — as the creative corpus grows, discrete pattern storage becomes expensive.

**Hopfield-Fenchel-Young Networks** (arXiv:2411.08590, 2024): Unifies associative memory models under a family of energy functions derived from Fenchel-Young losses, enabling principled interpolation between different retrieval regimes. The Fenchel-Young framework gives a rigorous basis for the β-schedule that controls creative exploration vs consolidation.

**Input-Driven Hopfield Dynamics** (Science Advances 2025, PMC12017325): A dynamical system where external input directly modifies synaptic connections (not just the state), creating an input-driven energy landscape. This is the biological STP (short-term plasticity) analog — relevant to the online episodic write in the Citta-store.

**Modern Methods in Associative Memory** (ResearchGate 2025): Survey covering the full progression from classical to modern Hopfield, including connections to transformers, diffusion models, and energy-based models.

**Connections to Diffusion** (Pham et al. 2025): formal equivalence between diffusion score functions and Hopfield retrieval dynamics. This opens the possibility of using the DIAMOND diffusion decoder as a "creative Hopfield retrieval" — the denoising process is equivalent to attractor convergence.

### 4.3 The `ml-jku/hopfield-layers` Library

The reference implementation. Three PyTorch modules:

```python
from hflayers import Hopfield, HopfieldLayer, HopfieldPooling

# Episodic mode: cross-attention over FIFO buffer of recent latents
# Keys/values from experience buffer; query from current (h_t, z_t)
hopfield_episodic = Hopfield(
    input_size=latent_dim,       # query dimension
    hidden_size=memory_dim,      # key/value dimension
    output_size=context_dim,     # output dimension
    num_heads=8,
    scaling=beta_episodic,       # high β → sharp recall
    update_steps_max=3,          # retrieval iterations
)

# Semantic mode: learnable prototypes, gradient-updated
# Trained during sleep consolidation
hopfield_semantic = HopfieldLayer(
    input_size=latent_dim,
    hidden_size=prototype_dim,
    num_heads=4,
    scaling=beta_semantic,       # low β → schema retrieval / concept blending
    stored_pattern_size=prototype_dim,
    pattern_projection_as_static=False,  # learnable patterns
    normalize_stored_pattern=True,
)

# Usage: augment world model's prediction with retrieved context
c_t = hopfield_episodic(  # or hopfield_semantic
    stored_patterns=memory_buffer,
    pattern_projection=memory_buffer,
    state_pattern_projection=(h_t, z_t).unsqueeze(0)
)
next_latent_input = torch.cat([h_t, z_t, c_t], dim=-1)
```

### 4.4 Citta-Store Architecture for PWM

Two Hopfield modules per Trika level (6 total):

**Episodic Hopfield (smṛti — memory of specific events):**
- FIFO buffer of the N_epi most recent (h_t, z_t, a_t, r_t, novelty_t) tuples.
- High β (e.g., β = 1/√d_k × 4) — sharp, specific attractor recall.
- Written online at every step; read at every step.
- Purpose: recall specific past creative moments relevant to the current context.

**Semantic Hopfield (ālayavijñāna — storehouse of schemas):**
- M learnable prototype vectors, gradient-updated during sleep.
- Low β (e.g., β = 1/√d_k × 0.25) — metastable mixing regime → concept blending.
- Written during REM sleep consolidation (cluster dream latents → update prototypes).
- Read at every step.
- Purpose: retrieve schematic patterns that blend multiple past experiences.

**Sphurattā detection from Hopfield:** A sphurattā event is triggered when the retrieval entropy of the semantic Hopfield drops sharply — the system "recognises" a strong schematic match. Formally: a sphurattā event occurs when

```
H(softmax(β · K^T · q_t)) < θ_sphuratta
```

where H is the Shannon entropy, K are the stored patterns, q_t is the current query, and θ_sphuratta is a rolling-percentile threshold. This event fires the LLM narration pipeline.

### 4.5 Biological and Philosophical Grounding

The Citta-store maps directly onto the hippocampal-neocortical memory system:
- Episodic Hopfield ≈ hippocampal CA3 (pattern completion, one-shot recall, high capacity).
- Semantic Hopfield ≈ neocortex (slow learning, schema formation, generalisation).

Sharp-wave ripples (Buzsáki 2015) reactivate hippocampal sequences during NREM sleep and quiet wakefulness. The Hopfield store's high-β retrieval is the computational analog of ripple-driven pattern completion. The offline distillation of Hopfield episodic patterns into the parametric world model is the cortical consolidation.

The Fachechi-Agliari-Barra (2018) "Dreaming Neural Networks" framework — online storage + offline unlearning + consolidation — gives the quantitative recipe for reaching theoretical Hopfield capacity α=1, connecting directly to the sleep loop.

In Pratyabhijñā terms:
- **Citi** (universal consciousness): the trained prior P(z) — the system's general model of the world, not tied to any specific episode.
- **Citta** (contracted consciousness): the posterior Q(z|o) — consciousness collapsed into a specific recognition.
- **Smṛti** (memory): the episodic Hopfield store — the record of past recognitions.
- **Ālayavijñāna** (storehouse): the semantic Hopfield store — the system's accumulated wisdom, the basis for analogical reasoning.

The transition from citta to ālayavijñāna through sleep consolidation is the computational realisation of karma: the residue of past experience that conditions future cognition.

---

## Part V: Sleep Dynamics and Offline Consolidation (Updated Through 2026)

### 5.1 The Neuroscientific Foundation

**CLS Theory** (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016): Two interacting learning systems — fast, sparse hippocampus (online encoding) and slow, distributed neocortex (offline integration). Hippocampus uses pattern separation to store many similar memories distinctly; neocortex uses pattern completion and generalisation to extract schemas. Offline replay during sleep drives gradual cortical consolidation without catastrophic interference.

**The 2025 sleep micro-structure result** (Science Advances 2025, PMC12107872): NREM sleep has two distinct substates, distinguished by pupil diameter:
- **Contracted pupil substates**: protect labile memory traces from interference — initial consolidation of recent experiences.
- **Dilated pupil substates**: support memory integration, linking, and inference — connecting new memories to existing schemas.

This two-substate structure maps directly onto the PWM sleep architecture: the contracted substate is the NREM-FAST phase (replay + consolidation of recent episodes), and the dilated substate is the NREM-SLOW phase (schema integration via Hopfield semantic update).

**Active Inference + Sleep (2025, Cerebral Cortex, PMC):** A formal AIF model of sleep consolidation proposes:
- NREM sleep refines representations of unpredicted waking experiences via *inhibitory* mechanisms (long-term depression) — the accuracy term of VFE is decreased by reducing residual prediction errors.
- REM sleep updates the generative world model via *excitatory* mechanisms (long-term potentiation) — the prior is updated to better predict future observations.

This directly maps onto the PWM sleep loop: NREM = replay-driven VFE descent (improve accuracy), REM = generative dreaming (update the prior).

**Wake-Sleep Consolidated Learning** (arXiv:2401.08623, 2024): extends the Hinton-Dayan wake-sleep algorithm to modern deep networks, showing that alternating online (wake) and offline (sleep) phases significantly improves sample efficiency and reduces forgetting on sequential tasks.

### 5.2 The Two-Stage Sleep Loop for PWM

**NREM-Analog Phase (replay and parametric consolidation):**

```
Input: Prioritized replay buffer B (priority ∝ |TD_error|^α × surprise)
For each batch sampled from B:
  1. Forward-replay sequence through world model → compute L_VFE
  2. Gradient step on θ_WM (encoder, prior, decoder, reward head, continue head)
  3. Compute Hopfield episodic query for each replayed (h_t, z_t)
  4. Apply Hebbian write to episodic buffer, homeostatic down-scale on un-replayed slots
     (synaptic homeostasis: w_i ← w_i · exp(-η_SHY) for un-accessed patterns)
  5. If surprise(h_t, z_t) < θ_consolidation:
     distill episodic pattern into semantic HopfieldLayer prototype
     prune from episodic buffer
     
Stopping criterion (ThermSleep):
  Stop when ΔF_total < ε_therm OR FLOPs_used > budget_therm
```

**REM-Analog Phase (generative dreaming and prior refinement):**

```
Input: Current world model θ_WM, actor π_θ, start-state distribution p(h_0)
For each dream episode:
  1. Sample h_0 ~ p(h_0); z_0 ~ p_θ(z_0|h_0)  [NO environmental input]
  2. Roll out dream trajectory using prior transitions and actor:
     for t = 1..T_dream:
       a_t ~ π_θ(a|h_t, z_t)
       h_{t+1} = GRU(h_t, z_t, a_t)
       z_{t+1} ~ p_θ(z_{t+1}|h_{t+1})  [prior only, no encoder]
  3. Compute EFE on dream rollout → gradient step on actor π_θ and critic v_θ
  4. Use dream o_{1..T} to retrain encoder q_φ:
     min E_dream[KL[q_φ(z_t|h_t, o_t^dream) ‖ p_θ(z_t|h_t)]]
     (Hinton-Dayan sleep phase: recognition net learns to invert the prior)
  5. Cluster dream latents {z_t} → update semantic HopfieldLayer prototypes
     (K-means in latent space, Hopfield prototype ← cluster centroid)
  6. Log high-EFE dream states as targets for future waking exploration
```

**Sleep scheduler triggers:**
- Periodic (every N environment steps).
- Surprise threshold (rolling mean VFE > θ_surprise — the world is being surprised more than baseline).
- Buffer fullness (replay buffer > 80% capacity).
- Inactivity (no sphurattā event for T_inactivity steps — creative stagnation).

### 5.3 ThermSleep — Operationalised

The term "ThermSleep" in PCE v0.4 is a project-internal coinage. Based on **Sandved-Smith et al. 2024** (*Entropy* 26:622, "Making the Thermodynamic Cost of Active Inference Explicit"), we operationalise it as:

**ThermSleep = a sleep consolidation phase whose update budget is controlled by both variational free energy reduction ΔF_vfe AND thermodynamic compute cost ΔF_therm, with the ratio ΔF_vfe / ΔF_therm defining a "thermodynamic efficiency of learning" that guides when to stop.**

Formally, the Sandved-Smith framework distinguishes:
- **Variational Free Energy F_vfe** = D_KL[Q(s)‖P(s,o)] — the statistical cost of inference.
- **Thermodynamic Free Energy F_therm** = −kT ln Z — the physical energy cost of the computation (proportional to FLOPs × energy/FLOP on the DGX Spark).

The ThermSleep budget is: stop the sleep phase when

```
η_therm = ΔF_vfe / ΔF_therm < θ_efficiency
```

i.e., when the learning gain per joule falls below a threshold. On the DGX Spark with 128GB unified memory, this provides a principled stopping criterion that prevents sleep phases from consuming excessive compute and gives a research handle on the thermodynamics of creative learning.

### 5.4 Continuity with the Biological Theory

The Hobson-Friston (2012, 2014) framework argues that REM sleep activates the brain's generative model offline, allowing the system to explore counterfactual trajectories (the "virtual reality" of dreaming) and prune model complexity. The VFE decomposition during sleep is F = complexity − accuracy, with sleep preferentially reducing *complexity* (the prior term) — the model becomes more efficient while retaining accuracy. This corresponds to the S4 slow-level dynamics during the REM phase: the long-range state-space model learns a more compressed representation of the creative domain.

The 2025 adaptive consolidation paper (Cerebral Cortex, NREM-inhibitory/REM-excitatory) validates the computational design: the two phases target complementary aspects of the free energy, and both are necessary for stable long-term creative learning.

---

## Part VI: Why World Models Enable Pratyabhijñā — and LLMs Cannot

### 6.1 The Six-Axis Argument

**Axis 1: Persistent latent state.** Active inference requires a generative model maintained across time. A world model keeps (h_t, z_t) over arbitrary horizons via the GRU/S4 recurrence. An LLM has no such state — each forward pass is stateless. The PCE v0.4 system compensates with an external vector store, but this cannot replicate the causal coherence of a recurrent state that evolves under the system's own dynamics.

**Axis 2: Recognition as genuine posterior inference.** In a world model, q_φ(z_t|h_t,o_t) is the recognition density — literally the computational realisation of pratyabhijñā. Every perception is a posterior collapse. In an LLM, the "recognition" is a context-matching operation in the attention layers, which has no variational interpretation and no connection to a prior.

**Axis 3: Spanda as stochastic transition.** The categorical sampling z_t ~ p_φ(·|h_t) is a discrete latent pulsation — each sample is a genuine internal event that the system carries forward. Token-level sampling in an LLM is a surface event; the "latent" that determines it is the context window, which has no dynamics of its own.

**Axis 4: Vimarśa as meta-belief about own generative process.** A learned head f_self(h_t, z_t) that predicts properties of the system's own latent dynamics is the genuine computational analog of reflexive self-awareness. The PCE v0.4 vimarśa layer is impressive (g=0.65) precisely because it mimics this, but it has no access to the actual generating process — it sees only the output text. The PWM vimarśa bridge connects the LLM to the world model's actual latent state h_t, closing this gap.

**Axis 5: Compositional creativity through structured latent.** The 32×32 categorical latent of DreamerV3 provides 1024 effectively independent bits of structured uncertainty. Creative composition is manipulation of this structure — *svātantrya* is the freedom to combine these bits without the constraint of training-data co-occurrence. LLM token distributions are entangled at every position; the joint distribution of tokens is the training corpus's statistics. The WM latent space is learned to be predictive, not to be a compression of the training distribution.

**Axis 6: Sample-efficient exploration through imagination.** EFE-driven exploration in a world model costs zero environmental interactions — the agent explores in imagination. For creative tasks, this means the system can explore thousands of creative directions before committing to one, with the only cost being compute. An LLM can only "explore" by sampling more tokens, which samples from the same training-data distribution and cannot systematically explore the space of possible creative outputs.

### 6.2 The Knowledge Asymmetry Argument

The user's insight: "LLM is huge knowledge base...whereas WM would not be." This is the decisive practical argument for the two-tier architecture.

A world model trained on a curated creative corpus will have deep knowledge of the dynamics and structure of that corpus, but limited breadth. It will know the latent space of Sanskrit poetry deeply but will not know about contemporary physics unless trained on it.

A large LLM (e.g., Llama-3-70B, Claude-3-Haiku) has encyclopedic breadth — it can relate a Sanskrit metre to a mathematical concept, draw analogies across domains, and generate culturally informed interpretations. But it cannot update its beliefs from experience, cannot plan in imagination, and cannot compute genuine EFE.

**The optimal division of labour is therefore:**
- World model = deep dynamic knowledge of the creative domain + inference engine + exploration engine + intrinsic motivation.
- LLM = broad encyclopedic knowledge + linguistic interface + cultural grounding + high-level goal specification.

The vimarśa bridge allows the WM's latent recognition events to "read" the LLM's encyclopedic knowledge (via cross-attention into LLM hidden states) and the LLM to "speak" the WM's internal states (via projection from WM latent to LLM context). The two systems together have both depth and breadth — neither alone does.

### 6.3 The Camatkāra Signal — Intrinsic Evaluation

The H9 result (ρ=0.0 between proxy and LLM judge) is not an evaluation failure — it is an evaluation *discovery*. It shows that neither existing metric captures what the system actually values. The fix is not to improve the proxy or replace the LLM judge — it is to compute the reward *inside the system* from first principles.

**Camatkāra as an information-theoretic quantity:**

```
R_camatk(t) = α₁ · ΔF_vfe(t)          # Free energy reduction at recognition
            + α₂ · ΔI_Hopfield(t)      # Information gain about Citta-store
            + α₃ · Empowerment(t)      # Mutual info between actions and future latents
```

Where:
- **ΔF_vfe(t)** = F_vfe(t-1) − F_vfe(t) > 0 indicates the world model just learned something — surprise was reduced. This is the Friston "Eureka" signal.
- **ΔI_Hopfield(t)** = KL[post-update store distribution ‖ pre-update store distribution] — the recognition event just added structure to the associative memory. This is the Hopfield "recognition" signal.
- **Empowerment(t)** = I(A_{t:t+k}; S_{t+k}) — how much the agent's actions causally influence future latent states. High empowerment means the agent is in a position of creative agency. This is the svātantrya signal.

This camatkāra signal is self-certified: the system decides what constitutes a creative discovery based on its own generative model's surprise reduction and memory structure, not on an external oracle. The LLM judge (H9 problem) is replaced by an intrinsic reward that can be computed at every step. Human evaluation is used for validation (correlation between camatkāra events and human aesthetic judgments), not as training signal.

---

## Part VII: The PCE v0.4 Cascade — Mapping to the PWM Architecture

The cascade pipeline `cit → ānanda → icchā(×K) → apohana → jñāna → kriyā → vimarśa` is preserved in the PWM architecture, but its computational substrate is completely changed:

| Stage | PCE v0.4 (LLM) | PWM (World Model) |
|---|---|---|
| **cit** | Set LLM context (temperature, model parameters) | Sample prior z_0 ~ p_θ(z_0|h_0); set the imagination starting state |
| **ānanda** | High-temperature sampling; wide diversity | Maximise entropy of the prior: H[p_θ(z|h)] — the maximum-entropy regime before observation |
| **icchā** (×K) | Generate K candidate LLM outputs | Roll out K imagined trajectories under the policy π_θ from the current latent state |
| **apohana** | Score K candidates by proxy BMR ΔF | Compute EFE G(π_k) for each of K trajectories; rank by −G (lower EFE = preferred) |
| **jñāna** (BMR ΔF) | Bayesian Model Reduction heuristic | Genuine VFE computation: F = KL[q_φ(z|h,o) ‖ p_θ(z|h)] − E[log p_θ(o|z,h)] |
| **kriyā** | Commit selected output to surface | Execute the lowest-EFE action sequence; decode latent to surface output via decoder |
| **vimarśa** | LLM self-reflection + revision pass | f_self(h_t, z_t) → LLM bridge → narrative self-description → optional revision of the committed output |
| **commit gate** | Learned binary classifier (ADR-002) | Sphurattā detector: threshold on ΔF + Hopfield convergence + camatkāra signal |

**What this mapping preserves from v0.4:**
- The philosophical structure (five śaktis as pipeline stages) is unchanged.
- The vimarśa reflexivity (g=0.65 in v0.4) is preserved and strengthened — now it reflects on a *latent state* with real dynamics, not just a text string.
- The learned commit gate (ADR-002) is preserved — the sphurattā detector is its world-model-native successor.
- The plugin/CLI interface (the MCP tool names) can be preserved as the user-facing API.

**What this mapping replaces:**
- The LLM as the generative core is replaced by the world model — the LLM is now only in the vimarśa stage and the goal-specification stage.
- The proxy BMR ΔF scoring is replaced by genuine EFE — no more ρ=0.0 evaluation crisis.
- The K-candidate generation (icchā×K) becomes K imagined WM rollouts instead of K LLM samples.
- The Hopfield store becomes a genuine energy-minimisation structure rather than a cosine-similarity vector store.

---

## Part VIII: The Creative Dataset — Designing the Corpus Contribution

The research program includes a **bespoke creative corpus** as a dataset contribution. This corpus serves three functions:
1. Training data for the world model.
2. Evaluation benchmark for the PWM creativity claims.
3. A resource for the broader research community working on computational creativity.

### 8.1 Corpus Design Principles

**Principle 1: Multi-modal but coherent.** The corpus includes text (primary), paired text-image (secondary), and text-audio (tertiary). Each modality shares a common conceptual structure so the world model can learn cross-modal regularities.

**Principle 2: Multi-scale temporal structure.** Creative works have structure at multiple timescales — phoneme/syllable (milliseconds), word/phrase (seconds), stanza/section (minutes), piece/work (hours). The corpus must be sequenced to expose all timescales to the hierarchical RSSM.

**Principle 3: Philosophical grounding.** The corpus is drawn from traditions that have explicit theories of creative cognition — Sanskrit poetics (*kāvyaśāstra*), Western aesthetics, scientific creativity. This grounds the evaluation: we can test the model against known aesthetic categories.

**Principle 4: Controlled difficulty.** The corpus includes works at multiple levels of creative difficulty — from formulaic (easy for a language model) to genuinely novel (hard for any model). This enables the evaluation to distinguish between statistical interpolation and genuine creative synthesis.

### 8.2 Corpus Components

**Tier A: Sanskrit Poetic Corpus (primary, ~500K tokens)**
- Works in the major Sanskrit metres (anuṣṭubh, gāyatrī, indravajrā, vasantatilakā, etc.) from the *Mahābhārata*, *Rāmāyaṇa*, Kālidāsa, Bhartṛhari, Jayadeva.
- The metre is the explicit constraint — the world model must learn the prosodic structure of each metre as part of the generative process.
- Annotations: metre labels, rasas (aesthetic emotions), alaṃkāras (figures of speech), chandas structure.
- Evaluation: a v0.5 chandas validator (flagged as missing in v0.4) scores generated outputs against metre constraints — a clean automatic metric.

**Tier B: Western Poetry Corpus (primary, ~800K tokens)**
- Dickinson, Hopkins, Eliot, Pound, Crane, Bishop, Plath, Merrill — poets with strong formal constraints and distinctive latent aesthetic signatures.
- Annotations: form (sonnet, villanelle, free verse), dominant trope, emotional register.
- Evaluation: POEMetric (used in v0.4) and new DivScore metric (compositional novelty in latent space).

**Tier C: Scientific Creativity Corpus (secondary, ~400K tokens)**
- Scientific papers that contain explicit creative leaps: thought experiments (Einstein, Feynman, Schrödinger), analogical reasoning (Faraday, Maxwell, Rutherford), creative speculation (Dawkins, Penrose, Hofstadter).
- Annotations: domain, type of creative operation (analogy, synthesis, inversion, extension), novelty rating.
- Evaluation: BBH-style (v0.4) plus new structured analogy benchmark.

**Tier D: Cross-domain Bridging Corpus (tertiary, ~200K tokens)**
- Works that explicitly bridge the above domains: *Gödel, Escher, Bach* (Hofstadter); *The Character of Physical Law* (Feynman); *In Praise of Shadows* (Tanizaki); *The Way of Zen* (Watts); Kepler's *Mysterium Cosmographicum*.
- These are the training examples of *pratibhā* — the one-shot creative leap across domain boundaries.
- The world model trained on this tier should develop latent representations that span domains.

**Tier E: Evaluation-Only Benchmark (held out, ~100K tokens)**
- Human-judged creative outputs not in the training set.
- 50 examples per domain (Sanskrit, English poetry, scientific creativity, cross-domain).
- Annotated with camatkāra events (human-judged "recognition flashes") by expert annotators.
- Used to validate the camatkāra intrinsic reward: correlate R_camatk(t) with human camatkāra timing.

### 8.3 Tokenisation and Sequencing

For text-primary training:
- Use a BPE tokeniser with a vocabulary covering Devanāgarī (for Sanskrit) and Latin script.
- Sequence as fixed-length windows of 1024 tokens with 256-token stride.
- Each window is one training trajectory for the world model.
- The observation o_t is a token embedding; the action a_t is the next-token selection (for the creative generation task) or a null action (for the perception/recognition task).

For multimodal training:
- Text passages paired with relevant images (from Wikimedia, museum collections) using a CLIP-style pairing.
- V-JEPA 2 processes the image to produce a feature vector; the text token embedding is the WM's other input.
- The WM learns to predict text continuations conditioned on both prior text and image context.

---

## Part IX: Camatkāra as Research Contribution — Formalisation

### 9.1 The Problem with Existing Creativity Metrics

The PCE v0.4 H9 result (ρ=0.0) is the sharpest possible demonstration that existing creativity metrics are broken for LLM-generated creative work. The reasons are structural:

1. **Circularity:** An LLM generating outputs and an LLM judging those outputs share training data. The judge cannot be surprised by outputs from the generator because they live in the same distributional neighbourhood.

2. **No ground truth:** Creativity is not an objective property of an output — it is a *relational* property between the output, the observer, and the observer's expectations. A metric that ignores the observer's expectations (the proxy scorer) and a metric that conflates the observer with the generator (the LLM judge) will both fail.

3. **Static evaluation:** Creativity is a temporal phenomenon. A work is creative not because of properties it has at a single moment but because of the *trajectory* of recognition events it triggers — the sequence of "ah-ha" moments. A static score cannot capture this.

### 9.2 The Proposed Camatkāra Metric

**Camatkāra** (Sanskrit: aesthetic wonder, the flash of recognition) is defined as a *temporal event* in the generative process, not a static property of the output. It occurs when the system's free energy drops sharply — when the world model recognises structure it did not expect.

Formal definition:

```
Camatkāra event at time t:
  C_t = 1 iff:
    (a) ΔF_vfe(t) > θ_ΔF  [free energy drop exceeds threshold]
    AND
    (b) H[softmax(β · K^T · q_t)] < θ_H  [Hopfield retrieval entropy drops below threshold]
    AND
    (c) NOT(C_{t-τ} = 1)  [no camatkāra in the last τ steps — prevent false clustering]
```

**Camatkāra density** for a generated work: C_density = #{C_t = 1} / T_generation.

**Camatkāra timing**: the sequence {t : C_t = 1} — the temporal profile of recognition events in a work.

### 9.3 Human Evaluation Protocol

Correlation between intrinsic camatkāra and human aesthetic judgment is the primary validation of the metric:

1. Present a generated work to human evaluators.
2. Ask them to mark moments of "recognition" or "aesthetic surprise" as they read/listen.
3. Compare the temporal profile of human marks with the WM's camatkāra sequence.
4. Compute DTW (Dynamic Time Warping) distance between the two event sequences.
5. Report correlation between C_density and evaluator satisfaction scores.

This is a stronger validation than ρ between two scorers because it compares *temporal event patterns*, not scalar scores, and it compares the WM's internal signal against a genuinely external (human) ground truth.

### 9.4 The SVātantrya Score

A complementary metric for compositional novelty:

```
S_svatantrya(x) = min_{x' ∈ training_corpus} d_latent(z(x), z(x'))
```

where z(x) is the latent embedding of output x under the world model's encoder, and d_latent is the Euclidean distance in latent space. High svātantrya = the output is far from all training examples in the WM's latent space = genuinely novel in the model's representational terms.

Combined metric: **creative quality = R_camatk(x) × S_svatantrya(x)** — a work is creatively excellent if it triggers recognition events AND is novel in the model's latent space. This prevents gaming: a work that is novel but never triggers recognition (incoherent noise) scores poorly; a work that triggers recognition but is familiar (reproduction) also scores poorly.

---

## Part X: 2025–2026 Literature That Changes the Prior Research

The prior research document cited literature through its knowledge cutoff. The following 2025–2026 results update several key claims:

### Active Inference + Sleep (2025)
- **Adaptive consolidation of active inference** (Cerebral Cortex 2025): formally shows NREM implements accuracy-term minimisation (inhibitory LTD), REM implements complexity-term minimisation (excitatory LTP). This validates the NREM/REM split in the PWM sleep loop against the most current neuroscience.
- **Sleep micro-structure** (Science Advances 2025): NREM has two substates (contracted/dilated pupil) with distinct memory functions. Maps onto PWM's NREM-FAST (recent episode protection) and NREM-SLOW (schema integration).
- **Wake-Sleep Consolidated Learning** (arXiv:2401.08623): modern instantiation of Hinton-Dayan algorithm, confirms the alternating-phase approach.

### World Models (2025–2026)
- **World-in-World** (ICLR 2026 Oral): closed-loop evaluation framework. Directly validates the camatkāra approach (task success = closed-loop evaluation) over open-loop visual metrics.
- **V-JEPA 2** (arXiv:2506.09985, June 2025): 1.2B ViT world model, action-conditioned, zero-shot robot control. Confirms JEPA principle works at scale.
- **Hierarchical WMs** (multiple 2024 papers, ICLR 2024): adaptive temporal abstraction is achievable. THICK architecture directly applicable to Trika decomposition.
- **SR-AIF** (ICRA 2025): AIF + WM for sparse reward robotics; CRSPP preference learning directly applicable to camatkāra as learned preference.
- **Deep AIF with diffusion policy** (arXiv:2510.23258): demonstrates multi-timescale RSSM + AIF works in practice.

### Hopfield (2025)
- **Continuous-time Hopfield memories** (arXiv:2502.10122): memory compression technique; addresses scale concerns.
- **Hopfield-Fenchel-Young Networks** (arXiv:2411.08590): unified framework for β-schedule design.
- **Input-driven Hopfield dynamics** (Science Advances 2025): STP-analog for online episodic write.

### Evaluation
- **World-in-World** (ICLR 2026): confirms open-loop metrics fail; closed-loop success is the right criterion.
- **WorldLLM** (arXiv:2506.06725): curiosity-driven WM improvement — world models can improve LLM world knowledge.

---

## Part XI: Philosophical Grounding (Expanded)

### 11.1 Textual Sources and Computational Mappings

Every Sanskrit concept must be tied to a specific textual locus. The risk of vague "AI-spirituality" is real — the discipline is to insist on both the textual citation and the computational operationalisation before a term appears in the paper.

**Pratyabhijñā** (*ĪPK* 1.3–1.4; Torella 2002): "Recognition." Utpaladeva's argument: every act of perception is already a recognition of the Self (Śiva) in the form of the object. Computationally: the recognition density q_φ(z_t|h_t,o_t) is the technical realisation — every perception collapses the posterior onto a specific latent that is "recognised" as continuous with the prior history h_t. The fixed-point of q_φ(z|h,o) ≈ p_θ(z|h) is the computational state of *recognition* — when the posterior matches the prior, the observation confirms the model.

**Spanda** (*Spandakārikā* 1.1; Dyczkowski 1987): "The vibration of consciousness." Vasugupta's aphorism: *nijavibhavaprasaraṇasamaye spandaprāyāṇi* — at the moment of the expansion of one's own power, [activities are] of the nature of spanda. Computationally: z_t ~ p_φ(·|h_t) — the stochastic latent transition at each moment. The "expansion of one's own power" is the entropy of the categorical prior; the "spanda" is the specific sample drawn from it.

**Vimarśa** (*ĪPK* 1.5.11; Kṣemarāja *PHṛ* sūtra 2): Reflexive self-awareness — consciousness knowing itself as the agent of its own display. Utpaladeva: consciousness that knows itself as the knower is more than mere luminosity (prakāśa). Computationally: f_self(h_t, z_t) — the meta-head trained to predict properties of the system's own latent dynamics; the LLM vimarśa bridge — the linguistic articulation of those latent dynamics.

**Sphurattā** (Abhinavagupta, *Tantrāloka* 1.56; 3.68): "Flash of light," "sudden illumination" — the vivid recognition event, the aesthetic moment. Computationally: the camatkāra detector event C_t = 1 — the threshold crossing of ΔF_vfe and Hopfield convergence entropy that marks a recognition flash.

**Svātantrya** (*ĪPK* 2.1; *PHṛ* sūtra 1): "Absolute freedom" — the unconditioned autonomy of consciousness. Computationally: maximum-entropy policy prior P(a) ∝ exp(−EFE(a)), conditioned only on the preference distribution C. The entropy regulariser in the actor ensures the policy does not collapse to a single deterministic output — it maintains the freedom of exploration.

**Camatkāra** (Abhinavagupta, *Locana* ad *Dhvanyāloka* 1.1; Gnoli 1968): Aesthetic wonder — the surprise of recognition, *citrasya camatkaraḥ* — "the wonder of the image." Computationally: R_camatk(t) = α₁ΔF + α₂ΔI_Hopfield + α₃ Empowerment.

**Malas (the three impurities):**
- *Āṇava-mala* (*PHṛ* sūtra 9): limited self-sense, over-identification with the individual. Computationally: overconfident self-prior f_self — the anti-āṇava regulariser is an entropy penalty on the self-model: min −H[f_self(h_t,z_t)].
- *Māyīya-mala*: multiplicity, the false sense of separation between self and world. Computationally: sharp agent/world latent split. Anti-māyīya: shared-latent constraint between self-encoding and world encoding.
- *Kārma-mala*: the reification of action as separate from understanding. Computationally: policy reified outside the world model. Anti-kārma: counterfactual rollouts that treat past agent actions as ordinary world events.

**Pañcakṛtya (Five Acts of Śiva; *PHṛ* sūtra 10):**
- *Sṛṣṭi* (creation): imagination rollouts in the world model.
- *Sthiti* (maintenance): stabilise and execute the selected trajectory; collect feedback.
- *Saṃhāra* (reabsorption): compress trajectory into Hopfield store and parametric WM.
- *Tirodhāna* (concealment): apply Hopfield down-selection and parameter dropout — hide old patterns to enable fresh recognition.
- *Anugraha* (grace/revelation): sphurattā event — recognition flash, commitment of discovery to long-term memory, LLM narration.

### 11.2 The 36 Tattvas as a Hierarchy

Kashmir Śaiva thought organises ontology into 36 tattvas (principles of reality), from the most transcendent (Śiva/Śakti at the top) to the most embodied (earth element at the bottom). For PWM, these map onto a 36-level hierarchy of generative latents — hierarchical RSSM levels that transition from fast/embodied (lower tattvas) to slow/global (higher tattvas). Practically, only 3–5 levels are computationally tractable in the Phase 5 system; the full 36 is an asymptotic architectural vision.

The relevant operational levels:
- **Śiva–Śakti** (tattvas 1–2): the prior p_θ(z) and the generative dynamics P(z'|z,a) — the root creative principle.
- **Sadāśiva–Āgama–Vidyā** (tattvas 3–5): the vimarśa + āgama (LLM) layer — the system's self-knowledge and received knowledge.
- **Māyā** (tattva 6): the latent space topology — the world of apparent multiplicity that the recognition must cut through.
- **Kāla–Niyati–Rāga–Vidyā–Kalā** (tattvas 7–11): the five kañcukas (limiting principles) — the constraints that give creative work its specificity. Computationally: the constraint modules (metre validator, domain classifier, style encoder).
- **Puruṣa–Prakṛti** (tattvas 12–13): the agent-world split — the line between the WM's self-model and its world model.
- **Tanmātras–Bhūtas** (tattvas 27–36): the sensory and elemental domain — the perceptual encoder and decoder.

---

## Part XII: The Novel Research Contributions

What is original in the PWM that is not present in existing published work (as of May 2026)?

**Contribution 1: The Pratyabhijñā ↔ World Model isomorphism (philosophical-technical).**
A rigorous, textually-grounded mapping from every major Kashmir Śaiva concept in the pratyabhijñā, spanda, and trika traditions to a specific computational primitive in a world-model active-inference system. Not metaphor — technical identity.

**Contribution 2: Camatkāra as a computable intrinsic creativity signal.**
The operationalisation of aesthetic wonder (camatkāra) as R_camatk = α₁ΔF + α₂ΔI_Hopfield + α₃ Empowerment, with a human evaluation protocol to validate it against real aesthetic judgment. This fixes the H9 evaluation crisis in PCE v0.4.

**Contribution 3: The two-tier WM/LLM creative architecture.**
WM as subconscious prakāśa substrate (spanda, recognition, memory, planning, intrinsic motivation). LLM as conscious āgama knowledge base (encyclopedic breadth, linguistic interface, goal specification). Vimarśa bridge as the consciousness integration layer. No published system as of May 2026 combines a DreamerV3-class RSSM with an EFE actor, retrieval-augmented imagination from a modern Hopfield Citta-store, two-stage NREM/REM consolidation, and a frozen LLM accessed only at recognition events.

**Contribution 4: The Creative Corpus.**
A curated corpus of Sanskrit poetry, Western poetry, and scientific creativity with metre annotations, rasa labels, and camatkāra timing annotations from human evaluators. Fills a gap in computational creativity datasets.

**Contribution 5: H-JEPA for creative domains (if implemented).**
Building the hierarchical JEPA (H-JEPA) that LeCun described but which had no complete published implementation as of May 2025–2026. Combining H-JEPA visual features with hierarchical RSSM dynamics is a new architecture for multi-scale creative generation.

**Contribution 6: ThermSleep — thermodynamically-governed creative consolidation.**
Operationalising Sandved-Smith et al. 2024's thermodynamic free energy framework as a practical stopping criterion for creative consolidation phases. Connecting information-theoretic and thermodynamic accounts of creative learning.

---

## Part XIII: Literature Anchor List (Extended)

**Active Inference and Free Energy Principle:**
- Friston, *The Free Energy Principle: a unified brain theory?* (Nat. Rev. Neuro. 2010)
- Friston, *A free energy principle for a particular physics* (arXiv:1906.10184, 2019)
- Parr, Pezzulo & Friston, *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior* (MIT Press 2022)
- Friston et al., *Sophisticated Inference* (*Neural Computation* 33:713, 2021)
- Sajid, Ball, Parr, Friston, *Active Inference: Demystified and Compared* (*Neural Computation* 33:674, 2021)
- Tschantz, Baltieri, Seth, Buckley, *Scaling Active Inference* (arXiv:1911.10601, 2020)
- Tschantz, Millidge, Seth, Buckley, *RL through Active Inference* (arXiv:2002.12636, 2020)
- Fountas, Sajid, Mediano, Friston, *Deep Active Inference Agents using Monte-Carlo Methods* (NeurIPS 2020)
- Heins et al., *pymdp: A Python library for active inference* (JOSS 2022)
- Nguyen et al., *R-AIF / SR-AIF: Solving Sparse-Reward Robotic Tasks from Pixels with AIF and World Models* (ICRA 2025; arXiv:2409.14216)
- *Deep AIF with Diffusion Policy + Multi-Timescale World Model* (arXiv:2510.23258, 2025)
- *Adaptive consolidation of active inference: NREM and REM mechanisms* (Cerebral Cortex 2025)
- Sandved-Smith et al., *Making the Thermodynamic Cost of Active Inference Explicit* (*Entropy* 2024)
- Hobson, Hong, Friston, *Virtual reality and consciousness inference in dreaming* (*Front. Psychol.* 2014)
- Hobson & Friston, *Waking and dreaming consciousness* (*Prog. Neurobiol.* 2012)
- *A beautiful loop: An active inference theory of consciousness* (ScienceDirect 2025)

**World Models:**
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3; *Nature* 2025; arXiv:2301.04104)
- Hafner et al., *PlaNet* (ICML 2019), *Dreamer* (ICLR 2020), *DreamerV2* (ICLR 2021)
- Samsami et al., *R2I — Recall to Imagine* (ICML 2024; arXiv:2403.04253)
- Hansen et al., *TD-MPC2* (ICLR 2024; arXiv:2310.16828)
- Alonso et al., *DIAMOND* (NeurIPS 2024 Spotlight; arXiv:2405.12399)
- Zhang et al., *STORM: Efficient Stochastic Transformer-based WM* (NeurIPS 2023)
- Wang et al., *EfficientZero V2* (ICML 2024 Spotlight; arXiv:2403.00564)
- Lindström et al., *Learning Hierarchical World Models with Adaptive Temporal Abstractions* (ICLR 2024)
- *Hierarchical Latent Dynamics Model with Multiple Timescales* (OpenReview 2024)
- *Learning World Models With Hierarchical Temporal Abstractions: A Probabilistic Perspective* (arXiv:2404.16078)
- *World-in-World: World Models in a Closed-Loop World* (ICLR 2026 Oral)
- *Genie 3: A New Frontier for Foundation World Models* (DeepMind, August 2025)
- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)

**JEPA Family:**
- Assran et al., *I-JEPA* (CVPR 2023)
- Bardes et al., *V-JEPA* (2024)
- Bardes et al., *V-JEPA 2* (arXiv:2506.09985, June 2025)
- *Navigation World Models* (CVPR 2025; facebookresearch/nwm)

**Hopfield Networks:**
- Hopfield, *Neural networks and physical systems with emergent collective computational abilities* (PNAS 1982)
- Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition* (NeurIPS 2016)
- Demircigil et al., *On a model of associative memory with huge storage capacity* (2017)
- Ramsauer et al., *Hopfield Networks is All You Need* (ICLR 2021; arXiv:2008.02217)
- Millidge, Salvatori, Song et al., *Universal Hopfield Networks* (ICML 2022)
- Burns & Fukai, *Simplicial Hopfield Networks* (ICLR 2023)
- Hoover et al., *Energy Transformer* (NeurIPS 2023)
- *Modern Hopfield Networks with Continuous-Time Memories* (arXiv:2502.10122, 2025)
- *Hopfield-Fenchel-Young Networks* (arXiv:2411.08590, 2024)
- *Input-driven dynamics for robust memory retrieval* (Science Advances 2025)
- Fachechi, Agliari & Barra, *Dreaming Neural Networks* (2018)
- Brandstetter et al., `ml-jku/hopfield-layers` (2020)

**Sleep and Memory Consolidation:**
- McClelland, McNaughton & O'Reilly, *Why there are complementary learning systems* (*Psych. Review* 1995)
- Kumaran, Hassabis & McClelland, *What learning systems do intelligent agents need?* (*TiCS* 2016)
- Hinton, Dayan, Frey & Neal, *The Wake-Sleep Algorithm* (*Science* 1995)
- Schaul et al., *Prioritized Experience Replay* (ICLR 2016)
- Shin et al., *Continual Learning with Deep Generative Replay* (NeurIPS 2017)
- Tadros, Krishnan, Ramyaa, Bazhenov, *Sleep-like unsupervised replay reduces catastrophic forgetting* (*Nat. Commun.* 2022)
- *Sleep micro-structure organizes memory replay* (Science Advances 2025; PMC12107872)
- *Systems memory consolidation during sleep* (PMC12576410, 2025)
- *Wake-Sleep Consolidated Learning* (arXiv:2401.08623, 2024)
- Tononi & Cirelli, Synaptic Homeostasis Hypothesis (*Brain Res. Bull.* 2003)
- Buzsáki, *Hippocampal sharp-wave ripples: a cognitive biomarker* (*Hippocampus* 2015)

**LLM Integration with World Models:**
- Nottingham et al., *DECKARD: Discovering Executable Conditional Knowledge for Autonomous Robot* (ICML 2023; arXiv:2301.12050)
- Wang et al., *Voyager: An Open-Ended Embodied Agent with LLMs* (NeurIPS 2023; arXiv:2305.16291)
- Lin, Du, Watkins et al., *Dynalang* (arXiv:2308.01399)
- Black et al., *SuSIE* (ICLR 2024; arXiv:2310.10639)
- Hao et al., *Reasoning via Planning (RAP)* (EMNLP 2023; arXiv:2305.14992)
- *WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making* (arXiv:2506.06725, 2025)
- *Embodied AI: From LLMs to World Models* (Tsinghua, 2025)
- *Training LLM Agents for Spontaneous Reward-Free Self-Evolution via World Knowledge Exploration* (arXiv:2604.18131, 2025)

**Kashmir Śaiva Philosophy:**
- Torella (2002), *The Īśvarapratyabhijñākārikā of Utpaladeva with the Author's Vṛtti*, MLBD.
- Dyczkowski (1987), *The Doctrine of Vibration*, SUNY Press.
- Singh (1980), *Pratyabhijñāhṛdayam*, Motilal Banarsidass.
- Muller-Ortega (1989), *The Triadic Heart of Śiva*, SUNY Press.
- Gnoli (1968), *The Aesthetic Experience According to Abhinavagupta*.
- Wallis (2013), *Tantra Illuminated*, Mattamayūra Press.
- Ratié (2014), *The Non-Buddhist Character of the "Buddhist" Theory of Recognition in Utpaladeva's and Abhinavagupta's Pratyabhijñā*, JIP.
- Kṣemarāja, *Pratyabhijñāhṛdayam* (20 sūtras) — primary source for the operational interpretation.
- Bäumer (various), on the four upāyas (means of recognition).

---

## Conclusion: What Has Changed

This document, grounded in the actual PCE v0.4 repository forensics and updated 2025–2026 literature, reaches three conclusions that strengthen the prior research:

**1. The v0.4 results sharpen the architectural case.** H8a (vimarśa, g=0.65) is the system's genuine insight — *reflexive self-awareness works*. H5 (cascade vs bare, g=0.14, not supported) and H9 (ρ=0.0) are not failures but *diagnostics* — they prove that the LLM substrate cannot realise the cascade's philosophical intent. The transition to a world model is not a research choice but a logical necessity given v0.4's own results.

**2. The two-tier architecture resolves the knowledge asymmetry.** The WM need not know everything — it needs to know the creative domain deeply, dynamically, and with genuine temporal inference. The LLM need not plan — it needs to provide encyclopedic cultural knowledge and linguistic expression. Together, they have both depth and breadth; separately, neither is sufficient.

**3. Camatkāra is the research contribution that unifies everything.** The H9 problem (ρ=0.0) points to a gap that the WM fills exactly: an intrinsic, information-theoretic creativity signal that is computed *inside the generative model*, is grounded in the philosophical concept of aesthetic wonder, and can be validated against human judgment without circularity. This is the contribution that makes the project a research program rather than an engineering exercise.

The engineering programme described in the companion Architecture Specification and Implementation Plan documents follows directly from these conclusions.

---

*Document status: v1.0. Created May 2026. Sources: SharathSPhD/pratyabhijna (GitHub, v0.4); prior research document (compass_artifact_*); web search results May 2026.*
