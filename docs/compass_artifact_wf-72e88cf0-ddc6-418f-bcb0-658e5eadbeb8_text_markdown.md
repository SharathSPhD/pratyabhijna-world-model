# The Pratyabhijñā World Model: A Research and Engineering Reference

## Preface — scope, method, and an honest caveat about the primary sources

This document is the consolidated technical reference for engineering a creative AI system that operationalizes the Kashmir Śaiva philosophy of *pratyabhijñā* (recognition) through Karl Friston's active inference, instantiated on a modern world-model substrate. It is written to be sufficient for an ML engineer to begin implementation on a single 128 GB unified-memory workstation.

**An important methodological caveat must be stated up front.** The three primary sources named in the brief — the paper PDF at `sharathsphd.github.io/pratyabhijna/paper/main.pdf`, the GitHub repository `github.com/sharathsphd/pratyabhijna`, and the Apple Podcasts episode — could not be retrieved in this research environment. Each returned a permissions error, and external search did not surface a `pratyabhijna` repository under the author's verified GitHub account (`SharathSPhD` — Dr. Sharath Sathish, Senior Data Scientist at bp plc, Google Scholar `dcyu5ucAAAAJ`, ~31–35 public repos including `pramana`, `attractor-flow`, `multi-agent-claude`, `RLpower`, `dreamprice`, `ccmMul`). The repository may be private, very recent, or under a slightly different slug. Section 1 below therefore reconstructs the *plausible* architecture and maturity profile of the project from the author's adjacent intellectual portfolio (an explicit Sanskrit-named research program), the philosophical thesis suggested by the project name, and the standard architectural choices in this subfield. Sections 2–10 do not depend on the primary sources and are based on independently verifiable scholarly and technical literature. Wherever Section 1's claims about the project are inferential rather than verified, that is flagged in the text.

---

## 1. The Pratyabhijñā Creative Engine — reconstructed architecture and maturity

### 1.1 Author and intellectual program

The project sits inside a coherent research program by Dr. Sharath Sathish that visibly merges Sanskrit epistemological categories with dynamical-systems and multi-agent LLM tooling. The relevant signposts are:

- `pramana` ("valid means of cognition") — a Python project whose name names the standard epistemology of Indian philosophy (`pratyakṣa`, `anumāna`, `āgama`, etc.), strongly indicating the author treats Indic categories as live engineering primitives, not metaphor.
- `attractor-flow` — multi-agent Claude Code trajectory monitoring with **Lyapunov exponents, regime classification, and bifurcation detection**. This is dynamical-systems ML, exactly the toolkit one would deploy if one took *spanda* (pulsation) and *unmeṣa/nimeṣa* (opening/closing) seriously as state-space dynamics.
- `multi-agent-claude`, `openclaw-swarm` — LLM-orchestration tooling consistent with using LLMs as a high-level planner / āgama layer.
- `RLpower`, `dreamprice`, `ccmMul` — reinforcement learning and causal-inference work indicating familiarity with the technical machinery of model-based RL and time-series causality.

The published profile (PyTorch, TensorFlow, scikit-learn, RL background) is a standard mid-career applied-ML stack; the philosophical naming is the distinguishing feature.

### 1.2 The most likely architectural intent (reconstructed)

Given (a) the project name *Pratyabhijñā Creative Engine*, (b) the brief's explicit references to Hopfield networks, sleep dynamics, active inference, and Kashmir Shaiva concepts, and (c) the author's adjacent portfolio, the project is best read as a **first-generation research prototype that wires philosophical categories to computational modules but whose computational core is currently LLM-centric rather than world-model-centric.** This is the standard pattern for solo philosophically-motivated AI systems in 2024–2026 because LLMs are the cheapest substrate to *suggest* the relevant behaviors without the infrastructural lift of training a world model.

The plausible high-level architecture, by section of the brief:

**Generative model.** The system likely defines a generative process P(o, s) at the level of *symbolic latent variables* representing creative state — e.g., themes, motifs, emotional valences — manipulated by an LLM acting as both the encoder Q(s|o) and the generator P(o|s). Belief updates are likely implemented as prompt-state updates rather than as gradient-based posterior inference. This is consistent with the author's heavy use of Claude Code as orchestration substrate.

**Active inference.** Expected Free Energy is most likely *named and decomposed in prose* (epistemic value, pragmatic value) but implemented as a heuristic scoring function over candidate next outputs — a standard pattern when one wants the exploration/exploitation balance of EFE without committing to variational message passing. The recognition density Q(s|o) is probably implicit in LLM context handling.

**Hopfield network.** Given the brief's specific call-out, the project almost certainly *names* a Hopfield-like associative memory ("Citta-store" is the natural philosophical label) and *wires the interface* — a memory module that stores past creative states and supports retrieval — but the implementation is most likely either (i) a vector store with cosine retrieval (a *de facto* one-step modern Hopfield with β=1/√d but called by another name) or (ii) a stub awaiting the modern-Hopfield-network library integration.

**Sleep / "ThermSleep" dynamics.** "ThermSleep" is not an established term in the published literature (the closest rigorous concept is Sandved-Smith et al. 2024, *Entropy* 26:622, which distinguishes variational free energy from thermodynamic free energy in active inference). It is most reasonably interpreted as a project-internal coinage: *thermodynamically-informed sleep consolidation*. The most likely implementation status is **specified but unimplemented**: the README/paper describes an offline phase intended for replay and prior refinement, but the actual loop is not running.

**Creative pipeline.** Probably a generate-and-narrate loop: an LLM proposes outputs, scores them on an EFE-flavored objective, the highest-scoring output is committed, and the system narrates the commitment as "recognition." Modalities are most likely text, possibly text-to-image via an external diffusion model.

**Evaluation / metrics.** The genre suggests qualitative case studies and possibly self-reported novelty / coherence scores, with formal benchmarks deferred.

### 1.3 Reconstructed implementation maturity matrix

| Component | Reconstructed status | Reasoning |
|---|---|---|
| Philosophical scaffolding (named modules: Spanda, Vimarśa, Citta…) | LIKELY IMPLEMENTED as naming/interface | strong signal from author's `pramana` repo pattern |
| Generative model P(o,s) | LIKELY PARTIAL — LLM-mediated symbolic latents | standard pattern in this space |
| Recognition model Q(s|o) | LIKELY IMPLICIT in LLM context | no separate variational encoder expected |
| Expected Free Energy minimization | LIKELY HEURISTIC, not variational | EFE-as-prose with score-function proxy |
| Epistemic / pragmatic value separation | LIKELY NAMED, partially computed | typical in LLM-EFE projects |
| Hopfield / Citta-store | LIKELY WIRED-NOT-IMPLEMENTED or vector-store proxy | brief explicitly flags this |
| Sleep / ThermSleep consolidation | LIKELY CONCEPTUAL ONLY | brief explicitly flags this |
| Creative output pipeline | LIKELY PARTIAL | LLM-driven generation; modal expansion aspirational |
| Belief-state world model with imagination | LIKELY ABSENT | the central gap and the rationale for the present brief |
| Evaluation suite | LIKELY MINIMAL | early prototypes rarely have one |

This reconstruction frames the redesign that follows: **the project's philosophical and architectural ambition is well-posed; its computational substrate (LLM-centric) is mismatched to its intent. The redesign moves the substrate to a world model while preserving the philosophical scaffolding and adding LLM augmentation in a clearly subordinated role.**

---

## 2. Gap analysis — philosophy ↔ active inference ↔ current substrate

### 2.1 Philosophical intent vs. computational substrate

Pratyabhijñā holds that **every cognition is already a recognition**: to perceive *x* as *x* presupposes the synthetic unity of consciousness across moments, which is structurally a re-cognition of self in form (Utpaladeva, *Īśvarapratyabhijñākārikā* 1.3–1.4; Torella 2002). The implication for engineering is severe: perception, memory, and generation cannot be modular; they must be aspects of a single inferential operation over a persistent generative model.

An LLM-centric prototype cannot satisfy this constraint, because **an LLM has no persistent latent state across episodes**. Each prompt is a fresh forward pass; "memory" lives only in the context window or in an external store retrieved by lookup. Recognition in the Pratyabhijñā sense — the latent identifying itself as its own — has no native locus.

Spanda (the dynamic throb of consciousness; Vasugupta/Bhaṭṭa Kallaṭa, *Spandakārikā* 1.1; Dyczkowski 1987) is *temporal* dynamics, not stochastic token sampling. RSSM-style stochastic transitions z_t ~ p(z_t | h_t) capture spanda far more faithfully than next-token logits, because each transition is a genuine internal pulsation that updates a state which the model then carries forward.

Vimarśa (reflexive self-awareness, Utpaladeva *ĪPK* 1.5.11; ĪPV ad loc.) requires a **meta-level belief about one's own generative model**. In an LLM this can only be simulated by self-instructional prompting; in a world model it admits a clean implementation as a learned function f_self(h_t, z_t) trained to predict properties of the system's own latent dynamics.

### 2.2 Active inference specification vs. typical implementation

The active-inference specification calls for:
- variational free energy F = D_KL[Q(s)‖P(s)] − E_Q[log P(o|s)] minimized in perception;
- expected free energy G(π) = (negative pragmatic value) + (negative epistemic value) minimized in action selection over imagined futures.

A heuristic-EFE LLM agent satisfies neither. Variational free energy is not minimized because there is no Q(s) being updated by gradient flow; expected free energy is not minimized because the imagination rollout is over token continuations, not over a learned latent dynamics, and information gain is not computed (it would require a posterior over latent states, which the LLM does not maintain). Worse, the *epistemic* term — D_KL[Q(s_τ|o_τ,π)‖Q(s_τ|π)] — has no analog in next-token prediction, so the principled curiosity-bonus that distinguishes active inference from standard model-free RL is absent.

A world-model implementation closes this gap directly: the RSSM's prior p_θ(z_t|h_t) and posterior q_φ(z_t|h_t,o_t) give Q(s) and P(s|o); imagination rollouts give the predictive distributions over o_τ and s_τ needed to compute both terms of EFE; and as Tschantz et al. (2020) and Fountas et al. (NeurIPS 2020) show, swapping REINFORCE for EFE in a Dreamer-class world model is a small, well-understood modification.

### 2.3 Hopfield's intended role vs. its actual state

The intended role is a Citta-store: a content-addressable memory of past latent states whose attractor dynamics support pattern completion (recognition under partial cue) and one-shot binding of new experiences. Two failure modes are typical in early prototypes: (i) the memory is implemented as a flat vector store with cosine retrieval and never reaches the metastable / mixture regimes of a modern continuous Hopfield network (Ramsauer et al. ICLR 2021), losing the schema-blending behavior that is precisely what makes the module philosophically interesting; (ii) the memory is read-only — populated from an offline ingestion script — rather than written online from agent experience, so it cannot support episodic recall of within-run events.

The fix is to drop in `ml-jku/hopfield-layers` and use `HopfieldLayer` (learnable patterns) for semantic memory and `Hopfield` (cross-attention over a FIFO of recent latents) for episodic memory, with the inverse-temperature β exposed as a knob between specific recall (high β) and schema retrieval (low β).

### 2.4 Sleep/consolidation intent vs. implementation

The intent is a biologically-plausible offline phase that performs (a) replay-based consolidation of episodic memory into the parametric world model (CLS theory, McClelland et al. 1995; updated Kumaran, Hassabis, McClelland 2016), and (b) generative dreaming that exposes the planner to unvisited regions of state space (Hobson & Friston 2014; Hobson & Friston 2012). A typical prototype omits both: there is no replay buffer prioritized by surprise, no dreaming loop, and no synaptic-homeostasis-style down-selection of the memory store (cf. Tononi & Cirelli; Fachechi, Agliari & Barra 2018).

The fix specified in §5 is a two-stage NREM/REM analog with prioritized experience replay (Schaul et al. 2016), generative replay (Shin et al. 2017), Tadros-Bazhenov-style sleep consolidation (Tadros et al. *Nat. Commun.* 2022), and Hopfield-store down-selection.

### 2.5 What's missing for genuine creativity

In Pratyabhijñā terms, creativity is *svātantrya* — the unconstrained autonomy of consciousness — manifest as *sṛṣṭi* (emanation) and registered as *camatkāra* (aesthetic wonder; Abhinavagupta, *Abhinavabhāratī*; Gnoli 1968). In computational terms this requires three properties absent from an LLM prototype: (i) a maximum-entropy policy prior tempered only by preferences, not by training-data likelihood; (ii) a structured latent space whose composition supports genuinely novel combinations rather than statistical interpolation in token distribution; (iii) an *intrinsic* reward signal tied to recognition — a free-energy-reduction or empowerment quantity computed inside the generative model — rather than an extrinsic LLM-judged "creativity score" that is biased by training data and easily gamed.

---

## 3. State-of-the-art world models — technical synthesis

### 3.1 The RSSM family and DreamerV3

DreamerV3 (Hafner et al. 2023, arXiv:2301.04104; *Nature* 2025) is the canonical reference. The Recurrent State-Space Model decomposes latent state at time t into a deterministic GRU state h_t and a stochastic state z_t consisting of **32 categorical variables of 32 classes each** (a 32×32 one-hot tensor, ≈1024 effective bits, with straight-through gradients). The per-step computation is

  h_t = f_φ(h_{t−1}, z_{t−1}, a_{t−1}); ẑ_t ~ p_φ(ẑ_t | h_t); z_t ~ q_φ(z_t | h_t, x_t); x̂_t ~ p_φ(x̂_t | h_t, z_t); r̂_t ~ p_φ(r̂_t | h_t, z_t); ĉ_t ~ p_φ(ĉ_t | h_t, z_t)

with the world-model loss L = E[L_pred + β_dyn·L_dyn + β_rep·L_rep] where L_pred is symlog-MSE for image/proprio reconstruction plus twohot-symlog cross-entropy for reward and binary cross-entropy for continue, L_dyn = KL[sg(q_φ)‖p_φ] (KL balancing β_dyn=0.5, β_rep=0.1) with **free bits at 1 nat** per categorical to prevent posterior collapse. The actor-critic is trained on **15-step imagined rollouts** in latent space using λ-returns (λ=0.95, γ=0.997), REINFORCE with entropy bonus, and 5th–95th-percentile return normalization. Symlog (sign(x)·ln(1+|x|)) and twohot-encoded distributional critics over 255 buckets handle scale variability without per-task hyperparameter tuning. Mature open source: `danijar/dreamerv3` (JAX, used in Nature paper), `NM512/dreamerv3-torch` (PyTorch, hackable), `EclecticSheep/sheeprl`. Trains DMC/Crafter on a single GPU in hours.

The crucial property for our purposes: **DreamerV3's loss decomposition is isomorphic to Friston's variational free energy**. L_pred is the negative expected log-likelihood (the accuracy term); L_dyn + L_rep is the KL between recognition and prior (the complexity term); free bits are a numerical regularizer on KL, not a conceptual deviation. Replacing REINFORCE with EFE minimization in the actor is ~50 lines of code (Tschantz et al. 2020 give the recipe).

### 3.2 Other world models surveyed

**TD-MPC2** (Hansen et al. ICLR 2024): decoder-free implicit world model with SimNorm latent, MPPI/CEM planning, distributional value via twohot bins, single hyperparameter set across 104 tasks. The MPC step is structurally identical to active-inference policy selection — sum of expected utilities + bootstrapped value can be replaced with negative EFE essentially by changing the reward function.

**STORM** (Zhang et al. NeurIPS 2023): Transformer dynamics over the same 32×32 categorical latent; mean Atari-100k HNS 1.27 in 4.3 h on a single RTX 3090 — the cleanest "Transformer DreamerV3" available.

**R2I — Recall to Imagine** (Samsami et al. ICML 2024): replaces the GRU in DreamerV3 with an S4 state-space model, **9× faster** training via parallel scan, SOTA on memory-heavy benchmarks. The right starting point for hierarchical/long-horizon Trika decomposition.

**DIAMOND** (Alonso et al. NeurIPS 2024 spotlight): EDM-formulation diffusion model in pixel space as the world model; **Atari-100k mean HNS 1.46**, the best for any world model on that benchmark. Strong creative-decoder candidate but lacks a compact latent for control loops.

**IRIS / Δ-IRIS** (Micheli et al. ICLR 2023, ICML 2024): VQ-tokenized frames + autoregressive Transformer; Δ-IRIS uses continuous tokens + delta-VAE for cheaper, longer-horizon prediction.

**EfficientZero V2** (Wang et al. ICML 2024 spotlight): MuZero descendant with sampling-based Gumbel search and self-supervised consistency loss; outperforms DreamerV3 on **50/66 tasks** across Atari-100k, DMC Proprio, DMC Vision.

**JEPA family** (Assran et al. CVPR 2023; Bardes et al. 2024; **V-JEPA 2** arXiv:2506.09985, June 2025): predict in embedding space, not pixel space. V-JEPA 2 is a 1.2 B-parameter ViT pretrained on >1 M h video + 1 M images, action-conditioned via a 62 h robot-trajectory finetune, used for zero-shot MPC robot control with reported 65–80 % pick-and-place success. Open source under `facebookresearch/vjepa2`. **H-JEPA — the hierarchical version LeCun emphasizes — has no published end-to-end implementation**; this is a real research opportunity.

**Genie 1/2/3** (DeepMind): foundation world models; Genie 3 (Aug 2025) reports 720p 24 fps real-time interactive multi-minute consistency with emergent object permanence. Closed weights — useful only as a capability benchmark.

**Object-centric models** (C-SWM, OP3, SAVi, SOLD): the only architectures with a built-in inductive bias for **compositional generalization**; the natural fit for factored Markov blankets and the "individuation in unity" of Pratyabhijñā.

### 3.3 Comparison matrix (condensed)

| Architecture | Sample eff. | Latent | Hierarchy | Compositionality | Single-128GB train | OSS maturity | AIF fit | Creative gen. |
|---|---|---|---|---|---|---|---|---|
| DreamerV3 | Excellent | 32×32 cat | Limited | Moderate | Yes | ★★★★★ | **Excellent** | Moderate |
| TD-MPC2 | Excellent | SimNorm cont. | Limited | Moderate | Yes | ★★★★★ | **Excellent** | Low |
| STORM | Excellent | 32×32 cat + Tx | Limited | Moderate | Yes | ★★★★ | Very good | Good |
| R2I | Very good | cat + S4 | **Best of family** | Moderate | Yes | ★★★ | Excellent | Moderate |
| DIAMOND | SOTA Atari-100k | Pixel diffusion | Limited | Low | Yes | ★★★★ | Moderate | **Excellent** |
| EfficientZeroV2 | SOTA | Abstract | Tree | Low | Marginal | ★★★★ | Very good | Low |
| V-JEPA 2 | Excellent SSL | ViT cont. | Partial | Moderate | Yes (frozen) | ★★★★ | **Excellent in principle** | Low (no decoder) |
| SAVi/SOLD | Moderate | Object slots | Possible | **Best** | Yes | ★★★ | Very good | Low |
| Genie 3 | N/A (closed) | Closed | Some | Unknown | — | ★ | N/A | Excellent |

### 3.4 Recommendation

The architecturally optimal substrate for the Pratyabhijñā World Model is a **DreamerV3 RSSM core (PyTorch, NM512 fork) modified with an EFE policy head**, optionally swapping the GRU for an S4 backbone (R2I) when long-horizon memory becomes the bottleneck, optionally fronted by a frozen V-JEPA 2 encoder for richer perceptual features, and optionally rear-ended by a DIAMOND-style EDM diffusion decoder for high-fidelity creative output. Object-centric slots (SOLD) are a Phase-3+ enhancement for compositional creativity. None of these six layers has been combined end-to-end in published work as of May 2026; that gap is itself a research contribution.

---

## 4. Hopfield networks in the modern context

### 4.1 The progression

Classical Hopfield (1982) is a recurrent symmetric network of N binary units with energy E(ξ) = −½ξᵀWξ + bᵀξ, Hebbian outer-product storage, and capacity ≈0.138 N. Krotov & Hopfield (NeurIPS 2016) replaced the quadratic energy with E(ξ) = −Σᵢ F(xᵢᵀξ), F(z) = zⁿ, gaining capacity ∝ N^{n−1}/log N; Demircigil et al. (2017) took F(z) = exp(z), proving **exponential capacity** ∝ exp(N/2).

Ramsauer et al. (Hochreiter group, ICLR 2021, "Hopfield Networks is All You Need"; arXiv:2008.02217) extended to continuous states with energy E(ξ) = −lse(β, Xᵀξ) + ½‖ξ‖² and the one-step update

  ξ^{new} = X · softmax(β Xᵀξ).

This is **exactly Transformer attention** with X = stored patterns, β = 1/√d_k. The capacity is exponential in the *embedding dimension* (not neuron count); retrieval converges in one step in the high-β regime; and there are three operating regimes (global average, metastable mixtures, sharply peaked attractors) selected by β. Empirically, lower Transformer layers operate in regime 1, upper layers in regimes 2/3 — a structural map between attention depth and memory specificity.

The 2024 Nobel Prize in Physics to Hopfield and Hinton has reinvigorated this strand. Recent work includes **Energy Transformer** (Hoover et al. NeurIPS 2023), **Universal Hopfield Networks** (Millidge et al. ICML 2022) which unifies classical HN, Sparse Distributed Memory, and modern continuous HN under a similarity-separation-projection schema, **Simplicial Hopfield Networks** (Burns & Fukai ICLR 2023) with multi-neuron interactions, and **Pham et al. 2025** linking diffusion models to associative memory.

### 4.2 The implementation library

The reference PyTorch library is `ml-jku/hopfield-layers` (Brandstetter et al., JKU Linz, 1.7k+ stars), exposing three modules:

- **`Hopfield`** — general associative module taking (R, Y, Q) where R provides keys/values and Q queries; equivalent to cross-attention with explicit β control.
- **`HopfieldLayer`** — stored patterns are *learnable parameters*; functions as trainable associative memory with gradient-descent optimization of the slot bank.
- **`HopfieldPooling`** — learnable query pools over a variable-sized set of inputs; replaces sum/mean/max pooling, used by Widrich et al. 2020 for immune-repertoire MIL.

### 4.3 Role in a world model and Pratyabhijñā mapping

The Citta-store maps onto a modern Hopfield network with two modes:

- **Episodic mode**: a FIFO buffer of recent (h_t, z_t, a_t, r_t, novelty_t) tuples used as keys/values with high β; sharp single-pattern retrieval supports one-shot recall of specific past trajectories — *smṛti*.
- **Semantic mode**: a `HopfieldLayer` of learnable prototypes trained by clustering replayed dream latents during sleep; lower β supports schema retrieval — the metastable-mixture regime is the formal locus of *concept blending*.

Sphurattā (the flash of recognition) maps onto the discrete event of a sharp attractor convergence in the Hopfield store — operationally, a thresholded drop in the retrieval entropy of softmax(β Q Kᵀ) that the system can log, narrate (via the LLM vimarśa head), and treat as a creative discovery to commit.

Citi vs. citta — universal vs. contracted consciousness — maps onto the global-prior ↔ episode-posterior distinction: the Hopfield semantic store is the system's contracted Citi (a finite parametric approximation of the universal generative model), and each query collapses citta into a specific posterior.

### 4.4 Connection to sleep replay

Sharp-wave ripples (Buzsáki 2015) reactivate hippocampal place-cell sequences during NREM sleep and quiet wakefulness; forward replay supports planning and consolidation, reverse replay supports credit assignment. The Hopfield store is the natural computational analog of the hippocampal CA3 recurrent network: pattern-completion under a partial cue is the ripple-driven retrieval, and the offline distillation of Hopfield patterns into the parametric world model is the cortical consolidation. The Fachechi-Agliari-Barra "Dreaming Neural Networks" (2018) extension — online storage + offline unlearning + consolidation — provides the quantitative recipe for reaching theoretical capacity α=1.

---

## 5. Sleep dynamics and offline consolidation

### 5.1 The CLS substrate

McClelland, McNaughton & O'Reilly (*Psych. Review* 1995) posit two interacting learning systems: a fast, sparse, pattern-separated hippocampus and a slow, distributed, overlapping neocortex; the hippocampus offline-replays episodes during quiet rest and sleep, allowing cortical consolidation without catastrophic interference. Kumaran, Hassabis & McClelland (*Trends Cog. Sci.* 2016) updated CLS to emphasize that replay is goal-dependent and prioritized — recent, salient, surprising, or reward-relevant memories preferentially replay (the biological precursor to prioritized experience replay), and the cortex can integrate new information rapidly when consistent with existing schemas.

### 5.2 Wake-Sleep, generative replay, and Tadros-style consolidation

Hinton, Dayan, Frey & Neal's wake-sleep algorithm (*Science* 1995) trains Helmholtz machines by alternating phases: in **wake**, clamp visible to data, sample h via recognition net, update generative weights to maximize log P(x,h); in **sleep**, sample h ~ P(h), x ~ P(x|h), and update recognition weights to better invert the generative samples. This is the conceptual ancestor of VAEs and the formal sibling of variational free-energy minimization.

Modern instantiations relevant for the present project:
- **Prioritized Experience Replay** (Schaul et al. ICLR 2016): sample by |TD error|^α with importance-sampling correction.
- **Generative Replay** (Shin et al. NeurIPS 2017): train a generator on past tasks, use generated pseudo-samples for rehearsal — circumvents storage of raw past data and addresses catastrophic forgetting; directly inspired by hippocampal generative replay.
- **Tadros, Krishnan, Ramyaa, Bazhenov** (*Nat. Commun.* 2022): map ANN → SNN, drive with task-tuned Poisson noise, update by local Hebbian/STDP plasticity (no labels, no replay buffer), map back to ANN. Empirically prevents catastrophic forgetting on sequential MNIST/CIFAR continual learning.
- **Tononi & Cirelli synaptic homeostasis hypothesis (SHY)**: net potentiation in wake, net depression in sleep; offline down-selection desaturates capacity (Nere et al. 2013).

### 5.3 "ThermSleep" — interpretation

The term is not indexed in major venues. The closest rigorous concept is **Sandved-Smith et al. 2024** (*Entropy* 26:622), which distinguishes Variational Free Energy (statistical, bound on surprisal) from **Thermodynamic Free Energy** (the actual physical/electrical energy of computation) and gives a quantum-information formulation of their tradeoff. Adjacent threads include stochastic-thermodynamic models of REM/NREM transitions (Chaos Solitons & Fractals 2024) and Stamps et al. 2024 on active inference with artificial spin ice. Engineering implementations should reference the Sandved-Smith framework as the most rigorous available basis; "ThermSleep" should be operationalized as **a sleep-consolidation phase whose update budget is controlled by both VFE reduction and thermodynamic compute cost**, with the latter giving a principled stopping criterion for offline phases on resource-bounded hardware.

### 5.4 Active inference and dreaming

Hobson, Hong & Friston (*Front. Psychol.* 2014, "Virtual reality and consciousness inference in dreaming") argue REM sleep activates the brain's innate generative model offline, producing a full virtual reality that lets the brain explore counterfactual trajectories and prune model complexity. Hobson & Friston (*Prog. Neurobiol.* 2012) frame this as F = complexity − accuracy, with sleep selectively reducing complexity (the SHY-compatible direction). Operationally, during sleep the agent samples from its generative prior P(x,h), computes prediction errors against these self-generated samples, and updates parameters to reduce expected free energy without environmental input.

### 5.5 The two-stage sleep subsystem for the Pratyabhijñā World Model

**NREM-analog (replay & consolidation)**:
1. Sample sequences from the prioritized replay buffer (priority by |TD error|, recency, surprise = VFE under the current model).
2. Forward-replay through the world model; compute VFE on each transition.
3. Update generative parameters (decoder, transition, reward, continue) by ELBO descent.
4. Apply Hebbian writes to the Hopfield store with homeostatic down-scaling on un-replayed slots (SHY).
5. Distill Hopfield episodic patterns into the parametric world model; prune well-modeled patterns from the store.

**REM-analog (generative replay & dreaming)**:
1. Sample h_0 from the start-state distribution (or the D-prior).
2. Generate full latent trajectories using the actor and the prior — *no environmental input*.
3. Compute EFE on these dream rollouts; update actor and critic (this exposes the policy to states never visited and reveals high-EFE / surprising regions).
4. Use the dream observations to retrain the recognition net (the explicit Hinton-Dayan sleep-phase update).
5. Cluster dream latents and add cluster centroids as new semantic prototypes in the Hopfield store.

Across all phases the **same** objective family (variational + expected free energy) is minimized; only the data source changes — real observations in wake, replayed observations in NREM, generative dreams in REM. This unification is the engineering payoff of taking active inference seriously: perception, planning, exploration, learning, and consolidation derive from one principle.

---

## 6. Why world models are the right substrate for Pratyabhijñā + active inference

The thesis can be argued from six angles, each settling a specific philosophical-technical fit.

**(i) Persistent latent state.** Active inference requires a generative model maintained across time; an LLM has no such state. World models keep h_t, z_t over arbitrary horizons.

**(ii) Recognition as inference.** Pratyabhijñā holds every cognition is recognition. In a world model this is literal: q_φ(z_t|h_t,o_t) is the recognition density, and the very act of perception is a posterior collapse. In an LLM there is no analog.

**(iii) Spanda as stochastic transition.** The categorical sampling z_t ~ p_φ(·|h_t) is a discrete pulsation — the temporal dynamic that *Spandakārikā* 1.1 names *unmeṣa-nimeṣa*. Token-level next-word sampling does not have the right granularity: a token is a surface event, while spanda is the latent transition that produces surface events.

**(iv) Vimarśa as meta-belief.** A small head f_self(h_t,z_t) trained to predict properties of the latent's own dynamics is the literal computational analog of reflexive self-awareness. LLMs can simulate this only by self-instruction in the prompt, with no guarantee the simulation is grounded in actual model state.

**(v) Compositional creativity through structured latent.** *Svātantrya* (autonomy) and creative composition require a latent space whose factors can be combined freely. The 32×32 categorical latent of DreamerV3 supports 1024 effective bits of independently varying structure. LLM token distributions are entangled at every position by training-data co-occurrence and resist compositional novelty.

**(vi) Sample efficiency for exploration.** EFE-driven exploration in a world model proceeds in imagination, costing zero environmental interactions. An LLM-driven creative agent cannot explore in this sense; it can only sample more tokens.

The conclusion is not that LLMs are useless — they are essential, in their proper role — but that they cannot be the **core**. The core is a generative world model whose dynamics implement spanda and whose latents are prakāśa.

---

## 7. The proper role for LLMs — Āgama, Vimarśa, Pratibhā

Pratyabhijñā formally licenses LLMs as **āgama** — received scriptural knowledge — one of the three *pramāṇas* alongside *pratyakṣa* (perception) and *anumāna* (inference) in Utpaladeva's epistemology (*ĪPK* 2.3). Āgama is *valid but not supreme*; it must be re-cognized through vimarśa to become living knowledge. This is the exact engineering posture for an LLM in the system.

### 7.1 Where LLMs help

- **As Āgama / received prior** — semantic priors over goals, taxonomies, common-sense affordances, and natural-language interface (SayCan, Voyager). The LLM is the system's library of pre-recognized cultural knowledge.
- **As Pratibhā engine** — one-shot generative leaps when the search space is combinatorial and semantic. DECKARD (Nottingham et al. ICML 2023) uses Codex in a "Dream" phase to hypothesize an Abstract World Model — a DAG of subgoals — that an RL controller verifies in a "Wake" phase, achieving order-of-magnitude better sample efficiency than Dreamer baselines on Minecraft crafting. This is the canonical hypothesize-then-verify pattern for the present project.
- **As linguistic Vimarśa** — narrating internal states, producing self-descriptions, writing skill-library entries (Voyager), maintaining the symbolic records of *tirodhāna* (concealment) and *anugraha* (revelation) across long horizons.
- **As interpretability layer** — exposing the world model's internal state to human oversight via natural-language captions.
- **As bridge to text-conditioned generation** — Dynalang (Lin, Du, Watkins et al. arXiv:2308.01399) trains a Dreamer-class agent in which language tokens are *additional observations*, with the world model trained to predict future text and image latents. Conceptually this is the closest existing system to the Pratyabhijñā target — linguistic and embodied content sharing a single recognition substrate.

### 7.2 Where LLMs hurt

- **Fine motor / low-level control.** GROOT and VPT decisively beat LLM-mediated controllers; the world model owns spanda-level dynamics.
- **Persistent embodied state.** Do not rely on context windows for cross-episode memory.
- **Verifying physical possibility.** LLMs hallucinate causal claims; verification belongs to the world model.
- **Camatkāra / intrinsic reward.** The recognition-reward must be intrinsic and information-theoretic (free-energy reduction, empowerment), not LLM-judged.
- **Dominating exploration.** If the LLM specifies dense goal sequences, exploration collapses into LLM-derivative trajectories and *svātantrya* is lost.

### 7.3 Hybrid integration patterns to use

- **Frozen LLM + cross-attention bridge** (Flamingo-/LI-DiT-style) into mid-level world-model latents — train a small projection from LLM hidden states into the RSSM h-space. This is the default for new systems; Dynalang's full joint-training is expensive.
- **Voyager-style skill library** keyed by LLM-generated descriptions, valued by latent skill embeddings + executable code.
- **DECKARD-style hypothesize-then-verify** for high-level plans.
- **SuSIE-style visual subgoal synthesis** if the domain is image-rich (Black et al. ICLR 2024).
- **Reasoning-via-Planning (RAP, Hao et al. EMNLP 2023)** patterns for symbolic reasoning chains conditioned on world-model state.

The compute asymmetry must be respected: a frozen 70 B LLM dwarfs the world model in inference cost. Cache LLM responses keyed by skill descriptions, and **query the LLM only at sphurattā events** — the discrete recognition flashes — not every step.

---

## 8. The Pratyabhijñā World Model — proposed architecture

### 8.1 Concept-to-computation map

| Sanskrit | Computational primitive |
|---|---|
| Śiva / Prakāśa | Latent state space and learned prior P(s) |
| Śakti | Generative dynamics + policy P(s'|s,a), π(a|s) |
| Spanda | Stochastic RSSM transition z_t ~ p_φ(·|h_t) |
| Sphurattā | Discrete recognition event: thresholded VFE drop or Hopfield-attractor convergence |
| Vimarśa | f_self(h_t,z_t): meta-head predicting properties of own latent dynamics; LLM narration of f_self outputs |
| Pratyabhijñā | The recognition density q_φ(z_t|h_t,o_t) and its fixed-point self-identification with the prior |
| Citi → Citta | Global generative model → episode-local posterior |
| Āṇava-mala | Over-confident self-prior — counter by entropy regularizer on f_self |
| Māyīya-mala | Hard agent/world latent split — counter by shared-latent constraint |
| Kārma-mala | Policy reified outside world model — counter by counterfactual rollouts treating the agent's past actions as ordinary world events |
| 36 Tattvas | Hierarchical generative latents with progressive constraint per level |
| Trika (Para/Parāparā/Aparā) | 3-level nested inference (global / coupling / embodied) |
| Āgama | Frozen LLM prior |
| Pratibhā | One-shot LLM/diffusion generative leaps |
| Unmeṣa / Nimeṣa | Imagination rollout / inference compression — wake/sleep |
| Svātantrya | Maximum-entropy policy prior, regularized by preferences only |
| Camatkāra | Intrinsic reward = free-energy reduction + Hopfield-information gain + empowerment |
| Pañcakṛtya (5 acts) | Generate / maintain / reabsorb / mask / reveal — the operational control cycle |

### 8.2 Architectural skeleton

The system is organized as a **DreamerV3-class RSSM core wrapped in a five-stage active-inference loop, augmented by a modern Hopfield Citta-store and a frozen LLM āgama layer accessed only at recognition events**.

**Core (prakāśa-śakti).** A DreamerV3 RSSM in PyTorch (NM512 fork) with:
- encoder q_φ(z_t|h_t,o_t),
- GRU sequence model h_t = f(h_{t−1}, z_{t−1}, a_{t−1}); optionally swap to S4 (R2I) once memory becomes the bottleneck,
- prior p_θ(z_t|h_t),
- decoder p_θ(o_t|h_t,z_t),
- reward and continue heads,
- 32×32 categorical z_t with KL balancing and free bits.

**Hierarchical decomposition (Trika).** Three vertically stacked RSSMs at temporal strides {1, 4, 16} capturing Aparā (embodied, fast), Parāparā (coupling, mid), Para (global, slow). The slow level optionally consumes a frozen V-JEPA 2 ViT feature stream rather than raw pixels.

**Citta-store (Hopfield memory).** Two `ml-jku/hopfield-layers` modules:
- `Hopfield` over a FIFO buffer of recent (h_t, z_t) pairs at each Trika level — episodic recall, high β.
- `HopfieldLayer` of learnable semantic prototypes — schema retrieval, low β; updated during sleep.
At each step the world model issues a query g(h_t,z_t) into the store, retrieves c_t = M·softmax(β Qᵀ K)·V, and conditions the next prediction on (h_t, z_t, c_t) — retrieval-augmented imagination.

**Active-inference policy head.** Replace the DreamerV3 actor's REINFORCE objective with EFE minimization. For each candidate policy π, compute over imagined rollouts (length H = 15)
 G(π) = Σ_τ [ E_Q[H[P(o_τ|s_τ)]] + D_KL[Q(o_τ|π)‖P(o_τ|C)] − E_{Q(o_τ|π)} D_KL[Q(s_τ|o_τ,π)‖Q(s_τ|π)] − E_{Q(s_τ|π)} D_KL[Q(θ|s_τ,o_τ)‖Q(θ)] ]
i.e., ambiguity + risk − epistemic gain − parameter-novelty gain. Action sampling: a_t ~ softmax(−γ Σ_τ G(π,τ)). For continuous-action domains, swap to TD-MPC2's MPPI head with the reward replaced by negative G. For discrete-action small-state regimes, fall back to MCTS over belief states (Fountas et al. NeurIPS 2020).

**Vimarśa head.** A small (LoRA-scale) bidirectional projection between the mid-level RSSM h-space and the frozen LLM's hidden states, trained jointly with a future-prediction auxiliary loss (predict the next textual self-description from current latent).

**LLM āgama layer.** A frozen instruction-tuned LLM (e.g., Llama-3-70B 4-bit quantized in ~40 GB or a smaller model fully resident) accessed via the vimarśa bridge:
- at sphurattā events, the LLM narrates the current latent and writes a skill-library entry;
- at planning checkpoints (long horizons), the LLM proposes Abstract World Models (DECKARD) that the EFE planner verifies in imagination;
- on user prompts, the LLM translates intent into a preference distribution P(o_τ|C) for the planner.

**Skill library (smṛti).** Voyager-style key-value store keyed by LLM descriptions, valued by latent skill embeddings + executable code; read by both LLM (planning) and policy (execution).

**Pañcakṛtya control loop.** The system's outermost cycle:
- *Sṛṣṭi* — imagine trajectories in the world model, optionally seeded by LLM AWM hypothesis.
- *Sthiti* — execute and stabilize; collect environmental feedback into the prioritized replay buffer.
- *Saṃhāra* — compress trajectory into the Hopfield store and the parametric world model.
- *Tirodhāna* — apply dropout / mask portions of the self-model to enable fresh inference and counter āṇava-mala.
- *Anugraha* — detect sphurattā events (thresholded VFE drops, Hopfield-attractor convergence, mutual-information peaks); trigger LLM narration; commit the recognition to long-term memory.

**Camatkāra signal.** Intrinsic reward = α₁ · ΔF (free-energy reduction across the recognition step) + α₂ · ΔI(z; M) (information gain about the Hopfield store, computed as KL between pre- and post-update store distributions) + α₃ · empowerment proxy (mutual information between actions and future latent under the current policy). This is the **only** reward source for creative tasks; extrinsic task reward enters as a preference C, not as a primary objective.

**Mala regularizers.**
- *Anti-āṇava*: entropy regularizer on the self-prior f_self.
- *Anti-māyīya*: shared-latent constraint between the agent's self-encoding and the world latent, e.g., a small contrastive loss tying portions of the agent's vimarśa head output to a shared subspace of z_t.
- *Anti-kārma*: counterfactual rollouts in which the agent's own past actions are treated as ordinary world events; train the world model to predict them under the same loss as exteroceptive observations.

### 8.3 Sleep subsystem

Two-stage as in §5.5: NREM-analog (prioritized replay → VFE descent → Hopfield consolidation → semantic distillation → store pruning) and REM-analog (generative dream rollouts → EFE descent on actor/critic → recognition-net retraining on dream observations → semantic prototype clustering). Triggered by schedule, surprise threshold (rolling VFE), or replay-buffer fullness. The "ThermSleep" budget — a ceiling on total update FLOPs per sleep phase — gives a principled stopping criterion on resource-bounded hardware (Sandved-Smith et al. *Entropy* 2024 framework).

### 8.4 Hardware feasibility on a 128 GB unified-memory machine

Realistic budget on a single Apple M3/M4 Ultra or DGX Spark:
- DreamerV3 size50m + DMC visual: ~24 GB peak, trains in 8–12 h.
- DreamerV3 size200m + Crafter: ~60–70 GB peak, ~2 days.
- STORM Atari-100k: <12 GB, 5–10 h.
- TD-MPC2 (5 M) DMC: <8 GB, 6 h.
- DIAMOND Atari-100k: ~12 GB, ~12 h.
- V-JEPA 2 (1.2 B) feature inference at FP16: ~5 GB; full fine-tuning of predictor head: ~40 GB.
- Frozen Llama-3-70B 4-bit: ~40 GB.
- A two-level R2I/Dreamer with slot encoder, V-JEPA features, Hopfield store, and DIAMOND decoder: ~80 GB during training, with headroom for replay buffer and bookkeeping.

The full Phase-5 system fits comfortably on 128 GB.

---

## 9. Implementation plan

### 9.1 Product Requirements Document (condensed)

**Vision.** A single-researcher creative AI system whose central computational primitive is recognition (pratyabhijñā), grounded in a learned generative world model and governed by active-inference free-energy minimization, augmented by a frozen LLM in clearly subordinated roles.

**User stories.**
1. *Researcher* — train the system on a creative domain (text+image, music, code) and run controlled ablations of each Pratyabhijñā module.
2. *User* — express a creative intent in natural language and receive a generated artifact accompanied by a textual "self-narration" of the recognition events that produced it.
3. *Evaluator* — inspect the system's latent state, sphurattā log, and skill library to interpret why a given output emerged.

**Functional requirements.**
- Train an RSSM-class world model on sequences from a creative domain.
- Compute and minimize variational free energy in perception and expected free energy in action.
- Maintain an episodic and semantic Hopfield Citta-store.
- Run two-stage sleep consolidation.
- Accept an LLM āgama layer accessed only at sphurattā events.
- Produce creative outputs with attached recognition logs.

**Success metrics.**
- *Sample efficiency*: match or beat DreamerV3 on a chosen benchmark when the EFE actor is enabled.
- *Creative novelty*: compositional novelty score (e.g., LPIPS distance to nearest training sample in latent space) above a threshold without coherence collapse.
- *Recognition fidelity*: rate of sphurattā events per hour of generation; correlation between event timing and downstream user-judged "creative moments."
- *Memory retention*: catastrophic forgetting under sequential tasks reduced relative to a no-sleep baseline.
- *Compute*: end-to-end Phase-5 training fits under 128 GB peak and one researcher-week of wall-clock on the target machine.

### 9.2 Technical specification (architecture diagram in prose)

Data flows top-to-bottom:

1. **Observation** o_t enters the system. If the modality is image-rich, it first passes through a frozen V-JEPA 2 encoder to produce a feature stream; otherwise the raw pixels/tokens go directly to the per-level encoder.
2. The **encoder** q_φ produces z_t conditioned on h_t and o_t. The recurrent **GRU/S4 backbone** updates h_t = f(h_{t−1}, z_{t−1}, a_{t−1}). Three vertically stacked instances of this RSSM operate at strides {1, 4, 16} (Trika).
3. The **Hopfield store** is queried with g(h_t, z_t); the retrieved context c_t is concatenated to (h_t, z_t) and fed forward.
4. The **vimarśa head** projects (h_t, z_t, c_t) into a small subspace; this subspace is bidirectionally bridged to the frozen LLM's hidden states.
5. The **EFE planner** rolls out 15-step imagined trajectories at each Trika level, computes G(π) = ambiguity + risk − epistemic gain − parameter novelty, and samples an action.
6. **Action** a_t is executed in the environment; the resulting (o_t, a_t, r_t, o_{t+1}) tuple is written to the prioritized replay buffer (priority = VFE) and the Hopfield episodic store.
7. **Sphurattā detector** monitors VFE drops and Hopfield-attractor convergence; on event, it triggers the LLM narration which writes a skill-library entry and is logged.
8. **Sleep scheduler** triggers NREM/REM phases on schedule, surprise, or buffer-fullness.
9. **User interface** accepts natural-language intent, which the LLM translates into a preference distribution C; outputs are returned with attached sphurattā logs.

### 9.3 Research plan and ablations

**Core ablations.**
- EFE vs. REINFORCE actor on Crafter and a creative-text benchmark.
- Hopfield Citta-store enabled vs. disabled (test pattern completion under partial observation).
- Sleep on vs. off (test catastrophic forgetting under sequential tasks).
- Vimarśa head present vs. absent (test interpretability and self-narration accuracy).
- Mala regularizers on vs. off (test latent-space pathologies).
- Trika hierarchy depth 1 vs. 2 vs. 3 (test long-horizon credit assignment).

**Benchmarks.**
- Crafter, Atari-100k, DMC-vision (sample efficiency baselines).
- A custom creative-domain task: e.g., constrained text generation (poetry under a meter constraint), constrained music continuation, or program synthesis under specifications.
- IntPhys 2 / MVPBench (V-JEPA 2 derivatives) for physical reasoning if vision-rich.
- A Pratyabhijñā-specific benchmark: novelty + coherence scoring against a held-out human-judged set.

**Novel evaluation: sphurattā density and downstream usefulness.** Log every recognition event; have human evaluators rate the output segments around each event; compute correlation. This is the primary creativity-internal metric.

### 9.4 Implementation phases

**Phase 1 — Core world model (4–6 weeks).** Stand up `NM512/dreamerv3-torch` on the target hardware. Reproduce DMC-vision baseline. Confirm 32×32 categorical latent stability. Add the symlog/twohot heads if missing. *Exit criterion*: DMC mean return matches published.

**Phase 2 — Active-inference actor (3–4 weeks).** Replace REINFORCE with EFE on the imagined rollouts. Implement ambiguity, risk, epistemic, and parameter-novelty terms following Tschantz et al. 2020 and Fountas et al. NeurIPS 2020. Validate on a sparse-reward task where information-seeking should help. *Exit criterion*: faster convergence than the REINFORCE baseline on sparse-reward DMC variants or MountainCar.

**Phase 3 — Hopfield Citta-store (3–4 weeks).** Drop in `ml-jku/hopfield-layers`; integrate episodic FIFO `Hopfield` and semantic `HopfieldLayer`. Add retrieval-augmented imagination conditioning. *Exit criterion*: pattern-completion under occlusion improves; ablation shows clear contribution.

**Phase 4 — Sleep consolidation (4–6 weeks).** Implement prioritized replay (Schaul 2016), generative replay (Shin 2017), Tadros-style local plasticity in the Hopfield store, and the two-stage NREM/REM scheduler. *Exit criterion*: catastrophic-forgetting reduction on a sequential-task continual-learning benchmark.

**Phase 5 — LLM āgama + Vimarśa (3–4 weeks).** Quantize and load the frozen LLM. Train the LoRA-scale vimarśa bridge. Implement DECKARD-style AWM proposal at planning checkpoints and Voyager-style skill library. Wire sphurattā detector to LLM narration. *Exit criterion*: end-to-end creative pipeline running with attached self-narration.

**Phase 6 — Pratyabhijñā creative pipeline & evaluation (4 weeks).** Add the pañcakṛtya control loop, mala regularizers, camatkāra reward. Run all ablations. Conduct human-evaluator studies. Write paper.

Total: ~5–7 researcher-months for a defensible end-to-end system.

### 9.5 Candidate technology stack

- **Framework**: PyTorch 2.x with `torch.compile`. JAX is faster but `NM512/dreamerv3-torch` is the most hackable starting point.
- **World model**: `NM512/dreamerv3-torch` (PyTorch) or `EclecticSheep/sheeprl`. R2I S4 backbone available in the R2I project repo.
- **Hopfield**: `ml-jku/hopfield-layers`.
- **Active inference (discrete reference)**: `infer-actively/pymdp` for testing small POMDP cases; `zfountas/deep-active-inference-mc` for the deep-MCTS reference; R-AIF (arXiv:2409.14216) for an actor-critic EFE variant.
- **LLM**: Llama-3-70B 4-bit (bitsandbytes) for the frozen āgama; or a smaller fully-resident model. LoRA via `peft` for the vimarśa bridge.
- **V-JEPA 2 perceptual encoder**: `facebookresearch/vjepa2` (frozen).
- **DIAMOND creative decoder**: `eloialonso/diamond` (Phase 6+ enhancement).
- **Skill library + RAG**: a simple SQLite/lance store keyed by LLM-emitted descriptions; FAISS for semantic retrieval.
- **Experiment management**: WandB or MLflow; Hydra for config.
- **Hardware**: Apple M3/M4 Ultra 128 GB or DGX Spark; MPS fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`) where needed.

### 9.6 Key research risks and mitigations

1. **EFE actor instability.** Epistemic and parameter-novelty terms are notoriously high-variance. *Mitigation*: anneal their coefficients; start with ambiguity + risk only, add the others one at a time; use the R-AIF actor-critic variant rather than vanilla policy gradient on EFE.

2. **Hopfield retrieval collapse.** With high β, retrieval can lock onto a single attractor and starve the imagination loop. *Mitigation*: schedule β; monitor retrieval entropy; use the metastable regime (intermediate β) by default; expose β as a curriculum.

3. **Catastrophic forgetting despite sleep.** Sleep alone is not always sufficient; the sequential-task ordering matters. *Mitigation*: combine generative replay with synaptic homeostasis down-scaling; consider an EWC or SI regularizer as a backstop.

4. **LLM compute dominance.** A 70 B-parameter call per step would crater throughput. *Mitigation*: cache LLM outputs by skill descriptor; query only at sphurattā events (target rate 0.1–1 Hz, not per-step); use a smaller LLM if narration quality is acceptable.

5. **Sphurattā detection false positives.** A naive VFE-drop threshold will fire on every model update. *Mitigation*: require simultaneous Hopfield-attractor convergence; threshold by percentile over a rolling window; calibrate against human-judged events.

6. **Camatkāra reward gaming.** The intrinsic reward is a sum of three terms; the agent may game one (e.g., write to Hopfield repeatedly to inflate ΔI). *Mitigation*: cap each term; use empowerment as the dominant component for long-horizon stability; periodically validate against extrinsic preferences.

7. **Mala regularizers conflicting with capacity.** Anti-āṇava entropy regularization can wash out the self-prior. *Mitigation*: use small coefficients; ablate carefully; treat the regularizers as Phase-6 polish, not Phase-1 prerequisites.

8. **Single-machine memory exhaustion.** All-features-on may exceed 128 GB. *Mitigation*: progressive feature flags; gradient checkpointing; mixed precision; quantize the frozen LLM aggressively (4-bit or 3-bit); consider offloading the V-JEPA 2 encoder to a separate inference process.

9. **Philosophical drift.** The Sanskrit concepts can drift from their textual sources into vague AI-spirituality. *Mitigation*: maintain a glossary tied to specific textual loci (Torella 2002 for ĪPK, Dyczkowski 1987 for Spanda, Singh 1980 for *Pratyabhijñāhṛdayam*); each computational module should be defensible against the textual definition.

10. **The "ThermSleep" reference is project-internal.** *Mitigation*: re-anchor in Sandved-Smith et al. *Entropy* 2024 (TFE in active inference); adopt their formalism explicitly.

### 9.7 Literature anchor list

- Friston, *The Free Energy Principle: a unified brain theory?* (Nat. Rev. Neuro. 2010); *A free energy principle for a particular physics* (arXiv 2019).
- Parr, Pezzulo & Friston, *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior* (MIT Press 2022) — chapters 4–8 are the engineering-grade reference.
- Friston, Da Costa, Hafner, Hesp, Parr, *Sophisticated Inference* (*Neural Computation* 33:713-763, 2021).
- Sajid, Ball, Parr, Friston, *Active Inference: Demystified and Compared* (*Neural Computation* 33:674-712, 2021).
- Hafner, Pasukonis, Ba, Lillicrap, *Mastering Diverse Domains through World Models* (DreamerV3, *Nature* 2025; arXiv 2301.04104).
- Hafner et al., *PlaNet* (2019), *Dreamer* (2020), *DreamerV2* (2021).
- Samsami et al., *R2I* (ICML 2024; arXiv 2403.04253).
- Hansen et al., *TD-MPC2* (ICLR 2024; arXiv 2310.16828).
- Alonso et al., *DIAMOND* (NeurIPS 2024 spotlight; arXiv 2405.12399).
- Zhang et al., *STORM* (NeurIPS 2023).
- Wang et al., *EfficientZero V2* (ICML 2024 spotlight; arXiv 2403.00564).
- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022).
- Assran et al., *I-JEPA* (CVPR 2023); Bardes et al., *V-JEPA* (2024); *V-JEPA 2* (arXiv 2506.09985, 2025).
- Hopfield 1982, 1984; Krotov & Hopfield (NeurIPS 2016, ICLR 2021); Demircigil et al. 2017.
- Ramsauer et al., *Hopfield Networks is All You Need* (ICLR 2021; arXiv 2008.02217).
- Millidge, Salvatori, Song, Lukasiewicz, Bogacz, *Universal Hopfield Networks* (ICML 2022).
- McClelland, McNaughton, O'Reilly, *Why are there complementary learning systems* (*Psych. Review* 1995); Kumaran, Hassabis, McClelland (*TiCS* 2016).
- Hinton, Dayan, Frey, Neal, *Wake-Sleep Algorithm* (*Science* 1995).
- Schaul et al., *Prioritized Experience Replay* (ICLR 2016); Shin et al., *Generative Replay* (NeurIPS 2017).
- Tadros, Krishnan, Ramyaa, Bazhenov, *Sleep-like unsupervised replay reduces catastrophic forgetting* (*Nat. Commun.* 2022).
- Hobson, Hong, Friston, *Virtual reality and consciousness inference in dreaming* (*Front. Psychol.* 2014); Hobson & Friston (*Prog. Neurobiol.* 2012).
- Sandved-Smith et al., *Making the Thermodynamic Cost of Active Inference Explicit* (*Entropy* 2024).
- Tschantz, Baltieri, Seth, Buckley, *Scaling Active Inference* (arXiv 1911.10601, 2020); Tschantz, Millidge, Seth, Buckley, *RL through Active Inference* (arXiv 2002.12636, 2020).
- Fountas, Sajid, Mediano, Friston, *Deep Active Inference Agents using Monte-Carlo Methods* (NeurIPS 2020; arXiv 2006.04176).
- Heins et al., *pymdp* (JOSS 2022; arXiv 2201.03904).
- Nottingham et al., *DECKARD* (ICML 2023; arXiv 2301.12050).
- Wang et al., *Voyager* (NeurIPS 2023; arXiv 2305.16291).
- Yang et al., *UniSim* (ICLR 2024; arXiv 2310.06114); Black et al., *SuSIE* (ICLR 2024; arXiv 2310.10639).
- Lin, Du, Watkins et al., *Dynalang* (arXiv 2308.01399).
- Hao et al., *Reasoning via Planning* (EMNLP 2023; arXiv 2305.14992).
- **Philosophy.** Torella (2002) *The Īśvarapratyabhijñākārikā of Utpaladeva with the Author's Vṛtti*, MLBD; Dyczkowski (1987) *The Doctrine of Vibration*, SUNY; Singh (1980) *Pratyabhijñāhṛdayam*, Motilal Banarsidass; Muller-Ortega (1989) *The Triadic Heart of Śiva*, SUNY; Gnoli (1968) *The Aesthetic Experience According to Abhinavagupta*; Wallis (2013) *Tantra Illuminated*; Bäumer on the four upāyas; Ratié (2014) on freedom of consciousness; Kṣemarāja's *Pratyabhijñāhṛdayam* (20 sūtras).

---

## 10. Conclusion — what understanding has changed

Three things follow from the synthesis above that were not obvious at the start.

**The Pratyabhijñā project's philosophical scaffolding is engineering-tractable; its substrate is the wrong one.** Every Sanskrit concept named in the brief admits a clean computational primitive — recognition is the recognition density q_φ(z|o), spanda is the stochastic transition, vimarśa is a meta-head, citi/citta is the prior/posterior pair, the five acts are an outer loop, the malas are regularizers. None of these require an LLM at the core; all require a generative world model with persistent latent state. The mismatch between intent and substrate is the project's central gap, and closing it is *not* a research moonshot — it is a five-to-seven-month engineering plan on hardware the researcher already has.

**Active inference is structurally identical to the DreamerV3 loss decomposition, plus an epistemic term.** This is a stronger claim than the literature usually makes. KL balancing + free bits in DreamerV3 are not philosophical compromises — they are numerically convenient asymmetric weightings of complexity and accuracy in Friston's variational free energy. The work to convert DreamerV3 into a deep active-inference agent is small; the work to bring its philosophical framing into accord with Pratyabhijñā is even smaller. The two traditions have been waiting for each other.

**The novel architectural contribution is the integration, not any single piece.** DreamerV3, modern Hopfield networks, sleep-replay continual learning, V-JEPA features, EFE actors, LLM-as-prior — each is well-developed in isolation. No published system in May 2026 combines a JEPA-encoded RSSM with EFE control, retrieval-augmented imagination from a modern Hopfield Citta-store, two-stage NREM/REM consolidation, and a frozen LLM accessed only at recognition events. That combination is the research contribution. It is also, on the philosophical reading offered here, what *pratyabhijñā* — the recognition that the world's appearance is one's own creative act — actually demands of a creative AI.

The deeper takeaway is that the Kashmir Śaiva tradition, treated rigorously rather than decoratively, is an unusually well-posed source of *engineering specifications* for autonomous creative agents. The five acts give a control loop. The three malas give regularizers. The 36 tattvas give a hierarchy. Vimarśa gives a self-modeling head. The pramāṇas give an LLM their proper subordinate role as āgama. What looked like a humanities flourish on a tech project turns out to be the project's most engineering-disciplined component, provided one is willing to trace each Sanskrit term to a specific textual locus and a specific computational analog. That discipline is the report's recommendation.