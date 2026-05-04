# PWM Prior Works Integration
## context-engineering-harness · pratyaksha-context-eng-harness · pramāṇa

*Version 0.1 — May 2026*

---

## 1. Overview of Prior Works

Three prior repositories by the same author form a coherent prior-art foundation for PWM:

| Repo | Short Name | Description |
|---|---|---|
| `SharathSPhD/context-engineering-harness` | CEH | Avacchedaka notation, buddhi/manas agents, khyātivāda hallucination classifier, H1–H7 experiments. The *experiment harness* that surfaced PCE v0.4 numbers. |
| `SharathSPhD/pratyaksha-context-eng-harness` | PCEH | Long-context discipline for Claude Code: Avacchedaka-typed retrieval, sublation, witness invariants, event-boundary compaction, Khyātivāda taxonomy. 15 MCP tools, 3 skills, 3 agents, 4 commands, 3 hooks. The *context infrastructure* plugin. |
| `SharathSPhD/pramana` | Pramāṇa | Fine-tuning paper. Logical system of pramāṇa (perception/inference/testimony/analogy) as a training objective. Docker-compose pipeline. The *epistemological fine-tuning* framework. |

These are not merely historical artifacts. They are the *engineering substrate* from which PWM is built up. This document specifies how each component maps into the new multi-agent / world model architecture.

---

## 2. Brainstorm: Integration Philosophy

The prior works form a three-layer stack that maps onto PWM's architecture:

```
┌─────────────────────────────────────────────────────┐
│  PRAMĀṆA (Phase 4+)                                 │
│  Epistemological fine-tuning of Nemotron 120B A12B  │
│  Teaches: perceive / infer / know / recognise        │
└─────────────────────────────────────────────────────┘
        ▲
        │ weights specialisation
┌─────────────────────────────────────────────────────┐
│  PRATYĀKṢA HARNESS (PCEH)                           │
│  Context infrastructure substrate for all 9 agents  │
│  Tools: Avacchedaka store, sublation, witness (Sākṣī)│
│  compaction, khyātivāda classifier, budget tracking  │
└─────────────────────────────────────────────────────┘
        ▲
        │ context operations
┌─────────────────────────────────────────────────────┐
│  CONTEXT ENGINEERING HARNESS (CEH)                  │
│  Experiment scaffolding, agent prompt templates,     │
│  H1–H9 hypothesis tracking, manas/buddhi split      │
└─────────────────────────────────────────────────────┘
        ▲
        │ experiment design
┌─────────────────────────────────────────────────────┐
│  PWM CORE                                           │
│  Trika World Model + Pañcakṛtya śakti agent pipeline│
└─────────────────────────────────────────────────────┘
```

The key insight: PCEH is not "one component among many" — it is the **operating system** for inter-agent communication in PWM. Every agent writes to and reads from the Avacchedaka store. CEH provides the **experiment methodology** and prompt templates. Pramāṇa provides the **weight-level epistemological alignment** (future).

---

## 3. context-engineering-harness (CEH) → PWM Wiring

### 3.1 Avacchedaka Notation → Inter-Agent Message Schema

**CEH origin**: The avacchedaka (delimiter/qualifier) notation was developed in CEH to give LLM context windows typed, provenance-tracked structure. Each piece of knowledge is tagged with its *qualificand* — the epistemic category it belongs to.

**PWM wiring**: In the multi-agent pipeline, every agent's output is written to the Avacchedaka store under its own qualificand. This gives all downstream agents a typed, auditable, structured view of what each upstream agent contributed — rather than a flat string concatenation.

```python
# CEH avacchedaka pattern → PWM inter-agent schema
# Each agent writes:
store.context_insert(
    qualificand="jnana",          # epistemic category: knowledge
    key="belief_update",
    value={"ΔF": -0.23, "source": "Hopfield_recall"},
    precision=0.85,               # epistemic confidence
)
# Downstream agents read:
jnana_belief = store.context_retrieve(qualificand="jnana", key="belief_update")
```

The avacchedaka schema for PWM defines 9 qualificands (one per agent) plus cross-cutting ones:

| Qualificand | Agent | Content Type |
|---|---|---|
| `cit` | cit-agent | Perceptual encoding: h_t, z_t, obs_embedding |
| `ananda` | ananda-agent | Camatkāra score, sphurattā flag, EFE ambiguity |
| `icha` | icha-agent | K candidate intentions with EFE scores |
| `apohana` | WM (no LLM) | Pruned candidate set |
| `jnana` | jnana-agent | Recalled memories, BMR ΔF, sublated context |
| `kriya` | kriya-agent | Draft creative output |
| `vimarsha` | vimarsha-agent | Khyātivāda judgment, narrative, commitment decision |
| `memory` | memory-agent | Compaction report, Hopfield update stats |
| `sleep` | sleep-agent | NREM/REM stats, sleep narrative |
| `sakshi` | sākṣī-keeper | Witness prefix (cross-cutting, all agents receive it) |

### 3.2 Buddhi/Manas Agents → Jñāna/Icchā Fast-Slow Split

**CEH origin**: The manas (sense-mind, fast/intuitive) and buddhi (discriminative intellect, slow/deliberate) agent split implements the System 1 / System 2 cognitive architecture within an LLM harness.

**PWM wiring**: The fast-slow split is preserved but *repositioned* in the WM context:

| CEH Agent | Function | PWM Mapping |
|---|---|---|
| **manas** | Fast, intuitive, first drafts. Sets `needs_buddhi: true` when uncertain. | → **icchā-agent** (generates K candidate intentions quickly; flags candidates that need jñāna verification) |
| **buddhi** | Slow, deliberate verifier. Re-fetches evidence, sublates contradicted claims, emits citations. | → **jñāna-agent** (retrieves Hopfield memories, computes BMR ΔF, sublates contradicted context via Pratyākṣa `sublate_with_evidence`) |

The `needs_buddhi: true` flag from CEH becomes the **EFE epistemic term** in PWM: high epistemic uncertainty in icchā's candidates triggers the jñāna agent's full verification pathway rather than a fast passthrough.

```python
# CEH pattern: manas sets needs_buddhi flag
# PWM equivalent: icchā sets needs_jnana flag based on EFE epistemic term
class IchaAgent(PWMBaseAgent):
    def sample_intentions_tool(self):
        candidates = self.wm.efe_actor.sample_intentions(h_t, K=self.K)
        for c in candidates:
            c.needs_jnana = (c.efe_epistemic_term > EPISTEMIC_THRESHOLD)
        self.emit("candidates", [c.to_dict() for c in candidates])
```

### 3.3 Khyātivāda Classifier → Vimarśa Quality Gate

**CEH origin**: The khyātivāda hallucination classifier (from Mīmāṃsā/Nyāya epistemology) classifies LLM outputs into valid cognitions (pratyakṣa, anumāna, āgama) vs erroneous ones (anyathākhyāti — misrepresentation, akhyāti — non-apprehension, viparītakhyāti — inverted perception).

**PWM wiring**: This is the vimarśa gate's primary epistemic filter. After kriyā generates a draft output, vimarśa-agent calls `classify_khyativada` (via the PCEH MCP tool). The classification determines whether to commit or revise:

```python
# vimarsha_agent.py
def classify_output_tool(self, draft: str) -> dict:
    result = self.mcp.classify_khyativada(text=draft)
    # Valid cognitions → commit
    # anyathākhyāti (misrepresentation) → revise with evidence correction
    # akhyāti (non-apprehension) → trigger jñāna re-retrieval
    # viparītakhyāti (inversion) → restart cascade from icchā
    commit_map = {
        "pratyaksha": "commit",
        "anumana": "commit",
        "agama": "commit",
        "anyathakhyati": "revise",
        "akhyati": "rejnana",    # re-run jñāna
        "viparitakhyati": "reicha",  # re-run from icchā
    }
    action = commit_map.get(result["type"], "revise")
    return {"action": action, "type": result["type"], "confidence": result["confidence"]}
```

The H9 crisis (ρ=0.0, proxy-judge disagreement) from PCE v0.4 is directly addressed by this integration: khyātivāda replaces both the proxy scorer (which measured surface-level quality) and the LLM judge (which measured arbitrary preference). The khyātivāda classification is *epistemologically grounded* — it asks not "is this good?" but "is this a valid form of cognition?"

### 3.4 H1–H7 Experiments → H1–H9 Continuation

The CEH H1–H7 experiment infrastructure (config.toml, pyproject.toml, reproducible runs) is directly re-used in PWM for H1–H9 and the six WM ablations. The experiment runner structure from CEH (hypothesis → config → run → metrics → report) is preserved; only the model backend changes from pure-LLM PCE to WM+āgama PWM.

```toml
# configs/hypotheses.toml (extended from CEH pattern)
[H8a]
description = "Vimarśa revision improves creative quality (g=0.65 from PCE v0.4)"
test = "vimarsha_revision_effect"
metric = "camatkaara_reward_delta"
target = "g > 0.5"

[H9]
description = "Camatkāra reward resolves proxy-judge disagreement (ρ=0.0 crisis)"
test = "camatkaara_proxy_correlation"
metric = "spearman_rho"
target = "rho > 0.5"

[H10]
description = "WM imagination rollouts increase creative EFE over LLM baseline"
test = "wm_vs_llm_efe"
metric = "efe_mean_ratio"
target = "wm/llm > 1.5"
```

---

## 4. pratyaksha-context-eng-harness (PCEH) → PWM Wiring

### 4.1 PCEH as the Context Infrastructure Substrate

PCEH is not a single component that maps to one agent. It is the **context operating system** for all nine agents. Its 15 MCP tools, exposed under `mcp__pratyaksha_mcp__*`, are called throughout the cascade:

| PCEH Tool Family | Tools | PWM Usage |
|---|---|---|
| **Avacchedaka store** | `context_insert, context_retrieve, context_get, context_sublate, list_qualificands` | All agents use for inter-agent message passing |
| **Sublation** | `sublate_with_evidence, detect_conflict` | jñāna-agent uses for BMR-style belief update; vimarśa uses on contradicted commitments |
| **Compaction** | `compact, boundary_compact, context_window` | memory-agent uses before sleep cycles (tirodhāna gate) |
| **Witness (Sākṣī)** | `set_sakshi, get_sakshi` | sākṣī-keeper uses to maintain the ≤500-token vimarśa witness |
| **Hallucination** | `classify_khyativada` | vimarśa-agent quality gate |
| **Budget** | `budget_status, budget_record` | monitoring — tracks token expenditure per cascade step |

### 4.2 PCEH Agents → PWM Agent Roles

| PCEH Agent | Role | PWM Mapping |
|---|---|---|
| **manas** | Fast first-draft subagent; sets `needs_buddhi` | → icchā-agent (fast path, 49B model) |
| **buddhi** | Slow verifier; sublates contradictions; emits citations | → jñāna-agent (slow path, with Hopfield retrieval + BMR) |
| **sākṣī-keeper** | Maintains witness invariant ≤500 tokens | → sākṣī-keeper background process (cross-cutting, unchanged) |

### 4.3 PCEH Skills → PWM Agent Prompts

| PCEH Skill | Content | PWM Usage |
|---|---|---|
| `context-discipline` | When and how to use typed insertion, sublation, boundary compaction | Injected into sākṣī witness as standing instruction for all agents |
| `sublate-on-conflict` | Bādha decision procedure: provenance, precision, timestamps | jñāna-agent's sublation logic |
| `witness-prefix` | Sākṣī authoring rules: ≤500 tokens, stable, no reasoning content | sākṣī-keeper invariant |

### 4.4 PCEH Hooks → PWM Lifecycle Events

| PCEH Hook | Trigger | PWM Mapping |
|---|---|---|
| `SessionStart` | Bootstrap Sākṣī | PWM: initialise sākṣī-keeper at session start; load WM checkpoint |
| `PreToolUse (mcp__pratyaksha_mcp__*)` | Warn at ≥90% budget | PWM: trigger memory compaction at 75% budget; trigger sleep cycle at 90% |
| `Stop` | Compact-now nudge at ≥75% budget | PWM: memory-agent runs boundary compaction; WM saves checkpoint |

### 4.5 Installation in PWM Repository

PCEH installs as a Claude Code plugin:

```bash
# In the PWM project directory
/plugin marketplace add SharathSPhD/pratyaksha-context-eng-harness
/plugin install pratyaksha-context-eng-harness@pratyaksha-context-eng-harness
```

For the DGX Spark (potentially air-gapped):
```bash
git clone https://github.com/SharathSPhD/pratyaksha-context-eng-harness.git \
  ~/.claude/plugins/pratyaksha-context-eng-harness
/plugin marketplace add ~/.claude/plugins/pratyaksha-context-eng-harness
/plugin install pratyaksha-context-eng-harness@pratyaksha-context-eng-harness
```

The MCP server is PEP 723 self-installing — `uv` provisions `mcp`, `pydantic`, `tiktoken` on first call. No additional pip installs required.

---

## 5. pramāṇa (Fine-Tuning) → PWM Phased Plan

### 5.1 Pramāṇa Epistemological Framework

The pramāṇa framework formalises four valid sources of knowledge (Nyāya/Mīmāṃsā):

| Pramāṇa | English | WM Analog | LLM Role |
|---|---|---|---|
| **Pratyakṣa** | Direct perception | WM observation model: o_t → z_t | cit-agent: encodes raw input |
| **Anumāna** | Inference | WM dynamics: h_t → z_{t+1} | jñāna-agent: BMR belief update |
| **Āgama** | Testimony / reliable text | LLM knowledge base: encyclopedic priors | āgama layer: Nemotron 120B |
| **Upamāna** | Analogy / comparison | Hopfield semantic similarity | jñāna-agent: prototype recall |

This mapping justifies the two-tier WM+LLM architecture at the epistemological level: the WM handles pratyakṣa and anumāna (perception and inference over experience); the LLM handles āgama (testimony from the training corpus); analogical reasoning spans both via Hopfield upamāna.

### 5.2 Fine-Tuning is NOT Sanskrit-Specific

The user explicitly clarified: creativity is language-agnostic; pramāṇa fine-tuning is about *epistemological discipline*, not *Sanskrit fluency*. The fine-tune teaches the model to:

1. Correctly identify which of its outputs are perception-grounded (pratyakṣa) vs inferred (anumāna) vs recalled (āgama)
2. Apply appropriate confidence calibration per pramāṇa type
3. Resist anyathākhyāti (misrepresentation) — the primary hallucination mode in creative generation

The training data is **not** Sanskrit texts but the PWM creative output corpus annotated with pramāṇa-type labels (collected during Phases 0–3).

### 5.3 Phased Fine-Tuning Plan

**Phase 0–3 (No fine-tuning; prompt engineering only)**:
- Use CEH's avacchedaka notation to structure prompts with explicit pramāṇa-type markers
- Every LLM call includes a witness prefix identifying which pramāṇa type the agent is performing
- Example witness prefix for jñāna-agent:
  ```
  You are operating as the jñāna-agent in a creative pipeline.
  Your role is ANUMĀNA (inference): derive meaning from recalled memories and observed patterns.
  Do not fabricate. Do not assert beyond your evidence.
  Source every claim with its retrieval qualificand.
  ```
- Collect (prompt, output, camatkaara_reward, khyativada_type) triples as a training corpus

**Phase 4 (Pramāṇa LoRA)**:
- Minimum corpus size: 5,000 committed creative outputs with pramāṇa annotations
- Fine-tune Nemotron 120B A12B with LoRA using `pramana` docker-compose
- Training objective: maximise log P(pramana_type | output) subject to camatkaara_reward > threshold
- LoRA configuration:
  ```yaml
  # pramana/configs/pwm_ft.yaml
  base_model: nemotron-120b-fp4
  lora_rank: 64
  lora_alpha: 128
  target_modules: [q_proj, v_proj, gate_proj, up_proj]
  learning_rate: 2e-4
  epochs: 3
  batch_size: 4
  gradient_accumulation: 8
  dataset: /data/pwm_pramana_triples.jsonl
  output_dir: /models/nemotron-120b-pramana-lora
  ```

**Phase 5+ (Merged weights)**:
- Merge LoRA adapter into base weights
- Rebuild TRT-LLM engine
- Continue joint WM + āgama training with merged weights
- The fine-tuned model becomes the permanent āgama layer

### 5.4 Pramāṇa docker-compose Integration

```bash
# Clone pramana repo into PWM workspace
git clone https://github.com/SharathSPhD/pramana.git /workspace/pramana

# Phase 4: Fine-tune (run after collecting ≥5K triples)
cd /workspace/pramana
docker-compose run finetune \
  --env BASE_MODEL=/models/nemotron-120b-fp4 \
  --env DATASET=/data/pwm_pramana_triples.jsonl \
  --env CONFIG=/workspace/pramana/configs/pwm_ft.yaml \
  --env OUTPUT=/models/nemotron-120b-pramana-lora
```

---

## 6. Brainstorm: Additional Integration Opportunities

Beyond the explicit port decisions above, the following synergies emerged from careful cross-reading of all three repos:

### 6.1 Budget Tracking as VFE Proxy

PCEH's `budget_status / budget_record` tools track token-budget expenditure per cascade step. In PWM this becomes a proxy measure for cognitive effort — the token budget consumed by the vimarśa narration step is correlated with VFE: high VFE (uncertain, surprising states) → more tokens needed to narrate → higher budget burn. This gives a second, non-parametric proxy for the WM's epistemic state without requiring gradient readouts.

```python
# A high-budget vimarsha step signals high uncertainty → trigger extra sleep replay
budget = pratyaksha_mcp.budget_status()
if budget["fraction_used"] > 0.5 and vfe > VFE_SURPRISE_THRESHOLD:
    sleep_scheduler.flag_high_priority_replay(current_episode)
```

### 6.2 Sublation (Bādha) as Mala Regulariser

The khyātivāda sublation mechanism (bādha — superior cognition overriding inferior cognition) maps to the **māyīya mala** regulariser in the Trika WM. The māyīya mala is the error of mistaking the WM's own latent projections for world structure. When jñāna-agent's sublation detects a contradiction between the WM's current belief and a retrieved memory, this triggers the anti-māyīya regulariser:

```python
# Sublation event → anti-mayiya gradient signal
def on_sublation_event(contradicted_key, evidence):
    # Penalise the WM representation that generated the contradicted belief
    loss_mayiya = cosine_similarity(wm.self_latent, wm.world_latent)
    # This is the same as the anti_mayiya() mala regulariser in Architecture Spec
    return loss_mayiya
```

### 6.3 Witness Invariant as Ātmavāda Ground

The Sākṣī (witness) invariant in PCEH — ≤500 tokens, stable, no reasoning content, pushed as real system field — is philosophically equivalent to the Pratyabhijñā concept of **svaprakāśa** (self-luminosity): the self-revealing ground of awareness that is always already present, never derived, never argued for. Every LLM call in the pipeline fires with this constant ground, preventing the "amnesia" that causes multi-agent pipelines to lose coherence across turns.

This is not a coincidence of naming — it is a deliberate architectural principle. The sākṣī witness is the **ātmavāda ground** for the entire multi-agent system.

### 6.4 Event-Boundary Compaction as Tirodhāna

PCEH's event-boundary compaction (`boundary_compact`) triggers when the context window crosses a complexity threshold. In the Pratyabhijñā pañcakṛtya, tirodhāna (concealment/withdrawal) is the act by which Śiva veils completed experience to make room for new creation. The `boundary_compact` call at the end of each committed creative cycle is the technical implementation of tirodhāna: the completed creative act is compressed into a compact semantic summary (stored in CittaStore), and the detailed trace is released.

---

## 7. Integration Summary Table

| Prior Work Component | Source Repo | PWM Target | Phase |
|---|---|---|---|
| Avacchedaka notation | CEH + PCEH | Inter-agent message schema (all agents) | 0 |
| Khyātivāda classifier | CEH + PCEH | Vimarśa quality gate | 0 |
| Manas agent → icchā | CEH + PCEH | Icchā-agent (fast K candidates) | 0 |
| Buddhi agent → jñāna | CEH + PCEH | Jñāna-agent (slow BMR verification) | 0 |
| Sākṣī-keeper | PCEH | Cross-cutting witness invariant | 0 |
| Context-discipline skill | PCEH | Sākṣī witness instruction | 0 |
| Sublation tools | PCEH | Jñāna BMR + māyīya mala signal | 0 |
| Boundary compaction | PCEH | Tirodhāna / memory-agent | 0 |
| Budget tracking | PCEH | VFE proxy + sleep priority signal | 0 |
| H1–H7 experiment runner | CEH | Extended to H1–H9 + WM ablations | 0 |
| Pramāṇa epistemology | Pramāṇa | LLM system prompt structure | 0–3 |
| Pramāṇa fine-tuning | Pramāṇa | Nemotron 120B A12B LoRA specialisation | 4+ |
| docker-compose FT pipeline | Pramāṇa | Phase 4 fine-tune execution | 4+ |

---

## 8. Repository Structure (Updated)

```
/workspace/
├── pratyabhijna/              ← core PCE repo (v0.4 baseline)
├── context-engineering-harness/  ← CEH (experiment methodology)
├── pratyaksha-context-eng-harness/  ← PCEH (context infrastructure plugin)
├── pramana/                   ← pramāṇa fine-tuning pipeline
└── PWM/                       ← THIS PROJECT
    ├── pwm/
    │   ├── wm/                ← Trika World Model (PyTorch)
    │   ├── agents/            ← 9 śakti agents (smolagents)
    │   ├── context/           ← PCEH client wrapper
    │   ├── inference/         ← TRT-LLM / vLLM serving
    │   └── experiments/       ← H1–H9 + ablations
    ├── configs/
    │   ├── multiagent.yaml
    │   ├── inference_routing.yaml
    │   └── hypotheses.toml
    ├── data/
    │   └── pwm_pramana_triples.jsonl  ← collected Phase 0–3
    ├── models/
    │   ├── nemotron-120b-fp4/
    │   └── nemotron-120b-pramana-lora/  ← Phase 4+
    └── docs/
        ├── PWM_Master_Research.md
        ├── PWM_Architecture_Spec.md
        ├── PWM_PRD_Plan.md
        ├── PWM_Local_Models_Inference.md
        ├── PWM_MultiAgent_Architecture.md
        └── PWM_PriorWorks_Integration.md   ← THIS FILE
```
