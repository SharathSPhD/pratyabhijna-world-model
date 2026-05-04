# PWM Plugin Integration Guide

> *Plugins scaffold and steer — they do not replace the core śakti cascade.*

This document specifies exactly WHEN and HOW each plugin fires during PWM development.
The rule: invoke a plugin when its trigger condition is true, not on every step.

---

## Plugin Map

```
Development event
        │
        ├── Major design decision? ──────────────────► attractor-flow:attractor-orchestrator
        │   (before starting Phase N)                     ├── EXPLORING/STUCK → explorer-agent
        │                                                 └── CONVERGING → convergence-agent
        │
        ├── Architectural contradiction? ─────────────► triz-engine:analyze → :matrix → :principles
        │   (two good requirements conflict)
        │
        ├── Complex multi-step implementation done? ──► ralph-wiggum:ralph-loop
        │   (completed a phase or major module)
        │
        ├── Context growing unwieldy? ───────────────► pratyaksha-context-eng-harness:compact-now
        │   (long session, lots of tool output)           or :context-status to check budget
        │
        ├── Design exploration needed? ──────────────► superpowers:brainstorming
        │   (starting new phase design)
        │
        └── Implementing a plan? ────────────────────► superpowers:executing-plans
            (executing known design)
```

---

## 1. Attractor-Flow — Trajectory Health Monitor

**When:** Before each phase transition + whenever implementation feels stuck or directionless.

| Trigger | Plugin call | What it does |
|---------|-------------|-------------|
| Starting a new phase | `attractor-flow:attractor-orchestrator` | Assesses regime: CONVERGING → convergence-agent, STUCK → explorer-agent |
| 3+ failed implementation attempts | `attractor-orchestrator` → inject_perturbation | Breaks out of local minima |
| Phase design (pre-implementation) | `attractor-flow:explorer-agent` | Explores solution space with λ slightly > 0 |
| Phase implementation | `attractor-flow:convergence-agent` | Drives to completion with low temperature |

**Phase trigger map:**

| Phase boundary | Attractor action |
|---------------|-----------------|
| Phase 0 → 1 | Orchestrator: record "corpus ready, training loop built, wiring real env" |
| Phase 1 → 2 | Orchestrator: checkpoint Phase 1, assess EFE actor integration approach |
| Phase 2 → 3 | Orchestrator: checkpoint, assess Hopfield + memory routing design |
| Phase 3 → 4 | Orchestrator: checkpoint, assess sleep trigger placement |
| Phase 4 → 5 | Orchestrator: assess LLM integration + Mamba upgrade |
| Phase 5 → 6 | Orchestrator: full ablation trajectory — detect bifurcations |

**NOT needed for:** routine coding within a phase, bug fixes, test writing.

---

## 2. TRIZ — Architectural Contradiction Resolver

**When:** Two legitimate requirements pull in opposite directions.

**Known PWM contradictions and TRIZ triggers:**

| Contradiction | TRIZ trigger condition | Principles to try |
|--------------|----------------------|------------------|
| LLM quality ↔ latency (call on every step vs. never) | Sphurattā gating too coarse or too fine | #10 Prior Action, #34 Discarding |
| Mamba-2 speed ↔ CPU test compatibility | `_use_mamba(x)` coverage gaps | #1 Segmentation, #35 Parameter change |
| EFE expressivity ↔ training stability | Actor loss diverges | #11 Beforehand Cushioning, #10 Prior Action |
| Hopfield capacity ↔ memory budget | β too high → energy landscape collapses | #17 Another Dimension, #40 Composite |
| Sleep duration ↔ training throughput | ThermSleep threshold tuning | #37 Thermal Expansion, #10 Prior Action |
| Camatkāra sparsity ↔ signal quality | Sphurattā fires too rarely or too often | #24 Intermediary, #10 Prior Action |

**How to invoke:**
```
# Example: actor loss diverging during Phase 2
/triz-engine:analyze
# Describe: "EFE actor loss diverges (requirement: low variance) but
#            we need high policy entropy for creativity (requirement: diversity)"
# → :matrix for improvement/worsening parameters
# → :principles for the top-4 inventive principles
```

**NOT needed for:** hyperparameter tuning, debug cycles, routine refactoring.

---

## 3. Ralph Wiggum — Completion Promise Keeper

**When:** After completing any multi-step implementation milestone.

**Trigger list (invoke ralph-loop after each):**

| Milestone | Ralph trigger |
|-----------|--------------|
| Phase 1 gate written + pushed | After `phase_1_gate.json` committed |
| Phase 2 EFE actor activated + smoke trained | After Phase B training verified |
| Phase 3 Hopfield integrated | After CittaStore wired into PañcakṛtyaLoop |
| Phase 4 Sleep consolidator wired | After SleepAgent.maybe_sleep() in trainer |
| Phase 5 LLM bridge live | After end-to-end narration test passes |
| Phase 6 Ablations complete | Before paper draft |

**How to invoke:**
```
/ralph-wiggum:ralph-loop
# Ralph asks: "Did you really finish everything you promised?"
# It walks through the implementation checklist and surfaces incomplete items.
```

---

## 4. Pratyaksha PCEH — Context Manager

**When:** Proactively, not reactively.

| Trigger | Action |
|---------|--------|
| Session > 60 turns | `/pratyaksha-context-eng-harness:context-status` — check budget |
| Context budget > 80% | `/pratyaksha-context-eng-harness:compact-now` — compress |
| Starting a new session after long gap | PCEH already manages context via `avacchedaka.py` client |
| After vimarśa commit | `AvacchedakaStore.context_insert(qualificand="vimarsha", ...)` — already in VimarshaAgent |

**Runtime use:** `pwm/context/avacchedaka.py` wraps the PCEH client. All vimarśa commitments flow through it automatically.

---

## 5. Superpowers Brainstorming — Design Exploration

**When:** Before designing a new phase (≥2 valid approaches exist).

| Phase design moment | Brainstorm trigger |
|--------------------|--------------------|
| Phase 2: How to integrate EFE with existing critic? | Before writing efe_actor integration |
| Phase 3: Hopfield β scheduling strategy | Before setting β annealing schedule |
| Phase 5: Vimarśa bridge depth (n_prefix_tokens) | Before Phase 5 LLM integration design |
| Phase 6: Ablation order (statistical power) | Before setting A1–A6 run order |

**NOT needed for:** implementing already-designed modules, fixing bugs.

---

## 6. Superpowers Executing-Plans — Implementation Discipline

**When:** Whenever implementing a plan that spans ≥3 files or ≥2 hours of work.

The executing-plans skill enforces:
- One task at a time (TodoWrite tracking)
- Verification at each step
- `finishing-a-development-branch` at the end

**Always active during phase implementations.** Already invoked for Phase 1→2.

---

## Development Loop Template

```
Phase N start:
  1. superpowers:brainstorming        ← design (if not already designed)
  2. attractor-flow:attractor-orchestrator ← regime check
     → convergence-agent for implementation
  3. superpowers:executing-plans      ← discipline scaffold
  4. [implement, test, commit]
  5. [if architectural contradiction] → triz-engine:analyze
  6. ralph-wiggum:ralph-loop          ← completion check
  7. Write benchmarks/results/phase_N_gate.json
  8. git push origin phase-N/...
  9. attractor-flow:attractor-orchestrator → checkpoint
 10. Advance to Phase N+1
```

---

## What Plugins Do NOT Do

- **Do not fragment the śakti cascade** — PañcakṛtyaLoop stays one Python call stack.
- **Do not call LLM on every step** — only at sphurattā events.
- **Do not replace the core training loop** — plugins scaffold the meta-level workflow.
- **Do not over-invoke** — TRIZ is for real contradictions, not preference differences. Ralph is for milestone completion, not every commit. Attractor-flow is for regime changes, not every file edit.
