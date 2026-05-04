# PWM Multi-Agent Architecture
## Śakti Pipeline, Vimarśa Agent, and smolagents Orchestration

*Version 0.2 — May 2026 (revised: agent boundary decision)*

---

## 1. Architectural Principle: Where Agents Add Value

The śakti cascade (cit → ānanda → icchā → apohana → jñāna → kriyā → vimarśa) is tempting to implement as seven separate autonomous agents. This would be architecturally wrong for the WM substrate for a simple reason: **the five śaktis (cit through kriyā) are computations that share continuous latent state** — the recurrent WM state `(h_t, z_t)` flows uninterrupted through all six steps. Fragmenting this into separate agent instances would:

1. Break the WM's continuous latent flow — `h_t` would have to be serialized to a context store and deserialized at each agent boundary, introducing quantization error and latency.
2. Force six separate LLM calls where two or fewer suffice — most śakti steps involve WM computations (pure PyTorch) with occasional LLM assistance for knowledge retrieval, not autonomous LLM deliberation.
3. Create meaningless "agents" — cit is an encoder forward pass; ānanda is a reward computation; apohana is a threshold filter. None of these have tool-using, multi-turn behaviour.

**The right agent boundary** is the *deliberative transition* — the point where the system pauses to reflect, use external tools, and make a commitment. There are exactly three such points:

| Agent | Role | Why a true agent |
|---|---|---|
| **VimarshaAgent** | Reflexive witness; quality gate; commit/revise | Multi-turn deliberation; tool calls (classify_khyativada, context tools); decides whether to commit or restart cascade |
| **MemoryAgent** | CittaStore update + context compaction | Orchestrates multi-step memory operations; uses Pratyākṣa tools; runs asynchronously post-commit |
| **SleepAgent** | NREM/REM cycle orchestration | Long-running; uses WM replay + LLM narration; needs to stop/continue based on ThermSleep budget |

**Everything else** — the śakti steps — lives inside a single `PancakrtyaLoop` class whose methods call the WM and the LLM directly via LiteLLM. The LLM is not an agent here; it is a called function, exactly like the RSSM decoder or Hopfield retrieval.

---

## 2. PancakrtyaLoop: The WM Pipeline Class

```python
# pwm/pipeline/pancakrtya_loop.py

import torch
from pymdp.maths import compute_expected_utility, compute_info_gain
from pwm.wm import TrikaWorldModel
from pwm.context import AvacchedakaStore
from pwm.llm import LLMBackend          # LiteLLM wrapper

class PancakrtyaLoop:
    """
    Implements the five acts of Śiva (pañcakṛtya) as sequential WM pipeline steps.
    
    The śakti cascade runs as a single Python call stack sharing continuous
    WM state (h_t, z_t). No agent framework here — this is the WM forward loop.
    
    Only vimarśa exits this loop: it is a true agent (tool-using, deliberative).
    """

    def __init__(
        self,
        wm: TrikaWorldModel,
        store: AvacchedakaStore,
        llm: LLMBackend,
        config: dict,
    ):
        self.wm = wm
        self.store = store
        self.llm = llm
        self.cfg = config

    # ── Step 1: Sṛṣṭi / Cit ──────────────────────────────────────────────────
    def cit_step(self, raw_input: str) -> tuple:
        """
        Cit (pure awareness): encode raw text input via WM encoder.
        WM observe step: embeds token sequence → h_t, z_t.
        No LLM call. Pure neural forward pass.
        """
        tokens = self.wm.tokenize(raw_input)
        h_t, z_t = self.wm.observe(tokens)

        self.store.context_insert(
            qualificand="cit",
            key="h_t", value=h_t.tolist(), precision=1.0,
        )
        self.store.context_insert(
            qualificand="cit",
            key="z_t", value=z_t.tolist(), precision=1.0,
        )
        return h_t, z_t

    # ── Step 2: Ānanda ────────────────────────────────────────────────────────
    def ananda_step(self, h_t: torch.Tensor, z_t: torch.Tensor) -> dict:
        """
        Ānanda (aesthetic tension): compute camatkāra potential.
        R_camatk = α₁·ΔF_vfe + α₂·ΔI_Hopfield + α₃·Empowerment
        No LLM call. Pure WM reward computation.
        """
        r, components = self.wm.camatkaara_reward.compute(h_t, z_t)
        is_sphuratta = self.wm.citta.is_sphuratta()

        self.store.context_insert(
            qualificand="ananda",
            key="camatkaara_score", value=float(r), precision=0.9,
        )
        self.store.context_insert(
            qualificand="ananda",
            key="sphuratta_flag", value=is_sphuratta, precision=1.0,
        )
        return {"r": r, "components": components, "sphuratta": is_sphuratta}

    # ── Step 3: Icchā ─────────────────────────────────────────────────────────
    def icha_step(self, h_t: torch.Tensor, z_t: torch.Tensor) -> list:
        """
        Icchā (will/desire): sample K candidate intentions via EFE Actor.
        
        Uses pymdp's EFE computation on top of the RSSM belief distribution.
        The EFE Actor is a neural policy net trained to minimise expected free energy G.
        No LLM call — EFE is computed from WM beliefs alone.
        """
        K = self.cfg["icha"]["K_candidates"]
        candidates = []

        for _ in range(K):
            # Sample latent action from EFE actor
            a_t = self.wm.efe_actor.sample(h_t)

            # Imagine one step forward: p(s_{t+1} | h_t, a_t)
            h_next, z_next = self.wm.imagine_step(h_t, z_t, a_t)

            # Compute EFE components using pymdp math on RSSM beliefs
            G, G_terms = self._compute_efe(h_next, z_next)

            # Flag high-epistemic-uncertainty candidates for deep jñāna processing
            needs_jnana = (G_terms["epistemic"] > self.cfg["icha"]["epistemic_threshold"])

            candidates.append({
                "a_t": a_t.tolist(),
                "h_next": h_next.tolist(),
                "z_next": z_next.tolist(),
                "G": float(G),
                "G_terms": {k: float(v) for k, v in G_terms.items()},
                "needs_jnana": bool(needs_jnana),
            })

        self.store.context_insert(
            qualificand="icha", key="candidates", value=candidates, precision=0.7,
        )
        return candidates

    def _compute_efe(self, h_t: torch.Tensor, z_t: torch.Tensor) -> tuple:
        """
        Compute Expected Free Energy using pymdp math utilities on RSSM beliefs.
        G = ambiguity + risk − epistemic_value − parameter_novelty
        
        pymdp.maths functions operate on numpy arrays; we convert RSSM tensors.
        """
        import numpy as np

        qs = z_t.softmax(dim=-1).detach().cpu().numpy()   # RSSM posterior → pymdp belief

        # Ambiguity: expected surprise under the model = decoder entropy E[-log p(o|s)]
        ambiguity = self.wm.decoder_entropy(h_t)

        # Risk: KL between predicted outcomes and preferred outcomes C
        # C is the preference distribution (goal specification from vimarśa/user)
        C = self.wm.get_preference_distribution()         # shape: (n_outcomes,)
        risk_np = compute_expected_utility(C, qs)
        risk = torch.tensor(risk_np, device=h_t.device)

        # Epistemic value: information gain I[s; o | π]
        # Approximate A-matrix from RSSM decoder Jacobian (linearised)
        A = self.wm.get_likelihood_matrix(h_t)            # (n_obs, n_states)
        epistemic_np = compute_info_gain(A, qs)
        epistemic = torch.tensor(epistemic_np, device=h_t.device)

        # Parameter novelty: uses running Dirichlet concentration estimate
        novelty = self.wm.parameter_novelty(h_t, z_t)

        G = ambiguity + risk - epistemic - novelty
        return G, {
            "ambiguity": ambiguity,
            "risk": risk,
            "epistemic": epistemic,
            "novelty": novelty,
        }

    # ── Step 4: Apohana (negation/pruning) ───────────────────────────────────
    def apohana_step(self, candidates: list) -> list:
        """
        Apohana (exclusion/negation): prune candidates with negative EFE.
        Pure WM computation — no LLM, no agent.
        Corresponds to Saṃhāra (dissolution) in pañcakṛtya.
        """
        viable = [c for c in candidates if c["G"] < self.cfg["apohana"]["G_threshold"]]
        viable.sort(key=lambda c: c["G"])               # lowest EFE = best
        top = viable[:3] if viable else candidates[:3]  # always return top-3

        self.store.context_insert(
            qualificand="apohana", key="top_candidates", value=top, precision=0.8,
        )
        return top

    # ── Step 5: Jñāna ────────────────────────────────────────────────────────
    def jnana_step(self, h_t: torch.Tensor, z_t: torch.Tensor,
                   top_candidates: list) -> dict:
        """
        Jñāna (knowledge): Hopfield recall + LLM knowledge augmentation.
        
        Fast path: if no candidate needs deep jñāna processing (needs_jnana=False),
        only Hopfield recall is used (no LLM call).
        
        Slow path: if any candidate needs_jnana, LLM is called for knowledge
        retrieval and belief update. This is the manas→buddhi escalation point.
        """
        # Fast path: Hopfield episodic recall (always runs)
        memories = self.wm.citta.recall_episodic(z_t, top_k=3)
        belief_update = {"recalled_memories": [m.to_dict() for m in memories]}

        # Slow path: LLM knowledge augmentation (only if needed)
        needs_deep = any(c["needs_jnana"] for c in top_candidates)
        if needs_deep:
            # Build context from Avacchedaka store + Hopfield memories
            context = self.store.context_get(qualificands=["cit", "ananda", "icha", "apohana"])
            memory_text = "\n".join(m.text for m in memories)

            # Call LLM via LiteLLM (fast path model — sub-deliberative knowledge call)
            response = self.llm.call(
                role="jnana",      # routes to fast model in provider config
                system=self.store.get_sakshi(),
                prompt=self._jnana_prompt(context, memory_text, top_candidates),
            )
            belief_update["llm_knowledge"] = response
            belief_update["source"] = "anumana+agama"   # inference + testimony

            # Sublate any contradicted context
            conflicts = self.store.detect_conflict(qualificand="jnana")
            for conflict in conflicts:
                self.store.sublate_with_evidence(
                    key=conflict["key"],
                    evidence=response,
                )
        else:
            belief_update["source"] = "pratyaksha"      # direct perception only

        self.store.context_insert(
            qualificand="jnana", key="belief_update", value=belief_update, precision=0.85,
        )
        return belief_update

    def _jnana_prompt(self, context: dict, memory_text: str, candidates: list) -> str:
        return (
            f"You are the jñāna function — knowledge recognition from memory.\n\n"
            f"Current WM context:\n{context}\n\n"
            f"Retrieved memories:\n{memory_text}\n\n"
            f"Candidate intentions:\n{candidates}\n\n"
            f"Provide: (1) relevant knowledge that grounds these candidates, "
            f"(2) any contradictions with existing context, "
            f"(3) which candidate is best supported by knowledge. "
            f"Source each claim as pratyakṣa/anumāna/āgama."
        )

    # ── Step 6: Kriyā ────────────────────────────────────────────────────────
    def kriya_step(self, h_t: torch.Tensor, best_candidate: dict,
                   belief_update: dict) -> str:
        """
        Kriyā (action): WM imagination rollout → draft creative output.
        
        Uses the WM's imagination rollout (DreamerV3-style) to project the
        best candidate's action sequence forward H steps, then decodes to text.
        The LLM is called once to expand the WM's token-level draft into fluent prose.
        """
        H = self.cfg["kriya"]["imagination_horizon"]
        a_t = torch.tensor(best_candidate["a_t"])

        # WM imagination rollout: project H steps from best candidate
        imagined_states = self.wm.imagine_rollout(h_t, a_t, horizon=H)

        # Decode WM imagination to text tokens (WM decoder, not LLM)
        wm_draft_tokens = self.wm.decoder.decode_sequence(imagined_states)
        wm_draft = self.wm.tokenizer.decode(wm_draft_tokens)

        # LLM fluency pass: expand WM draft into coherent creative output
        # This is NOT generation from scratch — the WM scaffold constrains the output
        knowledge_context = belief_update.get("llm_knowledge", "")
        response = self.llm.call(
            role="kriya",          # fast model
            system=self.store.get_sakshi(),
            prompt=self._kriya_prompt(wm_draft, knowledge_context),
        )

        self.store.context_insert(
            qualificand="kriya", key="draft_output", value=response, precision=0.75,
        )
        self.store.context_insert(
            qualificand="kriya", key="wm_draft", value=wm_draft, precision=1.0,
        )
        return response

    def _kriya_prompt(self, wm_draft: str, knowledge: str) -> str:
        return (
            f"You are the kriyā function — actualising the world model's creative intent.\n\n"
            f"World model draft (do not deviate from this structure/trajectory):\n"
            f"{wm_draft}\n\n"
            f"Supporting knowledge:\n{knowledge}\n\n"
            f"Expand this draft into a complete, fluent creative output. "
            f"Preserve the WM's structural choices. Do not add content not implied by the draft."
        )

    # ── Full cascade ─────────────────────────────────────────────────────────
    def run(self, raw_input: str) -> str:
        """Execute the full śakti cascade. Returns draft for vimarśa evaluation."""
        h_t, z_t = self.cit_step(raw_input)
        ananda = self.ananda_step(h_t, z_t)
        candidates = self.icha_step(h_t, z_t)
        top = self.apohana_step(candidates)
        belief = self.jnana_step(h_t, z_t, top)
        best = min(top, key=lambda c: c["G"])    # lowest EFE = committed intention
        draft = self.kriya_step(h_t, best, belief)
        return draft
```

---

## 3. VimarshaAgent: The True Deliberative Agent

Vimarśa is the only part of the cascade that genuinely needs to be an autonomous agent. It:
- Has its own multi-turn decision loop (commit / revise / reject)
- Uses external tools (Pratyākṣa MCP, context store, narration)
- Calls the primary LLM (120B) with deep reasoning
- Can restart the entire cascade with a revised goal

```python
# pwm/agents/vimarsha_agent.py

from smolagents import CodeAgent, tool
from pwm.llm import LLMBackend
from pwm.context import AvacchedakaStore, PratyakshaClient
from pwm.wm import VimarshaWMBridge

class VimarshaAgent:
    """
    Vimarśa: reflexive self-awareness gate.
    
    The only true agent in the pipeline. Receives the WM cascade's draft output,
    judges it via khyātivāda classifier, and decides: commit / revise / reject.
    
    Uses the primary LLM (Nemotron 120B or configured provider) with deep reasoning.
    """

    MAX_REVISIONS = 3

    def __init__(
        self,
        store: AvacchedakaStore,
        pratyaksha: PratyakshaClient,
        llm: LLMBackend,
        bridge: VimarshaWMBridge,
        pipeline: 'PancakrtyaLoop',
    ):
        self.store = store
        self.mcp = pratyaksha
        self.llm = llm
        self.bridge = bridge
        self.pipeline = pipeline
        self._revision_count = 0

    def evaluate(self, draft: str, user_input: str) -> dict:
        """
        Core vimarśa function: judge draft, decide action.
        Returns: {"action": commit|revise|reject, "output": str, "narrative": str}
        """
        # 1. Khyātivāda epistemic classification (via Pratyākṣa MCP)
        khyati = self.mcp.classify_khyativada(text=draft)

        # 2. Narrate WM latent state (only at sphurattā events — saves tokens)
        narrative = ""
        if self.store.context_retrieve("ananda", "sphuratta_flag"):
            h_t = torch.tensor(self.store.context_retrieve("cit", "h_t"))
            narrative = self.bridge.narrate_latent(h_t)

        # 3. LLM deliberation: should we commit?
        action_response = self.llm.call(
            role="vimarsha",    # primary model (120B) with reasoning
            system=self.store.get_sakshi(),
            prompt=self._vimarsha_prompt(draft, khyati, narrative),
        )

        action = self._parse_action(action_response, khyati)

        # 4. Update store
        self.store.context_insert(
            qualificand="vimarsha",
            key="judgment", value={
                "khyati_type": khyati["type"],
                "action": action["decision"],
                "narrative": narrative,
            }, precision=0.95,
        )

        return action

    def run(self, draft: str, user_input: str) -> str:
        """Full vimarśa loop with revision support."""
        while self._revision_count <= self.MAX_REVISIONS:
            result = self.evaluate(draft, user_input)

            if result["decision"] == "commit":
                self._revision_count = 0
                return result["output"]

            elif result["decision"] == "revise":
                self._revision_count += 1
                # Revise the goal preference C and re-run the cascade
                revised_goal = result.get("revised_goal", user_input)
                self.pipeline.wm.update_preference(revised_goal)
                draft = self.pipeline.run(user_input)

            else:  # reject
                self._revision_count = 0
                return f"[cascade rejected after {self._revision_count} attempts]"

        return draft  # fallback: return last draft if max revisions reached

    def _vimarsha_prompt(self, draft: str, khyati: dict, narrative: str) -> str:
        wm_draft = self.store.context_retrieve("kriya", "wm_draft") or ""
        return (
            f"You are vimarśa — the reflexive self-awareness that witnesses creative acts.\n\n"
            f"WM imagination scaffold:\n{wm_draft}\n\n"
            f"Generated draft:\n{draft}\n\n"
            f"Epistemic classification: {khyati['type']} (confidence: {khyati['confidence']:.2f})\n"
            f"WM narrative: {narrative}\n\n"
            f"Decide:\n"
            f"- COMMIT: if the draft is epistemically valid and creatively sound\n"
            f"- REVISE <revised_goal>: if the draft is correctable with a new goal\n"
            f"- REJECT: if the draft is fundamentally misaligned\n\n"
            f"Reply with exactly: COMMIT, REVISE <goal>, or REJECT."
        )

    def _parse_action(self, response: str, khyati: dict) -> dict:
        resp = response.strip()
        if resp.startswith("COMMIT"):
            return {"decision": "commit", "output": self.store.context_retrieve("kriya", "draft_output")}
        elif resp.startswith("REVISE"):
            revised_goal = resp[7:].strip()
            return {"decision": "revise", "revised_goal": revised_goal}
        else:
            return {"decision": "reject"}
```

---

## 4. MemoryAgent and SleepAgent

These remain true agents because they orchestrate complex multi-step operations asynchronously, after the main cascade completes.

```python
# pwm/agents/memory_agent.py

from smolagents import CodeAgent
from pwm.context import AvacchedakaStore, PratyakshaClient
from pwm.wm import TrikaWorldModel

class MemoryAgent:
    """
    Post-commit memory consolidation.
    Runs asynchronously after vimarśa commits. Not on the critical path.
    
    Steps:
    1. Write committed episode to CittaStore episodic Hopfield buffer
    2. Compact context window (Pratyākṣa boundary_compact = tirodhāna gate)
    3. Update semantic Hopfield prototypes if triggered
    """

    def __init__(self, wm: TrikaWorldModel, store: AvacchedakaStore,
                 pratyaksha: PratyakshaClient, llm: LLMBackend):
        self.wm = wm
        self.store = store
        self.mcp = pratyaksha
        self.llm = llm

    def run(self, committed_output: str) -> dict:
        # 1. Write to episodic Hopfield
        h_t = torch.tensor(self.store.context_retrieve("cit", "h_t"))
        z_t = torch.tensor(self.store.context_retrieve("cit", "z_t"))
        r = self.store.context_retrieve("ananda", "camatkaara_score")
        self.wm.citta.write_episodic({"h_t": h_t, "z_t": z_t, "r": r,
                                       "output": committed_output})

        # 2. Boundary compact (tirodhāna — conceal completed episode)
        compact_result = self.mcp.compact(
            threshold=self.cfg["memory"]["compaction_threshold"],
            task_context="post_commit",
        )

        # 3. Semantic prototype update (every N commits)
        if self._should_update_semantic():
            self.wm.citta.update_semantic_prototypes()

        return {"episodic_written": True, "compact": compact_result}
```

```python
# pwm/agents/sleep_agent.py

class SleepAgent:
    """
    NREM/REM sleep cycle orchestrator.
    Triggered by commit count threshold or ThermSleep budget trigger.
    
    NREM: SHY down-scaling + prioritised replay + Hopfield consolidation (VFE descent)
    REM: generative dreaming + EFE actor-critic update + recognition-net retraining
    """

    def run_full_cycle(self, wm: TrikaWorldModel, llm: LLMBackend) -> dict:
        nrem = wm.sleep_scheduler.run_nrem()
        rem = wm.sleep_scheduler.run_rem()

        # LLM narrates sleep insights (for skill library + DECKARD AWM proposals)
        if nrem.get("novel_prototypes"):
            insight = llm.call(
                role="sleep",
                system=self.store.get_sakshi(),
                prompt=f"Summarise these NREM consolidation insights in ≤200 words: {nrem}",
            )
            self.store.context_insert(
                qualificand="sleep", key="nrem_insight", value=insight, precision=0.8,
            )

        return {"nrem": nrem, "rem": rem}
```

---

## 5. LLM Backend: LiteLLM Unified Abstraction

The `LLMBackend` class wraps LiteLLM so that all LLM calls in the pipeline are provider-agnostic. The `role` parameter routes to the appropriate model tier (primary vs fast) based on the active provider config.

```python
# pwm/llm/backend.py

import litellm
from pwm.config import LLMConfig

class LLMBackend:
    """
    Unified LLM interface via LiteLLM.
    Provider (local Nemotron / Claude API / OpenAI / Gemini) is set in config.
    Commercial API providers bypass TRT-LLM/vLLM entirely.
    """

    ROLE_TIER = {
        "vimarsha": "primary",   # deep reasoning; 120B or equivalent
        "memory":   "primary",
        "sleep":    "primary",
        "jnana":    "fast",      # knowledge call; 49B or equivalent
        "kriya":    "fast",
    }

    def __init__(self, config: LLMConfig):
        self.cfg = config
        self.provider = config.provider         # e.g. "nemotron-local"
        litellm.drop_params = True

    def call(self, role: str, system: str, prompt: str,
             temperature: float = None, max_tokens: int = None) -> str:
        tier = self.ROLE_TIER.get(role, "fast")
        model_cfg = self.cfg.providers[self.provider][tier]

        response = litellm.completion(
            model=model_cfg["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            api_base=model_cfg.get("api_base"),
            api_key=model_cfg.get("api_key"),
            temperature=temperature or model_cfg.get("temperature", 0.7),
            max_tokens=max_tokens or model_cfg.get("max_tokens", 1024),
        )
        return response.choices[0].message.content
```

---

## 6. Orchestration Entry Point

```python
# pwm/main.py

from pwm.wm import build_world_model
from pwm.context import AvacchedakaStore, PratyakshaClient, SakshiKeeper
from pwm.pipeline import PancakrtyaLoop
from pwm.agents import VimarshaAgent, MemoryAgent, SleepAgent
from pwm.llm import LLMBackend
from pwm.config import load_config

def build_pwm_system(config_path: str):
    cfg = load_config(config_path)

    # Core components
    wm      = build_world_model(cfg)
    pratyak = PratyakshaClient(mcp_server=cfg["pceh"]["mcp_server"])
    store   = AvacchedakaStore(pratyak)
    llm     = LLMBackend(cfg["llm"])

    # Bootstrap Sākṣī witness
    sakshi = SakshiKeeper(pratyak)
    sakshi.bootstrap(
        claude_md=open("CLAUDE.md").read(),
        user_intent=cfg.get("creative_intent", ""),
    )

    # WM pipeline (śakti cascade — not agents)
    pipeline = PancakrtyaLoop(wm=wm, store=store, llm=llm, config=cfg)

    # True agents (deliberative, tool-using)
    vimarsha = VimarshaAgent(
        store=store, pratyaksha=pratyak,
        llm=llm, bridge=wm.vimarsha_bridge, pipeline=pipeline,
    )
    memory  = MemoryAgent(wm=wm, store=store, pratyaksha=pratyak, llm=llm)
    sleep   = SleepAgent(wm=wm, store=store, llm=llm)

    return pipeline, vimarsha, memory, sleep, sakshi


def run(user_input: str, config_path: str = "configs/default.yaml") -> str:
    pipeline, vimarsha, memory, sleep, sakshi = build_pwm_system(config_path)

    # Run śakti cascade
    draft = pipeline.run(user_input)

    # Vimarśa gate: judge + commit/revise
    output = vimarsha.run(draft, user_input)

    # Post-commit: memory consolidation (async / background)
    memory.run(output)

    # Sleep trigger
    if pipeline.wm.sleep_scheduler.should_sleep():
        sleep.run_full_cycle(pipeline.wm, pipeline.llm)

    return output


if __name__ == "__main__":
    import sys
    result = run(user_input=sys.argv[1])
    print(result)
```

---

## 7. pymdp Integration: EFE on RSSM Beliefs

The `_compute_efe` method above uses `pymdp.maths` utilities. Here is the full dependency and compatibility note:

```bash
# Install pymdp (active inference library)
pip install inferactively-pymdp

# Verify EFE functions are available
python -c "from pymdp.maths import compute_expected_utility, compute_info_gain; print('OK')"
```

**Why EFE-module-only (not full pymdp POMDP)**:

pymdp implements a discrete POMDP with explicit A (likelihood), B (transition), C (preference), D (prior) matrices. The Trika WM uses a continuous-latent RSSM, which is a parametric neural world model — incompatible with pymdp's matrix-based approach as a full replacement.

The right integration is **surgical**:

| pymdp component | Used in PWM? | How |
|---|---|---|
| `pymdp.maths.compute_info_gain(A, qs)` | Yes | Epistemic value in icchā step; A ≈ linearised RSSM decoder Jacobian |
| `pymdp.maths.compute_expected_utility(C, qs)` | Yes | Risk term; C = preference distribution from vimarśa goal spec |
| `pymdp.algos.run_active_inference_loop` | No | Full POMDP loop — replaced by `PancakrtyaLoop` |
| `pymdp.envs` | No | Environments — replaced by the WM itself |
| `pymdp.Agent` | No | Full agent class — replaced by WM + VimarshaAgent |

The RSSM's continuous categorical posterior `z_t` (32×32 classes) is treated as a discrete belief state `qs` for the pymdp math functions, bridging the neural and symbolic AIF worlds.

---

## 8. Configuration

```yaml
# configs/multiagent.yaml

pipeline:
  icha:
    K_candidates: 5
    epistemic_threshold: 0.3       # above this → trigger deep jñāna

  apohana:
    G_threshold: 0.0               # negative EFE = expected free energy reducing
    top_k: 3

  kriya:
    imagination_horizon: 15        # WM rollout steps before decoding

  jnana:
    hopfield_top_k: 3
    fast_path_default: true        # only go to LLM when needs_jnana=true

  vimarsha:
    max_revision_iterations: 3
    narrate_on_sphuratta: true

  memory:
    compaction_threshold: 2.5
    semantic_update_every_n: 10

  sleep:
    trigger_every_n_commits: 50
    therm_budget_threshold: 0.1

# pymdp EFE settings
efe:
  decoder_entropy_method: "mc_samples"   # or "analytic"
  n_mc_samples: 10
  A_approximation: "jacobian"            # linearised decoder Jacobian
  novelty_method: "dirichlet_kl"

# LLM backend (see configs/llm_backend.yaml for full provider spec)
llm:
  provider: "nemotron-local"
```

---

## 9. Pañcakṛtya → Pipeline Step Mapping

| Act | Sanskrit | Step | Implementation |
|---|---|---|---|
| Creation | Sṛṣṭi | `cit_step + ananda_step` | WM observe + camatkāra computation |
| Preservation | Sthiti | `jnana_step` | Hopfield recall + BMR belief stabilisation |
| Dissolution | Saṃhāra | `apohana_step` | EFE-based candidate pruning |
| Concealment | Tirodhāna | `memory.run()` | Boundary compaction: completed episode veiled |
| Grace | Anugraha | `kriya_step + vimarsha.run()` | WM imagination rollout + reflexive release |

---

## 10. What smolagents Is and Isn't Used For

| Task | smolagents? | Alternative |
|---|---|---|
| śakti cascade steps (cit–kriyā) | No | `PancakrtyaLoop` Python methods |
| vimarśa deliberation | Yes | `VimarshaAgent` wraps deliberation loop |
| Memory consolidation | Lightweight | `MemoryAgent` uses direct MCP calls |
| Sleep orchestration | Lightweight | `SleepAgent` orchestrates WM scheduler |
| Inter-agent messaging | No (use PCEH store) | Avacchedaka store (Pratyākṣa MCP) |
| Hypothesis ablations | No | Direct config flag + `PancakrtyaLoop` bypass |
