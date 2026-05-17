# PWM Wow-Architecture Design Spec
## Pratyabhijñā World Model — Production Rewrite for neo-fm-web

**Date:** 2026-05-17  
**Status:** Approved — ready for implementation planning  
**Product context:** This API is the creative lyrics/poem/song core for [neo-fm-web.vercel.app](https://neo-fm-web.vercel.app/)  
**Research context:** arXiv paper, GitHub Pages gallery, H1–H9 pre-registered hypotheses  
**Sprint scope:** Sprints 8–18 (Phases 3–6)

---

## Executive Summary

The current PWM production system has three critical gaps that prevent it from being the genuine creative AI it claims to be:

1. **WM on CPU** — `DEVICE = torch.device("cpu")` means the world model never uses the GB10 GPU during generation. The WM is cosmetic, not computational.
2. **Dead vimarśa bridge** — `format_prefix_text()` (text concatenation) runs in production; the real VimarsaBridge cross-attention is dead code. LLM is not conditioned by WM state.
3. **PancakrtyaLoop unused** — The full 6-act Pañcakṛtya loop is implemented but never called by the API. Generation bypasses it entirely via `engine.py`.

This spec defines the architecture that makes all three live on GPU, wires them together through a real token-level conditioning pipeline, and ships it as a production API for the neo-fm-web music/song generation product.

---

## Part 1 — Architectural Contracts (Non-Negotiable Invariants)

These three contracts are INVIOLABLE. No sprint may ship code that breaks them. Any apparent contradiction must be resolved with TRIZ before implementation proceeds.

### Contract 1: Pañcakṛtya Invariant

The śakti cascade **must** execute in full, in order, in every generation call:

| Act | Sanskrit | Computational Realisation | GPU Required |
|-----|----------|--------------------------|--------------|
| 1 | Cit (sṛṣṭi) | `wm.observe_step(obs_t)` → posterior `z_t` | ✓ WM |
| 2 | Ānanda (sthiti) | `efe_actor(h_t, z_t)` → EFE score, hold state | ✓ WM |
| 3 | Icchā (saṃhāra) | `citta_store.recall(z_t)` → episodic resonance | ✓ WM |
| 4 | Apohana (tirodhāna) | VFE entropy gate: `H[q(z_t)] > threshold` → sphurattā | ✓ WM |
| 5 | Jñāna (anugraha slow) | `vimarsa_bridge.encode(h_t)` → logit bias tensor | ✓ WM+Bridge |
| 6 | Kriyā | `llm.generate(prompt, logits_processor=bias_fn)` → tokens | ✓ LLM |

**Rule:** If any act is stubbed, mocked, or bypassed, the implementation is in violation of Contract 1. The EFEActor must compute real EFE (not a placeholder loss). The Hopfield recall must run a real matrix product. The logit bias must be a real projection `h_t → R^{vocab_size}`.

**Sphurattā definition:** A sphurattā event occurs when `H[q(z_t)] > τ_sphuratta` (default τ=0.65) AND `efe_score > efe_threshold`. At a sphurattā event, the full jñāna act re-runs (fresh h_t encoding), and the per-stanza generation cache is invalidated.

### Contract 2: Layer Boundary

Every module lives in exactly one layer. Crossing the boundary requires explicit translation via `CamatkaraNarrator`.

```
╔═══════════════════════════════════════════════════════╗
║  INTERNAL LAYER  (Śaiva, computational, private)      ║
║  EFE, VFE, h_t, z_t, sphurattā events               ║
║  VimarsaBridge, CittaStore, PancakrtyaLoop            ║
╠═══════════════════════════════════════════════════════╣
║  BOUNDARY: CamatkaraNarrator  (translate, never leak) ║
╠═══════════════════════════════════════════════════════╣
║  EXTERNAL LAYER  (domain-neutral, product, public)    ║
║  Lyrics text, register labels, rāga/scale names       ║
║  JSON API responses, SSE token stream, UI display     ║
╚═══════════════════════════════════════════════════════╝
```

**Rule:** Sanskrit terms (sphurattā, vimarśa, camatkāra) may appear in internal logs and research output but MUST NOT appear in generated lyrics, API JSON responses to neo-fm-web, or any end-user-visible text. The CamatkaraNarrator translates `sphurattā_event` → `creative_peak`, `camatkāra_score` → `aesthetic_quality`, `vimarśa` → `reflective_revision`.

### Contract 3: WM is Primary, LLM is Secondary

The world model **drives** generation. The LLM **renders** what the WM directs.

- WM computes creative trajectory (EFE minimisation)
- WM decides when to generate (sphurattā trigger)
- WM conditions every token via logit bias (VimarsaBridgeV2)
- LLM is a fluency renderer, not the creative agent

**Operational rule:** If llama.cpp or Ollama is unavailable, the WM must still run, produce EFE scores, generate sphurattā events, and output a degraded-quality stub. The system cannot fail because the LLM is unavailable. The WM is the product's core IP.

---

## Part 2 — Hardware Architecture

### GB10 Blackwell Allocation

| Resource | Allocated To | Budget | Notes |
|----------|-------------|--------|-------|
| Stream 0 (low priority) | TrikaWorldModel + EFEActor + CittaStore | ~0.3GB | All torch, CUDA 12.1 |
| Stream 1 (high priority) | Nemotron 120B Q5_K_M (llama.cpp) | ~80GB | llama-server via HTTP |
| Pinned host memory | dlpack bridge tensors | ~0.1GB | Zero-copy WM ↔ JAX |
| Remaining | System + OS | ~50GB | 130.7GB total unified |

### llama.cpp Migration (Ollama → llama.cpp)

**Why:** Ollama adds ~200ms HTTP overhead per request and does not expose a `logits_processor` hook. llama-cpp-python exposes a native Python callback that fires on every token — this is the mechanism VimarsaBridgeV2 requires.

**Existing binary:** `/home/sharaths/llama.cpp/build/bin/llama-server` (compiled with CUDA)

**Nemotron GGUF location:** Ollama blob cache contains Q4_K_M blobs; need to locate and symlink or re-export as single `.gguf`. Sprint 8 task: find blobs, use `llama-export` or `llama-gguf-merge` to produce `nemotron-120b-q5km.gguf`.

**llama-server startup:**
```bash
./llama-server \
  --model nemotron-120b-q5km.gguf \
  --n-gpu-layers 999 \
  --flash-attn \
  --cont-batching \
  --port 8080 \
  --n-predict 512 \
  --n-ctx 4096
```

**llama-cpp-python (Python binding):** Must install to enable `logits_processor` callback:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### Dual CUDA Stream Protocol

```python
stream_wm  = torch.cuda.Stream(priority=-1)   # low: WM
stream_llm = torch.cuda.Stream(priority=0)    # high: reserved for NCCL/LLM if needed

# WM computes on stream_wm
with torch.cuda.stream(stream_wm):
    z_t, h_t, vfe = wm.observe_step(obs_t)
    efe_score = efe_actor(h_t, z_t)

# Sync before logits_processor is registered
torch.cuda.current_stream().wait_stream(stream_wm)

# LLM runs on default stream; logits_processor uses CPU-pinned h_t projection
bias_fn = vimarsa_bridge.as_logits_processor(h_t)
tokens = llm.generate(prompt, logits_processor=[bias_fn])
```

---

## Part 3 — VimarsaBridgeV2 (Core Technical Novelty)

This is the architectural innovation that makes PWM genuinely novel versus all prior work. No published system conditions an LLM's token logits with a World Model's hidden state at inference time via a trained projection layer.

### Design

```python
class VimarsaBridgeV2(nn.Module):
    """
    Sanskrit concept: Vimarśa (ĪPK 1.5.11) — reflexive self-awareness.
    Computational: h_t → linear → vocab_size logit bias.
    Applied at every token via llama-cpp-python logits_processor hook.

    Training: supervised on (h_t, domain_label) pairs from WM training corpus.
    Loss: cross-entropy on next-token distribution (teacher-forcing from corpus).
    """
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        # ~66MB for hidden_dim=512, vocab_size=128256 (Llama-3 tokenizer)
        # ~33MB for hidden_dim=256, vocab_size=128256

    def as_logits_processor(self, h_t: torch.Tensor) -> Callable:
        """Returns a logits_processor function for llama-cpp-python."""
        with torch.no_grad():
            bias = self.proj(h_t.squeeze(0)).cpu().numpy()   # (vocab_size,)
        def _processor(token_ids: list[int], logits: np.ndarray) -> np.ndarray:
            return logits + bias
        return _processor

    def train_step(self, h_t, target_token_ids, optimizer):
        logits = self.proj(h_t)
        loss = F.cross_entropy(logits, target_token_ids)
        loss.backward()
        optimizer.step()
        return loss.item()
```

### Two-Phase Conditioning (Latency Optimisation)

**Problem:** WM observe_step on 60 tokens takes ~200ms. First token cannot wait.

**Solution:**
- **Phase 1 (token 0):** Serve `h_{t-1}` from previous stanza immediately. Logit bias is from last step.
- **Phase 2 (token ~30):** Fresh `h_t` is ready. Register new logits_processor mid-stream. From this token onward, WM state is current-stanza-conditioned.

```python
async def generate_stanza(prev_h: Tensor, obs_t: Tensor) -> AsyncIterator[str]:
    # Phase 1: immediate start with prev_h
    bridge_old = vimarsa_bridge.as_logits_processor(prev_h)
    llm_task = asyncio.create_task(llm.astream(prompt, logits_processor=[bridge_old]))

    # Phase 2: compute fresh h_t concurrently
    with torch.cuda.stream(stream_wm):
        z_t, h_t, vfe = wm.observe_step(obs_t)
        efe_score = efe_actor(h_t, z_t)
    torch.cuda.current_stream().wait_stream(stream_wm)
    bridge_new = vimarsa_bridge.as_logits_processor(h_t)

    # Swap at token 30
    token_count = 0
    async for token in llm_task:
        if token_count == 30:  # token 30 of this stanza (not global count)
            llm_task.update_logits_processor(bridge_new)  # hook swap
        yield token
        token_count += 1
```

### VimarsaBridgeV2 Training

- **Training data:** WM corpus (120 domains × seed phrases), run observe_step to collect (h_t, next_token) pairs
- **Training objective:** Next-token prediction cross-entropy (supervised)
- **Epochs:** 50 epochs on corpus, batch_size=256, lr=1e-4
- **Expected convergence:** Loss < 0.3 on held-out set
- **Checkpoint:** `checkpoints/vimarsa_bridge_v2.pt`

---

## Part 4 — Per-Stanza Pañcakṛtya Execution Model

### Loop Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PancakrtyaLoopV2                                       │
│                                                         │
│  for each stanza:                                       │
│    Act 1 (Cit):     z_t, h_t = wm.observe_step(obs)   │
│    Act 2 (Ānanda):  efe = efe_actor(h_t, z_t)          │
│    Act 3 (Icchā):   mem = citta.recall(z_t, top_k=5)   │
│    Act 4 (Apohana): sphuratta = entropy_gate(z_t)       │
│    Act 5 (Jñāna):   bias_fn = bridge.as_lp(h_t)        │
│    Act 6 (Kriyā):   tokens = llm.stream(prompt, bias_fn)│
│                                                         │
│    Update obs with generated stanza text                │
│    citta.store(z_t, stanza_text)                        │
│    emit SSE: camatk_score                               │
└─────────────────────────────────────────────────────────┘
```

### SSE Event Protocol (neo-fm-web API Contract)

Each generation emits a structured SSE stream. neo-fm-web must handle all event types:

```
event: wm_state
data: {"energy": 11.4, "aesthetic_quality": 0.73, "creative_peak": false, "register": "ardent", "epoch": 1}

event: stanza_start
data: {"stanza": 2, "scale": "Bhairavi", "section": "antara", "tempo": "medium"}

event: token
data: {"text": "बर", "token_id": 4821}

event: stanza_end
data: {"aesthetic_quality": 0.91, "memory_resonance": 0.64, "efe_score": -2.3, "vfe": 0.14}

event: complete
data: {"total_stanzas": 4, "mean_aesthetic_quality": 0.86, "creative_peaks": 2, "generation_ms": 4200}
```

**neo-fm-web integration notes:**
- `aesthetic_quality` = internal camatkāra score (domain-neutral label)
- `creative_peak` = sphurattā event (domain-neutral label)
- `memory_resonance` = Hopfield retrieval cosine similarity
- `scale` = music scale/rāga name (domain-specific, controlled by `domain` param)

### Domain Parameter Mapping

| `domain` param | Internal WM label | Output register | scale suggestion |
|---------------|-------------------|-----------------|------------------|
| `kannada_film` | `kf` | Tamil/Kannada emotional | Bhairavi, Charukeshi |
| `carnatic` | `ca` | Classical devotional | Todi, Bhairavi, Kharaharapriya |
| `hindi_film` | `hf` | Hindi romantic/filmi | Yaman, Bhupali |
| `english_pop` | `ep` | English contemporary | C major, Dorian |
| `jazz` | `jz` | English/scat jazz | Lydian, Mixolydian |
| `world_fusion` | `wf` | Multilingual fusion | Maqam, Pentatonic |

---

## Part 5 — Sprint Plan (Phases 3–6)

### Phase 3: WM on GPU + Real Conditioning (Sprints 8–11)

**Sprint 8 — WM CUDA + llama.cpp Server** (1 week)
- Move TrikaWorldModel to CUDA, verify shapes/gradients unchanged
- Locate Nemotron GGUF blobs in Ollama cache, export as single file
- Start llama-server with `--n-gpu-layers 999 --flash-attn`
- Install llama-cpp-python with CUDA build
- Update `engine.py`: `DEVICE = cuda`, replace `call_ollama()` with llama-cpp-python
- Gate metric: WM observe_step <10ms on GPU; llama-server first-token <500ms
- Phase gate: `benchmarks/results/sprint8_gate.json`

**Sprint 9 — PancakrtyaLoop wired to API** (1 week)
- Replace `engine.py` generate path with `PancakrtyaLoopV2`
- Wire Acts 1-4 (observe, EFE, recall, entropy gate)
- Acts 5-6 initially use text-prefix (VimarsaBridgeV1 fallback) — logits_processor next sprint
- SSE stream emits `wm_state`, `stanza_start`, `token`, `stanza_end`, `complete`
- Gate metric: API returns correct SSE events, PancakrtyaLoop runs for every request
- Phase gate: `benchmarks/results/sprint9_gate.json`

**Sprint 10 — EFEActor + CittaStore on GPU** (1 week)
- Wire EFEActor into PancakrtyaLoopV2 Act 2
- Wire CittaStore Hopfield recall into Act 3
- Install pymdp with JAX backend; test dlpack bridge on GB10 unified memory
- Gate metric: EFE score varies per domain/seed; recall cosine sim > 0.3 on held-out
- Phase gate: `benchmarks/results/sprint10_gate.json`

**Sprint 11 — VimarsaBridgeV2 (logits_processor)** (1.5 weeks)
- Train VimarsaBridgeV2 projection layer on WM corpus
- Integrate `as_logits_processor()` into Act 5 of PancakrtyaLoopV2
- Verify logit bias changes token distributions (KL-divergence > 0.05 vs no-bias)
- Two-phase conditioning (Phase 1: h_{t-1}, Phase 2: h_t swap at token 30)
- Gate metric: KL-div > 0.05; bridge checkpoint <100MB
- Phase gate: `benchmarks/results/sprint11_gate.json`

### Phase 4: Quality + Differentiation (Sprints 12–14)

**Sprint 12 — VFE Curriculum Training** (1 week)
- Train WM 5K steps VFE-only (no EFE), then anneal EFE with entropy bonus
- Goal: fix degenerate attractor (WM energy should vary 5-25 across domains)
- Verify: observe_step on "jazz" vs "carnatic" produces statistically different h_t
- Gate metric: mean pairwise cosine distance of domain h_t vectors > 0.4
- Phase gate: `benchmarks/results/sprint12_gate.json`

**Sprint 13 — EAGLE3 Speculative Decoding** (1 week)
- Train EAGLE3 draft head for Nemotron vocab
- Integrate with llama.cpp speculative decoding path
- Target: 2.5× decode speedup (from ~15 tok/s to ~37 tok/s)
- Gate metric: token/s improvement ≥2× measured on 10 generation runs
- Phase gate: `benchmarks/results/sprint13_gate.json`

**Sprint 14 — LiveViz Dual-Mode** (1 week)
- WebSocket endpoint `/ws/viz` emitting raw WM state tensors
- React component (or vanilla HTML) with dual-mode toggle:
  - Clean mode: animated quality score, creative peak glow, stanza progress bar
  - Research mode: EFE graph, VFE graph, latent t-SNE, Hopfield resonance heatmap
- Embed in GitHub Pages; link from neo-fm-web settings panel
- Gate metric: sub-50ms WebSocket latency on local network

### Phase 5: Product Polish + Paper (Sprints 15–17)

**Sprint 15 — neo-fm-web Integration** (1 week)
- Document SSE API contract (OpenAPI spec)
- CORS + auth token for neo-fm-web calls
- Production endpoint: `POST /v1/generate` with neo-fm schema
- Rate limiting, queue depth monitoring
- Test with actual neo-fm-web frontend (Vercel → DGX network path)
- Gate metric: end-to-end neo-fm-web request → lyrics card in UI, <5s first stanza

**Sprint 16 — H1–H9 Ablations + Paper Figures** (1.5 weeks)
- Run all 9 pre-registered hypotheses (≥3 seeds each)
- Generate TikZ/matplotlib figures for paper
- Update `paper/main.tex` with Phase 3-5 results
- PCA scatter plot of domain h_t embeddings (Sprint 12 gate figure)
- VimarsaBridgeV2 KL-div figure (Sprint 11 gate figure)
- Gate metric: latexmk clean build, zero undefined references

**Sprint 17 — arXiv Submission** (0.5 weeks)
- Final paper review (2 external readers)
- arXiv submission (cs.AI + cs.LG + cs.NE)
- GitHub release v1.0 with model checkpoints on HuggingFace
- Gate metric: arXiv submission ID confirmed

### Phase 6: Commercial Hardening (Sprint 18+)

**Sprint 18 — Production API Hardening**
- Structured logging (request/response with WM state snapshot)
- Health endpoint `/health` with GPU utilisation, queue depth, WM energy
- Auto-restart on CUDA OOM or llama.cpp crash
- Docker/systemd packaging for reliable startup
- SLA target: 99% of requests complete within 10s, p99 latency <15s

---

## Part 6 — Product Differentiators (Commercialisation)

PWM's creative API has genuine technical differentiation from existing music AI:

| Differentiator | PWM | Competitors |
|---------------|-----|-------------|
| World-model-conditioned generation | Token-level logit bias from GPU WM state | Prompt engineering only |
| Intrinsic aesthetic reward (camatkāra) | Real-time EFE + Hopfield resonance score | Post-hoc quality rating |
| Per-stanza iterative refinement | Pañcakṛtya 6-act loop per stanza | Single LLM pass |
| Episodic memory coherence | CittaStore Hopfield recall across stanzas | Stateless generation |
| Multilingual domain steering | LoRA domain adapters (15 domains) | Few-shot prompting |
| Speculative decoding | EAGLE3 draft head (2.5× speed) | Standard autoregressive |

### neo-fm-web API Contract

**Base URL:** `http://[DGX-IP]:8000` (production: reverse proxy to HTTPS)

**Primary endpoint:**
```
POST /v1/generate
Content-Type: application/json
X-API-Key: <token>

{
  "domain": "kannada_film",
  "seed": "ಮಳೆಯೊಳಗೆ",
  "n_stanzas": 4,
  "language": "kn",
  "style": "romantic",
  "stream": true
}
```

**Response (SSE):** See Part 4 SSE Event Protocol above.

**Secondary endpoints:**
- `GET /v1/domains` — list available domains with metadata
- `GET /v1/health` — system health, GPU state, queue depth
- `GET /v1/examples` — curated example gallery (links to GitHub Pages)
- `WebSocket /ws/viz` — real-time WM visualisation stream

---

## Part 7 — Constraints and Failure Modes

### Handled Failures

| Failure | Handling |
|---------|---------|
| llama.cpp OOM | Fallback to Q3_K_M quantisation; retry once |
| WM CUDA OOM | WM fits in ~0.3GB; this should not occur |
| EFE divergence | Clip EFE to [-10, 10]; log warning; continue |
| Hopfield recall timeout | Skip Act 3, continue with zero mem tensor |
| logits_processor exception | Disable bias_fn for stanza; log; restore next stanza |
| neo-fm-web network error | SSE connection closes; client reconnects with `Last-Event-ID` |

### Known Limitations (Paper Limitations Section)

- WM trained on ~120 seed phrases per domain — corpus is small; domain separation is model-level, not dataset-level
- VimarsaBridgeV2 is a linear projection — not cross-attention (higher capacity, deferred to Sprint 11 or later)
- EFE computed with pymdp.jax on CPU (dlpack bridge); ~5ms overhead per stanza
- No acoustic model — neo-fm-web must supply its own music synthesis from lyrics

---

## Appendix A — Files to Create / Modify

| File | Action | Sprint |
|------|--------|--------|
| `pwm/generation/engine.py` | Modify: CUDA device, llama-cpp-python backend | S8 |
| `pwm/pipeline/pancakrtya_loop.py` | Modify: PancakrtyaLoopV2, wire all 6 acts | S9 |
| `pwm/active_inference/efe_actor.py` | Modify: wire into loop, return efe_score | S10 |
| `pwm/memory/citta_store.py` | Modify: ensure GPU tensors, add recall() return | S10 |
| `pwm/vimarsa/bridge.py` | Add: VimarsaBridgeV2 class, as_logits_processor() | S11 |
| `pwm/scripts/train_vimarsa_bridge.py` | Create: training loop for bridge projection | S11 |
| `pwm/scripts/train_wm_curriculum.py` | Create: VFE-first then EFE curriculum | S12 |
| `api/main.py` | Modify: SSE protocol, PancakrtyaLoopV2, health endpoint | S9 |
| `api/ws_viz.py` | Create: WebSocket WM visualisation endpoint | S14 |
| `configs/phase3_production.yaml` | Create: CUDA device, llama-cpp params | S8 |
| `scripts/start_llama_server.sh` | Create: llama-server launch with CUDA flags | S8 |
| `paper/main.tex` | Update: Phase 3-5 results, new figures | S16 |

## Appendix B — TRIZ Resolved Contradictions

Three contradictions were identified and resolved via TRIZ analysis:

**C1 — Philosophical structure vs content neutrality:**  
Resolution: CamatkaraNarrator hard boundary (Principle 3, Local Quality). Śaiva structure is internal computation; output layer is domain-neutral translation. Both layers have local quality without interference.

**C2 — Streaming latency vs generation quality:**  
Resolution: Two-phase conditioning (Principle 10, Prior Action). Serve h_{t-1} immediately; compute h_t asynchronously; swap logits_processor at token 30. First-token latency and WM conditioning are separated in time.

**C3 — Multilingual coverage vs WM corpus specificity:**  
Resolution: LoRA domain adapters (Principle 3, Local Quality). Each domain trains independent (A,B) pair. WM base geometry preserved; domain adaptation is additive, not destructive. Already implemented in Sprint 6 with Fisher ratio=3.10.
