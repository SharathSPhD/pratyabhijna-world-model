# PWM API — Project Status

*Updated: 2026-05-19*

---

## What It Is

The **Pratyabhijñā World Model (PWM)** is a creative-generation backend for multilingual poetry and song lyrics. It pairs a Kashmir Śaiva-inspired **Recurrent State Space Model (RSSM)** world model with a frozen **Nemotron-3-Super 120B** LLM (via Ollama) to steer text generation through aesthetic and musical structure rather than pure token prediction.

The API wrapper (`api/main.py`) exposes this system to neo-fm and other callers via a FastAPI/SSE interface.

---

## Architecture

```
Caller (neo-fm-web or curl)
        │
        ▼ POST /v1/generate  (SSE)
┌──────────────────────────────────┐
│          FastAPI API             │
│  (api/main.py, port 8000)        │
│                                  │
│  PancakrtyaLoopV2 ──────────────►│──► Ollama (localhost:11434)
│  └─ Phase 1: WM warm-up (60 steps│    Nemotron-3-Super 120B
│  └─ Phase 2: LLM token streaming │
│  └─ Phase 3: scoring + finalise  │
│                                  │
│  AppState (in-memory)            │
│  └─ jobs dict (not persisted)    │
└──────────────────────────────────┘
        │ events (SSE)
        ▼
  wm_state → token → stanza_end → complete
```

### Core Modules

| Module | Path | Role |
|--------|------|------|
| **World Model (RSSM)** | `pwm/world_model/` | 3-level Trika RSSM, 512-dim hidden state |
| **Active Inference** | `pwm/active_inference/` | EFE actor (Expected Free Energy), CRSPP planner |
| **Memory** | `pwm/memory/` | CittaStore (episodic), Hopfield memory |
| **Sleep consolidation** | `pwm/sleep/` | NREM/REM replay for offline learning |
| **VimarsaBridge** | `pwm/vimarsa/` | World model ↔ LLM context bridge |
| **Rewards** | `pwm/rewards/` | Camatkāra aesthetic reward, Mala regularisers |
| **Generation engine** | `pwm/generation/` | PancakrtyaLoopV2, domain metadata, creative specs |
| **Pipeline** | `pwm/pipeline/` | v1 end-to-end loop |
| **Perception** | `pwm/perception/` | Text embedding, V-JEPA encoder |

### Checkpoints (required at runtime)

| Checkpoint | Path | Purpose |
|-----------|------|---------|
| `step_1000000.pt` | `pwm-phase2/checkpoints/` | Main RSSM world model |
| `step_multilingual.pt` | `pwm-phase2/checkpoints/` | Multilingual RSSM variant |
| `lora_final.pt` | `pwm-phase2/checkpoints/` | LoRA domain adapters |
| `vimarsa_bridge_v2.pt` | `pwm-phase4/checkpoints/` | VimarsaBridge trained weights |

---

## API Endpoints

### Canonical (neo-fm contract)

| Method | Route | Notes |
|--------|-------|-------|
| `POST` | `/v1/generate` | SSE stream via `PancakrtyaLoopV2`. Primary endpoint for neo-fm. |
| `GET` | `/v1/health` | Health check; returns `wm_ready`, `domains_prewarmed`, `llama_server_ok`, `ttft_profile`. neo-fm polls this before enabling the Generate button. |
| `WS` | `/v1/ws/generate` | WebSocket alternative (Sprint 15); implemented, stability testing may be needed. |

### Legacy (functional but not preferred)

| Method | Route | Notes |
|--------|-------|-------|
| `POST` | `/generate` | Legacy; uses WM-derived prompt prefix only |
| `GET` | `/stream/{job_id}` | SSE stream (v0 protocol) |
| `GET` | `/result/{job_id}` | Poll for completed result |
| `POST` | `/refine/{job_id}` | Feedback → adjusted WM state → regen |
| `POST` | `/batch` | Submit ≤10 jobs, returns `batch_id` |
| `GET` | `/batch/{batch_id}` | Poll batch |
| `GET` | `/domains` | List creative domains + predefined specs |
| `GET` | `/health` | Legacy health check |

### SSE Event Contract (enforced whitelist)

Only these keys are allowed out per event type — Śaiva vocabulary never leaks to callers:

| Event | Allowed keys |
|-------|-------------|
| `wm_state` | `energy`, `aesthetic_quality`, `creative_peak`, `entropy`, `prediction_error`, `stanza` |
| `token` | `text` |
| `stanza_end` | `stanza`, `aesthetic_quality`, `memory_resonance`, `selection_score`, `prediction_error` |
| `complete` | `total_stanzas`, `mean_aesthetic_quality`, `creative_peaks`, `generation_complete` |

---

## Supported Domains (15+)

`kannada_film`, `hindi_film`, `carnatic`, `hindustani`, `english_pop`, `english_jazz`, `sanskrit_classical`, `english_romantic`, `english_modernist`, `english_beat`, `bengali_lyric`, `tamil_classical`, `telugu_padyam`, `world_fusion`, `generic`

**Pre-warmed on startup** (zero TTFT): `kannada_film`, `hindi_film`, `carnatic`, `english_pop`, `english_romantic`, `world_fusion`

---

## Performance Profile

| Path | TTFT |
|------|------|
| Pre-warmed domain | 0 ms |
| Seed-specific warm | ~67 ms |
| Cold start (new domain) | ~5,247 ms |

---

## Sprint History

| Sprint | Work | Status |
|--------|------|--------|
| S1–S5 | RSSM world model, active inference, memory | Done |
| S6–S10 | VimarsaBridge, reward model, corpus ingestion | Done |
| S11–S13 | Multilingual support, LoRA adapters, domain specialisation | Done |
| S14 | Pre-warmed singleton states (eliminates cold-start for 6 domains) | Done |
| S15 | LiveViz WebSocket (`/v1/ws/generate`) + neo-fm SSE contract v2 | Done |
| S16 | End-to-end Ollama + PancakrtyaLoopV2 integration test | Done |
| S17 | Cascade model (Nemotron-Mini 4B → Super 120B) for fast TTFT | Done |
| S18 | Streaming robustness (Issue-5 fix: `put_nowait` via `call_soon_threadsafe`) | Done |
| S19 | CUDA tensor release fix (Issue-2: explicit `del loop`) | Done |
| S20 | Full pipeline E2E validation + production hardening | Done |

---

## Known Limitations & Pending Work

### Research Gaps

1. **H5b ablation result**: Live text superiority test shows PWM < bare 120B (effect size g = −0.47) on English-script domains. Near-parity on Kannada. Root cause: WM conditioning may be counterproductive for English creative tasks with no Indic musical structure. No fix committed.

2. **TTFT measurement confounds**: Aggregate cold-start/Ollama startup confounds make TTFT benchmarks unreliable. Warm-path measurements are clean but cold-start figures need a dedicated isolation test harness.

3. **No human evaluation**: Camatkāra (aesthetic reward) proxies are used instead of human ratings. No study designed or run yet.

### Engineering Debt

| Issue | Location | Impact |
|-------|----------|--------|
| Hardcoded checkpoint paths | `api/engine.py` lines 43–49 | Breaking if directory structure changes; no env-var override |
| In-memory job state | `AppState.jobs` dict | All in-flight job state lost on restart; no persistence |
| CORS allows all origins | `api/main.py` line 199 | Should restrict to `neo-fm.vercel.app` for production |
| Legacy endpoints untested | `/generate`, `/refine`, `/batch` | May drift from canonical `/v1/generate` behaviour |
| WebSocket path stability | `/v1/ws/generate` | Implemented in Sprint 15 but may need production stress testing |

### Immediate Next Actions

- [ ] Restrict CORS to `neo-fm.vercel.app` (one-line change, `main.py:199`)
- [ ] Replace hardcoded checkpoint paths with env-var overrides in `engine.py`
- [ ] Add persistence layer (Redis or Supabase) for in-memory job state
- [ ] Design human evaluation protocol for camatkāra calibration
- [ ] Run clean TTFT benchmark with warm-path isolation

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Web framework | FastAPI ≥ 0.115 + Uvicorn |
| Data validation | Pydantic ≥ 2.0 |
| ML runtime | PyTorch ≥ 2.10 (CUDA) |
| Active inference | `inferactively-pymdp` ≥ 0.0.7 |
| LLM backend | Ollama @ localhost:11434 (`nemotron-3-super:120b`) |
| LLM interface | `litellm` ≥ 1.50, `smolagents` ≥ 1.9 |
| Memory / embedding | `faiss-cpu`, `sentence-transformers` |
| Config | Hydra-core + OmegaConf |
| Experiment tracking | W&B + MLflow (optional) |
| Storage | None (in-memory AppState + `.pt` checkpoints) |
| Database | None |

---

## Test Coverage

20 sprint-based test suites in `tests/`:

```
test_sprint14_prewarm_singleton.py      S14 pre-warmed domain state tests
test_sprint15_liveviz_ws.py             WebSocket contract + neo-fm integration
test_sprint16_e2e_integration.py        Ollama + PancakrtyaLoopV2 end-to-end
test_sprint18_cascade_streaming.py      Cascade model (Mini → Super 120B)
test_sprint20_pipeline_e2e.py           Full pipeline validation
... + 15 others (S1–S13, S17, S19)
```

Run all:
```bash
cd /home/sharaths/projects/PWM
source /home/sharaths/vllm-env/bin/activate
pytest tests/ -v
```
