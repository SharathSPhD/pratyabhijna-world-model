# PWM API — Startup Guide

## Prerequisites

- NVIDIA DGX box with CUDA environment
- Virtual environment at `/home/sharaths/vllm-env`
- Nemotron-3-Super 120B GGUF at `/home/sharaths/projects/pwm-phase3/models/nemotron-120b.gguf`
- Checkpoints at `/home/sharaths/projects/pwm-phase2/checkpoints/` and `/home/sharaths/projects/pwm-phase4/checkpoints/`
- Ollama installed (serves the 120B model)

---

## Step 1 — Activate Environment

```bash
source /home/sharaths/vllm-env/bin/activate
```

---

## Step 2 — Start the LLM Backend (Ollama)

```bash
bash /home/sharaths/projects/PWM/scripts/start_llama_server.sh
```

This starts Ollama at `http://localhost:11434` serving `nemotron-3-super:120b`.

Verify it's up:
```bash
curl http://localhost:11434/api/tags
```

Expected: JSON listing `nemotron-3-super:120b` in the models list.

---

## Step 3 — Start the PWM API

From the PWM project root:

```bash
cd /home/sharaths/projects/PWM
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Do not use `--reload` in production** — it reloads on file changes and destroys the pre-warmed CUDA state.

### Startup sequence (automatic)

The API self-initialises in the background on first start:

1. Loads RSSM world model from `pwm-phase2/checkpoints/step_1000000.pt`
2. Loads multilingual variant from `step_multilingual.pt`
3. Loads LoRA domain adapters from `lora_final.pt`
4. Loads VimarsaBridge from `pwm-phase4/checkpoints/vimarsa_bridge_v2.pt`
5. Pre-warms 6 domain states in parallel (kannada_film, hindi_film, carnatic, english_pop, english_romantic, world_fusion)
6. Sets `wm_ready = true` (~3–5 seconds on CUDA)

### Verify startup is complete

```bash
curl http://localhost:8000/v1/health
```

Expected response when ready:
```json
{
  "wm_ready": true,
  "domains_prewarmed": ["kannada_film", "hindi_film", "carnatic", "english_pop", "english_romantic", "world_fusion"],
  "llama_server_ok": true,
  "ttft_profile": { "prewarmed_ms": 0, "seed_ms": 67, "cold_ms": 5247 }
}
```

Do not send generation requests until `wm_ready` is `true`.

---

## Step 4 — Test a Generation

```bash
curl -N -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "kannada_film",
    "seed": "ಮಳೆಯ ಹನಿ",
    "n_stanzas": 2,
    "language": "kn",
    "stream": true
  }'
```

Expect an SSE stream with events: `wm_state` → `token` → `stanza_end` → `complete`.

---

## Environment Variables

All paths are currently hardcoded in `api/engine.py`. Optional overrides via `.env` (copy from `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `nemotron-local` | LLM backend selection |
| `LLM_PRIMARY_API_BASE` | `http://localhost:11434/v1` | Ollama endpoint |
| `LLM_FAST_API_BASE` | `http://localhost:8001/v1` | Cascade model (optional) |
| `LLM_PRIMARY_API_KEY` | `local` | No-op for local inference |
| `WANDB_PROJECT` | `pratyabhijna-world-model` | W&B tracking (optional) |
| `HF_HOME` | `/home/sharaths/models` | HuggingFace cache |

---

## Port Reference

| Port | Service |
|------|---------|
| `8000` | PWM API (FastAPI/Uvicorn) |
| `11434` | Ollama (Nemotron-3-Super 120B) |
| `8080` | llama.cpp fallback (optional) |

---

## Shutdown

`Ctrl-C` in the Uvicorn terminal. CUDA tensors are released immediately on exit (Issue-2 fix: explicit `del loop` after event exhaustion).

To also stop Ollama:
```bash
pkill ollama
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `wm_ready: false` after 10s | Checkpoint path wrong or CUDA OOM | Check paths in `engine.py` lines 43–49; check `nvidia-smi` |
| `llama_server_ok: false` | Ollama not running or model not loaded | Re-run step 2; check `curl localhost:11434/api/tags` |
| SSE stream hangs at `wm_state` | LLM not responding | Verify Ollama health; check if 120B model is fully loaded |
| CORS errors from neo-fm-web | Origins not allowed | Tighten `allow_origins` in `api/main.py` line 199 |
