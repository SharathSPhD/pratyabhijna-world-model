"""
PWM Creative Generation API — FastAPI backend.

Endpoints:
  POST /generate          — Request creative generation (returns job_id)
  GET  /stream/{job_id}  — SSE stream of tokens as generated (TRIZ C2: P28/P32)
  GET  /result/{job_id}  — Full structured result when complete
  POST /refine/{job_id}  — Submit feedback, regenerate with adjusted WM state
  GET  /health            — Health check + model/checkpoint status
  GET  /domains           — List available creative domains
  POST /batch             — Submit up to 10 specs, returns batch_id
  GET  /batch/{batch_id} — Poll batch status

Architecture (TRIZ C2 resolution — Streaming Latency vs Quality):
  - WM warm-up runs in a background asyncio task (Principle 28: Mechanics Substitution)
  - SSE emits wm_status events while warm-up runs, then content tokens follow
  - First SSE event arrives in <200ms; WM conditioning completes in 2-5s
  - Client sees progress from t=0ms regardless of WM warm-up time

Run on DGX:
  uvicorn pwm_api.main:app --host 0.0.0.0 --port 8000 --workers 1 --reload
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add pwm-phase2 to path for WM imports
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, "/home/sharaths/projects/pwm-phase2")

from pwm.generation.domain_metadata import Domain, WMStateDecoder  # type: ignore
from pwm.generation.engine import (  # type: ignore
    CHECKPOINT, MODEL, OLLAMA_URL,
    load_wm, score_camatk, warmup_wm_on_text,
)
from pwm.generation.creative_specs import ALL_SPECS  # type: ignore

# ─── FastAPI App ─────────────────────────────────────────────────────────────

# ─── Global State ─────────────────────────────────────────────────────────────

class AppState:
    wm = None
    decoder: WMStateDecoder | None = None
    jobs: dict[str, dict] = {}
    batches: dict[str, dict] = {}
    wm_loading: bool = False
    wm_ready: bool = False

state = AppState()


async def _load_wm_background() -> None:
    """Background task: loads WM without blocking API startup."""
    state.wm_loading = True
    try:
        loop = asyncio.get_event_loop()
        state.wm = await loop.run_in_executor(None, load_wm)
        state.decoder = WMStateDecoder()
        state.wm_ready = True
        print("✓ PWM world model loaded and ready")
    except Exception as e:
        print(f"✗ WM load failed: {e}")
        state.wm_ready = False
    finally:
        state.wm_loading = False


@asynccontextmanager
async def lifespan(app_: FastAPI):  # type: ignore[type-arg]
    """FastAPI lifespan — starts WM loading immediately on boot."""
    asyncio.create_task(_load_wm_background())
    yield
    # Shutdown cleanup (none needed for stateless WM)


app = FastAPI(
    title="PWM Creative Generation API",
    description=(
        "Pratyabhijñā World Model — multilingual creative generation backend. "
        "Streams poems, songs, and lyrics via SSE. Powered by PWM RSSM world model "
        "+ Nemotron-3-Super 120B on NVIDIA GB10 DGX Spark."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production (Vercel domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request body for POST /generate."""
    domain: Domain = Field("generic", description="Creative domain")
    language: str = Field("english", description="Target language")
    theme: str = Field("", description="Optional theme or subject")
    style: str = Field("", description="Optional style descriptor")
    music_context: dict = Field(default_factory=dict,
                                description="Optional music context: rāga, tāla, key, mode")
    num_predict: int = Field(900, ge=100, le=2000)
    temperature: float = Field(0.88, ge=0.0, le=2.0)
    top_p: float = Field(0.92, ge=0.0, le=1.0)
    seed_text: str = Field("", description="Optional seed text for WM warmup")
    spec_id: str = Field("", description="Use a predefined spec by ID (overrides other fields)")

    class Config:
        json_schema_extra = {
            "example": {
                "domain": "western_jazz",
                "language": "english",
                "theme": "late night rain on city streets",
                "style": "Coltrane, four-movement, call-and-response",
                "music_context": {"mode": "Dorian", "key": "D minor", "tempo": "slow"},
            }
        }


class RefineRequest(BaseModel):
    """Feedback for POST /refine/{job_id}."""
    feedback: str = Field(..., description="Human feedback on the previous generation")
    strength: float = Field(0.5, ge=0.0, le=1.0,
                            description="How much to adjust (0=subtle, 1=full regeneration)")
    preserve_wm_state: bool = Field(True,
                                    description="Keep same WM h_t (same creative state)")


class BatchRequest(BaseModel):
    """Request body for POST /batch."""
    specs: list[GenerateRequest] = Field(..., min_length=1, max_length=10)


# ─── Build Prompt from GenerateRequest ───────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a master poet, lyricist, and composer fluent in classical and contemporary "
    "creative traditions across cultures. You write directly in the requested form: "
    "no preamble, no explanation, no meta-commentary, no reasoning traces. "
    "Begin the creative work immediately on the first line. "
    "When given a [Creative state: ...] prefix, use it to set emotional register, "
    "pace, and section structure for the piece."
)


def _build_user_prompt(req: GenerateRequest, prefix: str) -> str:
    """Build domain-appropriate user prompt from request."""
    parts = [prefix]

    if req.theme:
        parts.append(f"Theme: {req.theme}.")
    if req.style:
        parts.append(f"Style: {req.style}.")

    # Domain-specific structural instructions
    domain_instructions = {
        "carnatic": (
            "Write a Carnatic kṛti: Pallavi (2 lines) + Anupallavi (2 lines) + "
            "Caraṇam (4 lines). Include tāla beat numbers. "
        ),
        "hindustani": (
            "Write a Hindustani composition: Sthāyi (4 lines) + Antarā (4 lines). "
            "Include rāga name, tāla, laya. "
        ),
        "western_pop": (
            "Write a pop song: Intro (2 lines) + Verse 1 (4 lines) + "
            "Chorus (4 lines) + Verse 2 (4 lines) + Bridge (2 lines) + Chorus repeat. "
        ),
        "western_jazz": (
            "Write a jazz poem: 4 movements (I–IV), each 6–8 lines. "
            "Use jazz vocabulary: chord, resolution, blue note, drone, overtone. "
        ),
        "kannada_film": (
            "Write a Kannada film song: Mukhara (2 lines) + Charaṇa 1 (4 lines) "
            "+ Charaṇa 2 (4 lines). Provide Kannada + English meaning in parentheses. "
        ),
        "hindi_film": (
            "Write a Hindi film song: Mukhra (2 lines) + Antara 1 (4 lines) "
            "+ Antara 2 (4 lines). Write in Devanāgarī. "
        ),
        "sanskrit_classical": (
            "Write an anuṣṭubh śloka (4 verses, 8 syllables each quarter). "
            "Write Devanāgarī with IAST transliteration. "
        ),
        "english_romantic": (
            "Write an ode (4 stanzas × 8 lines) on a single sustained subject. "
        ),
        "english_modernist": (
            "Write a fragmented poem (4 sections I–IV, 6–8 lines each). "
        ),
        "english_beat": (
            "Write a Beat poem (40–60 long breath lines) with jazz rhythms. "
        ),
        "bengali_lyric": (
            "Write a Bengali lyric poem (3 stanzas × 5 lines) in Bengali script. "
        ),
        "tamil_classical": (
            "Write a Tamil poem (8–12 lines) in Tamil script. "
        ),
        "telugu_padyam": (
            "Write a Telugu padyamu (4 stanzas) in Telugu script. "
        ),
        "world_fusion": (
            "Write a multilingual poem (one stanza per language, 5 languages). "
        ),
    }

    instruction = domain_instructions.get(req.domain, "Write a poem (20–40 lines). ")
    parts.append(instruction)

    if req.music_context:
        ctx_str = ", ".join(f"{k}={v}" for k, v in req.music_context.items())
        parts.append(f"Music context: {ctx_str}.")

    parts.append(
        f"Language: {req.language}. "
        "Begin immediately with the first line of the creative work."
    )

    return " ".join(parts)


# ─── SSE Stream Generator ─────────────────────────────────────────────────────

async def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _generation_stream(job_id: str, req: GenerateRequest) -> AsyncGenerator[str, None]:
    """
    SSE stream generator implementing TRIZ C2 solution:
    1. Immediately yield wm_status events while WM warms up
    2. Yield content tokens as the LLM generates them
    3. Yield result event when complete

    Client sees: wm_warming → wm_ready → token stream → result
    """
    job = state.jobs[job_id]
    job["status"] = "warming"

    # ── Phase 1: WM warm-up ──────────────────────────────────────────────────
    yield await _sse_event("wm_status", {
        "stage": "warming",
        "message": "World model warming up...",
        "pct": 0,
        "job_id": job_id,
    })

    # Run WM warm-up in executor (non-blocking)
    loop = asyncio.get_event_loop()

    h_t = None
    meta = None
    prefix = ""

    if state.wm_ready and state.wm is not None:
        try:
            seed = req.seed_text or req.theme or req.style or "creative music poetry"
            for pct in range(20, 100, 20):
                yield await _sse_event("wm_status", {
                    "stage": "warming",
                    "pct": pct,
                    "message": f"WM warm-up {pct}%...",
                })
                await asyncio.sleep(0)  # yield control

            h_t = await loop.run_in_executor(
                None, warmup_wm_on_text, state.wm, seed, 60
            )
            meta = state.decoder.decode(h_t, domain=req.domain,  # type: ignore
                                         step=hash(job_id) % 100,
                                         spec_id=job_id)
            prefix = state.decoder.format_for_llm(meta)  # type: ignore

            job["wm_energy"] = round(meta.energy, 4)
            job["wm_register"] = meta.register
            job["wm_section"] = meta.section_name
            job["wm_prefix"] = prefix

            yield await _sse_event("wm_status", {
                "stage": "ready",
                "pct": 100,
                "message": f"WM ready — register={meta.register}, section={meta.section_name}",
                "energy": round(meta.energy, 3),
                "register": meta.register,
                "section": meta.section_name,
            })

        except Exception as e:
            yield await _sse_event("wm_status", {
                "stage": "skipped",
                "message": f"WM unavailable ({e}); proceeding with base generation",
            })
    else:
        yield await _sse_event("wm_status", {
            "stage": "loading",
            "message": "WM still loading; proceeding without WM conditioning",
        })

    # ── Phase 2: Streaming LLM generation ────────────────────────────────────
    job["status"] = "generating"
    yield await _sse_event("generation_start", {
        "message": "Generating...",
        "job_id": job_id,
        "domain": req.domain,
        "language": req.language,
    })

    user_prompt = _build_user_prompt(req, prefix)

    try:
        # Use Ollama streaming endpoint via async queue (non-blocking event loop).
        # The sync requests.iter_lines() runs in a thread; tokens are pushed to
        # an asyncio.Queue so the async generator can yield them without blocking.
        full_text: list[str] = []
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": True,
            "think": False,
            "options": {
                "num_predict": req.num_predict,
                "temperature": req.temperature,
                "top_p": req.top_p,
            },
        }

        def _stream_worker(q: asyncio.Queue, ev_loop: asyncio.AbstractEventLoop) -> None:
            """Blocking Ollama stream — runs in a thread pool."""
            try:
                resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=360)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        tok = chunk.get("message", {}).get("content", "")
                        if tok:
                            asyncio.run_coroutine_threadsafe(q.put(tok), ev_loop)
                        if chunk.get("done", False):
                            break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    q.put(f"\n[stream error: {exc}]"), ev_loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), ev_loop)  # sentinel

        loop.run_in_executor(None, _stream_worker, token_queue, loop)

        token_count = 0
        while True:
            token = await asyncio.wait_for(token_queue.get(), timeout=360.0)
            if token is None:
                break
            full_text.append(token)
            token_count += 1
            yield await _sse_event("token", {
                "token": token,
                "n": token_count,
            })

        text = "".join(full_text)
        job["text"] = text

        # ── Phase 3: Score and finalise ──────────────────────────────────────
        scores = {}
        if meta is not None:
            scores = score_camatk(text, meta, req.domain)
        else:
            wc = len(text.split())
            scores = {
                "camatk_total": min(1.0, wc / 120.0),
                "word_count": wc,
                "unique_words": len(set(text.lower().split())),
                "note": "WM not available; VFE score omitted",
            }

        job.update({
            "status": "complete",
            "scores": scores,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "token_count": token_count,
            "music_context": req.music_context,
        })

        yield await _sse_event("result", {
            "job_id": job_id,
            "text": text,
            "scores": scores,
            "wm_register": job.get("wm_register", ""),
            "wm_section": job.get("wm_section", ""),
            "music_context": req.music_context,
            "domain": req.domain,
            "language": req.language,
            "generated_at": job["generated_at"],
        })

        yield await _sse_event("done", {"job_id": job_id, "status": "complete"})

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        yield await _sse_event("error", {"job_id": job_id, "error": str(e)})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Health check — returns model and WM status."""
    ollama_ok = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "wm_ready": state.wm_ready,
        "wm_loading": state.wm_loading,
        "ollama_ok": ollama_ok,
        "model": MODEL,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/domains")
async def list_domains() -> dict:
    """List available creative domains and their musical contexts."""
    from pwm.generation.domain_metadata import SECTION_LABELS, RAGA_BY_REGISTER
    return {
        "domains": list(SECTION_LABELS.keys()),
        "example_ragas": RAGA_BY_REGISTER,
        "predefined_specs": [
            {"id": s.id, "title": s.title, "language": s.language, "domain": s.domain}
            for s in ALL_SPECS
        ],
    }


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """
    Submit a generation request. Returns job_id immediately.
    Use GET /stream/{job_id} for SSE stream.
    """
    # If spec_id provided, use predefined spec's settings
    if req.spec_id:
        spec = next((s for s in ALL_SPECS if s.id == req.spec_id), None)
        if spec:
            req.domain = spec.domain
            req.language = spec.language
            if not req.theme:
                req.theme = spec.user_prompt[:100]

    job_id = str(uuid.uuid4())
    state.jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "request": req.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "job_id": job_id,
        "stream_url": f"/stream/{job_id}",
        "result_url": f"/result/{job_id}",
        "status": "queued",
    }


@app.get("/stream/{job_id}")
async def stream_generation(job_id: str) -> StreamingResponse:
    """
    SSE stream for a generation job.
    Events: wm_status, generation_start, token, result, done, error.

    TRIZ C2 (P28 + P32): WM warm-up runs asynchronously; SSE emits progress
    events so client perceives activity from t=0ms.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = state.jobs[job_id]
    req_data = job["request"]
    req = GenerateRequest(**req_data)

    return StreamingResponse(
        _generation_stream(job_id, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/result/{job_id}")
async def get_result(job_id: str) -> dict:
    """Get the full result of a completed job."""
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = state.jobs[job_id]
    if job["status"] not in ("complete", "error"):
        return {"job_id": job_id, "status": job["status"], "message": "Still generating"}
    return job


@app.post("/refine/{job_id}")
async def refine(job_id: str, req: RefineRequest) -> dict:
    """
    Submit feedback on a previous generation. Adjusts WM state and regenerates.

    TRIZ C1 (P34 — Discarding): the old generation is discarded; a new job
    is created with adjusted temperature and the feedback prepended to the prompt.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    old_job = state.jobs[job_id]
    old_req_data = old_job["request"]

    # Build refined request
    new_req = GenerateRequest(**old_req_data)
    # Incorporate feedback into theme
    feedback_prefix = f"Previous version had this feedback: '{req.feedback}'. "
    new_req.theme = feedback_prefix + new_req.theme
    # Adjust temperature based on strength
    new_req.temperature = min(2.0, new_req.temperature + req.strength * 0.3)
    # Keep same seed_text if preserve_wm_state
    if req.preserve_wm_state:
        new_req.seed_text = old_req_data.get("seed_text", "")

    new_job_id = str(uuid.uuid4())
    state.jobs[new_job_id] = {
        "id": new_job_id,
        "status": "queued",
        "request": new_req.model_dump(),
        "parent_job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "job_id": new_job_id,
        "parent_job_id": job_id,
        "stream_url": f"/stream/{new_job_id}",
        "result_url": f"/result/{new_job_id}",
        "status": "queued",
    }


@app.post("/batch")
async def submit_batch(req: BatchRequest) -> dict:
    """Submit up to 10 generation specs as a batch."""
    batch_id = str(uuid.uuid4())
    job_ids = []
    for gen_req in req.specs:
        job_id = str(uuid.uuid4())
        state.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "request": gen_req.model_dump(),
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        job_ids.append(job_id)

    state.batches[batch_id] = {
        "id": batch_id,
        "job_ids": job_ids,
        "n_jobs": len(job_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "n_jobs": len(job_ids),
        "stream_urls": [f"/stream/{jid}" for jid in job_ids],
        "status_url": f"/batch/{batch_id}",
    }


@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str) -> dict:
    """Poll batch completion status."""
    if batch_id not in state.batches:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    batch = state.batches[batch_id]
    jobs = [state.jobs.get(jid, {}) for jid in batch["job_ids"]]
    statuses = [j.get("status", "unknown") for j in jobs]
    n_complete = sum(1 for s in statuses if s == "complete")
    n_error = sum(1 for s in statuses if s == "error")
    return {
        "batch_id": batch_id,
        "n_jobs": batch["n_jobs"],
        "n_complete": n_complete,
        "n_error": n_error,
        "n_pending": batch["n_jobs"] - n_complete - n_error,
        "all_done": n_complete + n_error == batch["n_jobs"],
        "job_statuses": [
            {"job_id": jid, "status": state.jobs.get(jid, {}).get("status", "unknown")}
            for jid in batch["job_ids"]
        ],
    }
