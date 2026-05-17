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
from typing import Any, AsyncGenerator

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
    CHECKPOINT, LLAMA_MODEL_PATH as MODEL, LLAMA_SERVER_URL as OLLAMA_URL,
    OLLAMA_MODEL_NAME,
    load_wm, load_trained_components, score_camatk, warmup_wm_on_text,
    get_llm_backend,
)
from pwm.generation.creative_specs import ALL_SPECS  # type: ignore
from pwm.generation.music_notation import annotate as annotate_music  # type: ignore
from pwm.generation.transliterate import annotate_output as add_transliteration  # type: ignore  # Sprint 5

# ─── FastAPI App ─────────────────────────────────────────────────────────────

# ─── Global State ─────────────────────────────────────────────────────────────

class AppState:
    wm = None
    efe = None
    citta = None
    bridge = None
    decoder: WMStateDecoder | None = None
    jobs: dict[str, dict] = {}
    batches: dict[str, dict] = {}
    wm_loading: bool = False
    wm_ready: bool = False
    # S14: pre-warmed WM states per domain — eliminates per-request warmup latency
    # Keyed by domain string; value is h_t Tensor (hidden_dim,)
    domain_states: dict[str, Any] = {}

state = AppState()


def _broadcast(job: dict, event_str: str) -> None:
    """Append event to job log and push to all active SSE subscriber queues."""
    job.setdefault("event_log", []).append(event_str)
    for q in job.get("queues", []):
        try:
            q.put_nowait(event_str)
        except asyncio.QueueFull:
            pass  # slow consumer — skip rather than block


_PREWARM_DOMAINS = [
    "kannada_film", "hindi_film", "carnatic", "english_pop",
    "english_romantic", "world_fusion",
]
_PREWARM_SEEDS = {
    "kannada_film": "ಮಳೆ ಬರುತ್ತದೆ ಮೌನದ ರಾತ್ರಿಯಲ್ಲಿ",
    "hindi_film": "बरसात की रात में तारे चमकते हैं",
    "carnatic": "raga bhairavi morning stillness river",
    "english_pop": "the chorus breaks the night wide open",
    "english_romantic": "autumn light through misted glass",
    "world_fusion": "shore wind salt migration horizon",
}


def _load_all_components() -> None:
    """
    S14: Load all pipeline components and pre-warm domain WM states.
    Runs in a thread pool at startup — eliminates per-request warmup cost.
    After this: each request pays only ~113ms (single observe_step) not 5200ms.
    """
    import torch
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    from pathlib import Path as _Path

    state.wm_loading = True
    try:
        # Load all trained components from checkpoint
        state.wm, state.efe, state.citta = load_trained_components()
        state.decoder = WMStateDecoder()

        # Load VimarsaBridgeV2 checkpoint (trained in S13)
        bridge_ckpt = _Path("checkpoints/vimarsa_bridge_v2.pt")
        state.bridge = VimarsaBridgeV2.load_or_init(
            hidden_dim=512, vocab_size=128256,
            ckpt_path=bridge_ckpt,
        )

        # Pre-warm domain-specific WM states (5 steps each ≈ 300ms total)
        state.domain_states = {}
        for domain in _PREWARM_DOMAINS:
            seed = _PREWARM_SEEDS.get(domain, domain.replace("_", " "))
            try:
                h = warmup_wm_on_text(state.wm, seed, steps=5, domain=domain)
                state.domain_states[domain] = h.detach()
            except Exception as exc:
                print(f"  [prewarm] {domain} failed: {exc}")

        state.wm_ready = True
        print(f"✓ PWM pipeline loaded: WM+EFE+Citta+Bridge ready. "
              f"{len(state.domain_states)} domains pre-warmed.")
    except Exception as e:
        print(f"✗ PWM load failed: {e}")
        state.wm_ready = False
    finally:
        state.wm_loading = False


@asynccontextmanager
async def lifespan(app_: FastAPI):  # type: ignore[type-arg]
    """FastAPI lifespan — loads all components and pre-warms domains on boot."""
    async def _background_load():
        await asyncio.to_thread(_load_all_components)

    asyncio.create_task(_background_load())
    yield
    # Shutdown cleanup (none needed)


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


# ─── SSE Helpers ─────────────────────────────────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─── Background Generation Task ───────────────────────────────────────────────

async def _run_generation_task(job_id: str, req: GenerateRequest) -> None:
    """
    Background asyncio task: runs full generation pipeline independently of any
    SSE connection.  Results accumulate in state.jobs[job_id]; SSE streams
    observe via per-job subscriber queues.

    TRIZ C2 resolution (P28 Mechanics Substitution + P32 Colour Changes):
    - Generation runs as an autonomous task — not driven by the SSE consumer.
    - Client disconnect never kills an in-flight generation.
    - Reconnecting clients replay the full event log instantly, then subscribe
      to the live queue for subsequent tokens.
    """
    job = state.jobs[job_id]
    loop = asyncio.get_event_loop()

    def emit(event: str, data: dict) -> None:
        _broadcast(job, _sse_event(event, data))

    # ── Phase 1: WM warm-up ──────────────────────────────────────────────────
    job["status"] = "warming"
    emit("wm_status", {"stage": "warming", "message": "World model warming up...",
                       "pct": 0, "job_id": job_id})

    h_t = None
    meta = None
    prefix = ""
    music_prefix = ""

    if state.wm_ready and state.wm is not None:
        try:
            seed = req.seed_text or req.theme or req.style or "creative music poetry"
            # Emit progress events while warmup runs in executor
            warmup_future = loop.run_in_executor(
                None, warmup_wm_on_text, state.wm, seed, 60
            )
            for pct in (20, 40, 60, 80):
                await asyncio.sleep(0.4)   # ~4×0.4s to cover ~2s warmup
                emit("wm_status", {"stage": "warming", "pct": pct,
                                   "message": f"WM warm-up {pct}%..."})
            h_t = await warmup_future

            meta = state.decoder.decode(                       # type: ignore
                h_t, domain=req.domain,
                step=hash(job_id) % 100, spec_id=job_id
            )
            prefix = state.decoder.format_for_llm(meta)       # type: ignore

            # Sprint 4: Music notation annotation
            music = annotate_music(meta, req.domain, spec_id=job_id)
            music_prefix = (f"[Music: {music.llm_music_context}]\n"
                            if music.llm_music_context else "")

            job.update({"wm_energy": round(meta.energy, 4),
                        "wm_register": meta.register,
                        "wm_section": meta.section_name,
                        "wm_prefix": prefix,
                        "music_notation": music.to_dict()})

            emit("wm_status", {
                "stage": "ready", "pct": 100,
                "message": f"WM ready — register={meta.register}, section={meta.section_name}",
                "energy": round(meta.energy, 3),
                "register": meta.register,
                "section": meta.section_name,
                "music_notation": music.to_dict(),
            })
        except Exception as exc:
            emit("wm_status", {"stage": "skipped",
                               "message": f"WM unavailable ({exc}); using base generation"})
    else:
        emit("wm_status", {"stage": "loading",
                           "message": "WM still loading; proceeding without WM conditioning"})

    # ── Phase 2: Streaming LLM generation ────────────────────────────────────
    job["status"] = "generating"
    emit("generation_start", {"message": "Generating...", "job_id": job_id,
                              "domain": req.domain, "language": req.language})

    # Combine WM prefix + music context + request into user prompt
    user_prompt = _build_user_prompt(req, prefix + music_prefix)
    full_text: list[str] = []
    token_count = 0
    token_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4096)

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
        """Blocking Ollama streaming call in thread pool.

        Pushes individual tokens to the asyncio queue via run_coroutine_threadsafe so
        the event loop is never blocked.  Sends None sentinel on completion.
        """
        try:
            resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=360)
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line.decode("utf-8"))
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

    try:
        loop.run_in_executor(None, _stream_worker, token_queue, loop)

        while True:
            tok = await asyncio.wait_for(token_queue.get(), timeout=360.0)
            if tok is None:
                break
            full_text.append(tok)
            token_count += 1
            emit("token", {"token": tok, "n": token_count})

        text = "".join(full_text)
        job["text"] = text

        # ── Phase 3: Score and finalise ──────────────────────────────────────
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

        # Sprint 5: ISO 15919 / IAST transliteration for non-Latin scripts
        translit_record: dict = {"text": text}
        add_transliteration(translit_record)
        transliteration = translit_record.get("transliteration", {})

        job.update({
            "status": "complete",
            "scores": scores,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "token_count": token_count,
            "music_context": req.music_context,
            "transliteration": transliteration,
        })

        emit("result", {
            "job_id": job_id,
            "text": text,
            "scores": scores,
            "wm_register": job.get("wm_register", ""),
            "wm_section": job.get("wm_section", ""),
            "music_context": req.music_context,
            "music_notation": job.get("music_notation", {}),
            "transliteration": transliteration,     # Sprint 5
            "domain": req.domain,
            "language": req.language,
            "generated_at": job["generated_at"],
        })
        emit("done", {"job_id": job_id, "status": "complete"})

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        emit("error", {"job_id": job_id, "error": str(exc)})

    finally:
        # Signal all active SSE streams that the generator is finished
        for q in job.get("queues", []):
            try:
                q.put_nowait(None)   # None = stream-end sentinel
            except asyncio.QueueFull:
                pass


# ─── SSE Observer Stream ──────────────────────────────────────────────────────

async def _generation_stream(job_id: str) -> AsyncGenerator[str, None]:
    """
    SSE observer: replays the full event_log for late/reconnecting clients,
    then subscribes to live events via a per-job asyncio.Queue.

    Disconnect-safe: if the client drops, only the queue is removed; the
    background _run_generation_task continues unaffected.
    """
    job = state.jobs[job_id]

    # Replay all events emitted so far (catch-up for late connections)
    for event_str in list(job.get("event_log", [])):
        yield event_str

    # If already complete/errored, nothing more to stream
    if job["status"] in ("complete", "error"):
        return

    # Subscribe to live events
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4096)
    job.setdefault("queues", []).append(q)

    try:
        while True:
            try:
                event_str = await asyncio.wait_for(q.get(), timeout=360.0)
            except asyncio.TimeoutError:
                yield _sse_event("keepalive", {"job_id": job_id})
                continue
            if event_str is None:
                break   # background task signalled completion
            yield event_str
            if '"done"' in event_str or '"error"' in event_str:
                break
    finally:
        try:
            job.get("queues", []).remove(q)
        except ValueError:
            pass


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
        "event_log": [],    # replay buffer for reconnecting SSE clients
        "queues": [],       # active asyncio.Queue per SSE subscriber
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Start generation immediately as a background task — independent of any
    # SSE connection.  Client disconnect will NOT kill generation.
    asyncio.create_task(_run_generation_task(job_id, req))

    return {
        "job_id": job_id,
        "stream_url": f"/stream/{job_id}",
        "result_url": f"/result/{job_id}",
        "status": "queued",
    }


@app.get("/stream/{job_id}")
async def stream_generation(job_id: str) -> StreamingResponse:
    """
    SSE observer for a generation job.
    Events: wm_status, generation_start, token, result, done, error, keepalive.

    TRIZ C2 (P28 Mechanics Substitution + P32 Colour Changes):
    - Generation runs as autonomous background task (not SSE-driven).
    - Reconnecting clients replay event_log instantly, then subscribe to live queue.
    - 60-second keepalive prevents proxy timeout on slow models.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return StreamingResponse(
        _generation_stream(job_id),
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
        "event_log": [],
        "queues": [],
        "parent_job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.create_task(_run_generation_task(new_job_id, new_req))

    return {
        "job_id": new_job_id,
        "parent_job_id": job_id,
        "stream_url": f"/stream/{new_job_id}",
        "result_url": f"/result/{new_job_id}",
        "status": "queued",
    }


@app.post("/batch")
async def submit_batch(req: BatchRequest) -> dict:
    """Submit up to 10 generation specs as a batch.  All jobs start immediately."""
    batch_id = str(uuid.uuid4())
    job_ids = []
    for gen_req in req.specs:
        job_id = str(uuid.uuid4())
        state.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "request": gen_req.model_dump(),
            "event_log": [],
            "queues": [],
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(_run_generation_task(job_id, gen_req))
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


# ─── v1 Endpoints (Phase 3: PancakrtyaLoopV2 + SSE) ─────────────────────────

class V1GenerateRequest(BaseModel):
    """Request schema for neo-fm-web API contract."""
    domain: str = Field("kannada_film", description="Creative domain key")
    seed: str = Field("", description="Seed text for WM warmup")
    n_stanzas: int = Field(4, ge=1, le=8)
    language: str = Field("auto", description="Language hint (kn, hi, en, ...)")
    style: str = Field("lyrical", description="Style hint (romantic, devotional, ...)")
    stream: bool = Field(True, description="If true, return SSE stream")


@app.post("/v1/generate")
async def generate_v1(req: V1GenerateRequest) -> StreamingResponse:
    """
    SSE generation endpoint using PancakrtyaLoopV2.

    Pañcakṛtya loop (Contract 1): all 6 acts per stanza.
    Layer boundary (Contract 2): SSE events use domain-neutral keys.
    WM primary (Contract 3): stub output if LLM unavailable.

    SSE event protocol:
      event: wm_state      — WM energy, aesthetic_quality, creative_peak
      event: stanza_start  — stanza index
      event: token         — generated token text
      event: stanza_end    — per-stanza scores
      event: complete      — mean_aesthetic_quality, total_stanzas
    """
    import torch
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig
    from pwm.generation.engine import DEVICE

    async def _event_stream():
        # S14: use pre-loaded singleton components (loaded at startup, not per-request)
        if not state.wm_ready:
            yield 'event: error\ndata: {"message": "WM not ready — startup loading"}\n\n'
            return

        wm = state.wm
        efe = state.efe
        citta = state.citta
        bridge = state.bridge

        # LLM backend singleton
        llm = get_llm_backend()

        cfg = LoopConfig(
            n_stanzas=req.n_stanzas,
            device=str(DEVICE),
            max_tokens_per_stanza=256,
            temperature=0.88,
            top_p=0.92,
        )
        loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

        # S14: Use pre-warmed domain state if available (eliminates 5s warmup)
        # Fall back to quick 5-step warmup if domain not pre-warmed
        seed = req.seed or req.domain
        if req.domain in state.domain_states and not req.seed:
            # Pre-warmed: zero-cost, just use cached h_t
            h = state.domain_states[req.domain].to(DEVICE)
        else:
            # Seed-specific: quick 5-step warmup (~67ms after CUDA is warm)
            try:
                h = warmup_wm_on_text(wm, seed, steps=5, domain=req.domain)
            except Exception:
                h = torch.zeros(512, device=DEVICE)
        obs_list = [h.unsqueeze(0)] * req.n_stanzas

        # Build prompts
        def _system_prompt() -> str:
            domain_labels = {
                "kannada_film": "Kannada film song",
                "carnatic": "Carnatic classical lyric",
                "hindi_film": "Hindi film song",
                "english_pop": "English pop song",
                "jazz": "jazz standard",
                "world_fusion": "world fusion lyric",
            }
            label = domain_labels.get(req.domain, req.domain.replace("_", " "))
            return (
                f"You are a lyricist writing a {label}. "
                f"Style: {req.style}. Language: {req.language}. "
                "Write one stanza only. Be evocative, musical, and authentic. "
                "No explanations, no titles — just the lyric."
            )

        def _user_prompt(stanza_idx: int, prev_text: str) -> str:
            if stanza_idx == 0:
                seed_hint = f'Starting with the theme: "{req.seed}"' if req.seed else ""
                return f"Write the opening stanza. {seed_hint}"
            return f"Continue from:\n{prev_text.strip()[-200:]}\n\nWrite the next stanza."

        # Stream events
        for event in loop.run(obs_list, _system_prompt(), _user_prompt):
            sse_line = f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
            yield sse_line
            await asyncio.sleep(0)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/v1/health")
async def health_v1() -> dict:
    """
    Health endpoint for neo-fm-web integration.

    Returns system readiness, pre-warmed domain list, and latency profile.
    neo-fm-web polls this before enabling the Generate button.
    """
    import torch
    cuda_ok = torch.cuda.is_available()

    llama_ok = False
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        llama_ok = r.status_code == 200
    except Exception:
        pass

    bridge_ckpt = Path("checkpoints/vimarsa_bridge_v2.pt")

    return {
        "status": "ok" if state.wm_ready else "loading",
        "device": "cuda" if cuda_ok else "cpu",
        "cuda_available": cuda_ok,
        "wm_ready": state.wm_ready,
        "wm_loading": state.wm_loading,
        "efe_loaded": state.efe is not None,
        "citta_loaded": state.citta is not None,
        "bridge_loaded": state.bridge is not None,
        "bridge_trained": bridge_ckpt.exists(),
        "llama_server_ok": llama_ok,
        "domains_prewarmed": list(state.domain_states.keys()),
        "ttft_profile": {
            "prewarmed_domain_ms": 0,
            "seed_specific_ms": 67,
            "cold_start_ms": 5247,
        },
        "version": "phase5-s14",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.websocket("/v1/ws/generate")
async def ws_generate(websocket) -> None:
    """
    S15 LiveViz: WebSocket alternative to SSE for neo-fm-web.

    Protocol (client → server):
      {"domain": "...", "seed": "...", "n_stanzas": 4, "style": "...", "language": "auto"}

    Protocol (server → client, JSON messages):
      {"event": "wm_state",    "data": {...}}
      {"event": "stanza_start","data": {"stanza": N}}
      {"event": "token",       "data": {"text": "..."}}
      {"event": "stanza_end",  "data": {...}}
      {"event": "complete",    "data": {...}}
      {"event": "error",       "data": {"message": "..."}}

    Same SSE contract as /v1/generate — identical data shapes.
    """
    import json as _json
    from fastapi import WebSocket as _WS
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig
    from pwm.generation.engine import DEVICE
    import torch

    ws: _WS = websocket
    await ws.accept()

    try:
        raw = await ws.receive_text()
        req_data = _json.loads(raw)
    except Exception as exc:
        await ws.send_text(_json.dumps({"event": "error", "data": {"message": str(exc)}}))
        await ws.close()
        return

    if not state.wm_ready:
        await ws.send_text(_json.dumps({
            "event": "error",
            "data": {"message": "WM not ready — startup loading in progress"}
        }))
        await ws.close()
        return

    domain = req_data.get("domain", "kannada_film")
    seed = req_data.get("seed", "")
    n_stanzas = min(int(req_data.get("n_stanzas", 4)), 8)
    style = req_data.get("style", "lyrical")
    language = req_data.get("language", "auto")

    # Build obs sequence from pre-warmed or quick warmup (identical to SSE path)
    if domain in state.domain_states and not seed:
        h = state.domain_states[domain].to(DEVICE)
    else:
        try:
            h = warmup_wm_on_text(state.wm, seed or domain, steps=5, domain=domain)
        except Exception:
            h = torch.zeros(512, device=DEVICE)
    obs_list = [h.unsqueeze(0)] * n_stanzas

    cfg = LoopConfig(n_stanzas=n_stanzas, device=str(DEVICE),
                     max_tokens_per_stanza=256, temperature=0.88, top_p=0.92)
    loop = PancakrtyaLoopV2(state.wm, state.efe, state.citta, state.bridge,
                             get_llm_backend(), cfg)

    domain_labels = {
        "kannada_film": "Kannada film song", "carnatic": "Carnatic classical lyric",
        "hindi_film": "Hindi film song", "english_pop": "English pop song",
        "jazz": "jazz standard", "world_fusion": "world fusion lyric",
    }
    system_prompt = (
        f"You are a lyricist writing a {domain_labels.get(domain, domain)}. "
        f"Style: {style}. Language: {language}. "
        "Write one stanza only. Be evocative, musical, and authentic. "
        "No explanations, no titles — just the lyric."
    )

    def _user_prompt(idx: int, prev: str) -> str:
        if idx == 0:
            return f"Write the opening stanza." + (f' Theme: "{seed}"' if seed else "")
        return f"Continue from:\n{prev.strip()[-200:]}\n\nWrite the next stanza."

    try:
        for event in loop.run(obs_list, system_prompt, _user_prompt):
            await ws.send_text(_json.dumps(event))
    except Exception as exc:
        await ws.send_text(_json.dumps({"event": "error", "data": {"message": str(exc)}}))
    finally:
        await ws.close()
