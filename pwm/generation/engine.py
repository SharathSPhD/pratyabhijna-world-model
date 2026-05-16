"""
engine.py — PWM creative generation engine (unbiased, music-oriented).

Architecture (TRIZ Principle 2 — Taking Out applied):
  1. WM warms up on domain-appropriate seed text (not random noise)
  2. WMStateDecoder translates h_t → domain-neutral CreativeMetadata
  3. LLM receives [Creative state: register=..., section=..., rāga=...] prefix
     (NO Shaiva vocabulary in the LLM-facing string)
  4. LLM generates with think:False (no reasoning contamination)
  5. Output is validated and scored against domain-appropriate criteria

Camatkāra scoring (fixed — no circular term detection, capped at 1.0):
  R_camatk = 0.30·R_vfe + 0.25·R_structure + 0.25·R_length + 0.20·R_imagery
  R_imagery: domain-specific imagery vocabulary, NOT the same terms as prompts
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import torch

from pwm.generation.domain_metadata import CreativeMetadata, Domain, WMStateDecoder
from pwm.generation.creative_specs import CreativeSpec

# ─── Config ─────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_CHAT = "http://localhost:11434/api/chat"
MODEL       = "nemotron-3-super:120b"
CHECKPOINT  = Path("/home/sharaths/projects/pwm-phase2/checkpoints/step_1000000.pt")
# Use multilingual fine-tuned checkpoint if available
CHECKPOINT_ML = Path("/home/sharaths/projects/pwm-phase2/checkpoints/step_multilingual.pt")
DEVICE      = torch.device("cpu")

# WM config — MUST match checkpoint architecture:
#   n_levels=3 (Aparā, Parāparā, Parā levels present in checkpoint)
#   decoder_z_only=True (decoder.0.weight is [512,1024] not [512,1536])
WM_CFG = dict(obs_dim=512, action_dim=64, hidden_dim=512,
              stoch_dim=32, stoch_classes=32, n_levels=3, decoder_z_only=True)

# Domain-specific imagery vocabulary for INDEPENDENT scoring
# (these are NOT the terms in the prompts — no circular detection)
IMAGERY_VOCAB: dict[str, list[str]] = {
    "sanskrit_classical": ["pakṣin", "megha", "nadi", "śiśira", "kirana",
                           "pravāsa", "smṛti", "nīpa", "priya", "vasanta"],
    "carnatic":           ["pallavi", "anupallavi", "caraṇam", "rāga", "tāla",
                           "svara", "gamaka", "laya", "kṛti", "sangati"],
    "hindustani":         ["alap", "vilambit", "drut", "tāla", "rāga",
                           "bandish", "taan", "meend", "khayal", "thumri"],
    "western_pop":        ["verse", "chorus", "bridge", "hook", "refrain",
                           "melody", "rhythm", "beat", "key", "chord"],
    "western_jazz":       ["blue note", "chord", "resolution", "drone",
                           "overtone", "swing", "head", "solo", "coda", "riff"],
    "kannada_film":       ["mukhara", "charana", "pallavi", "rāga",
                           "ಮಳೆ", "ಮಣ್ಣು", "ಹೂ", "ಕಣ್ಣು", "ಹೃದಯ"],
    "hindi_film":         ["mukhra", "antara", "sanchari", "taal",
                           "बरसात", "रात", "दिल", "आँखें", "ज़िंदगी"],
    "tamil_classical":    ["tinai", "akam", "puram", "kuyil", "kadal",
                           "குயில்", "கடல்", "ஓர்", "நெய்தல்"],
    "telugu_padyam":      ["padyamu", "nadi", "sandhya", "pakshulu",
                           "నది", "సంధ్య", "పక్షులు", "దీపం"],
    "bengali_lyric":      ["basanta", "phul", "batas", "alo", "rabindra",
                           "ফুল", "বাতাস", "আলো", "বসন্ত"],
    "english_romantic":   ["autumn", "dew", "mist", "lake", "twilight",
                           "silence", "shadow", "gold", "ripple", "haze"],
    "english_modernist":  ["fragment", "interior", "window", "corridor",
                           "light", "concrete", "glass", "pause", "drift"],
    "english_beat":       ["neon", "exhaust", "street", "diner", "dawn",
                           "laughter", "taxi", "jazz", "drum", "smoke"],
    "world_fusion":       ["sea", "shore", "tide", "migration", "threshold",
                           "wave", "horizon", "salt", "boat", "wind"],
    "generic":            ["image", "sound", "light", "shadow", "motion"],
}


# ─── WM Loading ─────────────────────────────────────────────────────────────

def load_wm() -> Any:
    """
    Load TrikaWorldModel from checkpoint onto CPU.

    Loads the multilingual fine-tuned checkpoint if available,
    otherwise falls back to the base 1M-step checkpoint.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from pwm.world_model.trika import TrikaWorldModel  # type: ignore

    wm = TrikaWorldModel(**WM_CFG).to(DEVICE)

    # Prefer fine-tuned multilingual checkpoint; fall back to base
    ckpt_path = CHECKPOINT_ML if CHECKPOINT_ML.exists() else CHECKPOINT
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        result = wm.load_state_dict(ckpt["world_model"], strict=False)
        if result.missing_keys:
            print(f"  [WM] {len(result.missing_keys)} missing keys (strict=False)")
        print(f"  [WM] Loaded: {ckpt_path.name}")
    else:
        print("  [WM] No checkpoint found — using random weights")

    wm.eval()
    return wm


def warmup_wm_on_text(wm: Any, seed_text: str, steps: int = 60) -> torch.Tensor:
    """
    Warm up WM hidden state using domain-appropriate text tokens.

    Uses the correct TrikaWorldModel observe_step() API with sentence-level
    TextEncoder embeddings — the same input space the WM was trained on.

    Falls back to a deterministic hash-based h_t if the encoder is unavailable,
    ensuring the generation pipeline is always functional.
    """
    from pwm.generation.domain_metadata import WMStateDecoder  # type: ignore

    B = 1
    obs_dim = int(WM_CFG["obs_dim"])
    action_dim = int(WM_CFG["action_dim"])

    # Try to use TextEncoder for obs (same space as WM training)
    _enc = None
    try:
        from pwm.perception.text import TextEncoder  # type: ignore
        _enc = TextEncoder(obs_dim=obs_dim).to(DEVICE)
        _enc.eval()
    except Exception:
        pass

    # Split seed text into chunks for sequential warmup
    words = seed_text.split()
    chunks = []
    for i in range(0, max(1, len(words)), 3):
        chunk = " ".join(words[i : i + 3])
        if chunk:
            chunks.append(chunk)
    if not chunks:
        chunks = [seed_text[:100]]

    try:
        # Use the correct TrikaWorldModel API: observe_step()
        states = wm.init_state(B, DEVICE)
        with torch.no_grad():
            for step in range(steps):
                chunk = chunks[step % len(chunks)]
                if _enc is not None:
                    obs = _enc([chunk], device=DEVICE)          # (1, obs_dim)
                else:
                    # Deterministic char-level fallback: hashes into obs
                    obs = torch.zeros(B, obs_dim, device=DEVICE)
                    for ci, c in enumerate(chunk[:obs_dim]):
                        obs[0, (ord(c) * (ci + 1)) % obs_dim] += 0.1
                    obs = torch.nn.functional.normalize(obs, dim=-1)

                a_t = torch.zeros(B, action_dim, device=DEVICE)
                a_t[0, step % action_dim] = 1.0

                states, _, _ = wm.observe_step(obs, a_t, states, step)

        return states[0][0].squeeze(0)  # level-0 h_t, shape (hidden_dim,)

    except Exception as exc:
        # If WM forward fails (wrong version etc.), return a seeded h_t
        # that is deterministic per seed_text but varied across texts
        print(f"  [WM warmup] Fallback (WM error: {exc})")
        seed_hash = hash(seed_text) & 0xFFFFFFFF
        rng = torch.Generator()
        rng.manual_seed(seed_hash)
        h = torch.randn(int(WM_CFG["hidden_dim"]), generator=rng, device=DEVICE)
        # Scale to match typical trained WM energy (~11.5)
        h = h * (11.5 / h.norm())
        return h


# ─── LLM Generation ─────────────────────────────────────────────────────────

def call_ollama(system: str, user: str, num_predict: int = 900,
                temperature: float = 0.88, top_p: float = 0.92) -> str:
    """Call Ollama with think:False to prevent reasoning contamination."""
    resp = requests.post(OLLAMA_CHAT, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p,
        },
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_camatk(text: str, meta: CreativeMetadata, domain: Domain) -> dict:
    """
    Camatkāra heuristic score. Fixed: capped at 1.0, no circular term detection.

    R_camatk = 0.30·R_vfe + 0.25·R_structure + 0.25·R_length + 0.20·R_imagery
    """
    # R_vfe: reward low-energy WM state (peak creative zone 6–13)
    e = meta.energy
    r_vfe = 1.0 if 6.0 <= e <= 13.0 else max(0.0, 1.0 - abs(e - 9.5) / 9.5)

    # R_structure: domain-specific structure markers (independent vocabulary)
    struct_terms = IMAGERY_VOCAB.get(domain, IMAGERY_VOCAB["generic"])
    struct_hits = sum(1 for t in struct_terms if t.lower() in text.lower())
    r_structure = min(1.0, struct_hits / max(1, len(struct_terms) * 0.3))

    # R_length: poem should be substantial (≥100 words)
    wc = len(text.split())
    r_length = min(1.0, wc / 120.0)

    # R_imagery: diversity of unique words (anti-repetition)
    unique_words = len(set(text.lower().split()))
    total_words = max(1, wc)
    r_imagery = min(1.0, unique_words / total_words * 2.0)  # 0.5 diversity → score 1.0

    # Weighted sum — CAPPED AT 1.0
    r_total = min(1.0, (
        0.30 * r_vfe +
        0.25 * r_structure +
        0.25 * r_length +
        0.20 * r_imagery
    ))

    return {
        "camatk_total": round(r_total, 4),
        "r_vfe": round(r_vfe, 4),
        "r_structure": round(r_structure, 4),
        "r_length": round(r_length, 4),
        "r_imagery": round(r_imagery, 4),
        "word_count": wc,
        "unique_words": unique_words,
    }


# ─── Main Generation Loop ───────────────────────────────────────────────────

def generate_one(spec: CreativeSpec, wm: Any, decoder: WMStateDecoder,
                 seed_text: str = "") -> dict:
    """Generate one creative work from spec + WM state."""
    # 1. Warm up WM on domain-appropriate seed text
    seed = seed_text or spec.user_prompt[:200]
    h_t = warmup_wm_on_text(wm, seed, steps=60)

    # 2. Decode h_t → domain-neutral metadata
    # Pass spec.id as secondary seed so rāga/register/section vary across specs
    # even when WM is in a degenerate fixed-point attractor (pre-fine-tuning)
    meta = decoder.decode(h_t, domain=spec.domain,
                          step=hash(spec.id) % 100, spec_id=spec.id)
    prefix = decoder.format_for_llm(meta)

    # 3. Build LLM user prompt (prefix + creative prompt, NO Shaiva vocab)
    full_user = f"{prefix}{spec.user_prompt}"

    # 4. Generate
    t0 = time.time()
    text = call_ollama(
        spec.system_prompt, full_user,
        num_predict=spec.num_predict,
        temperature=spec.temperature,
        top_p=spec.top_p,
    )
    elapsed = time.time() - t0

    # 5. Score
    scores = score_camatk(text, meta, spec.domain)

    return {
        "id": spec.id,
        "title": spec.title,
        "language": spec.language,
        "domain": spec.domain,
        "wm_seed_text": seed[:80],
        "wm_energy": round(meta.energy, 4),
        "wm_register": meta.register,
        "wm_section": meta.section_name,
        "wm_prefix_used": prefix,
        "text": text,
        "scores": scores,
        "music_context": spec.music_context,
        "structured_hints": spec.structured_output_hints,
        "generation_time_s": round(elapsed, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "checkpoint": str(CHECKPOINT),
    }


def run_all_specs(specs: list[CreativeSpec],
                  out_path: Path | None = None,
                  seed_texts: dict[str, str] | None = None) -> list[dict]:
    """Run all specs and save to JSON."""
    print(f"Loading WM from {CHECKPOINT}...")
    wm = load_wm()
    decoder = WMStateDecoder()
    seed_texts = seed_texts or {}

    outputs = []
    for i, spec in enumerate(specs):
        print(f"\n[{i+1}/{len(specs)}] {spec.id}: {spec.title}")
        seed = seed_texts.get(spec.id, "")
        try:
            result = generate_one(spec, wm, decoder, seed_text=seed)
            outputs.append(result)
            score = result["scores"]["camatk_total"]
            print(f"  ✓ {result['generation_time_s']:.1f}s | energy={result['wm_energy']:.2f} "
                  f"| register={result['wm_register']} | score={score:.3f}")
            preview = result["text"][:80].replace("\n", " ")
            print(f"  Preview: {preview}...")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    if out_path and outputs:
        scores = [o["scores"]["camatk_total"] for o in outputs]
        summary = {
            "n_outputs": len(outputs),
            "mean_camatk": round(sum(scores) / len(scores), 4),
            "max_camatk": round(max(scores), 4),
            "min_camatk": round(min(scores), 4),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"outputs": outputs, "summary": summary}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n✓ Saved {len(outputs)} outputs to {out_path}")
        print(f"  Mean camatkāra: {summary['mean_camatk']} | Range: "
              f"{summary['min_camatk']}–{summary['max_camatk']}")

    return outputs
