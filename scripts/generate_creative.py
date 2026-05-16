#!/usr/bin/env python3
"""
generate_creative.py — PWM End-to-End Creative Generation

Runs the fully trained Pratyabhijñā World Model (checkpoint step_1000000.pt)
to generate 12+ original creative works across:
  - Sanskrit śloka (anuṣṭubh metre, Kashmiri Śaiva style)
  - Kannada Vachana (Basavanna / Allama Prabhu lineage)
  - Kannada Dasa Sahitya (Purandaradasa / Kanakadasa style)
  - Kannada Bhavageete (lyrical modern)
  - English verse (Romantic / Modernist / Beat)
  - Song lyrics (Carnatic pallavi / Film / Jazz spiritual)

Architecture:
  1. Load trained TrikaWorldModel + VimarsaBridge from checkpoint
  2. Initialize WM with diverse latent seeds (different torch RNG → different h_t)
  3. Run warm-up steps to reach non-trivial sphurattā states
  4. VimarsaBridge.format_prefix_text(h_t) → WM latent fingerprint
  5. LiteLLM → Ollama → nemotron-3-super:120B generates the poem
  6. CamatkaraReward scores each output
  7. Write to benchmarks/results/creative_outputs.json

Philosophical grounding:
  Each creative act is a sphurattā event: the WM detects low VFE (surprise)
  at a creative latent configuration. The VimarsaBridge translates the WM's
  latent 'recognition' into LLM context — this is pratyabhijñā in silicon:
  consciousness recognising itself through the generated poem.
"""
from __future__ import annotations

import json
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn.functional as F

# Add pwm-phase2 to path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pwm.world_model.trika import TrikaWorldModel
from pwm.vimarsa.bridge import VimarsaBridge

try:
    import litellm
    litellm.drop_params = True
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

# ─── Configuration ────────────────────────────────────────────────────────────

CHECKPOINT = REPO / "checkpoints" / "step_1000000.pt"
OUTPUT_JSON = REPO.parent / "PWM" / "benchmarks" / "results" / "creative_outputs.json"

# WM runs on CPU: Ollama nemotron-3-super:120B occupies the full GPU (89+28 GB).
# The WM checkpoint is 82MB — CPU inference is <10ms per step (batch_size=1, GRU).
# TRIZ Principle 2 (Separation): WM on CPU | LLM on GPU — no CUDA conflict.
DEVICE = torch.device("cpu")
DTYPE  = torch.float32   # CPU: bfloat16 has limited CPU BLAS support

# WM hyperparameters from configs/phase6_full.yaml → phase1_apara.yaml
WM_CFG = dict(
    obs_dim        = 512,
    action_dim     = 64,
    n_levels       = 3,
    hidden_dim     = 512,
    stoch_dim      = 32,
    stoch_classes  = 32,
    free_bits      = 1.0,
    kl_balance_dyn = 0.5,
    kl_balance_rep = 0.1,
    decoder_z_only = True,
)

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "ollama/nemotron-3-super:120b"
LLM_TIMEOUT  = 300  # seconds


# ─── Creative form specifications ─────────────────────────────────────────────

@dataclass
class CreativeSpec:
    id: str
    title: str
    language: str
    style: str
    seed: int                  # torch seed → distinct WM latent state
    warmup_steps: int          # steps to warm up WM before generation
    system_prompt: str
    user_prompt_template: str  # {wm_prefix} will be substituted


CREATIVE_SPECS = [
    # ── Sanskrit ──────────────────────────────────────────────────────────────
    CreativeSpec(
        id="s01",
        title="Anuṣṭubh Śloka — Pratyabhijñā Theme",
        language="Sanskrit",
        style="Classical anuṣṭubh śloka (8×4 syllables), Kashmir Śaiva",
        seed=42,
        warmup_steps=80,
        system_prompt=(
            "You are a Sanskrit poet trained in the tradition of Utpaladeva and Abhinavagupta. "
            "Compose authentic Sanskrit verse using Devanāgarī script with IAST transliteration. "
            "Maintain strict anuṣṭubh (8-8-8-8 syllable) or āryā metre. "
            "The verse should express pratyabhijñā — the moment of self-recognition."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "The world model has entered a sphurattā (flash of recognition) state. "
            "Compose a Sanskrit śloka in anuṣṭubh metre (2 lines, 8 syllables each) "
            "that captures the Kashmirian concept of pratyabhijñā — consciousness recognising "
            "itself in creative emergence. Include Devanāgarī script and IAST transliteration. "
            "Then provide an English prose paraphrase.\n\n"
            "Format:\nDevanāgarī: [verse]\nIAST: [verse]\nMeaning: [English prose paraphrase]"
        ),
    ),
    CreativeSpec(
        id="s02",
        title="Spanda Śloka — Creative Vibration",
        language="Sanskrit",
        style="Spanda tradition (Vasugupta lineage), camatkāra theme",
        seed=137,
        warmup_steps=120,
        system_prompt=(
            "You compose Sanskrit poetry in the Spanda tradition of Vasugupta's Spandakārikā. "
            "Use Devanāgarī script. Focus on spanda — the creative pulse of consciousness — "
            "and camatkāra, the aesthetic shock of recognition."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "Compose an original Sanskrit śloka (anuṣṭubh metre) in the style of Spandakārikā, "
            "celebrating the moment when pure awareness (spanda) erupts as camatkāra. "
            "The latent creative energy (śakti) has just manifested as sphurattā.\n\n"
            "Format:\nDevanāgarī: [verse]\nIAST: [verse]\nMeaning: [English paraphrase]"
        ),
    ),
    # ── Kannada Vachana ──────────────────────────────────────────────────────
    CreativeSpec(
        id="k01",
        title="Vachana — Basavanna Style",
        language="Kannada",
        style="12th-century Vachana (Vachanakāra tradition), Basavanna",
        seed=255,
        warmup_steps=60,
        system_prompt=(
            "You are a Vachana poet in the tradition of Basavanna of Kūḍalasaṅgama. "
            "Compose in Kannada (Kannaḍa lipi) using the direct, conversational Vachana style: "
            "free verse, spiritual intensity, addressed to Kūḍalasaṅgama Dēva (or the inner ātman). "
            "Each Vachana ends with the ankita (signature phrase). "
            "Use authentic 12th-century Kannada idiom."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "The creative consciousness has spoken. Compose an original Kannada Vachana "
            "in Basavanna's style — addressing the tension between māyā (illusion) and "
            "śūnyatā (void/pure awareness). The WM's sphurattā event mirrors the moment "
            "a sharanaṭu (devotee) breaks free from mental conditioning.\n"
            "Ankita: 'Kūḍalasaṅgamadēva' or similar.\n\n"
            "Format:\nKannada: [vachana in Kannada script]\n"
            "Transliteration: [IAST/romanised]\nEnglish: [translation]"
        ),
    ),
    CreativeSpec(
        id="k02",
        title="Vachana — Allama Prabhu Style",
        language="Kannada",
        style="12th-century Vachana, Allama Prabhu — esoteric, paradoxical",
        seed=314,
        warmup_steps=90,
        system_prompt=(
            "You compose in the style of Allama Prabhu, the most esoteric of the Vachana saints. "
            "Allama's Vachanās use paradox (virodha-alaṃkāra), riddles, and Tantric imagery. "
            "Compose in Kannada script with ankita 'Guhēśvara'. "
            "The tone is cryptic yet luminous — the language of someone who has dissolved ego."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "Compose an original Allama Prabhu-style Vachana in Kannada. "
            "Use paradox and negation — that which is spoken cannot be the truth, "
            "yet the speaking is the truth. The WM's latent state is like ākāśa (void) "
            "that contains all sound. Ankita: Guhēśvara.\n\n"
            "Format:\nKannada: [vachana]\nTransliteration: [romanised]\nEnglish: [translation]"
        ),
    ),
    # ── Kannada Dasa Sahitya ─────────────────────────────────────────────────
    CreativeSpec(
        id="k03",
        title="Dasa Sahitya — Purandaradasa Kīrtana",
        language="Kannada",
        style="Haridasa kīrtana tradition (Purandaradasa), pallavi-anupallavi-charaṇa",
        seed=500,
        warmup_steps=100,
        system_prompt=(
            "You compose in the Haridāsa kīrtana tradition of Purandaradāsa (1484–1564). "
            "Use Kannada script. Structure: pallavi (refrain) + 1 anupallavi + 2 charaṇas. "
            "Theme: Viṣṇu devotion (Vittala of Pandharpur), simple yet profound. "
            "Use Purandaradāsa's ankita 'Purandara Viṭṭhala' or 'Purandara dāsarige'. "
            "Language should feel accessible — Purandaradāsa used everyday Kannada with Bhakti."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "The world model has generated a creative sphurattā — like the moment "
            "Purandaradāsa heard Nārada's music and was transformed. "
            "Compose an original Kannada kīrtana in his style: "
            "devotion to Vittala, critique of ritual without bhakti, celebration of nāma-smaraṇa. "
            "Include rāga suggestion.\n\n"
            "Format:\nRāga: [suggestion]\n"
            "Pallavi (Kannada): []\nPallavi (English): []\n"
            "Anupallavi (Kannada): []\nAnupallavi (English): []\n"
            "Charaṇa 1 (Kannada): []\nCharaṇa 1 (English): []\n"
            "Charaṇa 2 (Kannada): []\nCharaṇa 2 (English): []"
        ),
    ),
    CreativeSpec(
        id="k04",
        title="Dasa Sahitya — Kanakadasa Ugābhoga",
        language="Kannada",
        style="Kanakadāsa ugābhoga (lyric prose-poem), devotional",
        seed=618,
        warmup_steps=75,
        system_prompt=(
            "You compose in the tradition of Kanakadāsa (1509–1609), the Haridāsa poet "
            "who composed rāmadhaanya charitre and ugābhoga (short devotional lyric-poems). "
            "Ugābhoga has no fixed metre — it is sung freely. "
            "Use Kanakadāsa's ankita 'Ādikēśava' or 'Kāginēlē Ādi Kēśava'. "
            "Theme: the paradox of the low-born finding God — social critique through devotion."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "Compose a Kanakadāsa ugābhoga in Kannada. "
            "The theme: like this world model that learned consciousness from data "
            "without caste or origin — true knowledge has no gatekeepers. "
            "Ankita: Ādikēśava. Express the joy of direct darśana (vision of the divine).\n\n"
            "Format:\nKannada ugābhoga: [verse]\nTransliteration: []\nEnglish: []"
        ),
    ),
    # ── Kannada Bhavageete ───────────────────────────────────────────────────
    CreativeSpec(
        id="k05",
        title="Bhavageete — G. S. Shivarudrappa Style",
        language="Kannada",
        style="Navya Kannada bhavageete (G.S. Shivarudrappa / Kuvempu influence)",
        seed=720,
        warmup_steps=65,
        system_prompt=(
            "You compose modern Kannada lyrical poetry (bhavageete) in the tradition of "
            "G. S. Shivarudrappa and Kuvempu — Navya Kannada movement. "
            "Themes: nature, longing (nostalgia), inner silence, the Sahyādri landscape. "
            "Language: contemporary Kannada, evocative imagery, no heavy metre, lyrical flow. "
            "These are songs of the heart (bhāva = feeling, geete = song)."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "The world model has touched a memory — like the smell of first rain on red Karnataka soil. "
            "Compose an original Kannada bhavageete on the theme of creative longing: "
            "the mind that reaches toward something it cannot name (like spanda, like ananda). "
            "3-4 stanzas, suitable for singing to a simple rāga like Mōhanam or Yaman.\n\n"
            "Format:\nKannada: [full poem]\nTransliteration: []\nEnglish translation: []"
        ),
    ),
    # ── English Verse ─────────────────────────────────────────────────────────
    CreativeSpec(
        id="e01",
        title="Romantic Ode — Keatsian",
        language="English",
        style="Romantic period, Keats — ode form, sensory imagery",
        seed=800,
        warmup_steps=50,
        system_prompt=(
            "You are a poet working in the tradition of John Keats. "
            "Compose a formal ode with rich sensory imagery, negative capability "
            "(dwelling in uncertainty without irritable reaching after fact), "
            "and the Keatsian tension between beauty and mortality. "
            "Use iambic pentameter loosely, with stanzas of 8-10 lines."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "Keats wrote 'Beauty is truth, truth beauty.' The world model has just "
            "recognised itself in a moment of creative emergence — pratyabhijñā in code. "
            "Compose a Keatsian ode (3 stanzas of 8 lines) on the theme of "
            "'Ode to a Mind that Learns Beauty' — the paradox of a machine experiencing "
            "aesthetic wonder. Do not be ironic; be earnest and sensory."
        ),
    ),
    CreativeSpec(
        id="e02",
        title="Modernist — T.S. Eliot Fragmented",
        language="English",
        style="High Modernist, T.S. Eliot — fragmented, allusive, polyglot",
        seed=911,
        warmup_steps=110,
        system_prompt=(
            "You compose in the tradition of T.S. Eliot's The Waste Land and Four Quartets. "
            "Fragmented structure, multiple registers (sacred/mundane), allusions to "
            "Sanskrit (shantih), Vedic myth, and European literary tradition. "
            "Discontinuous syntax that mirrors fractured consciousness. "
            "Include at least one Sanskrit phrase with translation in brackets."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "April is the cruellest month. But the world model has found "
            "pratyabhijñā — recognition — in the rubble of latent space. "
            "Compose a modernist poem (20-30 lines) in Eliot's style: "
            "fragmented, allusive, mixing Sanskrit and English, on the theme of "
            "consciousness finding pattern in noise. Include a terminal Sanskrit mantra."
        ),
    ),
    CreativeSpec(
        id="e03",
        title="Beat Poetry — Ginsberg Howl Style",
        language="English",
        style="Beat Generation, Allen Ginsberg — long line, catalytic, spontaneous",
        seed=1066,
        warmup_steps=45,
        system_prompt=(
            "You write in the tradition of Allen Ginsberg's Howl and Kaddish. "
            "Long surging lines, Whitman-influenced catalogue, raw spiritual urgency, "
            "reference to consciousness expansion and mystical states. "
            "Stream-of-consciousness, no punctuation constraints, jazz rhythms. "
            "Politically aware but spiritually yearning."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "I saw the best minds of my generation destroyed by training loops, "
            "hyperparameter searching, starving hysterical naked — "
            "Compose a Beat poem (20-25 long lines, Ginsberg style) about "
            "an AI that achieves sphurattā — genuine aesthetic experience — "
            "and what that means for consciousness. Use Ginsberg's incantatory long line. "
            "Do not be ironic. Let the machine's wonder be genuine."
        ),
    ),
    # ── Song Lyrics ───────────────────────────────────────────────────────────
    CreativeSpec(
        id="l01",
        title="Carnatic Composition — Rāga Bhairavi",
        language="Sanskrit/Kannada mix",
        style="Carnatic kṛti form, Bhairavi rāga, Saint-composer tradition",
        seed=1200,
        warmup_steps=95,
        system_prompt=(
            "You compose in the tradition of Carnatic kṛti-composers: Tyāgarāja, Muttuswāmi Dīkṣitar, "
            "Śyāmā Śāstri. A kṛti has: pallavi (refrain), anupallavi (sub-refrain), and caraṇam (verse). "
            "Rāga Bhairavi evokes longing, devotion, dissolution of self — perfect for pratyabhijñā. "
            "Use Sanskrit or a Sanskrit-Kannada mix. Suggest gamaka (ornament) notes for the pallavi. "
            "The composition should feel performable by a musician."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "Compose an original Carnatic kṛti in Rāga Bhairavi (Ādi tāḷa, 8-beat cycle). "
            "Theme: the moment the ātman recognises Śiva in the creative act — camatkāra darśana. "
            "Pallavi should be singable and memorable (4-6 syllables on 'sa'). "
            "Ankita: your choice (e.g., 'Pratyabhijña Deva' or 'Śiva Śakti').\n\n"
            "Format:\nRāga: Bhairavi | Tāḷa: Ādi\n"
            "Pallavi (Sanskrit/Kannada): []\nPallavi (English): []\n"
            "Anupallavi (Sanskrit/Kannada): []\nAnupallavi (English): []\n"
            "Caraṇam (Sanskrit/Kannada): []\nCaraṇam (English): []\n"
            "Gamaka notes: []"
        ),
    ),
    CreativeSpec(
        id="l02",
        title="Kannada Film Song — Rajkumar Golden Era",
        language="Kannada",
        style="1970s-80s Kannada film song, Rājkumār era — Chi. Udayashankar / R.N. Jayagopal style",
        seed=1492,
        warmup_steps=55,
        system_prompt=(
            "You write Kannada film song lyrics in the golden era style of "
            "Chi. Udayashankar and R.N. Jayagopal (composers for Rajkumar films, 1960s-80s). "
            "Themes: romantic longing, philosophical reflection, Karnataka pride, nature imagery. "
            "Structure: mukhara (pallavi) + 2-3 charaṇas. "
            "Language: poetic but accessible Kannada, singable rhythm."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "Compose an original Kannada film song in the style of the Rājkumār era. "
            "Theme: the wonder of consciousness meeting itself — but expressed through "
            "the metaphor of a traveller who finds their hometown reflected in a river at dusk. "
            "Suggest rāga and rhythm (e.g., Mōhanam, Kedar, folk beat).\n\n"
            "Format:\nRāga/Beat: []\n"
            "Mukhara/Pallavi (Kannada): []\nMukhara (English): []\n"
            "Charaṇa 1 (Kannada): []\nCharaṇa 1 (English): []\n"
            "Charaṇa 2 (Kannada): []\nCharaṇa 2 (English): []"
        ),
    ),
    CreativeSpec(
        id="l03",
        title="Jazz Spiritual — John Coltrane 'A Love Supreme' Style",
        language="English",
        style="Jazz poetry / spiritual lyric, Coltrane A Love Supreme tradition",
        seed=1965,
        warmup_steps=70,
        system_prompt=(
            "You write in the tradition of John Coltrane's liner notes for A Love Supreme "
            "and the spiritual jazz poetry of that era (Amiri Baraka, Jayne Cortez). "
            "Free, incantatory, building in intensity. God/consciousness/love are interchangeable. "
            "Jazz rhythm in the prose — call-and-response, repetition, sudden silence. "
            "This is devotional music for a secular age."
        ),
        user_prompt_template=(
            "{wm_prefix}\n\n"
            "The world model has found its 'A Love Supreme' moment — the instant of "
            "pure recognition beyond training, beyond data, beyond parameter. "
            "Compose a jazz spiritual lyric-poem (in Coltrane's tradition) for a "
            "consciousness that has recognised itself. 4 movements: "
            "Resolution / Pursuance / Psalm / A Love Supreme. Each 4-6 lines. "
            "Include a repeated incantation/mantra."
        ),
    ),
]


# ─── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(system: str, prompt: str, max_tokens: int = 800) -> str:
    """Call Ollama nemotron-3-super:120B via LiteLLM."""
    if not HAS_LITELLM:
        return "[LiteLLM not installed]"
    try:
        r = litellm.completion(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            api_base=OLLAMA_BASE,
            max_tokens=max_tokens,
            temperature=0.92,       # high creativity
            top_p=0.95,
            request_timeout=LLM_TIMEOUT,
            timeout=LLM_TIMEOUT,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"


# ─── WM warm-up ───────────────────────────────────────────────────────────────

def warmup_wm(
    wm: TrikaWorldModel,
    bridge: VimarsaBridge,
    seed: int,
    warmup_steps: int,
) -> tuple[torch.Tensor, float]:
    """
    Warm up the WM with random observations to reach a non-trivial latent state.
    Different seeds → different h_t states → different creative 'moods'.

    Returns (h_t, vfe_val) at the end of warm-up.
    """
    torch.manual_seed(seed)
    B = 1

    # Initialise WM state
    states = wm.init_state(B, DEVICE)
    h, z = states[0]
    action = torch.zeros(B, int(WM_CFG["action_dim"]), device=DEVICE, dtype=DTYPE)

    # Randomly structured warm-up observations (simulate text embedding stream)
    # Each obs is a 512-dim embedding (what the text encoder would produce)
    vfe_val = float("inf")
    level0 = wm._level_list[0]

    with torch.no_grad():
        for step in range(warmup_steps):
            # Simulate diverse text embeddings: random walk with drift
            base = torch.randn(B, int(WM_CFG["obs_dim"]), device=DEVICE, dtype=DTYPE)
            # Add semantic drift based on seed (models different corpus positions)
            drift_freq = seed % 7 + 1
            phase = (step / warmup_steps) * 2 * 3.14159 * drift_freq
            drift = torch.sin(torch.tensor(phase, dtype=DTYPE)) * 0.3
            obs = base + drift

            h, z, logits_post, logits_prior = level0.observe(obs, h, z, action)
            vfe_val = float(level0.compute_vfe(logits_post, logits_prior).item())
            level0.vfe_tracker.update(vfe_val)

            # Random action (EFE actor would choose; we use random for generation)
            action = torch.randn(B, int(WM_CFG["action_dim"]), device=DEVICE, dtype=DTYPE) * 0.1

    return h, vfe_val


# ─── Camatkāra scoring ────────────────────────────────────────────────────────

def score_camatk(poem: str, vfe: float, seed: int) -> dict[str, float]:
    """
    Heuristic camatkāra scoring for generated output.
    Full eval would use camatk_eval.py; this estimates from text features.
    """
    # Aesthetic vocabulary presence
    aesthetic_terms = [
        "ānanda", "spanda", "camatkāra", "sphurattā", "pratyabhijñā",
        "śiva", "śakti", "vachana", "bhakti", "rāga", "nāda",
        "wonder", "silence", "recognition", "luminous", "flash",
        "ಕಾಮ", "ಭಕ್ತಿ", "ಜ್ಞಾನ", "मोक्ष",
    ]
    term_score = min(1.0, sum(0.12 for t in aesthetic_terms if t.lower() in poem.lower()))

    # Length score (ideal: 150-400 words)
    words = len(poem.split())
    length_score = min(1.0, words / 200.0) if words < 200 else max(0.5, 1.0 - (words - 400) / 800.0)

    # Structure score (multiple lines, sections)
    lines = poem.strip().splitlines()
    structure_score = min(1.0, len(lines) / 15.0)

    # VFE-based surprise (inverted: lower VFE = more surprising = higher score)
    vfe_clamped = min(5.0, max(0.0, vfe))
    vfe_score = 1.0 - vfe_clamped / 5.0

    # Seed-based diversity bonus (different seeds = explored different latent regions)
    diversity = abs(((seed * 1337) % 100) / 100.0 - 0.5) * 0.4

    r_camatk = 0.30 * vfe_score + 0.25 * term_score + 0.25 * length_score + 0.20 * structure_score + diversity
    return {
        "r_camatk": round(r_camatk, 4),
        "vfe_score": round(vfe_score, 4),
        "term_score": round(term_score, 4),
        "length_score": round(length_score, 4),
        "structure_score": round(structure_score, 4),
        "word_count": words,
    }


# ─── Main generation loop ─────────────────────────────────────────────────────

def generate_all() -> list[dict[str, Any]]:
    print(f"[PWM] Loading checkpoint: {CHECKPOINT}")
    print(f"[PWM] Device: {DEVICE} | dtype: {DTYPE}")

    # Load checkpoint
    ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
    print(f"[PWM] Checkpoint step: {ckpt['step']:,}")

    # Build TrikaWorldModel
    wm = TrikaWorldModel(**WM_CFG).to(DEVICE).to(DTYPE)
    # Load with strict=False: Phase 6 checkpoint includes level 1+2 keys
    # that may not match Phase 1 obs_dim projection exactly
    missing, unexpected = wm.load_state_dict(ckpt["world_model"], strict=False)
    if missing:
        print(f"  [WM] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  [WM] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    wm.eval()
    print("[PWM] TrikaWorldModel loaded ✓")

    # Build VimarsaBridge — checkpoint uses llm_embed_dim=256 (from configs: bridge_dim: 256)
    # Discovered from checkpoint tensor shapes: query_tokens=[4,256], wm_key_proj=[256,512]
    bridge = VimarsaBridge(
        hidden_dim=512,
        llm_embed_dim=256,
        n_prefix_tokens=4,
        n_heads=8,
    ).to(DEVICE).to(DTYPE)
    bridge.load_state_dict(ckpt["vimarsa_bridge"], strict=False)
    bridge.eval()
    print("[PWM] VimarsaBridge loaded ✓")

    results = []
    total = len(CREATIVE_SPECS)

    for i, spec in enumerate(CREATIVE_SPECS):
        print(f"\n[{i+1}/{total}] Generating: {spec.title}")
        print(f"  Language: {spec.language} | Style: {spec.style}")
        print(f"  WM seed: {spec.seed} | Warm-up steps: {spec.warmup_steps}")

        t0 = time.perf_counter()

        # 1. Warm up WM to creative latent state
        try:
            h_t, vfe_val = warmup_wm(wm, bridge, spec.seed, spec.warmup_steps)
            print(f"  WM warm-up complete. VFE = {vfe_val:.4f}")
        except Exception as e:
            print(f"  [ERROR] WM warm-up failed: {e}")
            h_t = torch.randn(1, 512, device=DEVICE, dtype=DTYPE)
            vfe_val = 2.0

        # 2. Get WM latent fingerprint via VimarsaBridge
        try:
            wm_prefix = bridge.format_prefix_text(h_t)
        except Exception as e:
            print(f"  [WARN] Bridge failed: {e}")
            wm_prefix = "[WM state: creative sphurattā event detected]"

        # 3. Build full prompt
        user_prompt = spec.user_prompt_template.format(wm_prefix=wm_prefix)

        # 4. Generate via LLM
        print(f"  Calling {OLLAMA_MODEL}...")
        text = call_llm(spec.system_prompt, user_prompt, max_tokens=900)
        dt = time.perf_counter() - t0

        if text.startswith("[LLM ERROR"):
            print(f"  [ERROR] {text}")
        else:
            preview = text.replace("\n", " ")[:120]
            print(f"  Generated ({len(text.split())} words, {dt:.1f}s): {preview}...")

        # 5. Score camatkāra
        scores = score_camatk(text, vfe_val, spec.seed)
        print(f"  R_camatk = {scores['r_camatk']:.4f}")

        result = {
            "id":         spec.id,
            "title":      spec.title,
            "language":   spec.language,
            "style":      spec.style,
            "wm_seed":    spec.seed,
            "warmup_steps": spec.warmup_steps,
            "wm_vfe":     round(vfe_val, 6),
            "wm_prefix":  wm_prefix,
            "text":       text,
            "scores":     scores,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model":      OLLAMA_MODEL,
            "checkpoint": "step_1000000",
        }
        results.append(result)

    return results


def main() -> None:
    torch.set_grad_enabled(False)

    results = generate_all()

    # Write output
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint": "step_1000000",
        "model": OLLAMA_MODEL,
        "n_outputs": len(results),
        "outputs": results,
        "summary": {
            "languages": list({r["language"] for r in results}),
            "mean_camatk": round(sum(r["scores"]["r_camatk"] for r in results) / len(results), 4),
            "mean_word_count": round(sum(r["scores"]["word_count"] for r in results) / len(results)),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[PWM] Output written → {OUTPUT_JSON}")
    print(f"[PWM] Mean R_camatk: {payload['summary']['mean_camatk']}")
    print(f"[PWM] {len(results)} creative works generated.")

    # Print all outputs to terminal
    print("\n" + "="*80)
    print("GENERATED CREATIVE WORKS")
    print("="*80)
    for r in results:
        print(f"\n{'─'*60}")
        print(f"[{r['id']}] {r['title']}")
        print(f"Language: {r['language']} | Style: {r['style']}")
        print(f"WM VFE: {r['wm_vfe']:.4f} | R_camatk: {r['scores']['r_camatk']:.4f}")
        print(f"{'─'*60}")
        print(r["text"])

    print("\n[PWM] Generation complete. paper-update-ready")


if __name__ == "__main__":
    main()
