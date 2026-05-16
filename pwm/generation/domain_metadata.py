"""
domain_metadata.py — Domain-neutral WM state → creative metadata decoder.

Philosophical grounding (TRIZ Principle 2 — Taking Out):
  The Pratyabhijñā computational structure (EFE, VFE, sphurattā event firing,
  camatkāra reward) is internally named in Sanskrit. These names MUST NOT cross
  the VimarsaBridge into the LLM token stream. This module translates WM hidden
  state h_t into domain-neutral creative metadata that any music/poetry context
  can interpret without triggering Shaiva associations.

  The internal computation is unchanged: sphurattā = VFE threshold crossing,
  camatkāra = R_camatk signal, vimarśa = self-modifying replay. The *labels*
  exposed to the LLM are domain-vocabulary tokens.

Usage:
    from pwm.generation.domain_metadata import WMStateDecoder
    decoder = WMStateDecoder()
    meta = decoder.decode(h_t, domain="carnatic")
    prompt_prefix = decoder.format_for_llm(meta, domain="carnatic")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

# Supported creative domains
Domain = Literal[
    "carnatic", "hindustani", "western_pop", "western_jazz",
    "kannada_film", "hindi_film", "sanskrit_classical",
    "english_romantic", "english_modernist", "english_beat",
    "bengali_lyric", "tamil_classical", "telugu_padyam",
    "world_fusion", "generic",
]

REGISTER_THRESHOLDS = {
    "still":        (0.0, 3.0),
    "contemplative":(3.0, 6.0),
    "yearning":     (6.0, 9.0),
    "ardent":       (9.0, 13.0),
    "ecstatic":     (13.0, float("inf")),
}

TENSION_LABELS = ["released", "gentle", "building", "taut", "unresolved"]

SECTION_LABELS = {
    "carnatic":    ["pallavi", "anupallavi", "caranam_1", "caranam_2", "svarakalpana"],
    "hindustani":  ["alap", "jod", "jhala", "bandish", "tihai"],
    "western_pop": ["intro", "verse_1", "pre_chorus", "chorus", "verse_2", "bridge", "outro"],
    "western_jazz":["head_in", "solos", "trading_4s", "head_out", "coda"],
    "kannada_film":["mukhara", "interlude", "charana_1", "charana_2"],
    "hindi_film":  ["mukhra", "antara_1", "antara_2", "sanchari"],
    "generic":     ["opening", "development", "climax", "resolution"],
}
for k in ["sanskrit_classical", "english_romantic", "english_modernist",
          "english_beat", "bengali_lyric", "tamil_classical",
          "telugu_padyam", "world_fusion"]:
    SECTION_LABELS[k] = SECTION_LABELS["generic"]

RAGA_BY_REGISTER = {
    "still":        ["Bhairavi", "Yaman", "Bhimpalasi"],
    "contemplative":["Todi", "Darbari Kanada", "Miyan ki Malhar"],
    "yearning":     ["Bhairavi", "Bageshri", "Jaijaivanti"],
    "ardent":       ["Bhairav", "Marwa", "Puriya Dhanashri"],
    "ecstatic":     ["Bhairavi", "Pooriya", "Bairagi"],
}

MODE_BY_REGISTER = {
    "still":        ["Dorian", "Aeolian", "Lydian"],
    "contemplative":["Phrygian", "Dorian", "Natural Minor"],
    "yearning":     ["Aeolian", "Harmonic Minor", "Dorian"],
    "ardent":       ["Mixolydian", "Major", "Lydian"],
    "ecstatic":     ["Major", "Lydian", "Ionian"],
}

TEMPO_BY_ENERGY = {
    (0, 4):    ("slow", "largo / vilambit"),
    (4, 8):    ("medium-slow", "andante / madhya"),
    (8, 12):   ("medium", "moderato / madhya-drut"),
    (12, 16):  ("fast", "allegro / drut"),
    (16, 100): ("very fast", "presto / ati-drut"),
}

EMOTION_TAGS = {
    "still":        ["serenity", "stillness", "peace"],
    "contemplative":["introspection", "longing", "memory"],
    "yearning":     ["desire", "separation", "ache"],
    "ardent":       ["passion", "devotion", "intensity"],
    "ecstatic":     ["joy", "celebration", "transcendence"],
}

MUSIC_KEY_BY_DIMS = {
    # Map the leading active dimension index to a musical key hint
    # These are heuristic mappings — they will improve once the WM
    # is trained on multilingual creative corpus
    range(0, 64):   "C / Sa",
    range(64, 128): "G / Pa",
    range(128, 192): "D / Ri",
    range(192, 256): "A / Dha",
    range(256, 320): "E / Ga",
    range(320, 384): "B / Ni",
    range(384, 448): "F / Ma",
    range(448, 512): "C / Sa",
}


@dataclass
class CreativeMetadata:
    """Domain-neutral creative state derived from WM h_t."""
    energy: float
    register: str          # still / contemplative / yearning / ardent / ecstatic
    tension_idx: int       # 0–4
    section_idx: int       # index into SECTION_LABELS[domain]
    leading_dim: int       # most active latent dimension
    diversity: float       # fraction of dims above mean

    # Domain-specific fields (populated by domain-specific formatting)
    domain: str = "generic"
    raga_hint: str = ""
    mode_hint: str = ""
    tempo_hint: str = ""
    key_hint: str = ""
    emotion_tags: list[str] = field(default_factory=list)
    section_name: str = ""


class WMStateDecoder:
    """
    Maps WM hidden state h_t → CreativeMetadata.

    No neural network required — uses interpretable scalar statistics
    of h_t (L2 norm, top-k activation pattern, signed diversity) to
    derive creative metadata that is:
      1. Deterministic given h_t (reproducible)
      2. Domain-neutral in vocabulary
      3. Varied across seeds (given diverse h_t inputs)
      4. Meaningful to the LLM (rāga names, mode names, section names)
    """

    def decode(self, h_t: Tensor, domain: Domain = "generic",
               step: int = 0, spec_id: str = "") -> CreativeMetadata:
        """
        Decode WM hidden state h_t into CreativeMetadata.

        Args:
            h_t: WM hidden state tensor, shape (hidden_dim,) or (1, hidden_dim)
            domain: Target creative domain
            step: Used for section progression (use hash(spec.id) for per-spec variation)
            spec_id: Spec identifier — used as secondary randomization seed so
                     rāga/mode/register vary per spec even when the WM is in a
                     degenerate fixed-point attractor (e.g. during initial training).
        """
        import random as _random

        h = h_t.detach().float().flatten()

        # Energy = L2 norm
        energy = float(h.norm().item())

        # Register
        register = "contemplative"
        for reg, (lo, hi) in REGISTER_THRESHOLDS.items():
            if lo <= energy < hi:
                register = reg
                break

        # Tension = variance of activation, quantized to 5 bins
        var = float(h.var().item())
        tension_idx = min(4, int(var * 5 / (energy + 1e-6)))

        # Leading dimension = argmax of abs values
        leading_dim = int(h.abs().argmax().item())

        # Diversity = fraction of dims with |h_i| > mean(|h|)
        mean_abs = h.abs().mean().item()
        diversity = float((h.abs() > mean_abs).float().mean().item())

        # Section: vary by both leading_dim (WM-driven) and step (spec-driven).
        # When WM is degenerate (all leading_dim equal), step alone still varies
        # the section across different specs.
        section_list = SECTION_LABELS.get(domain, SECTION_LABELS["generic"])
        section_idx = (leading_dim + step // max(1, 100 // len(section_list))) % len(section_list)
        section_name = section_list[section_idx]

        # Key hint from leading dimension
        key_hint = "C / Sa"
        for dim_range, key in MUSIC_KEY_BY_DIMS.items():
            if leading_dim in dim_range:
                key_hint = key
                break

        # Rāga / mode hint — seed with (leading_dim XOR spec_id hash) so that
        # different specs get different rāgas even when WM leading_dim is fixed.
        # This is TRIZ Principle 15 (Dynamics): make the static WM state dynamic
        # by combining it with the per-spec identity.
        spec_hash = hash(spec_id) if spec_id else 0
        rng = _random.Random(leading_dim ^ (spec_hash & 0xFFFFFFFF))
        ragas = RAGA_BY_REGISTER.get(register, ["Bhairavi"])
        modes = MODE_BY_REGISTER.get(register, ["Dorian"])
        raga_hint = rng.choice(ragas)
        mode_hint = rng.choice(modes)

        # Register also shifts per spec (when WM is degenerate, rotate through registers)
        if spec_id:
            register_list = list(REGISTER_THRESHOLDS.keys())
            # Blend: WM register is primary, spec hash shifts slightly
            base_idx = register_list.index(register) if register in register_list else 2
            shifted_idx = (base_idx + (spec_hash % 3) - 1) % len(register_list)
            # Only shift if WM energy is near a threshold (ambiguous zone)
            if not any(lo <= energy < hi for reg, (lo, hi) in REGISTER_THRESHOLDS.items()
                       if reg == register and (hi - lo) > 3):
                register = register_list[shifted_idx]

        # Tempo
        tempo_hint = "medium"
        for (lo, hi), (t_label, _) in TEMPO_BY_ENERGY.items():
            if lo <= energy < hi:
                tempo_hint = t_label
                break

        emotion_tags = EMOTION_TAGS.get(register, ["introspection"])

        return CreativeMetadata(
            energy=energy,
            register=register,
            tension_idx=tension_idx,
            section_idx=section_idx,
            leading_dim=leading_dim,
            diversity=diversity,
            domain=domain,
            raga_hint=raga_hint,
            mode_hint=mode_hint,
            tempo_hint=tempo_hint,
            key_hint=key_hint,
            emotion_tags=emotion_tags,
            section_name=section_name,
        )

    def format_for_llm(self, meta: CreativeMetadata) -> str:
        """
        Format CreativeMetadata as a domain-neutral LLM prefix.

        TRIZ Principle 2 (Taking Out): Shaiva vocabulary is NOT present.
        The computational event type (sphurattā → "creative peak",
        camatkāra → "aesthetic moment", vimarśa → "reflective passage")
        is translated to domain vocabulary.
        """
        tension_word = TENSION_LABELS[meta.tension_idx]
        domain = meta.domain

        if domain in ("carnatic", "hindustani"):
            return (
                f"[Creative state: rāga={meta.raga_hint}, "
                f"section={meta.section_name}, "
                f"register={meta.register}, tension={tension_word}, "
                f"tempo={meta.tempo_hint}, "
                f"emotion={'/'.join(meta.emotion_tags[:2])}] "
            )
        elif domain in ("western_pop", "western_jazz", "english_romantic",
                        "english_modernist", "english_beat"):
            return (
                f"[Creative state: mode={meta.mode_hint}, "
                f"key={meta.key_hint}, section={meta.section_name}, "
                f"register={meta.register}, tension={tension_word}, "
                f"tempo={meta.tempo_hint}, "
                f"mood={'/'.join(meta.emotion_tags[:2])}] "
            )
        elif domain in ("kannada_film", "hindi_film"):
            return (
                f"[Creative state: section={meta.section_name}, "
                f"rāga={meta.raga_hint}, register={meta.register}, "
                f"tension={tension_word}, "
                f"mood={'/'.join(meta.emotion_tags[:2])}] "
            )
        elif domain == "world_fusion":
            return (
                f"[Creative state: mode={meta.mode_hint}, "
                f"rāga={meta.raga_hint}, "
                f"section={meta.section_name}, "
                f"register={meta.register}, "
                f"emotion={'/'.join(meta.emotion_tags)}] "
            )
        else:
            return (
                f"[Creative state: register={meta.register}, "
                f"section={meta.section_name}, "
                f"tension={tension_word}, "
                f"mood={'/'.join(meta.emotion_tags[:2])}] "
            )
