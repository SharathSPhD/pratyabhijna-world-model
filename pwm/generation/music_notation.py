"""
music_notation.py — Music structure annotation for PWM creative outputs.

Sprint 4: Every creative output gets actionable music notation that a DAW,
notation software, or custom music app can parse directly.

Philosophical grounding:
  Nāda (Saṅgīta-Ratnākara 1.1, Śārṅgadeva): Sound as the primordial vibration
  that underlies all creative form.  The music notation layer translates the
  WM's latent creative state (register, energy, rāga hint) into the symbolic
  language of specific musical traditions — completing the arc from computational
  metaphysics to playable notation.

Architecture (TRIZ Principle 10 — Prior Action):
  Music metadata is computed from WMStateDecoder output BEFORE LLM generation,
  then injected into both the LLM prompt (as context) and the output record.
  The LLM has structural constraints that guide its creative decisions; the
  output record has machine-parseable notation for the music app.

Output format:
  {
    "notation_type": "carnatic" | "hindustani" | "western_chord" | "western_jazz",
    "tonic": "C" | "D" | "Bb" | ...,
    "raga": "Bhairavi" | "Yaman" | ...,
    "tala": "Adi" | "Rupaka" | ...,
    "tempo_bpm": 60-180,
    "svara_sequence": ["S", "R2", "G2", "M1", "P", "D2", "N2"],  # Carnatic
    "chord_progression": ["Dm", "Gm", "C7", "F"],  # Western
    "scale_degrees": [0, 2, 4, 5, 7, 9, 11],  # semitones from tonic
    "section_structure": ["pallavi", "anupallavi", "caranam"],
    "llm_music_context": "Rāga: Bhairavi | Tāla: Ādi (8 beats) | Tempo: adagio"
  }
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from pwm.generation.domain_metadata import Domain, CreativeMetadata

# ─── Svara Systems ────────────────────────────────────────────────────────────

# Carnatic svarasthānas (72-melakarta system): key svaras by rāga family
CARNATIC_RAGAS: dict[str, dict] = {
    "Bhairavi": {
        "svaras": ["S", "R1", "G1", "M1", "P", "D1", "N1"],
        "vadi": "M1", "samvadi": "S",
        "time": "morning", "emotion": "pathos",
        "scale_semitones": [0, 1, 3, 5, 7, 8, 10],
    },
    "Kambhoji": {
        "svaras": ["S", "R2", "G3", "M1", "P", "D2", "N2"],
        "vadi": "R2", "samvadi": "P",
        "time": "evening", "emotion": "devotion",
        "scale_semitones": [0, 2, 4, 5, 7, 9, 11],
    },
    "Yaman": {
        "svaras": ["S", "R2", "G3", "M2", "P", "D2", "N3"],
        "vadi": "G3", "samvadi": "N3",
        "time": "evening", "emotion": "romantic",
        "scale_semitones": [0, 2, 4, 6, 7, 9, 11],
    },
    "Todi": {
        "svaras": ["S", "R1", "G1", "M2", "P", "D1", "N2"],
        "vadi": "D1", "samvadi": "G1",
        "time": "morning", "emotion": "pathos",
        "scale_semitones": [0, 1, 3, 6, 7, 8, 10],
    },
    "Shankarabharanam": {
        "svaras": ["S", "R2", "G3", "M1", "P", "D2", "N3"],
        "vadi": "G3", "samvadi": "N3",
        "time": "afternoon", "emotion": "heroic",
        "scale_semitones": [0, 2, 4, 5, 7, 9, 11],  # matches major scale
    },
    "Kalyani": {
        "svaras": ["S", "R2", "G3", "M2", "P", "D2", "N3"],
        "vadi": "M2", "samvadi": "S",
        "time": "evening", "emotion": "serene",
        "scale_semitones": [0, 2, 4, 6, 7, 9, 11],  # Lydian
    },
    "Charukeshi": {
        "svaras": ["S", "R2", "G3", "M1", "P", "D1", "N1"],
        "vadi": "P", "samvadi": "S",
        "time": "any", "emotion": "contemplative",
        "scale_semitones": [0, 2, 4, 5, 7, 8, 10],
    },
    "Kiravani": {
        "svaras": ["S", "R2", "G1", "M1", "P", "D1", "N3"],
        "vadi": "N3", "samvadi": "G1",
        "time": "night", "emotion": "intense",
        "scale_semitones": [0, 2, 3, 5, 7, 8, 11],  # matches harmonic minor
    },
}

# Tāḷa system: beats and subdivisions
CARNATIC_TALAS: dict[str, dict] = {
    "Adi": {"beats": 8, "subdivisions": [4, 2, 2], "description": "8-beat cycle"},
    "Rupaka": {"beats": 6, "subdivisions": [2, 4], "description": "6-beat cycle"},
    "Misra Chapu": {"beats": 7, "subdivisions": [3, 2, 2], "description": "7-beat cycle"},
    "Khanda Chapu": {"beats": 5, "subdivisions": [2, 3], "description": "5-beat cycle"},
    "Dhruva": {"beats": 14, "subdivisions": [4, 2, 4, 4], "description": "14-beat cycle"},
}

HINDUSTANI_RAGAS: dict[str, dict] = {
    "Bhairav": {
        "svaras": ["S", "r", "G", "M", "P", "d", "N"],
        "time": "morning", "emotion": "devotion",
        "scale_semitones": [0, 1, 4, 5, 7, 8, 11],
    },
    "Yaman": {
        "svaras": ["S", "R", "G", "m", "P", "D", "N"],
        "time": "evening", "emotion": "romantic",
        "scale_semitones": [0, 2, 4, 6, 7, 9, 11],
    },
    "Bhairavi": {
        "svaras": ["S", "r", "g", "M", "P", "d", "n"],
        "time": "morning", "emotion": "pathos",
        "scale_semitones": [0, 1, 3, 5, 7, 8, 10],
    },
    "Desh": {
        "svaras": ["S", "R", "G", "M", "P", "D", "n"],
        "time": "monsoon night", "emotion": "longing",
        "scale_semitones": [0, 2, 4, 5, 7, 9, 10],
    },
    "Mian Ki Malhar": {
        "svaras": ["S", "R", "g", "M", "P", "D", "n"],
        "time": "monsoon", "emotion": "yearning",
        "scale_semitones": [0, 2, 3, 5, 7, 9, 10],
    },
}

HINDUSTANI_TALAS: dict[str, dict] = {
    "Teentaal": {"beats": 16, "vibhags": [4, 4, 4, 4], "sam": 1},
    "Ektaal": {"beats": 12, "vibhags": [2, 2, 2, 2, 2, 2], "sam": 1},
    "Jhaptaal": {"beats": 10, "vibhags": [2, 3, 2, 3], "sam": 1},
    "Rupak": {"beats": 7, "vibhags": [3, 2, 2], "sam": 2},
}

# ─── Western Chord Systems ────────────────────────────────────────────────────

# Map musical modes to chord progressions
WESTERN_MODES: dict[str, dict] = {
    "Ionian":    {"scale": [0,2,4,5,7,9,11], "character": "bright, major",
                  "progressions": [["I","IV","V","I"], ["I","vi","IV","V"], ["I","V","vi","IV"]]},
    "Dorian":    {"scale": [0,2,3,5,7,9,10], "character": "minor with bright 6th",
                  "progressions": [["i","IV","i","IV"], ["i","VII","IV","i"], ["i","IV","VII","III"]]},
    "Phrygian":  {"scale": [0,1,3,5,7,8,10], "character": "Spanish, intense",
                  "progressions": [["i","bII","bVII","i"], ["i","bII","i","bVII"]]},
    "Lydian":    {"scale": [0,2,4,6,7,9,11], "character": "dreamy, floating",
                  "progressions": [["I","II","I","II"], ["I","II","vii","I"]]},
    "Mixolydian":{"scale": [0,2,4,5,7,9,10], "character": "bluesy, dominant",
                  "progressions": [["I","bVII","IV","I"], ["I","bVII","I","bVII"]]},
    "Aeolian":   {"scale": [0,2,3,5,7,8,10], "character": "natural minor, melancholic",
                  "progressions": [["i","VII","VI","VII"], ["i","iv","VII","III"], ["i","VI","III","VII"]]},
    "Locrian":   {"scale": [0,1,3,5,6,8,10], "character": "unstable, dissonant",
                  "progressions": [["i°","bII","bvii°","i°"]]},
}

# Jazz chord vocabulary
JAZZ_VOICINGS: dict[str, list[str]] = {
    "bebop":   ["maj7", "m7", "7", "m7b5", "dim7", "aug7", "maj9", "m9", "13", "b9"],
    "modal":   ["maj7", "m7", "sus4", "add9", "11", "maj7#11", "m11"],
    "bossa":   ["maj7", "m7", "7", "m7b5", "maj9", "6/9", "m6"],
    "blues":   ["7", "9", "13", "b9", "#9"],
    "post_bop":["maj7", "7alt", "m7b5", "dim7", "aug", "7#11", "7b9"],
}

JAZZ_PROGRESSIONS: list[list[str]] = [
    ["IIm7", "V7", "Imaj7", "Imaj7"],         # II-V-I (major)
    ["IIm7b5", "V7b9", "Im7", "Im7"],          # II-V-i (minor)
    ["Imaj7", "VI7", "IIm7", "V7"],            # I-VI-II-V (rhythm changes)
    ["IIm7", "V7", "IIIm7", "VI7", "IIm7", "V7", "Imaj7", "Imaj7"],  # extended turnaround
    ["Imaj7", "bVIImaj7", "Imaj7", "bVIImaj7"],  # modal vamp
]

# ─── Tonics and Keys ─────────────────────────────────────────────────────────

WESTERN_TONICS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
CARNATIC_TONICS = ["C", "C#", "D", "D#", "E", "F"]  # Carnatic sa positions


# ─── Section Structures ───────────────────────────────────────────────────────

SECTION_STRUCTURES: dict[Domain, list[str]] = {
    "carnatic":          ["pallavi", "anupallavi", "caranam"],
    "hindustani":        ["alap", "jod", "jhala", "bandish-sthāyi", "bandish-antarā"],
    "western_pop":       ["intro", "verse", "chorus", "bridge", "outro"],
    "western_jazz":      ["head", "solo-I", "solo-II", "head-out"],
    "kannada_film":      ["mukhara", "charana-1", "charana-2"],
    "hindi_film":        ["mukhra", "antara-1", "antara-2"],
    "english_romantic":  ["stanza-1", "stanza-2", "stanza-3", "stanza-4"],
    "english_modernist": ["section-I", "section-II", "section-III", "section-IV"],
    "english_beat":      ["movement-I", "movement-II", "movement-III"],
    "bengali_lyric":     ["pratham-stabak", "dwitiya-stabak", "tritiya-stabak"],
    "tamil_classical":   ["akam", "puram", "kural-venpa"],
    "telugu_padyam":     ["mangalacaranam", "padyam-1", "padyam-2", "padyam-3"],
    "sanskrit_classical":["prathamah", "dvitiyah", "tritiyah", "caturthah"],
    "world_fusion":      ["verse-en", "verse-hi", "verse-ta", "verse-bn", "coda"],
    "generic":           ["section-1", "section-2", "section-3"],
}


# ─── Main Annotation Dataclass ────────────────────────────────────────────────

@dataclass
class MusicNotation:
    """Full music notation for one creative work."""
    domain: Domain
    notation_type: Literal["carnatic", "hindustani", "western_chord", "western_jazz", "generic"]
    tonic: str
    tempo_bpm: int
    section_structure: list[str]

    # Carnatic/Hindustani fields
    raga: str = ""
    tala: str = ""
    svara_sequence: list[str] = field(default_factory=list)
    vadi: str = ""
    samvadi: str = ""

    # Western fields
    mode: str = ""
    chord_progression: list[str] = field(default_factory=list)
    scale_degrees: list[int] = field(default_factory=list)
    jazz_style: str = ""

    # LLM-facing context string (no Shaiva vocabulary)
    llm_music_context: str = ""

    def to_dict(self) -> dict:
        return {
            "notation_type": self.notation_type,
            "tonic": self.tonic,
            "tempo_bpm": self.tempo_bpm,
            "raga": self.raga,
            "tala": self.tala,
            "svara_sequence": self.svara_sequence,
            "vadi": self.vadi,
            "samvadi": self.samvadi,
            "mode": self.mode,
            "chord_progression": self.chord_progression,
            "scale_degrees": self.scale_degrees,
            "jazz_style": self.jazz_style,
            "section_structure": self.section_structure,
            "llm_music_context": self.llm_music_context,
        }


# ─── Notation Generator ───────────────────────────────────────────────────────

class MusicNotationGenerator:
    """
    Generates domain-appropriate music notation from WM creative metadata.

    TRIZ Principle 10 (Prior Action): notation is computed before LLM generation
    so the LLM receives structural constraints that improve creative coherence.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate(self, meta: CreativeMetadata, domain: Domain,
                 spec_id: str = "") -> MusicNotation:
        """
        Generate music notation from WM metadata.

        The WM's creative state (register, energy, rāga hint) drives tonic
        selection, tempo, and emotional character of the notation.
        """
        # Seed deterministically per spec to ensure reproducibility
        rng = random.Random(hash(spec_id) ^ int(meta.energy * 100))

        sections = SECTION_STRUCTURES.get(domain, SECTION_STRUCTURES["generic"])
        tempo = self._energy_to_tempo(meta.energy, rng)

        if domain == "carnatic":
            return self._carnatic(meta, domain, sections, tempo, rng)
        elif domain == "hindustani":
            return self._hindustani(meta, domain, sections, tempo, rng)
        elif domain in ("western_jazz",):
            return self._jazz(meta, domain, sections, tempo, rng)
        elif domain in ("western_pop", "english_romantic", "english_modernist",
                        "english_beat", "world_fusion"):
            return self._western_chord(meta, domain, sections, tempo, rng)
        elif domain in ("kannada_film", "hindi_film", "bengali_lyric"):
            return self._film_song(meta, domain, sections, tempo, rng)
        elif domain in ("carnatic", "telugu_padyam", "tamil_classical"):
            return self._carnatic(meta, domain, sections, tempo, rng)
        else:
            return self._generic(meta, domain, sections, tempo, rng)

    def _energy_to_tempo(self, energy: float, rng: random.Random) -> int:
        """Map WM energy (0-20) to BPM. High energy = faster tempo."""
        # energy 6-13 = peak creative zone; map to 60-160 BPM
        if energy < 4:
            base = 40 + int(energy * 5)    # 40-60 BPM (very slow)
        elif energy < 8:
            base = 60 + int((energy - 4) * 15)   # 60-120 BPM
        elif energy < 12:
            base = 120 + int((energy - 8) * 10)  # 120-160 BPM
        else:
            base = 160 + int((energy - 12) * 5)  # 160-185 BPM (fast)
        # Add slight variation
        return base + rng.randint(-5, 5)

    def _pick_tonic(self, meta: CreativeMetadata, options: list[str],
                    rng: random.Random) -> str:
        """Pick tonic from energy+register."""
        # Higher energy → sharper keys; lower energy → flatter keys
        idx = int(meta.energy / 20.0 * len(options)) % len(options)
        return options[idx]

    def _carnatic(self, meta: CreativeMetadata, domain: Domain,
                  sections: list[str], tempo: int, rng: random.Random) -> MusicNotation:
        # Use rāga hint from meta, or pick from CARNATIC_RAGAS
        raga_name = meta.raga_hint if meta.raga_hint in CARNATIC_RAGAS else \
            rng.choice(list(CARNATIC_RAGAS.keys()))
        raga = CARNATIC_RAGAS[raga_name]

        tala_name = rng.choice(["Adi", "Rupaka", "Misra Chapu"])
        tala = CARNATIC_TALAS[tala_name]

        tonic = self._pick_tonic(meta, CARNATIC_TONICS, rng)

        ctx = (
            f"Rāga: {raga_name} | Sa: {tonic} | Tāḷa: {tala_name} ({tala['beats']} beats) | "
            f"Tempo: {tempo} BPM | Vādi: {raga['vadi']} | Saṁvādi: {raga['samvadi']} | "
            f"Section: {sections[0]}"
        )

        return MusicNotation(
            domain=domain,
            notation_type="carnatic",
            tonic=tonic,
            tempo_bpm=tempo,
            section_structure=sections,
            raga=raga_name,
            tala=tala_name,
            svara_sequence=raga["svaras"],
            vadi=raga["vadi"],
            samvadi=raga["samvadi"],
            scale_degrees=raga["scale_semitones"],
            llm_music_context=ctx,
        )

    def _hindustani(self, meta: CreativeMetadata, domain: Domain,
                    sections: list[str], tempo: int, rng: random.Random) -> MusicNotation:
        raga_name = meta.raga_hint if meta.raga_hint in HINDUSTANI_RAGAS else \
            rng.choice(list(HINDUSTANI_RAGAS.keys()))
        raga = HINDUSTANI_RAGAS[raga_name]

        tala_name = rng.choice(["Teentaal", "Ektaal", "Jhaptaal"])
        tala = HINDUSTANI_TALAS[tala_name]

        tonic = self._pick_tonic(meta, WESTERN_TONICS, rng)

        # Vilambit (slow) vs drut (fast) based on tempo
        laya = "vilambit" if tempo < 80 else ("madhya" if tempo < 130 else "drut")

        ctx = (
            f"Rāga: {raga_name} | Sa: {tonic} | Tāla: {tala_name} ({tala['beats']} mātrā) | "
            f"Laya: {laya} | Tempo: {tempo} BPM | Emotion: {raga['time']}"
        )

        return MusicNotation(
            domain=domain,
            notation_type="hindustani",
            tonic=tonic,
            tempo_bpm=tempo,
            section_structure=sections,
            raga=raga_name,
            tala=tala_name,
            svara_sequence=raga["svaras"],
            scale_degrees=raga["scale_semitones"],
            llm_music_context=ctx,
        )

    def _western_chord(self, meta: CreativeMetadata, domain: Domain,
                       sections: list[str], tempo: int, rng: random.Random) -> MusicNotation:
        # Pick mode based on energy and register
        if meta.register == "whisper":
            mode_name = rng.choice(["Dorian", "Aeolian", "Phrygian"])
        elif meta.register == "ardent":
            mode_name = rng.choice(["Ionian", "Lydian", "Mixolydian"])
        else:
            mode_name = rng.choice(list(WESTERN_MODES.keys()))

        mode = WESTERN_MODES[mode_name]
        progression = rng.choice(mode["progressions"])
        tonic = rng.choice(WESTERN_TONICS)

        ctx = (
            f"Key: {tonic} {mode_name} | Chord progression: {' - '.join(progression)} | "
            f"Tempo: {tempo} BPM | Character: {mode['character']}"
        )

        return MusicNotation(
            domain=domain,
            notation_type="western_chord",
            tonic=tonic,
            tempo_bpm=tempo,
            section_structure=sections,
            mode=mode_name,
            chord_progression=progression,
            scale_degrees=mode["scale"],
            llm_music_context=ctx,
        )

    def _jazz(self, meta: CreativeMetadata, domain: Domain,
              sections: list[str], tempo: int, rng: random.Random) -> MusicNotation:
        style = rng.choice(["bebop", "modal", "bossa", "blues", "post_bop"])
        progression = rng.choice(JAZZ_PROGRESSIONS)
        tonic = rng.choice(WESTERN_TONICS)
        mode_name = rng.choice(["Dorian", "Mixolydian", "Aeolian"])
        mode = WESTERN_MODES[mode_name]

        ctx = (
            f"Style: {style} jazz | Key center: {tonic} | "
            f"Progression: {' - '.join(progression)} | "
            f"Mode: {mode_name} | Tempo: {tempo} BPM"
        )

        return MusicNotation(
            domain=domain,
            notation_type="western_jazz",
            tonic=tonic,
            tempo_bpm=tempo,
            section_structure=sections,
            mode=mode_name,
            chord_progression=progression,
            scale_degrees=mode["scale"],
            jazz_style=style,
            llm_music_context=ctx,
        )

    def _film_song(self, meta: CreativeMetadata, domain: Domain,
                   sections: list[str], tempo: int, rng: random.Random) -> MusicNotation:
        """Hindi/Kannada film songs often blend Carnatic rāgas with Western harmony."""
        # Film songs primarily use Hindustani/Carnatic rāgas
        raga_name = rng.choice(["Bhairavi", "Desh", "Yaman", "Kambhoji", "Bhairav"])
        if raga_name in HINDUSTANI_RAGAS:
            raga = HINDUSTANI_RAGAS[raga_name]
            svaras = raga["svaras"]
            scale_sem = raga["scale_semitones"]
        elif raga_name in CARNATIC_RAGAS:
            raga = CARNATIC_RAGAS[raga_name]
            svaras = raga["svaras"]
            scale_sem = raga["scale_semitones"]
        else:
            svaras = ["S", "R", "G", "M", "P", "D", "N"]
            scale_sem = [0, 2, 4, 5, 7, 9, 11]

        tonic = rng.choice(["C", "D", "E", "F"])
        tala_name = "Keherwa (8)" if domain == "hindi_film" else "Adi (8)"

        ctx = (
            f"Rāga: {raga_name} | Sa: {tonic} | Tāla: {tala_name} | "
            f"Tempo: {tempo} BPM | Structure: {' → '.join(sections)}"
        )

        return MusicNotation(
            domain=domain,
            notation_type="carnatic",
            tonic=tonic,
            tempo_bpm=tempo,
            section_structure=sections,
            raga=raga_name,
            tala=tala_name,
            svara_sequence=svaras,
            scale_degrees=scale_sem,
            llm_music_context=ctx,
        )

    def _generic(self, meta: CreativeMetadata, domain: Domain,
                 sections: list[str], tempo: int, rng: random.Random) -> MusicNotation:
        tonic = rng.choice(WESTERN_TONICS)
        mode_name = "Ionian"
        mode = WESTERN_MODES[mode_name]
        progression = rng.choice(mode["progressions"])
        ctx = f"Key: {tonic} | Tempo: {tempo} BPM | Structure: {' → '.join(sections)}"
        return MusicNotation(
            domain=domain,
            notation_type="generic",
            tonic=tonic,
            tempo_bpm=tempo,
            section_structure=sections,
            mode=mode_name,
            chord_progression=progression,
            scale_degrees=mode["scale"],
            llm_music_context=ctx,
        )


# ─── Convenience function ─────────────────────────────────────────────────────

_DEFAULT_GENERATOR = MusicNotationGenerator()


def annotate(meta: CreativeMetadata, domain: Domain, spec_id: str = "") -> MusicNotation:
    """Annotate a creative work with music notation.  Thread-safe (per-call rng)."""
    return _DEFAULT_GENERATOR.generate(meta, domain, spec_id=spec_id)
