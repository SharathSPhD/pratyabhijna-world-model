"""
creative_specs.py — Domain-neutral creative generation specifications.

TRIZ Principle 2 (Taking Out): No Shaiva vocabulary appears in any system
prompt or user prompt. The WM structure (EFE/VFE/sphurattā) provides the
conditioning prefix via domain_metadata.WMStateDecoder; the LLM receives
only domain-appropriate creative framing.

Music orientation (per product brief): Each spec includes structured_output_hints
that guide the LLM toward singable, music-app-compatible output. The downstream
music app receives: text (poem/lyric), structure (sections), musical_context
(rāga/mode/tempo/key).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from pwm.generation.domain_metadata import Domain


@dataclass
class CreativeSpec:
    """Specification for one creative generation task."""
    id: str
    title: str
    language: str                   # iso code or descriptive
    domain: Domain
    system_prompt: str              # domain-neutral, no Shaiva vocab
    user_prompt: str                # specific creative task
    num_predict: int = 900
    temperature: float = 0.88
    top_p: float = 0.92
    structured_output_hints: list[str] = field(default_factory=list)
    music_context: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — domain-neutral master creative prompt
# ─────────────────────────────────────────────────────────────────────────────

MASTER_SYSTEM = (
    "You are a master poet, lyricist, and composer fluent in classical and contemporary "
    "creative traditions across cultures. You write directly in the requested form: "
    "no preamble, no explanation, no meta-commentary, no reasoning traces. "
    "Begin the creative work immediately on the first line. "
    "Every line must be complete — no ellipses as placeholders. "
    "When given a [Creative state: ...] prefix, use it to set emotional register, "
    "pace, and section structure for the piece. Ignore any technical labels in the "
    "prefix that are unfamiliar; focus on the register, mood, and section name."
)

# ─────────────────────────────────────────────────────────────────────────────
# SANSKRIT
# ─────────────────────────────────────────────────────────────────────────────

s01 = CreativeSpec(
    id="s01",
    title="Sanskrit Śloka — Anuṣṭubh, Nature and Perception",
    language="sanskrit",
    domain="sanskrit_classical",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Sanskrit śloka (4 verses, anuṣṭubh metre, 8 syllables per quarter) "
        "on the theme of perception awakening in nature — a bird call at dawn, dew on grass, "
        "the moment of recognising a familiar landscape. "
        "Vocabulary: classical Sanskrit with natural imagery (pakṣin, śiśira, madhura, sphurita, "
        "kirana, prasanna). No religious or philosophical vocabulary. "
        "Write Devanāgarī with IAST transliteration below each verse. "
        "Begin immediately with verse 1."
    ),
    music_context={"raga": "Bhairav", "tala": "Rupaka", "tempo": "vilambit"},
    structured_output_hints=["4 verses", "8-syllable quarters", "nature imagery"],
)

s02 = CreativeSpec(
    id="s02",
    title="Sanskrit Śloka — Monsoon, Memory, Longing",
    language="sanskrit",
    domain="sanskrit_classical",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Sanskrit śloka (4 verses, anuṣṭubh) on the monsoon as a metaphor "
        "for longing and memory. A traveller hears the first rain and remembers home. "
        "Classical imagery: megha (cloud), pravāsa (exile), smṛti (memory), nīpa (kadamba flower). "
        "Inspired by Kālidāsa's Meghadūta tradition — emotional, lyrical, not philosophical. "
        "Write Devanāgarī with IAST transliteration. Begin immediately with verse 1."
    ),
    music_context={"raga": "Miyan ki Malhar", "tala": "Teental", "tempo": "madhya"},
    structured_output_hints=["4 verses", "monsoon imagery", "emotional tone"],
)

# ─────────────────────────────────────────────────────────────────────────────
# KANNADA
# ─────────────────────────────────────────────────────────────────────────────

k01 = CreativeSpec(
    id="k01",
    title="Kannada Vachana — Secular, Inner Life",
    language="kannada",
    domain="kannada_film",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Kannada vachana (free-verse lyric poem, 8–12 lines) in the tradition "
        "of 12th-century Vachana Sahitya, but with a SECULAR theme: the inner life of a craftsperson, "
        "a farmer, a weaver, or an artisan — finding meaning in skilled work. "
        "No religious content. Use simple, direct Kannada. Short lines, free verse, direct address. "
        "Close with a signature phrase (ankita) using a craft or nature image. "
        "Write entirely in Kannada script. Begin immediately with the first line."
    ),
    music_context={"raga": "Bageshri", "tala": "Adi", "tempo": "madhya"},
    structured_output_hints=["free verse", "8-12 lines", "secular craft theme", "ankita signature"],
)

k02 = CreativeSpec(
    id="k02",
    title="Kannada Vachana — Doubt, Emptiness, City Life",
    language="kannada",
    domain="kannada_film",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Kannada vachana (10–14 lines) in the Allama Prabhu tradition of paradox "
        "and emptiness (śūnya), but applied to CONTEMPORARY urban life: the emptiness of a city "
        "apartment, commuting in silence, digital isolation. No religious vocabulary. "
        "Use paradoxical, imagistic style — short lines, rhetorical questions, stark images. "
        "Write entirely in Kannada script. Begin immediately."
    ),
    music_context={"raga": "Darbari Kanada", "tala": "Misra Chapu", "tempo": "slow"},
    structured_output_hints=["paradoxical imagery", "urban theme", "10-14 lines"],
)

k03 = CreativeSpec(
    id="k03",
    title="Kannada Kīrtana — Nature, Seasons, Devotion to Earth",
    language="kannada",
    domain="carnatic",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Kannada kīrtana (Dasa Sahitya format) with devotion to the EARTH "
        "and nature — not to a deity. Format: Pallavi (2 lines, repeated refrain), "
        "Anupallavi (2 lines, complementary), Caraṇam (4 lines, development). "
        "Theme: the farmer's gratitude to rain, soil, seasons. Use seasonal imagery: "
        "ಮಳೆ (rain), ಮಣ್ಣು (earth), ಬೆಳೆ (harvest), ಹೂ (flower). "
        "Close Caraṇam with an ankita using a natural image (e.g., 'ಭೂಮಿ ತಾಯಿ'). "
        "Write in Kannada script. Begin immediately with 'ಪಲ್ಲವಿ:' on line 1."
    ),
    music_context={"raga": "Kapi", "tala": "Adi", "tempo": "madhya"},
    structured_output_hints=["pallavi", "anupallavi", "caranam", "nature ankita"],
)

k04 = CreativeSpec(
    id="k04",
    title="Kannada Ugābhoga — Wanderer, Solitude, Starlight",
    language="kannada",
    domain="carnatic",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Kannada ugābhoga (free-form lyric prose-poem, 6–10 lines) "
        "in the Kanakadasa tradition — contemplative, wandering meditation. "
        "Theme: a solitary night walker observing stars, the river, sounds of the city sleeping. "
        "No religious content. Meditative, lyrical, free metre. "
        "Close with a quiet image of arrival or rest. "
        "Write in Kannada script. Begin immediately."
    ),
    music_context={"raga": "Kalyani", "tala": "free", "tempo": "slow"},
    structured_output_hints=["free metre", "6-10 lines", "nocturnal imagery"],
)

k05 = CreativeSpec(
    id="k05",
    title="Kannada Bhavageete — Monsoon, Red Earth, First Rain",
    language="kannada",
    domain="kannada_film",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Kannada bhavageete (lyric poem, 4 stanzas × 4 lines) "
        "in the style of G.S. Shivarudrappa — meditative, modern literary Kannada. "
        "Theme: the first rain on kumkuma-red earth, jasmine opening, a child running barefoot. "
        "Each stanza: one controlling image, developed through 4 lines. "
        "No spiritual content. Grounded in sensory imagery: smell, sound, colour, touch. "
        "Write in Kannada script. Begin immediately with stanza 1."
    ),
    music_context={"raga": "Kapi", "tala": "Adi", "tempo": "medium"},
    structured_output_hints=["4 stanzas × 4 lines", "sensory imagery", "monsoon theme"],
)

# ─────────────────────────────────────────────────────────────────────────────
# HINDI
# ─────────────────────────────────────────────────────────────────────────────

h01 = CreativeSpec(
    id="h01",
    title="Hindi Ghazal — City Rain, Longing, Monsoon Streets",
    language="hindi",
    domain="hindi_film",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Hindi ghazal (7 couplets, matla + 5 sher + maqta). "
        "Radif (refrain): 'baarish mein' (in the rain). "
        "Theme: the smell of wet city streets, neon lights in puddles, someone not there. "
        "Maqta: poet's name/signature couplet. Classical ghazal structure but contemporary imagery. "
        "Write in Devanāgarī with consistent rhyme (qafia) before the radif. "
        "Begin immediately with the matla (opening couplet)."
    ),
    music_context={"raga": "Bhimpalasi", "tala": "Teental", "tempo": "madhya"},
    structured_output_hints=["7 couplets", "consistent radif 'baarish mein'", "qafia rhyme"],
)

h02 = CreativeSpec(
    id="h02",
    title="Hindi Film Song — Romantic, Separation, Journey",
    language="hindi",
    domain="hindi_film",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Hindi film song (Bollywood style, circa 1970s–80s golden era). "
        "Format: Mukhra (2-line hook), Antara 1 (4 lines), Antara 2 (4 lines). "
        "Theme: a lover departing on a train, looking back at the platform, the last glimpse. "
        "Emotional register: bittersweet, lyrical. Use imagery: platform, fog, handkerchief, "
        "whistle, silhouette. Singable syllable counts — keep lines 8–12 syllables. "
        "Write in Devanāgarī. Begin immediately with 'मुखड़ा:' on line 1."
    ),
    music_context={"raga": "Yaman", "tala": "Dadra", "tempo": "medium-slow"},
    structured_output_hints=["mukhra 2 lines", "antara_1 4 lines", "antara_2 4 lines", "singable"],
)

# ─────────────────────────────────────────────────────────────────────────────
# TAMIL
# ─────────────────────────────────────────────────────────────────────────────

ta01 = CreativeSpec(
    id="ta01",
    title="Tamil Poem — Sangam Style, Sea, Separation",
    language="tamil",
    domain="tamil_classical",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Tamil poem (8–12 lines) in the Akam (interior/love) tradition "
        "of Sangam poetry. Tinai (landscape): Neytal (seashore, blue waterlily, heron). "
        "Theme: a woman waiting for her lover who has gone to sea; the sea at dusk. "
        "Use classical Tamil imagery: குயில் (kuyil/cuckoo), கடல் (kadal/sea), "
        "நெய்தல் (neytal/blue waterlily), ஓர் (longing). "
        "Modern literary Tamil, free verse, no archaic grammar needed. "
        "Write in Tamil script. Begin immediately with the first line."
    ),
    music_context={"raga": "Bhairavi", "tala": "Rupaka", "tempo": "slow"},
    structured_output_hints=["8-12 lines", "seashore imagery", "akam tradition"],
)

# ─────────────────────────────────────────────────────────────────────────────
# TELUGU
# ─────────────────────────────────────────────────────────────────────────────

te01 = CreativeSpec(
    id="te01",
    title="Telugu Padyamu — Nature, River, Evening",
    language="telugu",
    domain="telugu_padyam",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Telugu padyamu (classical Telugu verse, 4 stanzas). "
        "Theme: the evening hour on the banks of the Krishna river — fishermen returning, "
        "birds settling, lamps lit in distant huts. Sensory, lyrical, grounded in nature. "
        "Use literary Telugu vocabulary: నది (nadi/river), సంధ్య (sandhya/twilight), "
        "పక్షులు (pakshulu/birds), దీపం (deepam/lamp). "
        "Write in Telugu script. Begin immediately with stanza 1."
    ),
    music_context={"raga": "Hindolam", "tala": "Misra Chapu", "tempo": "medium-slow"},
    structured_output_hints=["4 stanzas", "evening imagery", "Krishna river"],
)

# ─────────────────────────────────────────────────────────────────────────────
# BENGALI
# ─────────────────────────────────────────────────────────────────────────────

bn01 = CreativeSpec(
    id="bn01",
    title="Bengali Lyric — Tagore Tradition, Seasonal Longing",
    language="bengali",
    domain="bengali_lyric",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Bengali lyric poem (3 stanzas × 5 lines) in the Rabindra Sangeet "
        "tradition — melodic, imageistic, full of seasonal feeling (Baul/Tagore blend). "
        "Theme: the arrival of Basanta (spring) — flowers, wind, bees, the feeling of "
        "something about to begin. Joyful, light, singable. "
        "Use seasonal Bengali: ফুল (phul/flower), বাতাস (batas/wind), আলো (alo/light), "
        "বসন্ত (Basanta/spring). "
        "Write in Bengali script. Begin immediately with stanza 1."
    ),
    music_context={"raga": "Pahadi", "tala": "Dadra", "tempo": "medium"},
    structured_output_hints=["3 stanzas × 5 lines", "spring theme", "singable"],
)

# ─────────────────────────────────────────────────────────────────────────────
# ENGLISH
# ─────────────────────────────────────────────────────────────────────────────

e01 = CreativeSpec(
    id="e01",
    title="English Ode — Romantic, Autumn, Transience",
    language="english",
    domain="english_romantic",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original English ode (4 stanzas × 8 lines) in the tradition of Keats's odes — "
        "rich sensory language, one sustained meditation on a single subject. "
        "Subject: autumn light on water — the specific quality of October afternoon light "
        "hitting a still lake. Develop the image through 4 stanzas: approach, stillness, "
        "reflection, departure. No abstract philosophy — only image and feeling. "
        "Begin immediately with stanza 1."
    ),
    music_context={"mode": "Dorian", "key": "D minor", "tempo": "slow", "time_sig": "6/8"},
    structured_output_hints=["4 stanzas × 8 lines", "sustained single image", "autumn light"],
)

e02 = CreativeSpec(
    id="e02",
    title="English Modernist — Fragmented Urban Interior",
    language="english",
    domain="english_modernist",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original English poem (4 sections, I–IV, 6–8 lines each) in the high Modernist "
        "tradition (Eliot, Stevens) — fragmented perspective, interior monologue, unexpected "
        "juxtaposition. Setting: a hospital waiting room, late afternoon. Fragments: the sound "
        "system, a magazine, a window with pigeons, another person's shoes. "
        "No narrative resolution. End in mid-thought. "
        "Begin immediately with 'I.' on the first line."
    ),
    music_context={"mode": "Phrygian", "key": "E minor", "tempo": "irregular", "time_sig": "free"},
    structured_output_hints=["4 sections", "fragmented", "no resolution", "interior monologue"],
)

e03 = CreativeSpec(
    id="e03",
    title="English Beat Poem — City Night, Jazz, Freedom",
    language="english",
    domain="english_beat",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original English Beat poem (40–60 lines) in the Ginsberg/Kerouac tradition — "
        "long breath lines, jazz rhythms, catalogues of American/global city life. "
        "Subject: a single night in a city — starting at dusk, moving through bars, "
        "streets, all-night diners, ending at dawn. Characters encountered. "
        "Energy builds then releases. No Eastern mysticism — grounded in urban physicality: "
        "neon, exhaust, laughter, the particular weight of night air. "
        "Begin immediately with the first long line."
    ),
    music_context={"mode": "Blues", "key": "Bb", "tempo": "medium-fast", "time_sig": "swing"},
    structured_output_hints=["40-60 lines", "long breath lines", "night to dawn arc"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SONG / MUSIC
# ─────────────────────────────────────────────────────────────────────────────

l01 = CreativeSpec(
    id="l01",
    title="Carnatic Kṛti — Rāga Bhairavi, Nature Devotion",
    language="sanskrit_telugu",
    domain="carnatic",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Carnatic kṛti (composition) for Rāga Bhairavi, Ādi Tāla. "
        "Language: Sanskrit with Telugu phrases (mix as Tyagaraja and Dikshitar did). "
        "Format: Pallavi (2 lines), Anupallavi (2 lines), Caraṇam (4 lines). "
        "Theme: devotion to MUSIC ITSELF as the divine — not to a personal deity. "
        "The rāga as the object of reverence. "
        "Include: gamaka hints (G for gamakas), svarakalpana note: where svara improvisation fits. "
        "Include tāla beat count (1–8) at the start of each line. "
        "Begin immediately with 'Pallavi:' on line 1."
    ),
    music_context={"raga": "Bhairavi", "tala": "Adi (8 beats)", "tempo": "madhya laya",
                   "gamaka_style": "Tyagaraja", "svara_section": "after caranam"},
    structured_output_hints=["pallavi", "anupallavi", "caranam", "tala beats", "gamaka marks"],
)

l02 = CreativeSpec(
    id="l02",
    title="Kannada Film Song — Romantic, Nature, Golden Era",
    language="kannada",
    domain="kannada_film",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original Kannada film song in the style of 1960s–80s golden era "
        "(Rajkumar films — composers Rajan-Nagendra, G.K. Venkatesh). "
        "Format: Mukhara (2-line opening hook), Charaṇa 1 (4 lines), Charaṇa 2 (4 lines). "
        "Each section in Kannada script followed by loose English meaning in parentheses. "
        "Theme: two people meeting at a village fair at dusk, colours and lamps and music. "
        "Rāga: Kāpi. Keep each line 8–10 syllables for singability. "
        "Begin immediately with 'ಮುಖಾರ:' on line 1."
    ),
    music_context={"raga": "Kapi", "tala": "Adi", "tempo": "medium",
                   "instrument_hint": "harmonium, tabla, violin"},
    structured_output_hints=["mukhara", "charana_1", "charana_2", "bilingual", "8-10 syllables"],
)

l03 = CreativeSpec(
    id="l03",
    title="Jazz Spiritual — Four Movements, Love and Sound",
    language="english",
    domain="western_jazz",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original jazz spiritual poem in the tradition of Coltrane's A Love Supreme. "
        "Four movements: I. RESOLUTION, II. PURSUANCE, III. ACKNOWLEDGEMENT, IV. PSALM. "
        "Each movement: 6–8 lines. "
        "Subject: SOUND ITSELF as the supreme experience — the relationship between "
        "musician and instrument, listener and tone, silence and note. "
        "No Eastern mysticism. Voice: ecstatic, searching, call-and-response. "
        "Use jazz vocabulary: blue note, chord, release, resolution, drone, overtone. "
        "Begin immediately with 'I. RESOLUTION' on line 1."
    ),
    music_context={"mode": "Mixolydian", "key": "F", "tempo": "variable", "time_sig": "free",
                   "instruments": "tenor saxophone, piano, bass, drums"},
    structured_output_hints=["4 movements", "6-8 lines each", "jazz vocabulary", "call-response"],
)

# ─────────────────────────────────────────────────────────────────────────────
# World fusion
# ─────────────────────────────────────────────────────────────────────────────

w01 = CreativeSpec(
    id="w01",
    title="World Fusion — Multilingual, Sea, Migration",
    language="multilingual",
    domain="world_fusion",
    system_prompt=MASTER_SYSTEM,
    user_prompt=(
        "Write an original multilingual poem (5 stanzas) — each stanza in a different language: "
        "Tamil, Bengali, Hindi, English, and Kannada (one stanza each). "
        "A single subject across all: the sea as threshold, migration, leaving home. "
        "Each stanza should stand alone yet connect thematically. "
        "Keep to ~5 lines per stanza. No translation needed. "
        "Write each stanza in its native script. "
        "Label each: [Tamil], [Bengali], [Hindi], [English], [Kannada]. "
        "Begin immediately with '[Tamil]' on line 1."
    ),
    music_context={"mode": "World", "raga": "Bilawal/Shankarabharanam", "tempo": "medium",
                   "instruments": "sitar, mridangam, violin, drums"},
    structured_output_hints=["5 stanzas", "5 languages", "native scripts", "sea/migration theme"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Full spec list
# ─────────────────────────────────────────────────────────────────────────────

ALL_SPECS: list[CreativeSpec] = [
    s01, s02,        # Sanskrit (2)
    k01, k02, k03, k04, k05,  # Kannada (5)
    h01, h02,        # Hindi (2)
    ta01,            # Tamil (1)
    te01,            # Telugu (1)
    bn01,            # Bengali (1)
    e01, e02, e03,   # English (3)
    l01, l02, l03,   # Song / music (3)
    w01,             # World fusion (1)
]
# Total: 19 creative specs across 7 languages and 7 domains
