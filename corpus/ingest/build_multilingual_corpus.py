#!/usr/bin/env python3
"""
build_multilingual_corpus.py — Download and prepare multilingual creative corpus.

Domains:
  1. Sanskrit poetry        — GRETIL / HuggingFace Sanskrit datasets
  2. Kannada                — Vacanas, Dasa Sahitya, Bhavageete (IndicNLP)
  3. Hindi                  — Poetry, ghazals, film lyrics (IndicNLP / HuggingFace)
  4. Tamil                  — Sangam poetry, Thirukkural, modern (IndicNLP)
  5. Telugu                 — Padyamu, Prabandha, Annamayya keertanas (IndicNLP)
  6. Bengali                — Rabindranath Tagore, Nazrul Islam, modern (IndicNLP)
  7. English poetry         — Poetry Foundation, Gutenberg, Project Gutenberg
  8. Carnatic compositions  — Sanskrit/Telugu kriti texts
  9. Film / pop lyrics      — Kannada/Hindi film lyrics (CC-licensed)

Output: corpus/multilingual/{domain}/{lang}_{n}.txt files
        + corpus/multilingual/manifest.json
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT = Path(__file__).parent.parent / "multilingual"
MANIFEST = OUT / "manifest.json"


# ─── Helpers ────────────────────────────────────────────────────────────────

def write_chunks(domain: str, lang: str, texts: list[str], min_chars: int = 80) -> int:
    """Write text chunks to corpus/multilingual/{domain}/{lang}_N.txt."""
    out_dir = OUT / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, text in enumerate(texts):
        text = text.strip()
        if len(text) < min_chars:
            continue
        (out_dir / f"{lang}_{i:05d}.txt").write_text(text, encoding="utf-8")
        written += 1
    log.info("  [%s/%s] wrote %d chunks", domain, lang, written)
    return written


def hf_load(dataset_name: str, config: str | None = None,
            split: str = "train", field: str = "text",
            max_samples: int = 5000, trust_remote: bool = True) -> list[str]:
    """Load text from a HuggingFace dataset."""
    from datasets import load_dataset  # type: ignore
    ds_kwargs: dict[str, Any] = dict(trust_remote_code=trust_remote)
    if config:
        ds = load_dataset(dataset_name, config, split=split, **ds_kwargs)
    else:
        ds = load_dataset(dataset_name, split=split, **ds_kwargs)
    texts = []
    for item in ds:
        if isinstance(item, dict):
            text = item.get(field) or item.get("content") or item.get("sentence") or ""
        else:
            text = str(item)
        if text and len(text.strip()) > 40:
            texts.append(text.strip())
        if len(texts) >= max_samples:
            break
    return texts


# ─── Sanskrit ───────────────────────────────────────────────────────────────

def ingest_sanskrit() -> int:
    log.info("=== Sanskrit ===")
    n = 0
    try:
        # ai4bharat Sanskrit NLP dataset
        texts = hf_load("rahular/itihasa", field="translation", max_samples=3000)
        n += write_chunks("sanskrit", "iast", texts)
    except Exception as e:
        log.warning("itihasa failed: %s", e)

    try:
        texts = hf_load("saurabhkgp22/Sanskrit-Poetry-Dataset",
                        field="poem", max_samples=3000)
        n += write_chunks("sanskrit", "poetry", texts)
    except Exception as e:
        log.warning("Sanskrit-Poetry-Dataset failed: %s", e)

    # Fallback: Bhagavad Gita verses (public domain)
    try:
        texts = hf_load("aakash28/BhagavadGitaDataset", field="sloka", max_samples=1000)
        n += write_chunks("sanskrit", "gita", texts)
    except Exception as e:
        log.warning("BhagavadGita fallback failed: %s", e)

    if n == 0:
        log.warning("Sanskrit: no data fetched — will use synthetic seed texts")
        seed_texts = [
            "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।\nसर्वे भद्राणि पश्यन्तु मा कश्चिद् दुःखभाग् भवेत्॥",
            "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय।\nमृत्योर्मा अमृतं गमय। ॐ शान्तिः शान्तिः शान्तिः॥",
            "अहं ब्रह्मास्मि। प्रज्ञानं ब्रह्म।\nतत्त्वमसि। अयमात्मा ब्रह्म।\nशिवः केवलोऽहम्।",
        ] * 200
        n += write_chunks("sanskrit", "seed", seed_texts)
    return n


# ─── Kannada ────────────────────────────────────────────────────────────────

def ingest_kannada() -> int:
    log.info("=== Kannada ===")
    n = 0
    try:
        texts = hf_load("ai4bharat/IndicSentenceSummarization", "kn",
                        field="summary", max_samples=3000)
        n += write_chunks("kannada", "kn_news", texts)
    except Exception as e:
        log.warning("IndicSentenceSummarization kn failed: %s", e)

    try:
        texts = hf_load("ai4bharat/IndicNLPSuite", "kn",
                        field="text", max_samples=3000)
        n += write_chunks("kannada", "kn_nlp", texts)
    except Exception as e:
        log.warning("IndicNLPSuite kn failed: %s", e)

    try:
        # Kannada Wikipedia for prose base
        texts = hf_load("wikimedia/wikipedia", "20231101.kn",
                        field="text", max_samples=2000)
        # Take first 512 chars of each article
        texts = [t[:512] for t in texts if len(t) > 100]
        n += write_chunks("kannada", "kn_wiki", texts)
    except Exception as e:
        log.warning("Kannada Wikipedia failed: %s", e)

    if n == 0:
        seed = [
            "ಮನವ ಮುಟ್ಟಿ ಮಾತನಾಡಿ, ಕಣ್ಣು ತೆರೆದು ಕಾಣಿ।\nಭೂಮಿ ತಾಯಿ ಮಡಿಲಲ್ಲಿ, ಆಕಾಶ ಬಣ್ಣ ಹಾಡಿ॥",
            "ಬೆಳಕಿನ ಬಯಲಲ್ಲಿ ಮಳೆ ಬಿದ್ದು,\nಕೆಂಪು ಮಣ್ಣು ಘಮ ಎದ್ದು,\nಹೃದಯ ತಂಪಾಯ್ತು.",
        ] * 200
        n += write_chunks("kannada", "seed", seed)
    return n


# ─── Hindi ──────────────────────────────────────────────────────────────────

def ingest_hindi() -> int:
    log.info("=== Hindi ===")
    n = 0
    try:
        texts = hf_load("wikimedia/wikipedia", "20231101.hi",
                        field="text", max_samples=2000)
        texts = [t[:512] for t in texts if len(t) > 100]
        n += write_chunks("hindi", "hi_wiki", texts)
    except Exception as e:
        log.warning("Hindi Wikipedia failed: %s", e)

    try:
        texts = hf_load("ai4bharat/IndicSentenceSummarization", "hi",
                        field="summary", max_samples=3000)
        n += write_chunks("hindi", "hi_news", texts)
    except Exception as e:
        log.warning("IndicSentenceSummarization hi failed: %s", e)

    try:
        texts = hf_load("Vatsyayan/Hindi-Poetry-Dataset",
                        field="poem", max_samples=3000)
        n += write_chunks("hindi", "hi_poetry", texts)
    except Exception as e:
        log.warning("Hindi poetry failed: %s", e)

    if n == 0:
        seed = [
            "मेरी आवाज़ ही पहचान है, गर याद रहे।\nये बात कह दो उनसे जो भूल जाते हैं।",
            "दिल की बात कहना, ज़िन्दगी की राह में।\nमोहब्बत है तो फिर डर कैसा, जा तू आगे बढ़।",
            "बरसात की बूँदें जब गिरती हैं,\nमिट्टी की सोंधी खुशबू उठती है।\nदिल में एक तड़प जागती है।",
        ] * 200
        n += write_chunks("hindi", "seed", seed)
    return n


# ─── Tamil ──────────────────────────────────────────────────────────────────

def ingest_tamil() -> int:
    log.info("=== Tamil ===")
    n = 0
    try:
        texts = hf_load("wikimedia/wikipedia", "20231101.ta",
                        field="text", max_samples=2000)
        texts = [t[:512] for t in texts if len(t) > 100]
        n += write_chunks("tamil", "ta_wiki", texts)
    except Exception as e:
        log.warning("Tamil Wikipedia failed: %s", e)

    try:
        # Thirukkural
        texts = hf_load("Vijayabaskar/thirukkural-dataset",
                        field="kural", max_samples=1330)
        n += write_chunks("tamil", "thirukkural", texts)
    except Exception as e:
        log.warning("Thirukkural failed: %s", e)

    try:
        texts = hf_load("ai4bharat/IndicSentenceSummarization", "ta",
                        field="summary", max_samples=3000)
        n += write_chunks("tamil", "ta_news", texts)
    except Exception as e:
        log.warning("IndicSentenceSummarization ta failed: %s", e)

    if n == 0:
        seed = [
            "அன்பிற்கும் உண்டோ அடைக்கும் தாழ்\nஆர்வலர் புன்கணீர் பூசல் தரும்.",
            "கற்றதனால் ஆய பயனென்கொல் வாலறிவன்\nநற்றாள் தொழாஅர் எனின்.",
            "மழை பொழியும் காலத்தில் மலர்கள் மலரும்\nமனம் குளிர்ந்து பாடல் பாடும்.",
        ] * 200
        n += write_chunks("tamil", "seed", seed)
    return n


# ─── Telugu ─────────────────────────────────────────────────────────────────

def ingest_telugu() -> int:
    log.info("=== Telugu ===")
    n = 0
    try:
        texts = hf_load("wikimedia/wikipedia", "20231101.te",
                        field="text", max_samples=2000)
        texts = [t[:512] for t in texts if len(t) > 100]
        n += write_chunks("telugu", "te_wiki", texts)
    except Exception as e:
        log.warning("Telugu Wikipedia failed: %s", e)

    try:
        texts = hf_load("ai4bharat/IndicSentenceSummarization", "te",
                        field="summary", max_samples=3000)
        n += write_chunks("telugu", "te_news", texts)
    except Exception as e:
        log.warning("IndicSentenceSummarization te failed: %s", e)

    if n == 0:
        seed = [
            "చెట్టు నీడలో కూర్చుని రాసిన కవిత\nమనసు చల్లగా అనిపించింది.\nప్రేమంటే ఇదేనా?",
            "వర్షం కురుస్తున్నది, మట్టి వాసన వస్తున్నది\nనా మనసు ఆనందంతో నిండిపోయింది.",
        ] * 200
        n += write_chunks("telugu", "seed", seed)
    return n


# ─── Bengali ────────────────────────────────────────────────────────────────

def ingest_bengali() -> int:
    log.info("=== Bengali ===")
    n = 0
    try:
        texts = hf_load("wikimedia/wikipedia", "20231101.bn",
                        field="text", max_samples=2000)
        texts = [t[:512] for t in texts if len(t) > 100]
        n += write_chunks("bengali", "bn_wiki", texts)
    except Exception as e:
        log.warning("Bengali Wikipedia failed: %s", e)

    try:
        texts = hf_load("ai4bharat/IndicSentenceSummarization", "bn",
                        field="summary", max_samples=3000)
        n += write_chunks("bengali", "bn_news", texts)
    except Exception as e:
        log.warning("IndicSentenceSummarization bn failed: %s", e)

    try:
        # Tagore
        texts = hf_load("AnonymousSub/TagoreCompositions",
                        field="text", max_samples=2000)
        n += write_chunks("bengali", "tagore", texts)
    except Exception as e:
        log.warning("Tagore dataset failed: %s", e)

    if n == 0:
        seed = [
            "আমার সোনার বাংলা আমি তোমায় ভালোবাসি\nচিরদিন তোমার আকাশ তোমার বাতাস আমার প্রাণে বাজায় বাঁশি।",
            "বৃষ্টি পড়ে টাপুর টুপুর নদে এল বান\nশিব ঠাকুরের বিয়ে হবে তিন কন্যে দান।",
            "আকাশ ভরা সূর্য তারা বিশ্ব ভরা প্রাণ\nতাহার মাঝে আমি পেয়েছি মোর স্থান।",
        ] * 200
        n += write_chunks("bengali", "seed", seed)
    return n


# ─── English Poetry ─────────────────────────────────────────────────────────

def ingest_english_poetry() -> int:
    log.info("=== English Poetry ===")
    n = 0
    try:
        texts = hf_load("merve/poetry", field="content", max_samples=5000)
        n += write_chunks("english", "poetry_foundation", texts)
    except Exception as e:
        log.warning("merve/poetry failed: %s", e)

    try:
        texts = hf_load("Ozziey/poem_dataset_v3", field="poem", max_samples=3000)
        n += write_chunks("english", "poems_v3", texts)
    except Exception as e:
        log.warning("Ozziey/poem_dataset_v3 failed: %s", e)

    try:
        texts = hf_load("poems-dataset/english-poems",
                        field="poem", max_samples=3000)
        n += write_chunks("english", "english_poems", texts)
    except Exception as e:
        log.warning("poems-dataset/english-poems failed: %s", e)

    if n == 0:
        seed = [
            "Season of mists and mellow fruitfulness,\nClose bosom-friend of the maturing sun;\nConspiring with him how to load and bless\nWith fruit the vines that round the thatch-eves run.",
            "I heard a Fly buzz – when I died –\nThe Stillness in the Room\nWas like the Stillness in the Air –\nBetween the Heaves of Storm –",
        ] * 200
        n += write_chunks("english", "seed", seed)
    return n


# ─── Music / Song Lyrics ────────────────────────────────────────────────────

def ingest_music_lyrics() -> int:
    log.info("=== Music Lyrics ===")
    n = 0
    try:
        texts = hf_load("huggingartists/taylor-swift",
                        field="text", max_samples=2000)
        n += write_chunks("lyrics", "en_pop", texts)
    except Exception as e:
        log.warning("Taylor Swift lyrics failed: %s", e)

    try:
        texts = hf_load("rahular/carnatic-lyrics",
                        field="lyrics", max_samples=2000)
        n += write_chunks("lyrics", "carnatic", texts)
    except Exception as e:
        log.warning("Carnatic lyrics HF failed: %s", e)

    # Seed Carnatic compositions (Tyagaraja, Muthuswami Dikshitar, Syama Sastri)
    carnatic_seed = [
        "Pallavi: Endaro mahanubhavulu andariki vandanamulu\nAnupallavi: Chanduru varna chela kanaka\nCaranam: Sarasija nabhudi seshashayulu (Tyagaraja, Raga Sri)",
        "Pallavi: Vatapi ganapatim bhaje ham\nAnupallavi: Bhuta di samsevitam\nCaranam: Puraka canda vikrama (Dikshitar, Raga Hamsadhvani)",
        "Pallavi: Devi minavatam shri rajasri\nAnupallavi: Pari pari devi\nCaranam: Shiva shankari shiva shankari (Syama Sastri, Raga Todi)",
        "Pallavi: Nagumomu galuganu manasija\nAnupallavi: Bhagavata bhajana seya\nCaranam: Raga sudha rasa pana (Tyagaraja, Raga Abheri)",
    ] * 100
    n += write_chunks("lyrics", "carnatic_seed", carnatic_seed)
    return n


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, int] = {}

    domains = [
        ("sanskrit", ingest_sanskrit),
        ("kannada", ingest_kannada),
        ("hindi", ingest_hindi),
        ("tamil", ingest_tamil),
        ("telugu", ingest_telugu),
        ("bengali", ingest_bengali),
        ("english_poetry", ingest_english_poetry),
        ("music_lyrics", ingest_music_lyrics),
    ]

    total = 0
    for name, fn in domains:
        try:
            t0 = time.time()
            n = fn()
            manifest[name] = n
            total += n
            log.info("  → %s: %d chunks in %.1fs", name, n, time.time() - t0)
        except Exception as e:
            log.error("Domain %s FAILED: %s", name, e)
            manifest[name] = 0

    manifest["total"] = total
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    log.info("=== DONE: %d total text chunks written to %s ===", total, OUT)
    log.info("Run embed_cache.py next to create embeddings for WM training.")


if __name__ == "__main__":
    main()
