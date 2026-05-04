# PWM Sanskrit ↔ Computational Concept Glossary

Canonical mapping between Kashmir Śaiva philosophical concepts and their computational realisations in the Pratyabhijñā World Model.

## Core Epistemic Concepts

| Sanskrit | IAST | Source | Computational Realisation |
|----------|------|--------|--------------------------|
| Pratyabhijñā | Pratyabhijñā | ĪPK 1.3–1.4 (Utpaladeva) | Recognition density q_φ(z_t\|h_t, o_t) — the posterior that "recognises" the observation |
| Vimarśa | Vimarśa | ĪPK 1.5.11; TĀ 1.24–1.33 (Abhinavagupta) | f_self(h_t, z_t) — self-reflexive evaluation; VimarshaAgent commit/revise/reject gate |
| Spanda | Spanda | SpandaK 1.1 (Vasugupta) | Stochastic latent z_t ~ Categorical(32×32) — the creative pulse/vibration |
| Sphurattā | Sphurattā | TĀ 1.56 (Abhinavagupta) | Camatkāra event C_t=1 — VFE < threshold AND Hopfield entropy drop |
| Svātantrya | Svātantrya | ĪPK 2.1 (Utpaladeva) | Max-entropy policy prior; EFE actor encouraging creative freedom |
| Camatkāra | Camatkāra | Locana ad DhvA 1.1 (Abhinavagupta) | R_camatk = α₁ΔF + α₂ΔI_Hopfield + α₃Empowerment |
| Dhvani | Dhvani | DhvA (Ānandavardhana) | Latent resonance between z_t and memory retrieval — Hopfield similarity |
| Rasa | Rasa | NŚ 6 (Bharata Muni) | Aesthetic emotion labels in corpus; DECKARD target_rasa proposals |

## Memory Concepts

| Sanskrit | IAST | Source | Computational Realisation |
|----------|------|--------|--------------------------|
| Ālayavijñāna | Ālayavijñāna | PHṛ sūtra 9 (Kṣemarāja) | Semantic Hopfield store — low β=0.25, learnable prototypes |
| Smṛti | Smṛti | — | Episodic Hopfield store — high β=4.0, FIFO with SHY down-scaling |
| Saṃskāra | Saṃskāra | PHṛ sūtra 9 | Latent traces stored in CittaStore via store_episodic() |
| Citta | Citta | PHṛ sūtra 9 | Posterior Q(z\|o) — the attending mind's recognition state |
| Citi | Citi | PHṛ sūtra 1 (Kṣemarāja) | Trained prior p_θ(z) — the substrate of consciousness |
| Cit | Cit | PHṛ sūtra 1 | World model itself — the knowing substrate |

## Action / Will Concepts

| Sanskrit | IAST | Source | Computational Realisation |
|----------|------|--------|--------------------------|
| Icchā-śakti | Icchā-śakti | ĪPK 2.3 | Creative intention — DECKARD AWM proposals; EFEActor action selection |
| Jñāna-śakti | Jñāna-śakti | ĪPK 2.4 | Inferential knowledge — LLM narration at sphurattā events |
| Kriyā-śakti | Kriyā-śakti | ĪPK 2.5 | Action execution — EFEActor policy output |
| Apohana | Apohana | — | Exclusion / inhibition — action masking, done-state reset |

## Consciousness / Śiva Concepts

| Sanskrit | IAST | Source | Computational Realisation |
|----------|------|--------|--------------------------|
| Śiva | Śiva | TĀ 1 | The world model as a whole (TrikaWorldModel) |
| Śakti | Śakti | ĪPK 1.6 | The śakti cascade (PañcakṛtyaLoop) — Śiva's dynamic power |
| Anuttara | Anuttara | TĀ 1.1 | The unconditioned prior p_θ — pre-observation latent distribution |
| Pañcakṛtya | Pañcakṛtya | TĀ 6 | Five acts = sṛṣṭi, sthiti, saṃhāra, tirodhāna, anugraha |

## Pañcakṛtya (Five Acts) Mapping

| Act | Sanskrit | Computational Step in PañcakṛtyaLoop |
|-----|----------|--------------------------------------|
| Creation | Sṛṣṭi | Generative imagination (posterior z_t, imagine_step) |
| Maintenance | Sthiti | CittaStore retrieval + memory blending |
| Dissolution | Saṃhāra | EFE actor → action selection → env step |
| Concealment | Tirodhāna | Sleep consolidation (NREM/REM) |
| Grace | Anugraha | VimarshaAgent commit → LLM narration |

## Sleep Concepts

| Sanskrit | IAST | Source | Computational Realisation |
|----------|------|--------|--------------------------|
| Svapna | Svapna | MU 2–3; MV 1.2.9 | REM phase — generative dreaming, recognition net retraining |
| Nidrā | Nidrā | — | NREM phase — replay + VFE descent + SHY down-scaling |
| Suṣupti | Suṣupti | MU 5; PHṛ sūtra 3 | Deep sleep = ThermSleep stopping criterion (efficiency < threshold) |

## Impurity (Mala) Concepts

| Sanskrit | IAST | Source | Computational Analogue |
|----------|------|--------|----------------------|
| Āṇavamala | Āṇavamala | TS 10.4; PHṛ sūtra 2 | Latent collapse (all z identical) — AnavaRegulariser |
| Māyīyamala | Māyīyamala | PHṛ sūtra 2 | Mode collapse (batch z identical) — MayiyaRegulariser |
| Kārmamala | Kārmamala | PHṛ sūtra 2 | Reward hacking (action collapse) — KarmaRegulariser |

## LLM Pramāṇa (Epistemic Sources)

| Pramāṇa | LLM Role | Usage |
|---------|----------|-------|
| Āgama (testimony) | `role="agama"` | System context, frozen knowledge |
| Anumāna (inference) | `role="jnana"` | Narration generation, sphurattā analysis |
| Icchā (will) | `role="icccha"` | DECKARD creative intention proposals |
| Vimarśa (self-reflection) | `role="vimarsha"` | VimarshaAgent revision |

---

## Abbreviations

| Abbreviation | Full Title |
|-------------|-----------|
| ĪPK | Īśvarapratyabhijñākārikā (Utpaladeva) |
| TĀ | Tantrāloka (Abhinavagupta) |
| SpandaK | Spandakārikā (Vasugupta) |
| PHṛ | Pratyabhijñāhṛdayam (Kṣemarāja) |
| MU | Māṇḍūkya Upaniṣad |
| MV | Māṇḍūkyavṛtti (Gauḍapāda) |
| DhvA | Dhvanyāloka (Ānandavardhana) |
| NŚ | Nāṭyaśāstra (Bharata Muni) |
| TS | Tantra-sāra (Abhinavagupta) |
