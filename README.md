# Pratyabhijñā World Model (PWM)

> *"Consciousness recognises itself in every creative act."*
> — Utpaladeva, *Īśvarapratyabhijñākārikā* 1.3

A creative AI research system operationalising Kashmir Śaiva philosophy through active inference on a DreamerV3-class world model with frozen LLM augmentation.

## Overview

PWM implements the five acts (pañcakṛtya) of Śiva as a computational pipeline:
- **Sṛṣṭi** (creation): generative imagination via 3-level Trika RSSM hierarchy
- **Sthiti** (maintenance): Hopfield CittaStore episodic + semantic memory
- **Saṃhāra** (dissolution): EFE actor with camatkāra intrinsic reward
- **Tirodhāna** (concealment): sleep consolidation (NREM + REM)
- **Anugraha** (grace): vimarśa deliberative gate + LLM narration

## Architecture

- **World Model**: 3-level Trika RSSM (Aparā/GRU → Parāparā/GRU → Parā/Mamba-2)
- **Memory**: Modern Hopfield networks (episodic smṛti + semantic ālayavijñāna)
- **Policy**: EFE actor (active inference replacing REINFORCE)
- **LLM Layer**: Frozen Nemotron/Claude via sphurattā-gated narration
- **Sleep**: NREM replay + REM dreaming with ThermSleep stopping criterion

## Setup

```bash
source /home/sharaths/vllm-env/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Phase 0: Corpus ingestion
python -m corpus.build --sources all --min-tokens 100000

# Phase 1: Train Aparā RSSM
python pwm/scripts/train.py

# Evaluate
python scripts/evaluate.py
```

## Citation

```bibtex
@article{pwm2025,
  title={Pratyabhijñā World Model: Creative AI through Recognition, Active Inference, and Associative Memory},
  author={SharathSPhD},
  year={2025}
}
```
