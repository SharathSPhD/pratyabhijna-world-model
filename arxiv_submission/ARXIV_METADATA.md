# arXiv Submission Metadata

## Title
Pratyabhijñā World Model: Creative AI Through Recognition, Active Inference, and Associative Memory

## Authors
Sharath S (University of York, qbz506@york.ac.uk)

## Category
Primary: cs.AI
Cross-list: cs.LG, cs.CL

## Abstract (copy-paste to arXiv form)
We present the Pratyabhijñā World Model (PWM), a creative AI system that
operationalises Kashmir Śaiva philosophy through active inference on a
DreamerV3-class world model with frozen large language model augmentation.
PWM implements the Pañcakṛtya (five-act) śakti cascade as a computational
loop: sṛṣṭi (world model prediction), sthiti (Hopfield associative memory
retrieval), saṃhāra (EFE-driven action selection), vilaya (sleep consolidation),
and anugraha (LLM narration at sphurattā events). The system resolves four
TRIZ engineering contradictions encountered during development, including
C4 (streaming latency vs 120B generation quality) via cascade streaming with
WM reasoning trace prefill — reducing switch latency from 65 s to 5 s (8×).
Phase-gated evaluation across seven development phases validates nine
pre-registered hypotheses (H1–H9) and six ablations (A1–A6). EFE outperforms
REINFORCE on sparse creative reward (H1, g = 0.71); Hopfield memory improves
pattern completion accuracy (H2, 91.3%); sleep consolidation reduces
catastrophic forgetting (H3, 23% reduction); VimarsaBridge improves narration
quality (H4, 78% human meaningful rate); PWM surpasses the PCE v0.4 baseline
on creative quality (H5, ratio 2.14); camatkāra correlates with human aesthetic
judgment (H6, DTW ρ = 0.78); and the three-level Trika hierarchy outperforms
ablated one-level variants (H7, VFE ratio 20.9×). All code, checkpoints, and
evaluation data are released open-source.

## Comments
21 pages, 9 figures. Code: https://github.com/SharathSPhD/pratyabhijna-world-model

## Submission files
- main.tex       — Paper source (IEEEtran conference format)
- refs_new.bib   — Bibliography (1183 lines, ~85 references)
- main.bbl       — Pre-compiled bibliography
- figures/       — 10 figure files (8 PNG, 2 PDF)

## Upload instructions
1. Go to https://arxiv.org/submit
2. Select cs.AI as primary category, cs.LG + cs.CL as cross-lists
3. Upload: pratyabhijna_world_model_arxiv.tar.gz
4. arXiv will auto-detect main.tex as the primary file
5. Verify compiled PDF matches paper/main.pdf (21 pages)
