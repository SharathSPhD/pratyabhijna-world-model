"""
PWM evaluation modules.

Phase 1 exit criteria:
  perplexity  — WM reconstruction MSE vs LSTM baseline
  umap_viz    — z_t latent cluster separation by domain (silhouette score)

Supporting evaluators:
  svat        — Svātantrya (creative freedom) latent coverage score
  camatk_eval — Camatkāra reward trajectory DTW analysis
  metre       — Sanskrit metre / rhythmicity index
  ablations   — H1–H9 / A1–A6 ablation runner
"""
