import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const resultsDir = path.join(repoRoot, "benchmarks/results");
const outDir = path.join(repoRoot, "site/src/data");
const outFile = path.join(outDir, "results.json");

function readJson(name) {
  const raw = fs.readFileSync(path.join(resultsDir, name), "utf8");
  return JSON.parse(raw.replace(/:\s*Infinity/g, ": null"));
}

function round(value, digits = 3) {
  if (typeof value !== "number" || Number.isNaN(value)) return value;
  return Number(value.toFixed(digits));
}

const p2 = readJson("phase_2_gate.json");
const p3 = readJson("phase_3_gate_step0300000.json");
const p4 = readJson("phase_4_gate_step0300000.json");
const p5 = readJson("phase_5_gate_step0500000.json");
const p6 = readJson("phase_6_gate_step1000000.json");
const h5Live = readJson("h5_live_ablation.json");
const a6 = readJson("ablation_a6_1level_wm.json");
const ttft = readJson("ttft_live_validation.json");
const phase7 = readJson("phase7_gate.json");
const phase8 = readJson("phase8_gate.json");

const domainRows = Object.entries(h5Live.domain_breakdown).map(([domain, data]) => ({
  domain,
  pwmMean: round(data.pwm_mean, 3),
  llmMean: round(data.llm_mean, 3),
  pwmWins: data.pwm_wins,
  llmWins: data.llm_wins,
  verdict: data.verdict,
}));

const payload = {
  generatedAt: new Date().toISOString(),
  sourceFiles: [
    "benchmarks/results/phase_2_gate.json",
    "benchmarks/results/phase_3_gate_step0300000.json",
    "benchmarks/results/phase_4_gate_step0300000.json",
    "benchmarks/results/phase_5_gate_step0500000.json",
    "benchmarks/results/phase_6_gate_step1000000.json",
    "benchmarks/results/h5_live_ablation.json",
    "benchmarks/results/ablation_a6_1level_wm.json",
    "benchmarks/results/ttft_live_validation.json",
    "benchmarks/results/phase7_gate.json",
    "benchmarks/results/phase8_gate.json"
  ],
  headline: {
    splitHypothesesPass: 9,
    splitHypothesesTotal: 10,
    h1RewardRatio: round(p2.h1_reward_ratio, 2),
    h5aRatio: round(p5.h5_reward_ratio, 3),
    h5bHedgesG: round(h5Live.live_result.hedges_g, 3),
    a6Advantage: round(a6.stats.advantage_factor_3level_over_1level, 1),
    h6Entropy: round(p6.h6_reward_entropy, 3),
    phase7Tests: phase7.cumulative_tests,
    phase7Passed: phase7.cumulative_passed,
  },
  hypotheses: [
    {
      id: "H1",
      label: "EFE actor",
      status: "pass",
      metric: `${round(p2.h1_reward_ratio, 2)}x mean reward ratio`,
      caveat: "Time-to-sphuratta diagnostic failed; mean reward is the primary gate.",
      source: "phase_2_gate.json"
    },
    {
      id: "H2",
      label: "Hopfield completion",
      status: "pass",
      metric: `${round(p3.completion.ratio, 3)}x completion ratio`,
      caveat: "Sphuratta-rate diagnostic overshot its target band.",
      source: "phase_3_gate_step0300000.json"
    },
    {
      id: "H3",
      label: "Sleep consolidation",
      status: "caveat",
      metric: "near-zero forgetting",
      caveat: "With-sleep and without-sleep values are identical at recorded precision.",
      source: "phase_4_gate_step0300000.json"
    },
    {
      id: "H4",
      label: "Vimarsha narration proxy",
      status: "pass",
      metric: `${round(p5.h4_meaningful_rate * 100, 1)}% meaningful proxy`,
      caveat: "Proxy metric, not a completed human evaluation.",
      source: "phase_5_gate_step0500000.json"
    },
    {
      id: "H5a",
      label: "Internal imagination reward",
      status: "pass",
      metric: `${round(p5.h5_reward_ratio, 3)}x over Phase 2 baseline`,
      caveat: "Internal reward/imagination protocol, not live text quality.",
      source: "phase_5_gate_step0500000.json"
    },
    {
      id: "H5b",
      label: "Bridge-bias text ablation",
      status: "fail",
      metric: `g=${round(h5Live.live_result.hedges_g, 2)}, with-bridge ${round(h5Live.live_result.mean_pwm, 3)} vs identity ${round(h5Live.live_result.mean_llm, 3)}`,
      caveat: "Ablation of the Vimarsa Bridge v2 logit-bias channel under a stripped text proxy. Same LLM both sides; near-parity on the lower-resource Kannada domain.",
      source: "h5_live_ablation.json"
    },
    {
      id: "H6",
      label: "Reward entropy",
      status: "pass",
      metric: `${round(p6.h6_reward_entropy, 3)} nats`,
      caveat: "Use phase_6_gate_step1000000.json as the current source of truth.",
      source: "phase_6_gate_step1000000.json"
    },
    {
      id: "H7",
      label: "Trika hierarchy",
      status: "caveat",
      metric: `${round(a6.stats.advantage_factor_3level_over_1level, 1)}x advantage`,
      caveat: "Two A6 seeds are partial timeout runs.",
      source: "ablation_a6_1level_wm.json"
    },
    {
      id: "H8",
      label: "Encoder stability",
      status: "pass",
      metric: `norm ${round(p6.h8_encoder_norm, 2)}`,
      caveat: "Within registered [1, 50] bounds.",
      source: "phase_6_gate_step1000000.json"
    },
    {
      id: "H9",
      label: "Policy diversity",
      status: "pass",
      metric: `${round(p6.h9_action_entropy, 3)} nats`,
      caveat: "Automated diversity proxy.",
      source: "phase_6_gate_step1000000.json"
    }
  ],
  h5Live: {
    meanPwm: round(h5Live.live_result.mean_pwm, 4),
    meanLlm: round(h5Live.live_result.mean_llm, 4),
    hedgesG: round(h5Live.live_result.hedges_g, 4),
    ci: h5Live.live_result.bca_ci_95.map((v) => round(v, 4)),
    pPermutation: h5Live.live_result.p_value_permutation,
    domainRows,
    interpretation: h5Live.scientific_interpretation.interpretation
  },
  ttft: {
    aggregatePass: ttft.gate_pass,
    aggregateMean: round(ttft.result.ttft_mean_s, 2),
    warmCondAMean: round(ttft.warm_model_measurements.cond_A_warm_ttft_mean_s, 2),
    warmCondBMean: round(ttft.warm_model_measurements.cond_B_warm_ttft_mean_s, 2),
    warmAdr001Pass: ttft.warm_model_measurements.adr001_warm_pass,
    warmReductionPct: round(ttft.warm_model_measurements.adr002_warm_reduction_pct, 1),
    note: ttft.scientific_interpretation.aggregate_result
  },
  a6: {
    seeds: a6.seed_results.map((seed) => ({
      seed: seed.seed,
      status: seed.status,
      steps: seed.steps_completed,
      finalVfe: round(seed.final_vfe, 6),
      ratio: round(seed.vfe_ratio_vs_phase3, 6)
    })),
    advantageFactor: round(a6.stats.advantage_factor_3level_over_1level, 1),
    note: a6.stats.note
  },
  phase8Overall: phase8.overall_gate_pass,
  wmRole: {
    summary:
      "The world model is the substrate. It learns latent state over text, holds memory, scores creative process, and never speaks.",
    bullets: [
      "Trika RSSM with three levels (Apara, Para-apara, Para) over text embeddings; emits (h, z) and prior/posterior logits each step.",
      "EFE actor selects actions on (h, z); imagination rolls forward without the LLM in the loop.",
      "Camatkara reward combines free-energy reduction, Hopfield-novelty gain and empowerment; fires sphuratta events on the recognised moments.",
      "Citta-store (episodic + semantic Hopfield banks) recalls and blends a retrieval vector with the current latent.",
      "NREM/REM sleep cycles consolidate replay and tune the prior; the LLM is silent throughout.",
      "All quantities (VFE delta, sphuratta rate, retrieval coherence) are substrate-internal and have no LLM analogue."
    ],
    sources: [
      "pwm/world_model/trika.py",
      "pwm/active_inference/efe_actor.py",
      "pwm/rewards/camatk.py",
      "pwm/memory/citta_store.py",
      "pwm/sleep/consolidation.py"
    ]
  },
  llmRole: {
    summary:
      "The frozen 120B LLM is the surface generator. Its weights are never trained; the world model couples to it through a single learned bias channel.",
    bullets: [
      "LLM weights are frozen everywhere; served via Ollama or llama.cpp.",
      "Vimarsa Bridge v2 is the only place a world-model signal directly shapes LLM output: it maps h_t to a vocabulary-sized additive logit bias.",
      "Narration is gated: the LLM is called for the jnana role on sphuratta events, not per step.",
      "Optional WMReasoningTrace prefill serialises h_t as the LLM's pre-completed reasoning in the v2 production loop.",
      "Embeddings for retrieval and goal-conditioning come from the LLM through encode().",
      "Surface fluency is the LLM's job. Creative process is not its job."
    ],
    sources: [
      "pwm/llm/backend.py",
      "pwm/vimarsa/bridge_v2.py",
      "pwm/vimarsa/narrator.py",
      "pwm/generation/llama_backend.py",
      "pwm/pipeline/pancakrtya_loop_v2.py"
    ]
  },
  philosophy: [
    { sanskrit: "Pratyabhijna", source: "Utpaladeva, IPK 1.3-1.4", primitive: "Recognition density q_phi(z | h, o)", module: "pwm/world_model/rssm.py" },
    { sanskrit: "Spanda", source: "Vasugupta, SpandaKarika 1.1", primitive: "Stochastic discrete latent z ~ Cat(32x32)", module: "pwm/world_model/rssm.py" },
    { sanskrit: "Vimarsa", source: "Utpaladeva, IPK 1.5.11", primitive: "Reflexive bridge from h to LLM logit bias", module: "pwm/vimarsa/bridge_v2.py" },
    { sanskrit: "Sphuratta", source: "Abhinavagupta, Tantraloka 1.56", primitive: "Recognised-moment event firing", module: "pwm/rewards/camatk.py" },
    { sanskrit: "Camatkara", source: "Abhinavagupta on Dhvanyaloka 1.1", primitive: "Intrinsic creative reward (dF + dI_Hopfield + Emp)", module: "pwm/rewards/camatk.py" },
    { sanskrit: "Citta", source: "Pratyabhijnahrdayam 9", primitive: "Posterior memory; Hopfield retrieval", module: "pwm/memory/citta_store.py" },
    { sanskrit: "Svatantrya", source: "Utpaladeva, IPK 2.1", primitive: "Max-entropy policy prior", module: "pwm/active_inference/efe_actor.py" }
  ],
  status: [
    { area: "Core world-model training", state: "stable", note: "Phase 6 checkpoint at 1M steps; A6 has two partial seeds queued for re-run." },
    { area: "Vimarsa Bridge v2", state: "stable", note: "Trained; in use in the API v1 path and the H5b runner." },
    { area: "API v1 (POST /v1/generate, WS /v1/ws/generate)", state: "stable", note: "Faithful surface over PancakrtyaLoopV2." },
    { area: "API legacy (POST /generate, /refine, /batch)", state: "deprecated", note: "Parallel orchestration that bypasses EFE/Citta/Bridge; documented only, not refactored this round." },
    { area: "Paper", state: "revised", note: "H5b reframed as a bridge-bias ablation; four fidelity bugs corrected." },
    { area: "GitHub Pages site", state: "rebuilt", note: "Compact hero, architecture, philosophy table, dual-audience copy." },
    { area: "H5c composition / null-bias / low-resource tests", state: "queued", note: "Reframed in the project review; not run this round." },
    { area: "Human evaluation cohort", state: "queued", note: "Pre-registered for the follow-up journal submission." }
  ],
  triz: [
    { principle: "1. Segmentation", move: "Report WM-substrate metrics and LLM-surface metrics separately. Stop collapsing a composite system into a single g." },
    { principle: "2. Take Out / Extract", move: "Add a null-bias condition (random or norm-matched) so the learned-bias claim is isolated from raw LLM strength." },
    { principle: "15. Dynamicity", move: "Make the conditioning channel selectable per phase (bridge only, bridge + Hopfield prefix, bridge + WMReasoningTrace prefill, bridge + LoRA)." },
    { principle: "25. Self-Service", move: "Fine-tune Vimarsa Bridge v2 against a richer dual proxy (substrate + surface), not only next-token CE on (h_t, next_token)." },
    { principle: "40. Composite", move: "Always present PWM as composite WM + frozen LLM. Stop framing H5b as 'WM vs LLM'." }
  ]
};

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outFile, JSON.stringify(payload, null, 2) + "\n");
console.log(`Wrote ${path.relative(repoRoot, outFile)}`);
