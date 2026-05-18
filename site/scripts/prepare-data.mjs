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
      label: "Live text ablation",
      status: "fail",
      metric: `g=${round(h5Live.live_result.hedges_g, 2)}, PWM ${round(h5Live.live_result.mean_pwm, 3)} vs LLM ${round(h5Live.live_result.mean_llm, 3)}`,
      caveat: "Bare 120B LLM wins on text-only camatkara score.",
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
  phase8Overall: phase8.overall_gate_pass
};

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outFile, JSON.stringify(payload, null, 2) + "\n");
console.log(`Wrote ${path.relative(repoRoot, outFile)}`);
