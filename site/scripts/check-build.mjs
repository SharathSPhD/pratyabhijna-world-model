import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(__dirname, "..");
const dist = path.join(siteRoot, "dist");
const index = path.join(dist, "index.html");

if (!fs.existsSync(index)) {
  throw new Error("Missing dist/index.html after Astro build");
}

const html = fs.readFileSync(index, "utf8");
const required = [
  "9 of 10 split hypotheses pass",
  "H5b is a real negative result",
  "benchmarks/results/h5_live_ablation.json",
  "fig10_h5_live_per_domain.png",
  "fig12_ttft_warm_cold.png"
];

const missing = required.filter((needle) => !html.includes(needle));
if (missing.length) {
  throw new Error(`Build content check failed. Missing: ${missing.join(", ")}`);
}

console.log("Site build content checks passed.");
