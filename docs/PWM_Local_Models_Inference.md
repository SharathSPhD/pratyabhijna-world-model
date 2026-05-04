# PWM Local Models & Inference Stack
## Zero Paid-API Architecture on DGX Spark GB10 Blackwell (128 GB)

*Version 0.1 — May 2026*

---

## 1. Motivation: Why Zero Paid APIs

The Pratyabhijñā World Model (PWM) is a research program in creative autonomy. Routing every inference call through a commercial API would:

1. **Violate svātantrya** — the system's own "unconstrained autonomy" cannot be predicated on an external provider's uptime, pricing, or rate limits.
2. **Leak training signal** — creative outputs and reward trajectories sent to third-party endpoints contaminate the experiment's epistemic independence.
3. **Prevent fine-tuning** — the pramāṇa-grounded specialisation plan (Phase 4+) requires access to model weights; that is only possible with local deployment.
4. **Hardware already available** — the DGX Spark GB10 Blackwell is purpose-built for this: 1 petaFLOP FP4, 273 GB/s memory bandwidth, 128 GB unified LPDDR5X. There is no cost argument for offloading.

The architecture uses **zero paid inference endpoints** at every layer — world model, āgama LLM, agent orchestration, and evaluation.

---

## 2. LLM Choice: Nemotron 3 Super 120B A12B (Primary)

### 2.1 Model Specification

| Property | Value |
|---|---|
| Full name | Nemotron 3 Super 120B A12B |
| Architecture | Hybrid Mamba-Transformer MoE |
| Total parameters | 120B |
| Active parameters per token | ~12B |
| Model weight footprint | ~87 GB (FP4/INT4 quantised) |
| Context window | 128K tokens |
| Agentic benchmark (PinchBench) | 85.6% |
| Tool use / function calling | Native |
| Availability | HuggingFace Hub (open weights) |
| License | NVIDIA Open Model License |

### 2.2 Why This Model

The 120B A12B is the strongest available open-weight model for the PWM āgama role for four reasons:

**Active parameter efficiency.** Only 12B parameters fire per forward pass (MoE routing). On the GB10's 273 GB/s bandwidth, this yields throughputs comparable to a 12B dense model while retaining the 120B knowledge capacity. This matters because the āgama layer runs *at sphurattā events only* (narration, goal encoding, AWM proposals) — low frequency but demanding latency.

**Hybrid Mamba-Transformer backbone.** The Mamba layers handle long-context temporal state cheaply (O(1) per token, not O(n²)), which maps cleanly to the āgama layer's function: maintaining the vimarśa narrative thread over long creative sessions without blowing the KV-cache budget.

**Agentic capability.** At 85.6% PinchBench agentic score, it leads all open-weight models on multi-step tool use — exactly what the vimarśa-agent and goal-encoding pathways need.

**Fits the hardware.** At 87 GB, it fits within the 128 GB unified memory alongside the world model components (see Section 5).

### 2.3 Nemotron-Super-49B as Secondary / Fast-Path

The dense **Llama-3.3-Nemotron-Super-49B-v1.5** (Arena Hard 88.3, ~28 GB in FP4) is deployed as a secondary tier for sub-agents that require fast iteration:

- **icchā-agent** (candidate generation): needs many fast samples, not deep reasoning
- **ānanda-agent** (camatkāra scoring): lightweight judgment calls
- **Fallback**: if the 120B model is under memory pressure (e.g. during sleep replay with large batch sizes), the 49B handles narration temporarily

The 49B's **dynamic reasoning toggle** (thinking on/off at inference time) is exploited: reasoning OFF for speed-critical paths, reasoning ON for vimarśa deliberation.

---

## 3. Inference Acceleration: TensorRT-LLM vs vLLM vs llama.cpp

### 3.1 Comparison on GB10 Blackwell

| Factor | TensorRT-LLM | vLLM | llama.cpp |
|---|---|---|---|
| **FP4 tensor core support** | Native (NV FP4) | Partial (via quantisation plugins) | No GB10 FP4 path |
| **GB10 optimisation** | First-class: fused MoE kernels, Blackwell-specific GEMM | Improving; needs `--dtype fp4` flag | CPU-offload only; no native GPU FP4 |
| **Throughput (120B MoE)** | ~2–3× higher vs vLLM on GB10 | Baseline | 0.3× (bottleneck on bandwidth) |
| **MoE routing** | Expert-parallel fused kernels | Token-level routing, no fusion | Single-device, sequential |
| **Ease of deployment** | Moderate: needs engine compilation | Simple: `vllm serve` one-liner | Very simple: `llama-server` |
| **OpenAI-compatible API** | Yes (via triton-server) | Native | Native |
| **Fine-tuning compatibility** | Post-FT quantisation required | Native (load HF weights) | GGUF fine-tune via `llama-finetune` |
| **Pramāṇa LoRA loading** | Supported (LoRA engine rebuild) | Hot-swap LoRA supported | LoRA via GGUF adapters |

### 3.2 Decision

**Primary inference engine: TensorRT-LLM** for the 120B A12B MoE model (āgama, vimarśa orchestration). The GB10's FP4 tensor cores are the primary differentiator; TRT-LLM's fused MoE kernels cut per-token latency by ~2.5× vs vLLM on this hardware, which matters for the multi-agent pipeline where each cascade step blocks on an LLM response.

**Secondary inference engine: vLLM** for the 49B dense model (fast sub-agent path). vLLM's single-command serving and hot-swap LoRA support make it preferable for the lower-stakes fast path, especially during Phase 4+ when pramāṇa LoRA adapters are being experimented with.

**llama.cpp: not used in production** — no path to GB10 FP4 acceleration. Retained for offline experimentation, model format conversion (GGUF), and the initial prototyping phase on CPU before the DGX is provisioned.

### 3.3 TensorRT-LLM Setup for Nemotron 120B A12B

```bash
# Step 1: Pull NVIDIA NGC container
docker pull nvcr.io/nvidia/tensorrt-llm:latest

# Step 2: Download model weights from HuggingFace
pip install huggingface_hub
huggingface-cli download nvidia/Nemotron-3-Super-120B-A12B \
  --local-dir /models/nemotron-120b

# Step 3: Quantise to FP4 (NV FP4 for Blackwell)
python3 -m tensorrt_llm.quantization.quantize \
  --model_dir /models/nemotron-120b \
  --dtype float16 \
  --qformat fp4 \
  --output_dir /models/nemotron-120b-fp4 \
  --calib_size 512

# Step 4: Build TRT-LLM engine
trtllm-build \
  --checkpoint_dir /models/nemotron-120b-fp4 \
  --output_dir /engines/nemotron-120b \
  --gemm_plugin fp4 \
  --max_batch_size 8 \
  --max_input_len 8192 \
  --max_output_len 2048 \
  --tp_size 1        \  # GB10 is a single SoC — no tensor parallel needed
  --moe_tp_size 1 \
  --moe_ep_size 1

# Step 5: Serve with OpenAI-compatible endpoint
python3 -m tensorrt_llm.serve \
  --engine_dir /engines/nemotron-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --max_beam_width 1
```

### 3.4 vLLM Setup for Nemotron 49B (Fast Path)

```bash
# Install vLLM with Blackwell support
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

# Serve on port 8001 (TRT-LLM takes 8000)
vllm serve meta-llama/Llama-3.3-Nemotron-Super-49B-v1.5 \
  --dtype bfloat16 \
  --quantization fp8 \          # FP8 on Blackwell (near-lossless)
  --max-model-len 32768 \
  --gpu-memory-utilization 0.30 \ # ~38 GB budget slot
  --enable-lora \
  --max-lora-rank 64 \
  --port 8001
```

---

## 4. smolagents Integration via LiteLLM

The multi-agent pipeline (see `PWM_MultiAgent_Architecture.md`) uses **HuggingFace smolagents** with **LiteLLM** as the abstraction layer. LiteLLM speaks OpenAI-compatible protocol, so both TRT-LLM and vLLM endpoints are transparent:

```python
from smolagents import CodeAgent, LiteLLMModel

# Primary āgama — TRT-LLM endpoint
agama_model = LiteLLMModel(
    model_id="openai/nemotron-120b",   # LiteLLM custom provider
    api_base="http://localhost:8000/v1",
    api_key="local",
    temperature=0.7,
    max_tokens=2048,
)

# Fast sub-agent path — vLLM endpoint
fast_model = LiteLLMModel(
    model_id="openai/nemotron-49b",
    api_base="http://localhost:8001/v1",
    api_key="local",
    temperature=0.9,
    max_tokens=512,
)

# Instantiate agents
vimarsba_agent = CodeAgent(
    tools=[...],
    model=agama_model,
    name="vimarsha",
    description="Reflexive self-awareness gate; orchestrates the śakti pipeline",
)

icha_agent = CodeAgent(
    tools=[...],
    model=fast_model,           # Fast path for candidate generation
    name="icha",
    description="Desire/will; generates K candidate intentions",
)
```

---

## 5. DGX Spark Memory Budget (128 GB Unified)

The memory layout across both model instances and all WM components:

| Component | Footprint | Notes |
|---|---|---|
| Nemotron 120B A12B (FP4) | ~44 GB | TRT-LLM engine + KV cache (8K ctx) |
| Nemotron 49B (FP8) | ~28 GB | vLLM; loaded only when cit/icchā/ānanda agents active |
| Trika World Model (3 levels) | ~25 GB | RSSM/S4 recurrent state + params |
| V-JEPA 2 visual encoder | ~5 GB | Frozen ViT-1.2B; inference only |
| DIAMOND diffusion decoder | ~8 GB | EDM decoder for imagination rollouts |
| CittaStore Hopfield layers | ~4 GB | Episodic + semantic memories |
| EFE Actor-Critic | ~2 GB | Per-level EFE + camatkāra reward head |
| Vimarśa Bridge (WM↔LLM) | ~0.5 GB | LoRA-scale projection layers |
| Prioritised Replay Buffer | ~6 GB | 100K transitions @ 60KB each |
| Optimizer states | ~4 GB | Adam for actively-trained WM components |
| **Total (both LLMs loaded)** | **~126.5 GB** | 1.5 GB headroom |
| **Total (49B swapped out)** | **~98.5 GB** | 29.5 GB free for larger batch / longer ctx |

**Scheduling policy**: The 49B model is resident during active cascade steps (cit→ānanda→icchā→jñāna→kriyā). During sleep phases (NREM/REM) and vimarśa narration, the 49B's weight pages are evicted to free headroom for the 120B's extended KV cache.

The GB10 Blackwell's unified LPDDR5X memory allows CPU↔GPU zero-copy moves; eviction is not a hard unload but a page deactivation.

---

## 6. Inference Routing Logic

The pipeline routes inference to the appropriate engine based on agent identity and urgency:

```python
# configs/inference_routing.yaml

routing:
  agama_120b:
    agents: [vimarsha, memory, sleep, sakshi_keeper]
    endpoint: "http://localhost:8000/v1"
    model: "nemotron-120b"
    max_tokens: 2048
    temperature: 0.7
    reason_mode: "auto"   # thinking toggle auto

  fast_49b:
    agents: [cit, ananda, icha, jnana, kriya]
    endpoint: "http://localhost:8001/v1"
    model: "nemotron-49b"
    max_tokens: 512
    temperature: 0.9
    reason_mode: "off"    # no thinking for fast path

  fallback_49b:
    trigger: "120b_memory_pressure"
    redirect: [vimarsha]  # vimarsha falls back to 49B if 120B KV cache full
    temperature: 0.6
    reason_mode: "on"     # compensate with thinking
```

---

## 7. Pramāṇa LoRA Fine-Tuning Plan (Phase 4+)

Per the user's design intent, fine-tuning is not central in early phases. The phased plan:

**Phase 0–3 (prompt engineering only)**:
- Use avacchedaka-typed system prompts with pramāṇa epistemological structure
- Classify agent outputs via khyātivāda (`classify_khyativada`) — no gradient updates
- Collect (prompt, output, reward) triples for future fine-tuning

**Phase 4 (pramāṇa LoRA)**:
- Use the `pramana` repository's docker-compose pipeline to fine-tune on collected triples
- LoRA rank 64, target modules: `q_proj, v_proj, gate_proj, up_proj`
- Training objective: language-agnostic creative quality (camatkāra reward as preference label)
- NOT Sanskrit-specific — the fine-tune teaches epistemological discipline (perceive/infer/know/recognise), not a language

```bash
# pramana fine-tuning (Phase 4+)
cd /workspace/pramana
docker-compose up finetune \
  -e BASE_MODEL=nemotron-120b-fp4 \
  -e LORA_RANK=64 \
  -e DATASET=/data/pwm_creative_triples.jsonl \
  -e OUTPUT=/models/nemotron-120b-pramana-lora
```

**Phase 5+ (merged weights)**:
- Merge LoRA into base weights: `python merge_lora.py --base nemotron-120b --lora nemotron-120b-pramana-lora`
- Rebuild TRT-LLM engine with merged weights
- Continue WM + āgama joint training

---

## 8. Monitoring and Observability

```yaml
# Prometheus + Grafana stack (no paid telemetry)
services:
  trtllm_metrics:
    endpoint: "localhost:8000/metrics"
    scrape_interval: 10s
    panels: [tokens_per_sec, kv_cache_utilisation, queue_depth, p50_latency]

  vllm_metrics:
    endpoint: "localhost:8001/metrics"
    scrape_interval: 10s
    panels: [tokens_per_sec, gpu_cache_usage, num_running_requests]

  pwm_wm_metrics:
    source: "pwm.metrics.WorldModelMetrics"
    panels: [vfe_train, efe_mean, hopfield_entropy, sphuratta_events_per_hour,
             camatkaara_reward_mean, nrem_duration_min, rem_duration_min]
```

---

## 9. Summary Decision Table

| Layer | Choice | Rationale |
|---|---|---|
| āgama LLM (default) | Nemotron 3 Super 120B A12B | Best open-weight agentic model; fits 128GB |
| fast sub-agent LLM | Nemotron-Super-49B-v1.5 | Dynamic reasoning toggle; speed/quality trade-off |
| inference engine (120B) | TensorRT-LLM | GB10 FP4 tensor cores; 2.5× speedup on Blackwell |
| inference engine (49B) | vLLM | Hot-swap LoRA; simpler ops; FP8 sufficient |
| LLM abstraction | LiteLLM | Single config key switches between all providers |
| commercial API path | LiteLLM direct (no TRT-LLM/vLLM) | Claude/OpenAI/Gemini via API — no local inference stack |
| agent framework | smolagents (vimarśa/memory/sleep only) | Only deliberative agents; śaktis are WM pipeline methods |
| fine-tuning pipeline | pramāṇa docker-compose (Phase 4+) | Language-agnostic epistemic specialisation |
| paid API calls | Zero by default; optional via config | `llm.provider=claude-api` enables; local is default |

---

## 10. Multi-Provider LiteLLM Configuration

### 10.1 Architecture Principle

All LLM calls in PWM route through a single `LLMBackend` class that wraps LiteLLM. Switching providers is a one-line config change. When a commercial API provider is selected, TRT-LLM and vLLM are not started at all — only LiteLLM is needed.

```
Local path:    PWM code → LLMBackend → LiteLLM → TRT-LLM (8000) or vLLM (8001)
API path:      PWM code → LLMBackend → LiteLLM → Anthropic/OpenAI/Gemini API
```

### 10.2 Provider Config

See `configs/llm_backend.yaml` (full spec in `PWM_Architecture_Spec.md` Section 13). Quick reference:

```bash
# DGX Spark (default — local Nemotron)
python -m pwm.main "..." --set llm.provider=nemotron-local

# Claude API (API key in env)
export ANTHROPIC_API_KEY=sk-ant-...
python -m pwm.main "..." --set llm.provider=claude-api

# OpenAI
export OPENAI_API_KEY=sk-...
python -m pwm.main "..." --set llm.provider=openai-api

# Gemini
export GOOGLE_API_KEY=AIza...
python -m pwm.main "..." --set llm.provider=gemini-api

# Any other LiteLLM-supported model (Ollama, Together, Replicate, etc.)
export LLM_PRIMARY_MODEL=ollama/llama3.1:70b
export LLM_FAST_MODEL=ollama/llama3.1:8b
python -m pwm.main "..." --set llm.provider=custom
```

### 10.3 When Local Inference Stack Is and Isn't Needed

| Provider | TRT-LLM needed? | vLLM needed? | Notes |
|---|---|---|---|
| `nemotron-local` | Yes (120B) | Yes (49B) | Full local stack; best performance on DGX Spark |
| `claude-api` | No | No | Direct HTTPS to Anthropic; LiteLLM only |
| `openai-api` | No | No | Direct HTTPS to OpenAI; LiteLLM only |
| `gemini-api` | No | No | Direct HTTPS to Google; LiteLLM only |
| `custom` (Ollama) | No | No | Ollama serves its own endpoint |
| `custom` (vLLM any model) | No | Yes | Use vLLM for any HuggingFace model |

### 10.4 Performance Expectations Across Providers

| Provider | Latency (vimarśa call) | Throughput | Cost | Privacy |
|---|---|---|---|---|
| Nemotron 120B local | ~2–5s | High | $0 | Full (local) |
| Claude Opus 4.6 API | ~5–15s | API rate limits | $$$ | Data sent to Anthropic |
| GPT-4o API | ~3–10s | API rate limits | $$ | Data sent to OpenAI |
| Gemini Flash API | ~1–3s | High | $ | Data sent to Google |
| Nemotron 49B local (fast path) | ~0.5–1s | Very high | $0 | Full (local) |

For research with sensitive creative training data, local Nemotron preserves full data sovereignty. API providers are appropriate for rapid prototyping and researchers without DGX access.
