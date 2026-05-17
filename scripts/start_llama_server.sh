#!/bin/bash
# Start llama-server for Nemotron 120B (CUDA, flash-attn, cont-batching)
# Usage: bash scripts/start_llama_server.sh [port]
PORT=${1:-8080}
MODEL="/home/sharaths/projects/pwm-phase3/models/nemotron-120b.gguf"
LLAMA_SERVER="/home/sharaths/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="/home/sharaths/llama.cpp/build/bin:$LD_LIBRARY_PATH"

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --n-gpu-layers 999 \
  --flash-attn \
  --cont-batching \
  --port "$PORT" \
  --n-predict 512 \
  --n-ctx 4096 \
  --host 0.0.0.0 \
  --log-disable
