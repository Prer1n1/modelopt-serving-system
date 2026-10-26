#!/usr/bin/env bash
set -euo pipefail

export PATH="/tmp/modelopt-serving-system-venv/bin:$PATH"
export HF_HOME="/tmp/modelopt-serving-system-hf"

exec /tmp/modelopt-serving-system-venv/bin/vllm serve \
  /tmp/modelopt-qwen2.5-7b-fp8 \
  --served-model-name baseline \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000
