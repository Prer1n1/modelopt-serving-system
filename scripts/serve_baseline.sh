#!/usr/bin/env bash
set -euo pipefail

export PATH="/tmp/modelopt-serving-system-venv/bin:$PATH"

export HF_HOME="/tmp/modelopt-serving-system-hf"

exec /tmp/modelopt-serving-system-venv/bin/vllm serve \
  Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name baseline \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000
