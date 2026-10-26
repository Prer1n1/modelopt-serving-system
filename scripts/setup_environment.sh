#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR=/tmp/modelopt-uv-cache
ENV_DIR=/tmp/modelopt-serving-system-venv

uv venv --python python3.12 --allow-existing "$ENV_DIR"

uv pip install \
  --python "$ENV_DIR/bin/python" \
  --link-mode copy \
  --extra-index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  "vllm==0.32.0" \
  "torch==2.13.0+cu132" \
  "torchvision==0.28.0+cu132" \
  "torchaudio==2.11.0+cpu" \
  "transformers==5.19.0" \
  "huggingface-hub==2.3.0" \
  "setuptools>=78.1.1,<81.0.0"

uv pip check --python "$ENV_DIR/bin/python"

"$ENV_DIR/bin/python" - <<'PY'
import torch
import torchaudio
from vllm.engine.arg_utils import EngineArgs

assert torch.cuda.is_available(), "CUDA is unavailable"
print("PyTorch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU computation:", (torch.ones(4, device="cuda") * 2).tolist())
PY

"$ENV_DIR/bin/ninja" --version
