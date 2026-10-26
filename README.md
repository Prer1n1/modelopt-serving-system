# ModelOpt Serving System

An independent project comparing original and ModelOpt-optimized LLM
serving using vLLM, with performance and quality evaluation.

## Current setup

- GPU: NVIDIA L40S
- Model: Qwen/Qwen2.5-7B-Instruct
- Model revision: a09a35458c702b33eeacc393d103063234e8bc28
- Baseline precision: BF16
- vLLM: 0.32.0

## Progress

- [x] Verify GPU execution
- [x] Serve the original model through an API
- [x] Measure initial sequential and streaming requests
- [x] Save dependency and environment information
- [ ] Restart the pinned baseline with prefix caching disabled
- [ ] Collect repeated streaming measurements
- [ ] Benchmark different workloads and concurrency levels
- [ ] Optimize and export using ModelOpt
- [ ] Compare performance and held-out answer quality
- [ ] Implement release checks and deployment packaging

## Resume point

Stage 2: baseline measurement.

A previous vLLM engine held GPU memory during restart.
Check GPU processes before starting scripts/serve_baseline.sh.
Verify /health returns HTTP 200 before benchmarking.

## Environment and storage

The working Python environment was created at:
  /tmp/modelopt-serving-system-venv

The Hugging Face cache was configured at:
  /tmp/modelopt-serving-system-hf

These temporary directories may need rebuilding after pod recreation.
The startup script currently references that environment path.

Initial measurements are exploratory, not final comparative benchmarks.
