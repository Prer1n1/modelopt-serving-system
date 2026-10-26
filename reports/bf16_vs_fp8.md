# Qwen2.5-7B-Instruct: BF16 vs ModelOpt FP8

## Outcome

On one NVIDIA L40S, ModelOpt FP8 achieved 1.35–1.55x higher
aggregate output throughput across the tested fixed-length workloads.
Weight-file storage decreased by 42.84%.
Numeric-answer accuracy on a fixed 100-question GSM8K sample
changed from 92% for BF16 to 91% for FP8.

These are initial experimental results, not production guarantees.

## Configuration

- Model: Qwen/Qwen2.5-7B-Instruct
- Source revision: a09a35458c702b33eeacc393d103063234e8bc28
- GPU: NVIDIA L40S, compute capability 8.9
- Serving engine: vLLM 0.32.0
- PyTorch: 2.13.0+cu132
- Serving Transformers: 5.19.0
- Quantization Transformers: 5.14.1
- ModelOpt: 0.46.1
- Recipe: FP8_DEFAULT_CFG, max calibration
- Calibration: 128 GSM8K training questions, maximum 512 input tokens
- Calibration examples were processed without generation
- No exact question overlap between calibration and evaluation
- KV-cache quantization: disabled
- Embeddings and lm_head excluded from FP8 quantization
- Maximum model length: 4096
- Maximum active sequences: 16
- GPU memory utilization setting: 0.80
- Prefix caching: disabled

The API model name was "baseline" for both checkpoints.
The startup script determined which checkpoint was served.

## Performance methodology

Each setting used 64 measured requests, following a warmup wave.
The client maintained up to the selected concurrency until all
requests completed.

Input lengths were 57, 841, and 3145 tokens.
Every measured request generated exactly 128 tokens, enforced using
min_tokens=128, max_tokens=128, and ignore_eos=True.

Longer prompts used repeated synthetic background text.
Temperature was zero.

Aggregate output throughput equals total generated tokens divided
by measurement wall time, including the final drain period.

Latency measurements were client-observed:
- TTFT: request start to first nonempty text chunk.
- End-to-end: request start to stream completion.
- TPOT estimate: first-to-last text arrival interval divided by
  output token count minus one.

Text chunks do not necessarily correspond one-to-one with tokens.
TPOT is therefore a client-side estimate, not GPU kernel timing.

## Fixed-length throughput

All requests generated 128 output tokens.

| Input tokens | Concurrency | BF16 output tokens/s | FP8 output tokens/s | Speedup |
|---:|---:|---:|---:|---:|
| 57 | 1 | 49.72 | 74.10 | 1.49x |
| 57 | 4 | 188.38 | 291.51 | 1.55x |
| 57 | 8 | 377.01 | 578.90 | 1.54x |
| 57 | 16 | 733.67 | 1114.00 | 1.52x |
| 841 | 1 | 48.98 | 72.75 | 1.49x |
| 841 | 16 | 548.23 | 793.60 | 1.45x |
| 3145 | 1 | 45.39 | 66.45 | 1.46x |
| 3145 | 16 | 299.26 | 403.87 | 1.35x |

## Median end-to-end latency

| Input tokens | Concurrency | BF16 seconds | FP8 seconds |
|---:|---:|---:|---:|
| 57 | 1 | 2.574 | 1.728 |
| 57 | 4 | 2.717 | 1.756 |
| 57 | 8 | 2.717 | 1.766 |
| 57 | 16 | 2.786 | 1.836 |
| 841 | 1 | 2.614 | 1.759 |
| 841 | 16 | 3.703 | 2.557 |
| 3145 | 1 | 2.822 | 1.927 |
| 3145 | 16 | 6.730 | 5.014 |

FP8 improved throughput at every tested setting.
The smaller advantage at long context and high concurrency is
consistent with a growing contribution from attention and KV-cache
work. Kernel profiling is needed to establish the actual bottleneck.

## Quality evaluation

A seeded sample of 100 GSM8K test questions was evaluated using
temperature=0 and max_tokens=512, with brief reasoning requested.

The scorer compared the final numeric answer following ####,
allowing the marker inline or on a separate line.
Reference answers were not included in model requests.

| Outcome | Questions |
|---|---:|
| Both correct | 88 |
| Both wrong | 5 |
| BF16 correct, FP8 wrong | 4 |
| BF16 wrong, FP8 correct | 3 |

- BF16 numeric-answer accuracy: 92/100
- FP8 numeric-answer accuracy: 91/100
- Both runs: zero parse failures under the final scoring rule
- Both runs: zero responses truncated by the token limit

BF16 initially scored 90/100 with a stricter separate-line parser.
Two numerically correct answers used inline markers.
Original responses and strict-score results were preserved.
The relaxed numeric-answer parser was used for the comparison.

The observed accuracy change is minus one percentage point.
This small sample does not establish general quality equivalence.

## Checkpoint storage

Only safetensors weight files were counted.

| Checkpoint | Weight bytes | Weight GiB |
|---|---:|---:|
| BF16 | 15231271888 | 14.185 |
| FP8 | 8706022488 | 8.108 |

- Storage reduction: 42.84%
- Compression ratio: 1.75x

This does not measure GPU memory savings.
Both serving configurations used a GPU memory utilization setting
of 0.80, so total GPU allocation is not a direct weight-size comparison.

## Result files

- results/fixed128_comparison.json
- results/longer_input_comparison.json
- results/checkpoint_size_comparison.json
- results/fp8_quantization_manifest.json
- results/bf16_gsm8k_20261026T165108Z/summary_rescored.json
- results/fp8_gsm8k_20261026T171906Z/paired_comparison.json

## Limitations and remaining work

- One GPU and one model.
- One full run per matched performance setting; no between-run
  uncertainty estimate.
- Synthetic repeated prompts and fixed 128-token outputs.
- Finite closed-loop workloads rather than production arrival traffic.
- Small, math-only quality sample; potential pretraining exposure
  to GSM8K was not assessed.
- Calibration was math-focused and did not reproduce all evaluation
  prompt formatting or generated reasoning.
- No isolated prefill, kernel, power, cost, or GPU-memory profiling.
- A clean environment restoration has not yet been validated.
- FP8 checkpoint weights need durable backup outside temporary storage.
- Automated acceptance checks and deployment packaging remain.
