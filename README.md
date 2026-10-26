# ModelOpt Serving System

An experimental workflow for quantizing Qwen2.5-7B-Instruct with
NVIDIA Model Optimizer, serving it through vLLM, and comparing
performance and answer accuracy against the original BF16 model.

## Measured results

Tested on one NVIDIA L40S.

| Measure | BF16 | ModelOpt FP8 |
|---|---:|---:|
| Weight-file size | 14.19 GiB | 8.11 GiB |
| GSM8K numeric-answer accuracy, 100 questions | 92% | 91% |
| Output throughput, 57-token input, concurrency 16 | 733.7 tokens/s | 1,114.0 tokens/s |
| Median response latency for that workload | 2.786 s | 1.836 s |

Across the tested fixed-output workloads, FP8 achieved
**1.35–1.55× aggregate output throughput** and
**42.84% smaller weight files**.

These are initial measurements, not production performance guarantees.
The quality evaluation is a small math-only sample.

## Workflow

1. Serve the original BF16 checkpoint through vLLM.
2. Measure streaming latency, prompt-length effects, and concurrency.
3. Evaluate a fixed sample of 100 GSM8K test questions.
4. Calibrate ModelOpt FP8 using 128 separate GSM8K training questions.
5. Export and serve the quantized checkpoint.
6. Repeat the same quality and performance evaluations.
7. Save raw responses, summaries, and comparison results.

## Quantization

- Model: Qwen/Qwen2.5-7B-Instruct
- Source revision: a09a35458c702b33eeacc393d103063234e8bc28
- ModelOpt: 0.46.1
- Recipe: FP8_DEFAULT_CFG with max calibration
- Calibration: 128 questions, maximum 512 input tokens each
- Embeddings and lm_head excluded from FP8 quantization
- KV-cache quantization disabled

Calibration and evaluation used separate dataset splits, with no
exact question overlap in the selected samples.

## Performance methodology

Matched comparisons used:

- Input lengths: 57, 841, and 3,145 tokens
- Output length: exactly 128 tokens
- Concurrency: 1, 4, 8, and 16 for short inputs;
  1 and 16 for medium and long inputs
- 64 measured requests per setting, following warmup
- Temperature: 0
- Prefix caching: disabled
- Same serving limits and GPU memory utilization setting

Output length was controlled with min_tokens, max_tokens, and
ignore_eos. These forced-length tests measure serving performance,
not answer quality.

TTFT, end-to-end latency, and TPOT estimates were measured from the
client. TPOT is based on text-chunk arrival times, not GPU profiling.

## Repository layout

| Directory | Purpose |
|---|---|
| scripts/ | Environment setup and server startup |
| benchmarks/ | Streaming and concurrency measurements |
| quantization/ | ModelOpt calibration and FP8 export |
| evaluation/ | GSM8K generation and scoring |
| data/ | Fixed calibration and evaluation samples |
| results/ | Raw responses, summaries, and comparisons |
| reports/ | Methodology, findings, and limitations |

## Running the project

The current scripts were developed for a Linux GPU pod with Python
3.12 and NVIDIA CUDA support. They use temporary paths under /tmp.

### Set up serving

```bash
bash scripts/setup_environment.sh
```

The setup procedure has not yet been validated through a clean
restoration test.

### Serve BF16

```bash
bash scripts/serve_baseline.sh
```

### Serve the exported FP8 checkpoint

Restore the checkpoint directory to:

```text
/tmp/modelopt-qwen2.5-7b-fp8
```

Then run:

```bash
bash scripts/serve_fp8.sh
```

Run one server at a time. Both scripts expose the API model name
`baseline`; the startup script determines the loaded checkpoint.

### Run fixed-output benchmarks

With the corresponding server running:

```bash
/tmp/modelopt-serving-system-venv/bin/python \
  benchmarks/measure_fixed_length.py \
  --label fp8 --prompt-length short
```

Use `--label bf16` when serving BF16.
The label names the results; it does not switch models.

### Run quality evaluation

```bash
/tmp/modelopt-serving-system-venv/bin/python \
  evaluation/evaluate_gsm8k.py \
  --model baseline --label fp8 --limit 100
```

The scorer accepts a final numeric answer following `####`,
either inline or on a separate line.

## Detailed results

See [the experiment report](reports/bf16_vs_fp8.md).

Comparison files:

- [Short-input performance](results/fixed128_comparison.json)
- [Medium- and long-input performance](results/longer_input_comparison.json)
- [Weight-file sizes](results/checkpoint_size_comparison.json)
- [Quantization manifest](results/fp8_quantization_manifest.json)

## Limitations

- One GPU, one model, and synthetic performance prompts.
- One full run per matched setting; between-run variability has
  not been quantified.
- The 100-question quality sample does not establish general
  quality preservation.
- Weight-file reduction does not establish GPU-memory savings.
- No production arrival-traffic, power, or kernel profiling.
- Clean restoration and automated acceptance checks remain.

## Model artifacts

Exported model weights are not stored in this Git repository.
The FP8 checkpoint was archived separately, downloaded, and verified
against its SHA-256 checksum.

## Next steps

- Validate restoration on a fresh GPU pod.
- Repeat benchmarks and report variability.
- Add automated workload and result validation.
- Expand quality evaluation beyond the initial math sample.
- Package deployment with explicit configuration and health checks.
