import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_dir = root / "results" / f"bf16_repeated_{timestamp}"
output_dir.mkdir(parents=True)

source = root / "results/baseline_streaming_single.json"
if source.exists():
    (output_dir / "before_repeat.json").write_text(source.read_text())

records = []

for index in range(23):
    warmup = index < 3
    label = f"warmup_{index + 1}" if warmup else f"request_{index - 2}"

    completed = subprocess.run(
        [sys.executable, str(root / "benchmarks/measure_streaming.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)

    result = json.loads(source.read_text())
    (output_dir / f"{label}.json").write_text(json.dumps(result, indent=2))

    print(f"{label}: TTFT={result['ttft_seconds'] * 1000:.2f} ms")
    if not warmup:
        records.append(result)

summary = {
    "warmup_requests": 3,
    "measured_requests": len(records),
    "concurrency": 1,
    "metrics": {},
}

for key in (
    "ttft_seconds",
    "end_to_end_seconds",
    "client_tpot_estimate_seconds",
):
    values = [r[key] * 1000 for r in records if r.get(key) is not None]
    if values:
        summary["metrics"][key.replace("_seconds", "_ms")] = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }

summary["output_token_counts"] = [
    r["usage"]["completion_tokens"] for r in records
]

(output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nSummary:")
print(json.dumps(summary, indent=2))
print("\nSaved to:", output_dir)
