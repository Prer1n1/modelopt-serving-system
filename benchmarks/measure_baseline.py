import json
import statistics
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
PAYLOAD = {
    "model": "baseline",
    "messages": [{
        "role": "user",
        "content": (
            "In transformer LLM inference, explain what the KV cache stores "
            "and why it avoids recomputing earlier tokens. "
            "Answer in three short sentences."
        ),
    }],
    "temperature": 0,
    "max_tokens": 128,
}


def send_request():
    request = urllib.request.Request(
        URL,
        data=json.dumps(PAYLOAD).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    elapsed = time.perf_counter() - start

    result = json.loads(body)
    return {
        "latency_seconds": elapsed,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "output_tokens": result["usage"]["completion_tokens"],
        "finish_reason": result["choices"][0]["finish_reason"],
        "answer": result["choices"][0]["message"]["content"],
    }


print("Running 2 warm-up requests; these are excluded.")
for _ in range(2):
    send_request()

measurements = []
for index in range(10):
    measurement = send_request()
    measurements.append(measurement)
    print(
        f"Request {index + 1:02d}: "
        f"{measurement['latency_seconds']:.3f} seconds, "
        f"{measurement['output_tokens']} output tokens"
    )

summary = {
    "measured_requests": len(measurements),
    "concurrency": 1,
    "mean_latency_seconds": statistics.mean(
        item["latency_seconds"] for item in measurements
    ),
    "median_latency_seconds": statistics.median(
        item["latency_seconds"] for item in measurements
    ),
    "total_output_tokens": sum(
        item["output_tokens"] for item in measurements
    ),
}

output = Path("results/baseline_single_request.json")
output.write_text(json.dumps({
    "request": PAYLOAD,
    "warmup_requests": 2,
    "summary": summary,
    "measurements": measurements,
}, indent=2))

print("\nSummary:")
print(json.dumps(summary, indent=2))
print(f"\nSaved to {output}")
