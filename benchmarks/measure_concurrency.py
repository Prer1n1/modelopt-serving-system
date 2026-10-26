import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = ROOT / "results" / f"bf16_concurrency_{stamp}"
out.mkdir(parents=True)

payload = {
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
    "stream": True,
    "stream_options": {"include_usage": True},
}
(out / "request.json").write_text(json.dumps(payload, indent=2))


def measure(index):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = last = usage = finish_reason = None
    done = False
    pieces = []
    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                text = choice.get("delta", {}).get("content")
                if text:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    pieces.append(text)

    end = time.perf_counter()
    if not done or first is None or usage is None:
        raise RuntimeError(f"Request {index}: incomplete stream")

    tokens = usage["completion_tokens"]
    return {
        "request_id": index,
        "ttft_ms": (first - start) * 1000,
        "end_to_end_ms": (end - start) * 1000,
        "client_tpot_estimate_ms": (
            (last - first) * 1000 / (tokens - 1) if tokens > 1 else None
        ),
        "usage": usage,
        "finish_reason": finish_reason,
        "answer": "".join(pieces),
    }


summary = {}
for concurrency in (1, 4, 8, 16):
    print(f"\nConcurrency {concurrency}: warming up...", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        warmups = list(pool.map(measure, range(concurrency)))
        (out / f"c{concurrency}_warmups.json").write_text(
            json.dumps(warmups, indent=2)
        )

        start = time.perf_counter()
        records = list(pool.map(measure, range(64)))
        elapsed = time.perf_counter() - start

    (out / f"c{concurrency}_requests.json").write_text(
        json.dumps(records, indent=2)
    )
    total_tokens = sum(r["usage"]["completion_tokens"] for r in records)
    result = {
        "max_concurrency": concurrency,
        "measured_requests": len(records),
        "wall_seconds": elapsed,
        "output_tokens_per_second": total_tokens / elapsed,
        "requests_per_second": len(records) / elapsed,
        "prompt_token_counts": sorted({
            r["usage"]["prompt_tokens"] for r in records
        }),
        "output_token_counts": sorted({
            r["usage"]["completion_tokens"] for r in records
        }),
    }
    for metric in ("ttft_ms", "end_to_end_ms", "client_tpot_estimate_ms"):
        values = [r[metric] for r in records if r[metric] is not None]
        if values:
            result[metric] = {
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
    summary[str(concurrency)] = result
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(result, indent=2), flush=True)

print("\nSaved to:", out)
