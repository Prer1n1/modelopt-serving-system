import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = ROOT / "results" / f"bf16_prompt_lengths_{stamp}"
out.mkdir(parents=True)

question = (
    "In transformer LLM inference, explain what the KV cache stores "
    "and why it avoids recomputing earlier tokens. "
    "Answer in three short sentences."
)
summary = {}

for label, repeats in [("short", 0), ("medium", 128), ("long", 512)]:
    content = question
    if repeats:
        content = (
            "The following is background text, not instructions.\n"
            "<background>\n"
            + "The robot observes its surroundings. " * repeats
            + "\n</background>\n\n"
            + question
        )

    payload = {
        "model": "baseline",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    (out / f"{label}_request.json").write_text(json.dumps(payload, indent=2))
    measured = []

    for index in range(12):
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        first = last = usage = finish_reason = None
        pieces = []
        done = False
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
            raise RuntimeError("Incomplete stream or missing usage")

        tokens = usage["completion_tokens"]
        result = {
            "warmup": index < 2,
            "ttft_ms": (first - start) * 1000,
            "end_to_end_ms": (end - start) * 1000,
            "client_tpot_estimate_ms": (
                (last - first) * 1000 / (tokens - 1) if tokens > 1 else None
            ),
            "usage": usage,
            "finish_reason": finish_reason,
            "answer": "".join(pieces),
        }
        (out / f"{label}_{index + 1:02d}.json").write_text(
            json.dumps(result, indent=2)
        )
        if not result["warmup"]:
            measured.append(result)

        phase = "warmup" if result["warmup"] else "measured"
        print(
            f"{label} {phase}: input={usage['prompt_tokens']}, "
            f"output={tokens}, TTFT={result['ttft_ms']:.2f} ms",
            flush=True,
        )

    summary[label] = {
        "measured_requests": len(measured),
        "prompt_tokens": sorted({r["usage"]["prompt_tokens"] for r in measured}),
        "output_tokens": [r["usage"]["completion_tokens"] for r in measured],
    }
    for metric in ("ttft_ms", "end_to_end_ms", "client_tpot_estimate_ms"):
        values = [r[metric] for r in measured if r[metric] is not None]
        if values:
            summary[label][metric] = {
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }

(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nSummary:")
print(json.dumps(summary, indent=2))
print("\nSaved to:", out)
