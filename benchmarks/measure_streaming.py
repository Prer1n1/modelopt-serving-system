import json
import time
import urllib.request
from pathlib import Path

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

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

first_text_time = None
last_text_time = None
usage = None
pieces = []
received_done = False

print("Streaming answer:\n")
start = time.perf_counter()

with urllib.request.urlopen(request, timeout=120) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()

        if not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            received_done = True
            break

        event = json.loads(data)

        if event.get("usage") is not None:
            usage = event["usage"]

        for choice in event.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if text:
                now = time.perf_counter()
                if first_text_time is None:
                    first_text_time = now
                last_text_time = now
                pieces.append(text)
                print(text, end="", flush=True)

end = time.perf_counter()

if not received_done or first_text_time is None or usage is None:
    raise RuntimeError("Incomplete stream, missing text, or missing token usage.")

output_tokens = usage["completion_tokens"]
tpot = (
    (last_text_time - first_text_time) / (output_tokens - 1)
    if output_tokens > 1 else None
)

result = {
    "request": payload,
    "ttft_seconds": first_text_time - start,
    "end_to_end_seconds": end - start,
    "client_tpot_estimate_seconds": tpot,
    "usage": usage,
    "answer": "".join(pieces),
}

output = Path("results/baseline_streaming_single.json")
output.write_text(json.dumps(result, indent=2))

print("\n\nMeasurements:")
print(f"TTFT: {result['ttft_seconds'] * 1000:.2f} ms")
print(f"End-to-end: {result['end_to_end_seconds']:.3f} seconds")
print(f"Output tokens: {output_tokens}")
if tpot is not None:
    print(f"TPOT estimate: {tpot * 1000:.2f} ms/token")
print(f"Saved to {output}")
