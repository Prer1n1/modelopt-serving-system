import argparse
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="baseline")
parser.add_argument("--label", default="bf16")
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
dataset = root / "data/evaluation/gsm8k_100.jsonl"
raw = dataset.read_bytes()
rows = [json.loads(line) for line in raw.decode().splitlines()]
if not 1 <= args.limit <= len(rows):
    raise SystemExit("--limit must be between 1 and 100")
rows = rows[:args.limit]

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = root / "results" / f"{args.label}_gsm8k_{stamp}"
out.mkdir(parents=True)

instruction = (
    "Solve the following math problem. Explain your reasoning briefly. "
    "End your response with a separate line containing #### followed by "
    "only the final numeric answer, for example: #### 18"
)

def extract(text):
    matches = re.findall(
        r"(?m)####[ \t]*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))[ \t]*$",
        text,
    )
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None

correct = parse_failures = truncated = 0
start = time.perf_counter()

with (out / "responses.jsonl").open("w") as output:
    for index, row in enumerate(rows, 1):
        payload = {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": instruction + "\n\n" + row["question"],
            }],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            event = json.load(response)

        if event.get("error"):
            raise RuntimeError(event["error"])

        choice = event["choices"][0]
        answer = choice["message"].get("content") or ""
        prediction = extract(answer)
        reference = Decimal(row["reference_answer"].replace(",", ""))
        passed = prediction is not None and prediction == reference
        correct += int(passed)
        parse_failures += int(prediction is None)
        truncated += int(choice.get("finish_reason") == "length")

        record = {
            **row,
            "request": payload,
            "answer": answer,
            "prediction": str(prediction) if prediction is not None else None,
            "correct": passed,
            "finish_reason": choice.get("finish_reason"),
            "usage": event.get("usage"),
        }
        output.write(json.dumps(record) + "\n")
        output.flush()
        print(
            f"{index}/{len(rows)}: {'PASS' if passed else 'FAIL'} "
            f"| predicted={prediction} | expected={reference}",
            flush=True,
        )

summary = {
    "model": args.model,
    "label": args.label,
    "dataset_sha256": hashlib.sha256(raw).hexdigest(),
    "questions": len(rows),
    "correct": correct,
    "accuracy_percent": 100 * correct / len(rows),
    "parse_failures": parse_failures,
    "truncated_responses": truncated,
    "temperature": 0,
    "max_tokens": 512,
    "instruction": instruction,
    "wall_seconds": time.perf_counter() - start,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nSummary:")
print(json.dumps(summary, indent=2))
print("\nSaved to:", out)
