import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
checks = 0

def require(condition, message):
    global checks
    if not condition:
        raise ValueError(message)
    checks += 1

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def source_directory(value):
    path = Path(value)
    if path.parts[0] == "results":
        return ROOT / path
    return RESULTS / path

def validate_pair(name, comparison):
    bf_dir = source_directory(comparison["bf16_source"])
    fp_dir = source_directory(comparison["fp8_source"])
    bf = load(bf_dir / "summary.json")
    fp = load(fp_dir / "summary.json")

    require(
        load(bf_dir / "request.json") == load(fp_dir / "request.json"),
        f"{name}: request payloads differ",
    )

    for concurrency, saved in comparison["results"].items():
        b, f = bf[concurrency], fp[concurrency]
        for label, directory, summary in (
            ("BF16", bf_dir, b), ("FP8", fp_dir, f)
        ):
            prefix = f"{name}, concurrency {concurrency}, {label}"
            require(summary["measured_requests"] == 64,
                    f"{prefix}: expected 64 measured requests")
            require(summary["max_concurrency"] == int(concurrency),
                    f"{prefix}: concurrency mismatch")
            require(summary["output_token_counts"] == [128],
                    f"{prefix}: output length is not fixed at 128")

            records = load(directory / f"c{concurrency}_requests.json")
            require(len(records) == 64, f"{prefix}: missing raw requests")
            require(len({r["request_id"] for r in records}) == 64,
                    f"{prefix}: duplicate request IDs")

            actual_inputs = sorted({
                r["usage"]["prompt_tokens"] for r in records
            })
            require(actual_inputs == summary["prompt_token_counts"],
                    f"{prefix}: input counts disagree with raw records")
            require(all(r["usage"]["completion_tokens"] == 128 for r in records),
                    f"{prefix}: raw output lengths differ")

            wall = summary["wall_seconds"]
            require(math.isfinite(wall) and wall > 0,
                    f"{prefix}: invalid measurement duration")
            throughput = sum(
                r["usage"]["completion_tokens"] for r in records
            ) / wall
            require(math.isclose(
                throughput, summary["output_tokens_per_second"],
                rel_tol=1e-9
            ), f"{prefix}: throughput calculation mismatch")

        require(b["prompt_token_counts"] == f["prompt_token_counts"],
                f"{name}: BF16/FP8 input lengths differ")

        for label, summary in (("bf16", b), ("fp8", f)):
            require(math.isclose(
                saved[f"{label}_output_tokens_per_second"],
                summary["output_tokens_per_second"], rel_tol=1e-9
            ), f"{name}: saved {label} throughput differs from source")

        speedup = (
            f["output_tokens_per_second"] / b["output_tokens_per_second"]
        )
        require(math.isclose(
            saved["throughput_speedup"], speedup, rel_tol=1e-9
        ), f"{name}: speedup calculation mismatch")

        print(f"PASS: {name}, concurrency {concurrency}")

def main():
    validate_pair("short", load(RESULTS / "fixed128_comparison.json"))
    longer = load(RESULTS / "longer_input_comparison.json")
    for name in ("medium", "long"):
        validate_pair(name, longer[name])
    print(f"\nAll {checks} checks passed.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError, ZeroDivisionError) as error:
        raise SystemExit(f"FAIL: {error}")
