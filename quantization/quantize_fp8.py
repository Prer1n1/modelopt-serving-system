import copy
import hashlib
import json
import os
import shutil
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/modelopt-serving-system-hf"

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
OUTPUT = Path("/tmp/modelopt-qwen2.5-7b-fp8")
DATA = ROOT / "data/calibration/gsm8k_train_128.jsonl"

if OUTPUT.exists():
    raise SystemExit(f"Output already exists; refusing to overwrite: {OUTPUT}")

assert torch.cuda.is_available(), "CUDA is unavailable"
free, total = torch.cuda.mem_get_info()
print(f"Free GPU memory: {free / 2**30:.1f} GiB", flush=True)
if free < 25 * 2**30:
    raise SystemExit("Free the GPU before quantization; expected at least 25 GiB.")

disk_free = shutil.disk_usage("/tmp").free
if disk_free < 12 * 2**30:
    raise SystemExit("Less than 12 GiB disk space remains; inspect storage first.")

raw = DATA.read_bytes()
records = [json.loads(line) for line in raw.decode().splitlines()]
assert len(records) == 128

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL, revision=REVISION, local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager",
)
model.eval()
config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
lengths = []

@torch.inference_mode()
def calibration_loop(calibration_model):
    for index, row in enumerate(records, 1):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to("cuda")
        lengths.append(inputs["input_ids"].shape[1])
        calibration_model(**inputs, use_cache=False)
        if index % 16 == 0:
            print(f"Calibrated {index}/{len(records)} questions", flush=True)

print("Starting FP8 calibration...", flush=True)
model = mtq.quantize(model, config, forward_loop=calibration_loop)
mtq.print_quant_summary(model)

print("Exporting FP8 checkpoint...", flush=True)
with torch.inference_mode():
    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=OUTPUT,
        save_modelopt_state=False,
        max_shard_size="2GB",
    )

tokenizer.save_pretrained(OUTPUT)

quant_config = OUTPUT / "hf_quant_config.json"
if not quant_config.exists():
    raise RuntimeError("Export did not produce hf_quant_config.json; inspect output.")

manifest = {
    "source_model": MODEL,
    "source_revision": REVISION,
    "modelopt": version("nvidia-modelopt"),
    "transformers": version("transformers"),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "recipe": "FP8_DEFAULT_CFG",
    "quantization_config": config,
    "calibration_questions": len(records),
    "calibration_sha256": hashlib.sha256(raw).hexdigest(),
    "calibration_max_length": 512,
    "calibration_token_lengths": lengths,
    "calibration_use_cache": False,
    "export_directory": str(OUTPUT),
    "exported_quantization_config": json.loads(quant_config.read_text()),
}
manifest_path = ROOT / "results/fp8_quantization_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))

size = sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())
print(f"\nCheckpoint saved: {OUTPUT}")
print(f"Checkpoint size: {size / 2**30:.2f} GiB")
print(f"Manifest saved: {manifest_path}")
print("\nExported quantization configuration:")
print(quant_config.read_text())
