#!/usr/bin/env python3
"""Execute the deployed AutoHDR Qwen checkpoint once on CUDA and log provenance."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    runtime = Path(args.runtime_root).resolve()
    # The official helper resolves bundled assets (including its legacy font)
    # relative to the process directory even though this probe never renders it.
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    canonical_autohdr = Path(__file__).resolve().parents[3] / "src" / "AutoHDR"
    sys.path.insert(0, str(canonical_autohdr))
    os.chdir(canonical_autohdr)
    import torch
    from utils_pipeline import model_init
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen validation requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = model_init(str(Path(args.model_path).resolve()))
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "你是古籍修复系统的文本候选生成器。"},
        {"role": "user", "content": "仅输出：模型已加载"},
    ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    sequence = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(sequence[0], skip_special_tokens=True)
    result = {
        "status": "PASS", "device": torch.cuda.get_device_name(0),
        "model_path": str(Path(args.model_path).resolve()), "response": text,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
