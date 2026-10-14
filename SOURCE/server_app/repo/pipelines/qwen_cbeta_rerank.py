#!/usr/bin/env python3
"""Run the deployed local Qwen model as a non-authoritative CBETA reranker."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    started = time.time()
    evidence_path = Path(args.evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    selected = evidence["selected_same_sutra_evidence"]
    alternatives = evidence["top_candidates"]

    candidate_lines = []
    seen = set()
    for side in ("a", "b"):
        for item in alternatives[side]:
            key = (item["sutra_id"], item["chunk_key"])
            if key in seen:
                continue
            seen.add(key)
            candidate_lines.append(
                f"{len(candidate_lines)+1}. {item['sutra_id']}《{item['canonical_title']}》"
                f" chunk={item['chunk_key']} records={item['record_start_id']}..{item['record_end_id']} "
                f"text={item['normalized_text'][:220]}"
            )
            if len(candidate_lines) >= 6:
                break
        if len(candidate_lines) >= 6:
            break

    a = evidence["ocr_inputs"]["a"]
    b = evidence["ocr_inputs"]["b"]
    prompt = (
        "你是敦煌残片文本证据重排器。OCR 有噪声；CBETA 是权威来源。"
        "只判断两片是否被同一连续 CBETA 段落支持，不判断邻接和几何。"
        "不得补造引文。请只输出一行 JSON，字段为 decision、sutra_id、reason。\n"
        f"A 的逐字 OCR 锚点：{a['anchors_observed_verbatim']}\n"
        f"B 的逐字 OCR 锚点：{b['anchors_observed_verbatim']}\n"
        f"确定性检索选择：A={selected['fragment_a']['sutra_id']} chunk "
        f"{selected['fragment_a']['chunk_key']}；B={selected['fragment_b']['sutra_id']} chunk "
        f"{selected['fragment_b']['chunk_key']}。\n"
        "候选：\n" + "\n".join(candidate_lines)
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the completion runbook")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    messages = [
        {"role": "system", "content": "输出必须是单行 JSON。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=160, do_sample=False,
            repetition_penalty=1.05,
        )
    response = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()
    parsed = None
    match = re.search(r"\{.*\}", response, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
    expected = selected["fragment_a"]["sutra_id"]
    agrees = bool(
        parsed
        and str(parsed.get("sutra_id", "")) == expected
        and str(parsed.get("decision", "")).upper() in {
            "PASS", "MATCH", "SAME_TEXT", "TEXTUAL_HOMOLOGY_PASS"
        }
    )
    payload = {
        "status": "PASS" if agrees else "ABSTAIN",
        "role": "NON_AUTHORITATIVE_LOCAL_QWEN_RERANKER",
        "authority_boundary": "Qwen cannot override or create CBETA evidence.",
        "model_path": str(Path(args.model).resolve()),
        "device": torch.cuda.get_device_name(0),
        "torch_dtype": "float16",
        "prompt": prompt,
        "raw_response": response,
        "parsed_response": parsed,
        "agrees_with_deterministic_cbeta": agrees,
        "deterministic_cbeta_sutra_id": expected,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "runtime_seconds": time.time() - started,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "device": payload["device"],
        "agrees": agrees, "response": response,
        "gpu_peak_bytes": payload["gpu_peak_bytes"],
        "runtime_seconds": payload["runtime_seconds"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
