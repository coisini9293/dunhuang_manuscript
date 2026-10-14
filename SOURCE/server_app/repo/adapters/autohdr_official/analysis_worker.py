from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from . import ADAPTER_VERSION, OFFICIAL_COMMIT
from .audit_writer import append_audit
from .candidate_exporter import finalize_candidate_bundle
from .common import (
    AdapterContractError,
    file_sha256,
    stable_region_id,
    verify_official_tree,
)


DAMAGE_SCORE_THRESHOLD = 0.3
DAMAGE_OCR_THRESHOLD = 0.9
OVERLAP_IOU_THRESHOLD = 0.5
MAX_DEGRADED_PLACEHOLDERS = 270


@contextmanager
def official_import_context(source: Path, runtime: Path):
    previous_cwd = Path.cwd()
    original_path = list(sys.path)
    if not runtime.is_dir():
        raise AdapterContractError(f"Official runtime is missing: {runtime}")
    os.chdir(runtime)
    sys.path.insert(0, str(source))
    try:
        yield
    finally:
        sys.path[:] = original_path
        os.chdir(previous_cwd)


def _iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )
    return inter / union if union else 0.0


def _context(ordered: list[dict], index: int, direction: int, limit: int = 16) -> str:
    chars: list[str] = []
    cursor = index + direction
    while 0 <= cursor < len(ordered) and len(chars) < limit:
        value = ordered[cursor].get("visible_text", "")
        if value:
            if direction < 0:
                chars.insert(0, value)
            else:
                chars.append(value)
        cursor += direction
    return "".join(chars)


def _qwen_predictions(
    model, tokenizer, prompt: str, device, multi_token_topk
) -> dict[str, dict]:
    import torch
    import torch.nn.functional as functional

    messages = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "请帮助恢复古籍中缺失的字。"},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([messages], padding=True, return_tensors="pt").to(device)
    output = model.generate(
        **model_inputs,
        max_new_tokens=4096,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequence = output["sequences"][0].cpu()
    input_length = int(model_inputs["input_ids"].shape[1])
    generated = sequence[input_length:]
    scores = output["scores"]
    result: dict[str, dict] = {}
    cursor = 0
    while cursor < len(generated):
        token = tokenizer.decode([generated[cursor]])
        if token.startswith("<|extra_") and token.endswith("|>"):
            placeholder = token
            char_start = cursor + 1
            char_ids = []
            while char_start + len(char_ids) < len(generated):
                candidate_id = generated[char_start + len(char_ids)]
                decoded = tokenizer.decode([candidate_id])
                if decoded.startswith("<|extra_") or decoded == "<|im_end|>":
                    break
                char_ids.append(candidate_id)
            predicted = tokenizer.decode(char_ids).strip()
            positions = list(range(char_start, char_start + len(char_ids)))
            ranked: list[tuple[str, float]] = []
            if len(positions) == 1 and positions[0] < len(scores):
                distribution = functional.softmax(scores[positions[0]][0], dim=0)
                values, ids = torch.topk(distribution, k=5)
                ranked = [
                    (tokenizer.decode([token_id]).strip(), float(probability))
                    for token_id, probability in zip(ids, values)
                    if tokenizer.decode([token_id]).strip()
                ]
            elif positions and positions[0] < len(scores):
                combinations = multi_token_topk(
                    [score[0] for score in scores],
                    tokenizer,
                    start_idx=positions[0],
                    num_tokens=len(positions),
                    k=5,
                )
                ranked = [
                    (text.strip(), float(probability))
                    for text, probability, _ in combinations
                    if text.strip()
                ]
            result[placeholder] = {
                "predicted_text": predicted,
                "ranked": ranked,
            }
            cursor = char_start + len(char_ids)
            continue
        cursor += 1
    return result


def run_analysis(
    image_path: Path,
    case_id: str,
    case_dir: Path,
    official_source: Path,
    official_runtime: Path,
) -> dict:
    """Run only Official damage/OCR/Qwen components and export candidates."""
    verify_official_tree(official_source, OFFICIAL_COMMIT)
    image_path = image_path.resolve()
    image_hash = file_sha256(image_path)
    analysis_dir = case_dir / "analysis"
    crop_dir = analysis_dir / "crops"
    mask_dir = analysis_dir / "masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    audit_path = case_dir / "audit" / "adapter_events.jsonl"

    with official_import_context(official_source, official_runtime):
        import cv2
        import numpy as np
        import torch
        from mmdet.apis import inference_detector, init_detector
        from utils.reader import get_sort
        from utils.recognition import batch_char_recog_exe
        from utils.reg_wrapper import reg_model
        from opencc import OpenCC
        from zhconv import convert
        from utils_pipeline import (
            get_topk_multi_tokens,
            invert_image,
            model_init,
            rank_probability_weighted_fusion,
        )
        from infer_pipeline import detect

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise AdapterContractError("GPU_REQUIRED_FOR_OFFICIAL_ANALYSIS")

        original = Image.open(image_path).convert("RGB")
        inverted = invert_image(original)
        temporary_input = analysis_dir / "official_inverted_input.png"
        inverted.save(temporary_input, format="PNG")
        gray_rgb = inverted.convert("L").convert("RGB")

        damage_model = init_detector(
            "./ckpt/damage_detect.py", "./ckpt/damage_detect.pth", device=device
        )
        damage_result = inference_detector(damage_model, np.asarray(gray_rgb))
        damage_boxes: list[list[int]] = []
        for box, score in zip(
            damage_result.pred_instances.bboxes.cpu().numpy(),
            damage_result.pred_instances.scores.cpu().numpy(),
        ):
            if float(score) > DAMAGE_SCORE_THRESHOLD:
                damage_boxes.append([int(round(value)) for value in box])

        options = SimpleNamespace(
            data=str(temporary_input),
            det_batch_size=1,
            reg_batch_size=36,
            img_size=2048,
            conf_thres=0.45,
            iou_thres=0.2,
            vague_det_config="./ckpt/damage_detect.py",
            vague_det_weights="./ckpt/damage_detect.pth",
            model_name_or_path="./ckpt/AutoHDR-Qwen2-7B",
        )
        detection = detect(options, str(temporary_input), 1, 2048, 0.45, 0.2)
        ocr_boxes = [
            [round(row[0]), round(row[1]), round(row[2]), round(row[3])]
            for rows in detection.values()
            for row in rows
        ]
        gray = cv2.imread(str(temporary_input), 0)
        dictionary = Path("ckpt/dic_31524.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        recognizer = reg_model("./dist/reg_model")
        recognizer.start()
        try:
            ocr_chars, ocr_probs = batch_char_recog_exe(
                recognizer,
                device,
                dictionary,
                [gray[y1:y2, x1:x2] for x1, y1, x2, y2 in ocr_boxes],
                bs=36,
            )
            damage_chars, damage_probs = batch_char_recog_exe(
                recognizer,
                device,
                dictionary,
                [gray[y1:y2, x1:x2] for x1, y1, x2, y2 in damage_boxes],
                bs=36,
            )
        finally:
            recognizer.cleanup()

        ocr_by_box = {
            tuple(box): {"chars": chars, "probs": probs}
            for box, chars, probs in zip(ocr_boxes, ocr_chars, ocr_probs)
        }
        damage_by_box = {
            tuple(box): {"chars": chars, "probs": probs}
            for box, chars, probs in zip(damage_boxes, damage_chars, damage_probs)
        }
        overlapping_ocr = {
            tuple(box)
            for box in ocr_boxes
            if any(_iou(box, damage) >= OVERLAP_IOU_THRESHOLD for damage in damage_boxes)
        }
        ordered_boxes = get_sort(
            damage_boxes
            + [box for box in ocr_boxes if tuple(box) not in overlapping_ocr],
            gray.shape[0],
            gray.shape[1],
        )

        ordered: list[dict] = []
        placeholder_count = 0
        visible_count = 0
        prompt_parts: list[str] = []
        for index, raw_box in enumerate(ordered_boxes):
            box = [int(value) for value in raw_box]
            key = tuple(box)
            if key in ocr_by_box:
                visible_count += 1
                text = str(ocr_by_box[key]["chars"][0])
                ordered.append(
                    {"box": box, "kind": "visible", "visible_text": text, "index": index}
                )
                prompt_parts.append(text)
                continue
            evidence = damage_by_box[key]
            top_char = str(evidence["chars"][0])
            top_prob = float(evidence["probs"][0])
            if top_prob < DAMAGE_OCR_THRESHOLD:
                placeholder = f"<|extra_{placeholder_count}|>"
                placeholder_count += 1
                prompt_parts.append(placeholder)
            else:
                placeholder = f"<|ocr_{visible_count + 1}|>"
                visible_count += 1
                prompt_parts.append(top_char)
            ordered.append(
                {
                    "box": box,
                    "kind": "damage",
                    "visible_text": "",
                    "index": index,
                    "placeholder": placeholder,
                    "ocr_chars": [str(value) for value in evidence["chars"][:5]],
                    "ocr_probs": [float(value) for value in evidence["probs"][:5]],
                    "ocr_top": top_char,
                    "ocr_top_probability": top_prob,
                }
            )
        if placeholder_count > MAX_DEGRADED_PLACEHOLDERS:
            raise AdapterContractError("TOO_MANY_DEGRADED_CHARACTERS")

        prompt = OpenCC("s2t").convert(convert("".join(prompt_parts), "zh-cn"))
        qwen = {}
        if placeholder_count:
            language_model, tokenizer = model_init("./ckpt/AutoHDR-Qwen2-7B")
            try:
                qwen = _qwen_predictions(
                    language_model,
                    tokenizer,
                    prompt,
                    device,
                    get_topk_multi_tokens,
                )
            finally:
                del language_model
                del tokenizer
                torch.cuda.empty_cache()

        damage_regions: list[dict] = []
        for item in ordered:
            if item["kind"] != "damage":
                continue
            index = item["index"]
            box = item["box"]
            region_id = stable_region_id(image_hash, index, box)
            x1, y1, x2, y2 = box
            crop_path = crop_dir / f"{region_id}.png"
            mask_path = mask_dir / f"{region_id}.png"
            original.crop((x1, y1, x2, y2)).save(crop_path, format="PNG")
            mask = Image.new("L", original.size, 0)
            mask.paste(255, (x1, y1, x2, y2))
            mask.save(mask_path, format="PNG")
            qwen_result = qwen.get(item["placeholder"], {})
            qwen_char = qwen_result.get("predicted_text", "")
            llm_ranked = qwen_result.get("ranked", [])
            ocr_ranked = list(zip(item["ocr_chars"], item["ocr_probs"]))
            fused_scores = {}
            if llm_ranked:
                try:
                    chosen, fused_scores = rank_probability_weighted_fusion(
                        ocr_ranked, llm_ranked
                    )
                except (ValueError, IndexError, KeyError):
                    chosen = item["ocr_top"]
            else:
                chosen = item["ocr_top"]
            candidates = []
            for char, score in fused_scores.items():
                candidates.append(
                    {
                        "text": char,
                        "source": "official_ocr_qwen_fusion",
                        "score": float(score),
                    }
                )
            if qwen_char and not any(
                candidate["text"] == qwen_char for candidate in candidates
            ):
                candidates.append({"text": qwen_char, "source": "official_qwen"})
            for char, probability in zip(item["ocr_chars"], item["ocr_probs"]):
                if not any(candidate["text"] == char for candidate in candidates):
                    candidates.append(
                        {
                            "text": char,
                            "source": "official_ocr",
                            "confidence": probability,
                        }
                    )
            damage_regions.append(
                {
                    "region_id": region_id,
                    "bbox": box,
                    "mask_path": str(mask_path.relative_to(case_dir)),
                    "crop_path": str(crop_path.relative_to(case_dir)),
                    "ocr_before": _context(ordered, index, -1),
                    "ocr_after": _context(ordered, index, 1),
                    "ocr_raw": item["ocr_top"],
                    "official_placeholder": item["placeholder"],
                    "autohdr_candidate": chosen,
                    "autohdr_confidence": (
                        float(fused_scores.get(chosen))
                        if chosen in fused_scores
                        else item["ocr_top_probability"]
                    ),
                    "autohdr_candidates": candidates,
                    "reading_order": index,
                }
            )
        del damage_model
        torch.cuda.empty_cache()

    payload = {
        "case_id": case_id,
        "image_sha256": image_hash,
        "official_commit": OFFICIAL_COMMIT,
        "worker_version": ADAPTER_VERSION,
        "official_thresholds": {
            "damage_score": DAMAGE_SCORE_THRESHOLD,
            "damage_ocr": DAMAGE_OCR_THRESHOLD,
            "overlap_iou": OVERLAP_IOU_THRESHOLD,
        },
        "damage_regions": damage_regions,
        "final_restore_generated": False,
    }
    bundle = finalize_candidate_bundle(payload, analysis_dir / "candidate_bundle.json")
    append_audit(
        audit_path,
        "official_analysis_completed",
        {
            "case_id": case_id,
            "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
            "region_count": len(damage_regions),
            "final_restore_generated": False,
        },
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--official-runtime", required=True, type=Path)
    args = parser.parse_args()
    run_analysis(
        args.input,
        args.case_id,
        args.case_dir,
        args.official_source,
        args.official_runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
