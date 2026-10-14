#!/usr/bin/env python3
"""Extract AutoHDR damage-character slots without running restoration.

The script reuses the official DINO damage detector, character recognizer and
reading-order function.  It emits exact boxes and top-k recognition evidence;
it never invokes Qwen, fonts, SD2, or the restoration compositor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--autohdr-root", required=True)
    parser.add_argument("--damage-threshold", type=float, default=0.3)
    parser.add_argument("--degraded-threshold", type=float, default=0.9)
    args = parser.parse_args()

    started = time.time()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    root = Path(args.autohdr_root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = output_path.parent / (output_path.stem + "_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    inverted_path = scratch / "official_inverted.jpg"

    os.chdir(root)
    sys.path.insert(0, str(root))
    from mmdet.apis import inference_detector, init_detector
    from utils.reader import get_sort
    from utils.recognition import batch_char_recog_exe
    from utils.reg_wrapper import reg_model
    from utils_pipeline import invert_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = Image.open(image_path).convert("RGB")
    invert_image(original).save(inverted_path)
    detector_input = np.asarray(Image.open(inverted_path).convert("L").convert("RGB"))

    config_path = root / "ckpt/damage_detect.py"
    weights_path = root / "ckpt/damage_detect.pth"
    detector = init_detector(str(config_path), str(weights_path), device=device)
    with torch.no_grad():
        result = inference_detector(detector, detector_input)
    detector_boxes = result.pred_instances.bboxes.detach().cpu().numpy()
    detector_scores = result.pred_instances.scores.detach().cpu().numpy()
    detector_labels = result.pred_instances.labels.detach().cpu().numpy()
    del detector, result
    torch.cuda.empty_cache()

    raw = []
    for box, score, label in zip(detector_boxes, detector_scores, detector_labels):
        if float(score) <= args.damage_threshold:
            continue
        xyxy = [int(round(float(value))) for value in box]
        raw.append((xyxy, float(score), int(label)))

    gray = cv2.imread(str(inverted_path), cv2.IMREAD_GRAYSCALE)
    valid = []
    crops = []
    detector_meta = {}
    for box, score, label in raw:
        x1, y1, x2, y2 = box
        crop = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if not crop.size:
            continue
        valid.append(box)
        crops.append(crop)
        detector_meta[str(box)] = {"damage_score": score, "damage_label": label}

    dictionary_path = root / "ckpt/dic_31524.txt"
    recognizer_path = root / "dist/reg_model"
    chars, probabilities = [], []
    if crops:
        dictionary = dictionary_path.read_text(encoding="utf-8").splitlines()
        recognizer = reg_model(str(recognizer_path))
        recognizer.start()
        chars, probabilities = batch_char_recog_exe(
            recognizer, device, dictionary, crops, bs=36
        )
        recognizer.cleanup()
        torch.cuda.empty_cache()

    recognized = {}
    for box, alternatives, probs in zip(valid, chars, probabilities):
        alternatives = plain(alternatives)
        probs = plain(probs)
        recognized[str(box)] = {
            "bbox_xyxy": box,
            "bbox_xywh": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
            **detector_meta[str(box)],
            "ocr_alternatives": alternatives,
            "ocr_probabilities": probs,
            "ocr_top1": alternatives[0] if alternatives else "",
            "ocr_top1_probability": float(probs[0]) if probs else 0.0,
        }

    height, width = gray.shape[:2]
    ordered_boxes = get_sort(valid, height, width) if valid else []
    ordered = [recognized[str(box)] for box in ordered_boxes if str(box) in recognized]
    mandatory = []
    for item in ordered:
        if item["ocr_top1_probability"] < args.degraded_threshold:
            copy = dict(item)
            copy["slot_id"] = f"extra_{len(mandatory)}"
            mandatory.append(copy)

    payload = {
        "status": "PASS" if mandatory else "FAIL_NO_DEGRADED_SLOTS",
        "mode": "OFFICIAL_AUTOHDR_DAMAGE_SLOT_EXTRACTION_READ_ONLY",
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "image_dimensions": [width, height],
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "damage_detector_config": str(config_path),
        "damage_detector_weights": str(weights_path),
        "recognizer_path": str(recognizer_path),
        "dictionary_path": str(dictionary_path),
        "settings": {
            "damage_threshold": args.damage_threshold,
            "degraded_threshold": args.degraded_threshold,
            "reading_order": "official utils.reader.get_sort",
        },
        "all_damage_box_count": len(ordered),
        "mandatory_slot_count": len(mandatory),
        "all_damage_boxes": ordered,
        "mandatory_slots": mandatory,
        "runtime_seconds": time.time() - started,
        "caveat": "Detector and OCR values are machine evidence; semantic text is assigned independently.",
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"],
        "device": payload["device"],
        "all_damage_box_count": payload["all_damage_box_count"],
        "mandatory_slot_count": payload["mandatory_slot_count"],
        "runtime_seconds": payload["runtime_seconds"],
    }, ensure_ascii=False))
    return 0 if mandatory else 2


if __name__ == "__main__":
    raise SystemExit(main())
