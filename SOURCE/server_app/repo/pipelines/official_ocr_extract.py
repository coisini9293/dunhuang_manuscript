#!/usr/bin/env python3
"""Read-only AutoHDR OCR extraction for closed-loop evidence.

This deliberately stops after the official detector/recognizer and never invokes
the restoration model.  It records character boxes, alternatives, confidence,
reading order, CUDA provenance, and model paths as JSON.
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
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, tuple):
        return [plain(x) for x in value]
    if isinstance(value, list):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--autohdr-root", required=True)
    ap.add_argument("--img-size", type=int, default=2048)
    ap.add_argument("--conf-thres", type=float, default=0.45)
    ap.add_argument("--iou-thres", type=float, default=0.2)
    args = ap.parse_args()

    started = time.time()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    root = Path(args.autohdr_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    from utils.datasets import create_dataloader
    from utils.general import check_img_size, colorstr, non_max_suppression, scale_coords
    from utils.reader import get_sort
    from utils.recognition import batch_char_recog_exe
    from utils.det_wrapper import det_model
    from utils.reg_wrapper import reg_model
    from utils_pipeline import invert_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch = output_path.parent / (output_path.stem + "_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    inverted_path = scratch / "inverted.png"
    image = Image.open(image_path).convert("RGB")
    invert_image(image).save(inverted_path)

    detector_path = root / "dist/det_model/det_model"
    recognizer_path = root / "dist/reg_model"
    dictionary_path = root / "ckpt/dic_31524.txt"

    detector = det_model(str(detector_path))
    stride = detector(device, mode=1)
    img_size = check_img_size(args.img_size, s=stride)
    loader = create_dataloader(
        str(inverted_path), img_size, 1, stride, argparse.Namespace(), pad=0.5,
        rect=True, prefix=colorstr("ocr: "), infer=True,
    )[0]
    boxes = []
    for batch, _targets, paths, shapes in loader:
        batch = batch.to(device, non_blocking=True)
        batch = batch.half() if device.type == "cuda" else batch.float()
        batch /= 255.0
        with torch.no_grad():
            predictions = detector(batch, mode=2)
            predictions = non_max_suppression(
                predictions, conf_thres=args.conf_thres,
                iou_thres=args.iou_thres, labels=[], multi_label=True,
            )
        for index, pred in enumerate(predictions):
            scale_coords(batch[index].shape[1:], pred[:, :4], shapes[index][0], shapes[index][1])
            boxes.extend([
                [round(float(row[0])), round(float(row[1])), round(float(row[2])), round(float(row[3]))]
                for row in pred[:, :5].detach().cpu().numpy().tolist()
            ])
    detector.cleanup()
    torch.cuda.empty_cache()

    gray = cv2.imread(str(inverted_path), 0)
    valid_boxes, crops = [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size:
            valid_boxes.append(box)
            crops.append(crop)

    dictionary = dictionary_path.read_text(encoding="utf-8").splitlines()
    recognizer = reg_model(str(recognizer_path))
    recognizer.start()
    alternatives, probabilities = batch_char_recog_exe(
        recognizer, device, dictionary, crops, bs=36,
    )
    recognizer.cleanup()
    torch.cuda.empty_cache()

    by_box = {}
    for box, chars, probs in zip(valid_boxes, alternatives, probabilities):
        by_box[str(box)] = {
            "bbox": box,
            "alternatives": plain(chars),
            "probabilities": plain(probs),
            "top1": plain(chars)[0] if plain(chars) else "",
            "top1_probability": float(plain(probs)[0]) if plain(probs) else 0.0,
        }
    height, width = gray.shape[:2]
    ordered_boxes = get_sort(valid_boxes, height, width) if valid_boxes else []
    ordered = [by_box[str(box)] for box in ordered_boxes if str(box) in by_box]
    text = "".join(item["top1"] for item in ordered)

    payload = {
        "status": "PASS" if ordered else "FAIL_NO_CHARACTERS",
        "mode": "OFFICIAL_AUTOHDR_DETECTOR_AND_RECOGNIZER_READ_ONLY",
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "image_dimensions": [width, height],
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "detector_path": str(detector_path),
        "recognizer_path": str(recognizer_path),
        "dictionary_path": str(dictionary_path),
        "settings": {
            "img_size": args.img_size,
            "conf_threshold": args.conf_thres,
            "iou_threshold": args.iou_thres,
        },
        "character_count": len(ordered),
        "ocr_text": text,
        "characters": ordered,
        "runtime_seconds": time.time() - started,
        "caveat": "OCR is machine evidence and is not treated as a ground-truth transcription.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "device", "character_count", "ocr_text", "runtime_seconds")}, ensure_ascii=False))
    return 0 if ordered else 2


if __name__ == "__main__":
    raise SystemExit(main())
