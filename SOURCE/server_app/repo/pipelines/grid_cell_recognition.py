#!/usr/bin/env python3
"""Recognize frozen layout cells with the official AutoHDR recognizer only."""
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
    digest.update(path.read_bytes())
    return digest.hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ink-mask", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--autohdr-root", required=True)
    parser.add_argument("--min-ink-pixels", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    image_path = Path(args.image).resolve()
    ink_path = Path(args.ink_mask).resolve()
    layout_path = Path(args.layout).resolve()
    output_path = Path(args.output).resolve()
    root = Path(args.autohdr_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from utils.recognition import batch_char_recog_exe
    from utils.reg_wrapper import reg_model
    from utils_pipeline import invert_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inverted = np.asarray(invert_image(Image.open(image_path).convert("RGB")).convert("L"))
    ink = np.asarray(Image.open(ink_path).convert("L")) > 0
    layout = json.loads(layout_path.read_text(encoding="utf-8"))

    metadata = []
    crops = []
    for column in layout["columns"]:
        for cell in column["cells"]:
            x1, y1, x2, y2 = [int(value) for value in cell["bbox"]]
            ink_pixels = int(ink[y1:y2, x1:x2].sum())
            record = {
                "cell_id": cell["cell_id"],
                "column_id": cell["column_id"],
                "reading_order": cell["reading_order"],
                "bbox": cell["bbox"],
                "cell_state": cell["cell_state"],
                "requires_text_completion": cell["requires_text_completion"],
                "damage_ratio": cell["damage_ratio"],
                "ink_pixels": ink_pixels,
            }
            if ink_pixels >= args.min_ink_pixels:
                crop = inverted[y1:y2, x1:x2]
                if crop.size:
                    record["recognized"] = True
                    crops.append(crop)
                else:
                    record["recognized"] = False
            else:
                record["recognized"] = False
            metadata.append(record)

    chars, probabilities = [], []
    if crops:
        dictionary = (root / "ckpt/dic_31524.txt").read_text(encoding="utf-8").splitlines()
        recognizer = reg_model(str(root / "dist/reg_model"))
        recognizer.start()
        chars, probabilities = batch_char_recog_exe(
            recognizer, device, dictionary, crops, bs=36
        )
        recognizer.cleanup()
        torch.cuda.empty_cache()

    cursor = 0
    for record in metadata:
        if not record["recognized"]:
            record.update({"alternatives": [], "probabilities": [], "top1": "", "top1_probability": 0.0})
            continue
        alternatives = plain(chars[cursor])
        probs = plain(probabilities[cursor])
        cursor += 1
        record.update({
            "alternatives": alternatives,
            "probabilities": probs,
            "top1": alternatives[0] if alternatives else "",
            "top1_probability": float(probs[0]) if probs else 0.0,
        })

    payload = {
        "status": "PASS",
        "mode": "OFFICIAL_AUTOHDR_RECOGNIZER_FROZEN_GRID_READ_ONLY",
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "layout_path": str(layout_path),
        "layout_sha256": sha256(layout_path),
        "ink_mask_path": str(ink_path),
        "ink_mask_sha256": sha256(ink_path),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "recognized_cell_count": len(crops),
        "cell_count": len(metadata),
        "cells": metadata,
        "runtime_seconds": time.time() - started,
        "caveat": "Cell OCR is alignment evidence only and cannot override CBETA continuity.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "device": payload["device"],
        "recognized_cell_count": payload["recognized_cell_count"],
        "cell_count": payload["cell_count"], "runtime_seconds": payload["runtime_seconds"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
