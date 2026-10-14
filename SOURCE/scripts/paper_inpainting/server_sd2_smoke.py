#!/usr/bin/env python3
"""Real GPU smoke test on the largest enclosed hole in BD2816."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline


BASE = Path("/path/to/dunhuang-autohdr")
REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
MODEL = BASE / "models" / f"sd2-inpainting-community-{REVISION[:12]}"
INPUT = BASE / "experiments" / "direct_single_bd2816_20260730" / "output" / "full_hypothetical_reconstruction.png"
OUTPUT = BASE / "experiments" / "paper_inpainting_bd2816_20260730" / "smoke"
PROMPT = (
    "a seamless patch of ancient aged tan parchment paper, warm ochre fibrous texture, "
    "flat museum conservation scan, subtle natural stains, no text, no writing, no symbols"
)
NEGATIVE = (
    "letters, calligraphy, writing, printed text, symbols, black hole, torn opening, "
    "modern object, bright colors, frame, shadow"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def largest_enclosed_hole(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    black = (gray < 18).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    choices = []
    height, width = black.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and area >= 100:
            choices.append((int(area), label, (int(x), int(y), int(w), int(h))))
    if not choices:
        raise RuntimeError("NO_ENCLOSED_HOLE_FOUND")
    area, label, (x, y, w, h) = max(choices)
    component = (labels == label).astype(np.uint8) * 255
    component = cv2.dilate(component, np.ones((11, 11), np.uint8), iterations=1)
    cx, cy = x + w // 2, y + h // 2
    left = min(max(cx - 256, 0), width - 512)
    top = min(max(cy - 256, 0), height - 512)
    return component, (left, top, left + 512, top + 512), area


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(INPUT).convert("RGB")
    image_np = np.asarray(image)
    full_mask, bbox, original_hole_pixels = largest_enclosed_hole(image_np)
    roi = image.crop(bbox)
    mask = Image.fromarray(full_mask).crop(bbox).convert("L")
    roi.save(OUTPUT / "input_roi.png")
    mask.save(OUTPUT / "mask.png")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        local_files_only=True,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")
    loaded = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(2816)
    generated = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=roi,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=generator,
    ).images[0].convert("RGB")
    inferred = time.perf_counter()
    generated.save(OUTPUT / "raw_generated.png")

    feather = mask.filter(ImageFilter.GaussianBlur(4))
    composited = Image.composite(generated, roi, feather)
    composited.save(OUTPUT / "composited_roi.png")
    result_gray = cv2.cvtColor(np.asarray(composited), cv2.COLOR_RGB2GRAY)
    mask_np = np.asarray(mask) > 127
    black_after = int(np.count_nonzero((result_gray < 18) & mask_np))
    mask_pixels = int(np.count_nonzero(mask_np))
    status = {
        "status": "SD2_INPAINTING_SMOKE_PASS"
        if mask_pixels and black_after / mask_pixels < 0.2
        else "SD2_INPAINTING_SMOKE_FAIL",
        "model": str(MODEL),
        "model_revision": REVISION,
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "roi_bbox": bbox,
        "original_enclosed_hole_pixels": original_hole_pixels,
        "mask_pixels": mask_pixels,
        "black_pixels_after": black_after,
        "black_ratio_after": black_after / mask_pixels if mask_pixels else 1,
        "model_load_seconds": loaded - started,
        "inference_seconds": inferred - loaded,
        "total_seconds": inferred - started,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "steps": 30,
        "guidance_scale": 7.0,
        "seed": 2816,
        "output_sha256": sha256(OUTPUT / "composited_roi.png"),
    }
    (OUTPUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if status["status"].endswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
