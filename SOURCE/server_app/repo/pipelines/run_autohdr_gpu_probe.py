#!/usr/bin/env python3
"""Run the official AutoHDR UNet pipeline on an externally controlled patch.

This is deliberately a diagnostic/adaptation stage: it does not claim to
recover text, does not render Unicode fonts, and never overwrites a page.
Its inputs and generated result are kept together for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--paper-image", required=True)
    p.add_argument("--source-image", required=True)
    p.add_argument("--missing-mask", required=True)
    p.add_argument("--text-mask", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--unet-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime_root).resolve()
    if not (runtime / "document" / "tools" / "build_HDR.py").is_file():
        raise FileNotFoundError("official AutoHDR runtime is incomplete")
    sys.path.insert(0, str(runtime))

    import torch
    import torchvision.transforms.functional as TF
    from types import SimpleNamespace
    from diffusers import UNet2DModel
    from document.tools.build_HDR import build_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("AutoHDR probe requires a real CUDA device")
    paper = Image.open(args.paper_image).convert("RGB")
    source = Image.open(args.source_image).convert("RGB")
    missing = np.asarray(Image.open(args.missing_mask).convert("L")) > 0
    text = np.asarray(Image.open(args.text_mask).convert("L")) > 0
    if paper.size != source.size or missing.shape != text.shape or paper.size != (missing.shape[1], missing.shape[0]):
        raise ValueError("paper/source/masks must have matching dimensions")
    candidate = missing & text
    if not candidate.any():
        candidate = missing
    ys, xs = np.where(candidate)
    center_x, center_y = int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2)
    left = min(max(center_x - 256, 0), max(paper.width - 512, 0))
    top = min(max(center_y - 256, 0), max(paper.height - 512, 0))
    right, bottom = min(left + 512, paper.width), min(top + 512, paper.height)
    # Make an exact 512 canvas even for a small original page.
    def canvas(im: Image.Image, fill: int) -> Image.Image:
        result = Image.new(im.mode, (512, 512), fill)
        result.paste(im.crop((left, top, right, bottom)), (0, 0))
        return result
    crop_mask = np.zeros((512, 512), dtype=np.uint8)
    crop_mask[: bottom - top, : right - left] = (candidate[top:bottom, left:right] * 255).astype(np.uint8)
    # Content condition comes exclusively from same-page source pixels; no fonts.
    paper_patch = canvas(paper, 255)
    source_patch = canvas(source.convert("L"), 255)
    content = Image.new("L", (512, 512), 255)
    content_arr = np.asarray(content).copy()
    src_arr = np.asarray(source_patch)
    content_arr[crop_mask > 0] = src_arr[crop_mask > 0]
    content = Image.fromarray(content_arr)
    char_mask = Image.fromarray(crop_mask, "L")
    white_degraded = np.asarray(paper_patch).copy()
    white_degraded[crop_mask > 0] = 255
    degraded = Image.fromarray(white_degraded)
    paper_patch.save(out / "paper_patch.png")
    source_patch.save(out / "same_page_source_patch.png")
    content.save(out / "same_page_glyph_condition.png")
    char_mask.save(out / "char_mask.png")
    degraded.save(out / "degraded_condition.png")

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    options = SimpleNamespace(
        pipeline="DPM-Solver++", ddpm_num_steps=1000, ddpm_beta_schedule="linear",
        prediction_type="sample", solver_order=2,
    )
    started = time.perf_counter()
    unet = UNet2DModel.from_pretrained(args.unet_path, local_files_only=True).to("cuda")
    pipeline = build_pipeline(options, unet).to("cuda")
    norm = lambda im: TF.normalize(TF.to_tensor(im), [0.5], [0.5]).unsqueeze(0).to("cuda")
    result = pipeline(
        degraded_image=norm(degraded), char_mask_image=norm(char_mask), content_image=norm(content),
        image_channel=3, classifier_free=True, content_mask_guidance_scale=1.5,
        degraded_guidance_scale=1.2, generator=torch.Generator(device="cuda").manual_seed(args.seed),
        batch_size=1, num_inference_steps=args.steps, output_type="pil",
    ).images[0].convert("RGB")
    result.save(out / "autohdr_generated_patch.png")
    record = {
        "status": "PASS", "case_id": args.case_id, "official_runtime": str(runtime),
        "unet_path": str(Path(args.unet_path).resolve()), "device": torch.cuda.get_device_name(0),
        "seed": args.seed, "steps": args.steps, "patch_bbox": [left, top, right, bottom],
        "condition_pixels": int((crop_mask > 0).sum()), "runtime_seconds": time.perf_counter() - started,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "paper_sha256": digest(Path(args.paper_image)), "generated_sha256": digest(out / "autohdr_generated_patch.png"),
        "font_policy": "NO_UNICODE_FONT; same-page source pixels only",
    }
    (out / "autohdr_gpu_probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
