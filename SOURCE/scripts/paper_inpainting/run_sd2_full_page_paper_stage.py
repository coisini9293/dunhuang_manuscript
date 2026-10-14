#!/usr/bin/env python3
"""Formal, local-only SD2 paper-stage runner for the Full-Page LOOP.

The runner never reconstructs masks and never changes a pixel outside the
provided missing mask.  It creates a new output directory for every run and
does not write to historical evidence directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROMPT = (
    "flat museum scan of ancient aged tan parchment paper, seamless warm ochre "
    "fibrous paper texture, subtle natural stains, historically weathered surface, "
    "no text, no writing, no symbols, no objects"
)
NEGATIVE_PROMPT = (
    "letters, calligraphy, writing, printed text, symbols, black opening, empty hole, "
    "hard seam, frame, shadow, modern object, bright color, illustration"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_rectangle(value: str | None) -> list[int] | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_file():
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
        value = json.dumps(loaded.get("target_rectangle", loaded))
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 4 or not all(isinstance(x, int) for x in parsed):
        raise ValueError("target rectangle must be a JSON [x1,y1,x2,y2] list or a JSON contract file")
    return parsed


def owned_tiles(width: int, height: int, mask: np.ndarray, tile_size: int):
    """Yield disjoint mask ownership tiles with a fixed-size context ROI."""
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            owned = np.zeros_like(mask, dtype=np.uint8)
            owned[top:bottom, left:right] = mask[top:bottom, left:right]
            if not np.any(owned):
                continue
            center_x, center_y = (left + right) // 2, (top + bottom) // 2
            roi_left = min(max(center_x - tile_size // 2, 0), max(width - tile_size, 0))
            roi_top = min(max(center_y - tile_size // 2, 0), max(height - tile_size, 0))
            roi_right, roi_bottom = min(roi_left + tile_size, width), min(roi_top + tile_size, height)
            yield {
                "ownership_bbox": [left, top, right, bottom],
                "roi_bbox": [roi_left, roi_top, roi_right, roi_bottom],
                "mask": owned,
            }


def validate(args: argparse.Namespace) -> tuple[Image.Image, np.ndarray, np.ndarray, list[int] | None]:
    input_path, missing_path, protected_path, model_path = map(
        Path, (args.input_image, args.missing_mask, args.protected_mask, args.model_path)
    )
    for path, label in ((input_path, "input image"), (missing_path, "missing mask"), (protected_path, "protected mask")):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"not a local SD2 model directory: {model_path}")
    image = Image.open(input_path).convert("RGB")
    missing = np.asarray(Image.open(missing_path).convert("L")) > 0
    protected = np.asarray(Image.open(protected_path).convert("L")) > 0
    if image.size != (missing.shape[1], missing.shape[0]) or missing.shape != protected.shape:
        raise ValueError("input image, missing mask and protected mask dimensions must match")
    if np.any(missing & protected):
        raise ValueError("protected mask overlaps missing mask")
    if not np.any(missing):
        raise ValueError("missing mask is empty")
    rect = parse_rectangle(args.target_rectangle)
    if rect:
        x1, y1, x2, y2 = rect
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("target rectangle is outside image bounds")
        rect_mask = np.zeros_like(missing, dtype=bool)
        rect_mask[y1:y2, x1:x2] = True
        if np.any(missing & ~rect_mask):
            raise ValueError("missing mask extends outside supplied target rectangle")
    return image, missing, protected, rect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--missing-mask", required=True)
    parser.add_argument("--protected-mask", required=True)
    parser.add_argument("--target-rectangle", help="JSON rectangle or geometry-contract JSON")
    parser.add_argument("--geometry-contract", help="alias for --target-rectangle")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.geometry_contract and not args.target_rectangle:
        args.target_rectangle = args.geometry_contract

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image, missing, protected, rect = validate(args)
    config = {
        "case_id": args.case_id,
        "input_image": str(Path(args.input_image).resolve()),
        "missing_mask": str(Path(args.missing_mask).resolve()),
        "protected_mask": str(Path(args.protected_mask).resolve()),
        "target_rectangle": rect,
        "model_path": str(Path(args.model_path).resolve()),
        "seed": args.seed,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "tile_size": args.tile_size,
        "local_files_only": True,
        "dry_run": args.dry_run,
        "missing_pixels": int(missing.sum()),
        "protected_overlap_pixels": int((missing & protected).sum()),
    }
    write_json(output / "run_config.json", config)
    image.save(output / "input_original.png")
    Image.fromarray((missing * 255).astype(np.uint8)).save(output / "final_missing_mask.png")
    Image.fromarray((protected * 255).astype(np.uint8)).save(output / "protected_mask.png")
    if args.dry_run:
        write_json(output / "dry_run.json", {"status": "PASS", "cuda_initialized": False, **config})
        return 0

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real paper-stage execution requires --device cuda and an attached CUDA device")
    if args.tile_size != 512:
        raise ValueError("the audited SD2 runner currently requires --tile-size 512")
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")
    current = image.copy()
    logs = []
    for index, tile in enumerate(owned_tiles(image.width, image.height, missing.astype(np.uint8), args.tile_size), 1):
        left, top, right, bottom = tile["roi_bbox"]
        local_mask = (tile["mask"][top:bottom, left:right] * 255).astype(np.uint8)
        mask_image = Image.fromarray(local_mask, "L")
        roi = current.crop((left, top, right, bottom))
        tile_dir = output / "tiles" / f"tile_{index:03d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        roi.save(tile_dir / "input.png")
        mask_image.save(tile_dir / "mask.png")
        generated = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=roi,
            mask_image=mask_image,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + index),
        ).images[0].convert("RGB")
        generated.save(tile_dir / "generated.png")
        current_np = np.asarray(roi).copy()
        generated_np = np.asarray(generated)
        take = local_mask > 0
        current_np[take] = generated_np[take]
        composed = Image.fromarray(current_np)
        composed.save(tile_dir / "composited.png")
        current.paste(composed, (left, top))
        logs.append({"tile": index, "roi_bbox": tile["roi_bbox"], "ownership_bbox": tile["ownership_bbox"], "mask_pixels": int(take.sum()), "generated_sha256": sha256(tile_dir / "generated.png")})

    final_path = output / "paper_stage.png"
    current.save(final_path)
    before = np.asarray(image)
    after = np.asarray(current)
    changed = np.any(before != after, axis=2)
    audit = {
        "status": "PASS" if current.size == image.size and int((changed & protected).sum()) == 0 and int((changed & ~missing).sum()) == 0 else "FAIL",
        "output_dimensions": list(current.size),
        "input_dimensions": list(image.size),
        "changed_pixels": int(changed.sum()),
        "protected_changed_pixels": int((changed & protected).sum()),
        "outside_missing_changed_pixels": int((changed & ~missing).sum()),
        "missing_pixels": int(missing.sum()),
        "tile_count": len(logs),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_seconds": time.perf_counter() - started,
        "input_sha256": sha256(Path(args.input_image)),
        "output_sha256": sha256(final_path),
        "tile_logs": logs,
    }
    write_json(output / "pixel_audit.json", audit)
    del pipe
    torch.cuda.empty_cache()
    if audit["status"] != "PASS":
        raise SystemExit("pixel safety audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
