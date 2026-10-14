#!/usr/bin/env python3
"""Combine sealed AutoHDR text restoration with SD2 paper completion."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline


BASE = Path("/path/to/dunhuang-autohdr")
EXP = BASE / "experiments" / "paper_inpainting_bd2816_20260730"
INPUT = EXP / "input"
OUTPUT = EXP / "output"
REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
MODEL = BASE / "models" / f"sd2-inpainting-community-{REVISION[:12]}"
OFFICIAL = BASE / "src" / "AutoHDR"
OFFICIAL_COMMIT = "d92b223b3d9274b03ca6f1e4c62b09253f378c98"
PROMPT = (
    "flat museum scan of ancient aged tan parchment paper, seamless warm ochre fibrous "
    "paper texture, subtle natural stains, historically weathered surface, no text, "
    "no writing, no symbols, no objects"
)
NEGATIVE = (
    "letters, calligraphy, writing, printed text, symbols, black opening, empty hole, "
    "hard seam, frame, shadow, modern object, bright color, illustration"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def git_state() -> tuple[str, bool]:
    import subprocess

    commit = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--porcelain"], text=True)
    return commit, not bool(status.strip())


def owned_tiles(width: int, height: int, mask: np.ndarray):
    x_edges = [0, width // 2, width]
    y_edges = [0, height // 3, 2 * height // 3, height]
    for row in range(3):
        for col in range(2):
            x1, x2 = x_edges[col], x_edges[col + 1]
            y1, y2 = y_edges[row], y_edges[row + 1]
            ownership = np.zeros_like(mask, dtype=np.uint8)
            ownership[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            if not np.any(ownership):
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            left = min(max(cx - 256, 0), width - 512)
            top = min(max(cy - 256, 0), height - 512)
            yield {
                "row": row,
                "column": col,
                "ownership_bbox": [x1, y1, x2, y2],
                "roi_bbox": [left, top, left + 512, top + 512],
                "global_mask": ownership,
            }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phases = []

    def phase(name: str, detail: str) -> None:
        item = {"phase": name, "detail": detail, "updated_at": now()}
        phases.append(item)
        write_json(OUTPUT / "phase.json", item)
        write_json(OUTPUT / "phase_timeline.json", {"events": phases})

    started = time.perf_counter()
    phase("PREPARING_INPUT", "Verify sealed text stage, paper mask, protected union and Official source")
    plan = json.loads((INPUT / "full_restoration_plan.json").read_text(encoding="utf-8"))
    expected = {
        "original_input.png": plan["original_png_sha256"],
        "text_stage_autohdr.png": plan["text_stage_sha256"],
        "paper_completion_mask.png": plan["paper_mask_sha256"],
        "protected_union_mask.png": plan["protected_union_mask_sha256"],
    }
    for name, expected_hash in expected.items():
        actual = sha256(INPUT / name)
        if actual != expected_hash:
            raise RuntimeError(f"INPUT_HASH_MISMATCH:{name}:{actual}:{expected_hash}")
    official_commit, official_clean = git_state()
    if official_commit != OFFICIAL_COMMIT or not official_clean:
        raise RuntimeError(f"OFFICIAL_SOURCE_INVALID:{official_commit}:clean={official_clean}")

    original = Image.open(INPUT / "original_input.png").convert("RGB")
    text_stage = Image.open(INPUT / "text_stage_autohdr.png").convert("RGB")
    paper_mask_image = Image.open(INPUT / "paper_completion_mask.png").convert("L")
    protected = np.asarray(Image.open(INPUT / "protected_union_mask.png").convert("L")) > 0
    paper_mask = (np.asarray(paper_mask_image) > 0).astype(np.uint8)
    if original.size != text_stage.size or original.size != paper_mask_image.size:
        raise RuntimeError("INPUT_DIMENSION_MISMATCH")

    tiles = list(owned_tiles(original.width, original.height, paper_mask))
    write_json(
        OUTPUT / "tile_plan.json",
        {
            "tile_size": 512,
            "tile_count": len(tiles),
            "tiles": [
                {
                    "tile": index + 1,
                    "ownership_bbox": item["ownership_bbox"],
                    "roi_bbox": item["roi_bbox"],
                    "mask_pixels": int(item["global_mask"].sum()),
                }
                for index, item in enumerate(tiles)
            ],
        },
    )

    phase("LOADING_MODEL", f"Load fixed SD2 Inpainting revision for {len(tiles)} paper tiles")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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

    phase("RUNNING_INFERENCE", f"Run real SD2 Inpainting over {len(tiles)} disjoint paper-mask tiles")
    current = text_stage.copy()
    tile_logs = []
    for index, item in enumerate(tiles, 1):
        left, top, right, bottom = item["roi_bbox"]
        tile_dir = OUTPUT / "tiles" / f"tile_{index:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        roi = current.crop((left, top, right, bottom))
        local_mask_np = item["global_mask"][top:bottom, left:right] * 255
        local_mask = Image.fromarray(local_mask_np.astype(np.uint8), "L")
        roi.save(tile_dir / "input.png")
        local_mask.save(tile_dir / "mask.png")
        tile_started = time.perf_counter()
        generated = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=roi,
            mask_image=local_mask,
            num_inference_steps=35,
            guidance_scale=6.5,
            generator=torch.Generator(device="cuda").manual_seed(281600 + index),
        ).images[0].convert("RGB")
        generated.save(tile_dir / "generated.png")

        # Feather inward only. Alpha is exactly zero outside the authorized mask.
        blurred = np.asarray(local_mask.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) / 255.0
        exact = np.asarray(local_mask) > 0
        alpha = np.where(exact, np.maximum(blurred, 0.18), 0.0)[..., None]
        roi_np = np.asarray(roi).astype(np.float32)
        generated_np = np.asarray(generated).astype(np.float32)
        composed_np = np.clip(generated_np * alpha + roi_np * (1 - alpha), 0, 255).astype(np.uint8)
        composed = Image.fromarray(composed_np)
        composed.save(tile_dir / "composited.png")
        current.paste(composed, (left, top))
        tile_logs.append(
            {
                "tile": index,
                "roi_bbox": item["roi_bbox"],
                "ownership_bbox": item["ownership_bbox"],
                "mask_pixels": int(item["global_mask"].sum()),
                "runtime_seconds": time.perf_counter() - tile_started,
                "generated_sha256": sha256(tile_dir / "generated.png"),
            }
        )
        write_json(
            OUTPUT / "live_progress.json",
            {
                "phase": "RUNNING_INFERENCE",
                "tiles_complete": index,
                "tiles_total": len(tiles),
                "updated_at": now(),
            },
        )

    phase("COMPOSITING", "Finalize text-stage plus paper-mask result with immutable outside-union pixels")
    final_path = OUTPUT / "final_restored.png"
    current.save(final_path, "PNG")
    original.save(OUTPUT / "original_input.png", "PNG")
    paper_mask_image.save(OUTPUT / "paper_completion_mask.png", "PNG")

    phase("PIXEL_AUDIT", "Recompute protected-union and paper-only pixel invariants")
    arr_original = np.asarray(original)
    arr_text = np.asarray(text_stage)
    arr_final = np.asarray(current)
    changed_from_original = np.any(arr_final != arr_original, axis=2)
    paper_changed = np.any(arr_final != arr_text, axis=2)
    gray_before = np.asarray(text_stage.convert("L"))
    gray_after = np.asarray(current.convert("L"))
    paper_bool = paper_mask > 0
    black_before = int(np.count_nonzero((gray_before < 18) & paper_bool))
    black_after = int(np.count_nonzero((gray_after < 18) & paper_bool))
    audit = {
        "status": "VISUAL_FULL_RESTORATION_TECHNICAL_PASS"
        if int(np.count_nonzero(changed_from_original & ~protected)) == 0
        and int(np.count_nonzero(paper_changed & ~paper_bool)) == 0
        and black_after < black_before
        else "VISUAL_FULL_RESTORATION_TECHNICAL_FAIL",
        "model_repo": "sd2-community/stable-diffusion-2-inpainting",
        "model_revision": REVISION,
        "official_commit": official_commit,
        "official_worktree_clean": official_clean,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_runtime_seconds": time.perf_counter() - loaded,
        "total_runtime_seconds": time.perf_counter() - started,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "tile_count": len(tiles),
        "steps_per_tile": 35,
        "input_sha256": sha256(INPUT / "original_input.png"),
        "text_stage_sha256": sha256(INPUT / "text_stage_autohdr.png"),
        "paper_mask_sha256": sha256(INPUT / "paper_completion_mask.png"),
        "output_sha256": sha256(final_path),
        "paper_mask_pixels": int(np.count_nonzero(paper_bool)),
        "paper_black_pixels_before": black_before,
        "paper_black_pixels_after": black_after,
        "paper_black_reduction_ratio": 1 - black_after / black_before if black_before else 0,
        "paper_changed_pixels": int(np.count_nonzero(paper_changed & paper_bool)),
        "changed_outside_paper_mask_from_text_stage": int(np.count_nonzero(paper_changed & ~paper_bool)),
        "changed_outside_protected_union_from_original": int(
            np.count_nonzero(changed_from_original & ~protected)
        ),
        "tile_logs": tile_logs,
        "textual_correctness": "NOT_EVALUATED",
        "scholarly_review": "PENDING",
    }
    write_json(OUTPUT / "run_audit.json", audit)
    phase("COMPLETE", "Full text plus paper visual reconstruction completed")
    print(json.dumps(audit, indent=2))
    if audit["status"].endswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
