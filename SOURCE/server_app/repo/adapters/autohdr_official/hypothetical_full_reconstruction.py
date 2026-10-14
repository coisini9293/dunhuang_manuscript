from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from adapters.autohdr_official import OFFICIAL_COMMIT
from adapters.autohdr_official.common import (
    AdapterContractError,
    canonical_sha256,
    file_sha256,
    load_json,
    verify_official_tree,
    write_json,
)


PATCH_SIZE = 448


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_phase(output_dir: Path, timeline: list[dict], phase: str, detail: str) -> None:
    event = {"phase": phase, "detail": detail, "updated_at": now()}
    timeline.append(event)
    write_json(output_dir / "phase.json", event)
    write_json(output_dir / "phase_timeline.json", {"events": timeline})


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


def patch_origin(bbox: list[int], width: int, height: int) -> tuple[int, int]:
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    return (
        max(0, min(center_x - PATCH_SIZE // 2, width - PATCH_SIZE)),
        max(0, min(center_y - PATCH_SIZE // 2, height - PATCH_SIZE)),
    )


def build_groups(regions: list[dict], width: int, height: int) -> list[dict]:
    remaining = sorted(regions, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    groups: list[dict] = []
    while remaining:
        seed = remaining[0]
        x0, y0 = patch_origin(seed["bbox"], width, height)
        patch_box = [x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE]
        members = []
        deferred = []
        for region in remaining:
            bx = region["bbox"]
            if (
                bx[0] >= patch_box[0]
                and bx[1] >= patch_box[1]
                and bx[2] <= patch_box[2]
                and bx[3] <= patch_box[3]
            ):
                members.append(region)
            else:
                deferred.append(region)
        groups.append({"patch_bbox": patch_box, "regions": members})
        remaining = deferred
    return groups


def fallback_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/path/to/dunhuang-autohdr/RUNTIME/AutoHDR_official_run/ckpt/simhei.ttf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def write_review_artifacts(
    original: Image.Image,
    final: Image.Image,
    mask: Image.Image,
    content: Image.Image,
    manifest: dict,
    output_dir: Path,
) -> None:
    original.save(output_dir / "original_panorama.png", "PNG")
    final.save(output_dir / "full_hypothetical_reconstruction.png", "PNG")
    mask.save(output_dir / "full_union_mask.png", "PNG")
    content.save(output_dir / "full_content_image.png", "PNG")

    original_array = np.asarray(original).astype(np.int16)
    final_array = np.asarray(final).astype(np.int16)
    difference = np.abs(final_array - original_array).astype(np.uint8)
    Image.fromarray(difference).save(output_dir / "difference.png", "PNG")
    amplified = np.clip(difference.astype(np.int16) * 4, 0, 255).astype(np.uint8)
    Image.fromarray(amplified).save(output_dir / "difference_amplified_x4.png", "PNG")

    wipe = original.copy()
    split = original.width // 2
    wipe.paste(final.crop((split, 0, original.width, original.height)), (split, 0))
    draw = ImageDraw.Draw(wipe)
    draw.line((split, 0, split, original.height), fill=(198, 135, 35), width=6)
    wipe.save(output_dir / "before_after_wipe.png", "PNG")

    overlay = original.convert("RGBA")
    numbered = original.convert("RGBA")
    overlay_layer = Image.new("RGBA", original.size, (0, 0, 0, 0))
    number_layer = Image.new("RGBA", original.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_layer)
    number_draw = ImageDraw.Draw(number_layer)
    label_font = fallback_font(19)
    colors = {
        "high": (44, 133, 92, 105),
        "medium": (218, 152, 42, 115),
        "low": (190, 63, 57, 120),
        "texture": (58, 111, 168, 110),
        "skip": (105, 105, 105, 75),
    }
    for item in manifest["regions"]:
        bbox = tuple(item["bbox"])
        if not item["enters_restore"]:
            level = "skip"
        elif item["region_type"] in {
            "paper_background_damage",
            "stain_or_texture_damage",
        }:
            level = "texture"
        else:
            value = float(item["confidence"])
            level = "low" if value < 0.70 else "medium" if value < 0.90 else "high"
        color = colors[level]
        overlay_draw.rectangle(bbox, fill=color, outline=color[:3] + (235,), width=2)
        number_draw.rectangle(bbox, outline=color[:3] + (255,), width=3)
        number_draw.text(
            (bbox[0] + 2, bbox[1] + 1),
            str(item["reading_order"]),
            font=label_font,
            fill=(45, 23, 10, 255),
            stroke_width=2,
            stroke_fill=(255, 245, 217, 255),
        )
    Image.alpha_composite(overlay, overlay_layer).convert("RGB").save(
        output_dir / "reconstruction_confidence_overlay.png", "PNG"
    )
    Image.alpha_composite(numbered, number_layer).convert("RGB").save(
        output_dir / "region_number_map.png", "PNG"
    )
    write_json(
        output_dir / "low_confidence_regions.json",
        {
            "regions": [
                item
                for item in manifest["regions"]
                if item["enters_restore"] and item["low_confidence"]
            ]
        },
    )


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict] = []
    write_phase(output_dir, timeline, "QUEUED", "Experimental job accepted")
    write_phase(
        output_dir,
        timeline,
        "PREPARING_INPUT",
        "Validating immutable input, plan, manifest and Official Analysis bundle",
    )
    plan = load_json(args.plan)
    manifest = load_json(args.manifest)
    bundle = load_json(args.bundle)
    if plan.get("state") != "hypothetical_visual_reconstruction":
        raise AdapterContractError("EXPERIMENT_STATE_INVALID")
    plan_core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if canonical_sha256(plan_core) != plan["plan_sha256"]:
        raise AdapterContractError("FULL_RECONSTRUCTION_PLAN_HASH_MISMATCH")
    input_hash = file_sha256(args.input)
    if input_hash != plan["input_sha256"] or input_hash != bundle["image_sha256"]:
        raise AdapterContractError("FULL_RECONSTRUCTION_INPUT_HASH_MISMATCH")
    if plan["source_candidate_bundle_sha256"] != bundle["candidate_bundle_sha256"]:
        raise AdapterContractError("FULL_RECONSTRUCTION_BUNDLE_HASH_MISMATCH")
    if plan["region_manifest_sha256"] != file_sha256(args.manifest):
        raise AdapterContractError("FULL_RECONSTRUCTION_MANIFEST_HASH_MISMATCH")
    official_tree = verify_official_tree(args.official_source, OFFICIAL_COMMIT)

    original = Image.open(args.input).convert("RGB")
    if original.width < PATCH_SIZE or original.height < PATCH_SIZE:
        raise AdapterContractError("IMAGE_SMALLER_THAN_OFFICIAL_PATCH")
    region_by_id = {item["region_id"]: item for item in bundle["regions"]}
    selected = []
    for item in manifest["regions"]:
        if not item["enters_restore"]:
            continue
        region = region_by_id.get(item["region_id"])
        if not region or list(region["bbox"]) != list(item["bbox"]):
            raise AdapterContractError(f"REGION_BUNDLE_MISMATCH:{item['region_id']}")
        selected.append(item)
    if {item["region_id"] for item in selected} != set(plan["approved_region_ids"]):
        raise AdapterContractError("PLAN_REGION_SET_MISMATCH")

    planned_rectangle_mask = Image.new("L", original.size, 0)
    content = Image.new("L", original.size, 255)
    ownership = Image.new("I", original.size, 0)
    mask_draw = ImageDraw.Draw(planned_rectangle_mask)
    for item in sorted(selected, key=lambda row: float(row["confidence"])):
        bbox = tuple(int(value) for value in item["bbox"])
        mask_draw.rectangle(bbox, fill=255)
        ownership.paste(int(item["reading_order"]), bbox)
    groups = build_groups(selected, original.width, original.height)
    write_json(
        output_dir / "tile_plan.json",
        {
            "patch_size": PATCH_SIZE,
            "tile_count": len(groups),
            "tiles": [
                {
                    "tile": index + 1,
                    "patch_bbox": group["patch_bbox"],
                    "region_ids": [item["region_id"] for item in group["regions"]],
                }
                for index, group in enumerate(groups)
            ],
        },
    )

    write_phase(
        output_dir,
        timeline,
        "LOADING_MODEL",
        f"Loading fixed Official AutoHDR models for {len(groups)} real GPU tiles",
    )
    with official_import_context(args.official_source, args.official_runtime):
        import torch
        import torchvision.transforms.functional as TF
        from diffusers import UNet2DModel
        from document.tools.build_HDR import build_pipeline
        from utils_pipeline import invert_image, render_char_with_font_T, restore_image

        if not torch.cuda.is_available():
            raise AdapterContractError("GPU_REQUIRED_FOR_FULL_RECONSTRUCTION")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        # Build the exact global content image with the upstream renderer.
        for item in selected:
            bbox = [int(value) for value in item["bbox"]]
            glyph = render_char_with_font_T(item["final_content"]).convert("L")
            if glyph.getbbox() is None:
                raise AdapterContractError(
                    f"OFFICIAL_FONT_CANNOT_RENDER:{item['region_id']}:{item['final_content']}"
                )
            box_width = bbox[2] - bbox[0]
            box_height = bbox[3] - bbox[1]
            glyph_width = max(1, round(box_width * 0.80))
            glyph_height = max(1, round(box_height * 0.80))
            glyph = glyph.resize(
                (glyph_width, glyph_height),
                Image.Resampling.LANCZOS,
            )
            content.paste(
                glyph,
                (
                    bbox[0] + (box_width - glyph_width) // 2,
                    bbox[1] + (box_height - glyph_height) // 2,
                ),
            )

        # The upstream model is conditioned with rectangular damage masks, but
        # replacing the entire rectangle produces visible paper-color blocks.
        # Build the final compositor mask from the exact rendered strokes and
        # keep it strictly inside the recorded plan rectangles.
        stroke = ImageOps.invert(content)
        stroke = stroke.filter(ImageFilter.MaxFilter(size=5))
        stroke_array = np.asarray(stroke).astype(np.uint8)
        planned_array = np.asarray(planned_rectangle_mask) > 0
        effective_array = np.where(
            planned_array & (stroke_array > 8), 255, 0
        ).astype(np.uint8)
        global_mask = Image.fromarray(effective_array, mode="L")
        soft_stroke = Image.fromarray(
            np.where(planned_array, stroke_array, 0).astype(np.uint8), mode="L"
        ).filter(ImageFilter.GaussianBlur(radius=0.8))

        unet = UNet2DModel.from_pretrained("ckpt/unet").to(device)
        options = argparse.Namespace(
            pipeline="DPM-Solver++",
            ddpm_num_steps=1000,
            ddpm_beta_schedule="linear",
            prediction_type="sample",
            solver_order=2,
        )
        pipeline = build_pipeline(options, unet)
        generator = torch.Generator(device=pipeline.device).manual_seed(123)
        inverted = invert_image(original)
        generated_canvas = original.copy()
        tile_logs = []
        write_phase(
            output_dir,
            timeline,
            "RUNNING_INFERENCE",
            f"Running Official diffusion over {len(groups)} grouped tiles",
        )
        try:
            for index, group in enumerate(groups, start=1):
                x0, y0, x1, y1 = group["patch_bbox"]
                local_mask = Image.new("L", (PATCH_SIZE, PATCH_SIZE), 0)
                local_mask_draw = ImageDraw.Draw(local_mask)
                for item in group["regions"]:
                    bx = item["bbox"]
                    local_mask_draw.rectangle(
                        (bx[0] - x0, bx[1] - y0, bx[2] - x0, bx[3] - y0),
                        fill=255,
                    )
                local_content = content.crop((x0, y0, x1, y1))
                degraded = inverted.crop((x0, y0, x1, y1))
                degraded_array = np.asarray(degraded).copy()
                degraded_array[np.asarray(local_mask) > 0] = 255
                degraded = Image.fromarray(degraded_array)
                tile_dir = output_dir / "tiles" / f"tile_{index:02d}"
                tile_dir.mkdir(parents=True, exist_ok=True)
                degraded.save(tile_dir / "degraded.png", "PNG")
                local_content.save(tile_dir / "content.png", "PNG")
                local_mask.save(tile_dir / "mask.png", "PNG")

                def to_tensor(image: Image.Image):
                    return TF.normalize(
                        TF.to_tensor(
                            image.resize((512, 512), Image.Resampling.LANCZOS)
                        ),
                        [0.5],
                        [0.5],
                    ).unsqueeze(0).to(device)

                tile_start = time.perf_counter()
                with torch.no_grad():
                    generated = pipeline(
                        degraded_image=to_tensor(degraded),
                        char_mask_image=to_tensor(local_mask),
                        content_image=to_tensor(local_content),
                        image_channel=3,
                        classifier_free=True,
                        content_mask_guidance_scale=1.5,
                        degraded_guidance_scale=1.2,
                        generator=generator,
                        batch_size=1,
                        num_inference_steps=20,
                        output_type="pil",
                    ).images[0]
                generated = restore_image(
                    generated.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
                )
                generated.save(tile_dir / "generated.png", "PNG")
                # Official output is returned in the restored (white-ink)
                # polarity for this corpus. Convert it back to dark ink, match
                # each generated glyph to the local manuscript ink colour, and
                # write it with confidence-aware opacity. The GPU luminance
                # texture remains in the accepted strokes; the paper background
                # is always inherited from the immutable input.
                dark_ink = ImageOps.invert(generated)
                styled = original.crop((x0, y0, x1, y1))
                styled_array = np.asarray(styled).copy()
                dark_array = np.asarray(dark_ink)
                local_alpha = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                for item in group["regions"]:
                    bx = [int(value) for value in item["bbox"]]
                    rx0, ry0 = bx[0] - x0, bx[1] - y0
                    rx1, ry1 = bx[2] - x0, bx[3] - y0
                    original_region = styled_array[ry0:ry1, rx0:rx1]
                    original_luma = np.asarray(
                        Image.fromarray(original_region).convert("L")
                    )
                    threshold = np.percentile(original_luma, 25)
                    ink_pixels = original_region[original_luma <= threshold]
                    if ink_pixels.size:
                        target_ink = np.median(ink_pixels, axis=0)
                    else:
                        target_ink = np.array([82, 77, 70], dtype=np.float32)
                    generated_region = dark_array[ry0:ry1, rx0:rx1]
                    generated_luma = np.asarray(
                        Image.fromarray(generated_region).convert("L")
                    ).astype(np.float32)
                    modulation = 0.58 + 0.42 * (generated_luma / 255.0)
                    matched = np.clip(
                        target_ink[None, None, :] * modulation[..., None],
                        0,
                        255,
                    ).astype(np.uint8)
                    global_alpha = np.asarray(
                        soft_stroke.crop(tuple(bx))
                    ).astype(np.float32)
                    confidence = float(item["confidence"])
                    opacity = (
                        0.36
                        if item["region_type"] == "multi_character_damage"
                        else 0.44
                        if confidence < 0.70
                        else 0.52
                    )
                    alpha_region = np.clip(
                        global_alpha * opacity, 0, 255
                    ).astype(np.uint8)
                    alpha_float = alpha_region[..., None].astype(np.float32) / 255.0
                    styled_array[ry0:ry1, rx0:rx1] = np.clip(
                        matched * alpha_float
                        + original_region.astype(np.float32) * (1.0 - alpha_float),
                        0,
                        255,
                    ).astype(np.uint8)
                    local_alpha[ry0:ry1, rx0:rx1] = np.maximum(
                        local_alpha[ry0:ry1, rx0:rx1], alpha_region
                    )
                styled = Image.fromarray(styled_array)
                styled.save(tile_dir / "styled.png", "PNG")
                generated_canvas.paste(
                    styled,
                    (x0, y0),
                    Image.fromarray(
                        np.where(local_alpha > 0, 255, 0).astype(np.uint8),
                        mode="L",
                    ),
                )
                tile_logs.append(
                    {
                        "tile": index,
                        "patch_bbox": group["patch_bbox"],
                        "region_ids": [
                            item["region_id"] for item in group["regions"]
                        ],
                        "runtime_seconds": time.perf_counter() - tile_start,
                        "output_sha256": file_sha256(tile_dir / "generated.png"),
                    }
                )
                write_json(
                    output_dir / "live_progress.json",
                    {
                        "phase": "RUNNING_INFERENCE",
                        "tiles_complete": index,
                        "tiles_total": len(groups),
                        "updated_at": now(),
                    },
                )
        finally:
            peak = int(torch.cuda.max_memory_allocated(device))
            del pipeline
            del unet
            del generator
            torch.cuda.empty_cache()

    write_phase(
        output_dir,
        timeline,
        "COMPOSITING",
        "Inside-mask feathering with an exactly zero outside-mask alpha",
    )
    binary = np.asarray(global_mask).astype(np.float32) / 255.0
    alpha = binary[..., None]
    original_array = np.asarray(original).astype(np.float32)
    generated_array = np.asarray(generated_canvas).astype(np.float32)
    final_array = np.clip(
        generated_array * alpha + original_array * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    final = Image.fromarray(final_array)
    write_review_artifacts(original, final, global_mask, content, manifest, output_dir)

    write_phase(
        output_dir,
        timeline,
        "PIXEL_AUDIT",
        "Verifying immutable pixels outside the full union mask",
    )
    changed = np.any(final_array != original_array.astype(np.uint8), axis=2)
    mask_array = binary > 0
    outside = int((changed & ~mask_array).sum())
    if outside:
        raise AdapterContractError("OUTSIDE_FULL_RECONSTRUCTION_MASK_CHANGED")
    mask_pixels = int(mask_array.sum())
    changed_inside = int((changed & mask_array).sum())
    total_pixels = int(mask_array.size)
    runtime = time.perf_counter() - start
    audit = {
        "status": "FULL_RECONSTRUCTION_PIXEL_AUDIT_PASS",
        "input_sha256": input_hash,
        "output_sha256": file_sha256(
            output_dir / "full_hypothetical_reconstruction.png"
        ),
        "plan_sha256": plan["plan_sha256"],
        "official_commit": official_tree["commit"],
        "official_worktree_clean": official_tree["clean"],
        "gpu": str(device),
        "gpu_peak_bytes": peak,
        "gpu_runtime_seconds": runtime,
        "tile_count": len(groups),
        "tile_logs": tile_logs,
        "selected_region_count": len(selected),
        "skipped_region_count": len(manifest["regions"]) - len(selected),
        "mask_pixels": mask_pixels,
        "mask_area_ratio": mask_pixels / total_pixels,
        "changed_pixels_inside_mask": changed_inside,
        "outside_mask_changed_pixels": outside,
        "textual_correctness": "NOT_EVALUATED",
        "scholarly_review": "PENDING",
    }
    write_json(output_dir / "pixel_audit.json", audit)
    write_phase(output_dir, timeline, "COMPLETE", "Real Official GPU job completed")
    write_json(output_dir / "gpu_run.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--official-runtime", required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
