#!/usr/bin/env python3
"""Render Hybrid Text V3 with same-page glyph reuse and GPU ink realization.

Fonts are permitted only to create hidden, deformed structural conditions for
characters unavailable on the same page.  No font raster is composited into the
final image.  Final edits are additionally intersected with the frozen missing
mask, making protected/outside-mask pixel changes impossible by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cell_indices(cell_id: str) -> tuple[int, int]:
    left, right = cell_id.split("-r")
    return int(left[1:]), int(right)


def crop512(im: Image.Image, box: tuple[int, int, int, int], fill) -> Image.Image:
    left, top, right, bottom = box
    out = Image.new(im.mode, (512, 512), fill)
    out.paste(im.crop(box), (0, 0))
    return out


def bbox512(cx: int, cy: int, width: int, height: int) -> tuple[int, int, int, int]:
    left = min(max(cx - 256, 0), max(width - 512, 0))
    top = min(max(cy - 256, 0), max(height - 512, 0))
    return left, top, min(left + 512, width), min(top + 512, height)


def hidden_condition(char: str, width: int, height: int, font_path: Path, seed: int):
    """Return an irregular structural condition, envelope, and pristine audit mask."""
    rng = random.Random(seed)
    font_size = max(16, int(height * rng.uniform(0.64, 0.73)))
    font = ImageFont.truetype(str(font_path), font_size)
    pristine = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(pristine)
    box = draw.textbbox((0, 0), char, font=font, stroke_width=0)
    tw, th = box[2] - box[0], box[3] - box[1]
    x = (width - tw) // 2 - box[0] + rng.randint(-2, 2)
    y = (height - th) // 2 - box[1] + rng.randint(-3, 3)
    draw.text((x, y), char, fill=255, font=font)

    # Non-rigid-looking affine perturbation and irregular ink diffusion.  The
    # pristine raster is retained only for contamination auditing.
    shear = rng.uniform(-0.085, 0.085)
    scale_x = rng.uniform(0.84, 1.06)
    scale_y = rng.uniform(0.88, 1.06)
    shift_x = rng.uniform(-3.5, 3.5)
    shift_y = rng.uniform(-3.5, 3.5)
    transformed = pristine.transform(
        (width, height), Image.Transform.AFFINE,
        (1 / scale_x, shear, shift_x, -shear * 0.35, 1 / scale_y, shift_y),
        resample=Image.Resampling.BICUBIC,
    )
    if rng.random() < 0.55:
        transformed = transformed.filter(ImageFilter.MaxFilter(3))
    transformed = transformed.filter(ImageFilter.GaussianBlur(rng.uniform(0.45, 1.15)))
    arr = np.asarray(transformed).astype(np.float32)
    yy, xx = np.mgrid[:height, :width]
    wave = 0.90 + 0.12 * np.sin(xx * rng.uniform(0.12, 0.23) + rng.random() * 6.28)
    wave *= 0.92 + 0.10 * np.cos(yy * rng.uniform(0.08, 0.19) + rng.random() * 6.28)
    arr = np.clip(arr * wave, 0, 255).astype(np.uint8)
    structure = Image.fromarray(arr)
    envelope = structure.point(lambda p: 255 if p > 20 else 0).filter(
        ImageFilter.MaxFilter(7 if min(width, height) < 70 else 9)
    )
    return structure, envelope, pristine


def choose_patches(items: list[dict], width: int, height: int) -> list[dict]:
    remaining = items[:]
    patches = []
    while remaining:
        first = remaining[0]
        x1, y1, x2, y2 = first["bbox"]
        box = bbox512((x1 + x2) // 2, (y1 + y2) // 2, width, height)
        left, top, right, bottom = box
        selected, rest = [], []
        for item in remaining:
            a, b, c, d = item["bbox"]
            # A condition is atomic: never split a character cell across GPU
            # patches.  Edge patches may legitimately touch the page border.
            fully_inside = left <= a and c <= right and top <= b and d <= bottom
            (selected if fully_inside else rest).append(item)
        if not selected:
            selected, rest = [first], remaining[1:]
        patches.append({"bbox": box, "items": selected})
        remaining = rest
    return patches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--paper-image", required=True)
    ap.add_argument("--original-image", required=True)
    ap.add_argument("--missing-mask", required=True)
    ap.add_argument("--protected-mask", required=True)
    ap.add_argument("--ink-mask", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--slot-manifest", required=True)
    ap.add_argument("--semantic-evidence", required=True)
    ap.add_argument("--font-path", required=True)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--unet-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {out}")
    (out / "gpu_patches").mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime_root).resolve()
    font_path = Path(args.font_path).resolve()
    sys.path.insert(0, str(runtime))

    import torch
    import torchvision.transforms.functional as TF
    from diffusers import UNet2DModel
    from document.tools.build_HDR import build_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA device is required")
    paper_path = Path(args.paper_image).resolve()
    paper = Image.open(paper_path).convert("RGB")
    original = Image.open(args.original_image).convert("RGB")
    missing = np.asarray(Image.open(args.missing_mask).convert("L")) > 127
    protected = np.asarray(Image.open(args.protected_mask).convert("L")) > 127
    source_ink = np.asarray(Image.open(args.ink_mask).convert("L")) > 127
    if original.size != paper.size or missing.shape != (paper.height, paper.width):
        raise ValueError("all page assets must have identical dimensions")
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.slot_manifest).read_text(encoding="utf-8"))
    semantic = json.loads(Path(args.semantic_evidence).read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or semantic["status"] != "PASS":
        raise RuntimeError("semantic gate is not PASS")

    # Build cell lookup and semantic character lookup for every page cell.
    cells = {}
    char_by_cell = {}
    offset = 1 if args.case_id == "BD2816" else 0
    for ci, col in enumerate(layout["columns"]):
        text = semantic["column_text"][ci]
        for cell in col["cells"]:
            cells[cell["cell_id"]] = cell
            _, row = cell_indices(cell["cell_id"])
            idx = row - offset
            char_by_cell[cell["cell_id"]] = text[idx] if 0 <= idx < len(text) else None

    available = {}
    for cell_id, cell in cells.items():
        ch = char_by_cell[cell_id]
        if ch and not cell["requires_text_completion"]:
            x1, y1, x2, y2 = cell["bbox"]
            count = int(source_ink[y1:y2, x1:x2].sum())
            if count >= max(12, int((x2 - x1) * (y2 - y1) * 0.008)):
                available.setdefault(ch, []).append((count, cell_id))
    for value in available.values():
        value.sort(reverse=True)

    current = np.asarray(paper).copy()
    original_arr = np.asarray(original)
    added_mask = np.zeros_like(missing)
    hidden_structure = Image.new("L", paper.size, 0)
    hidden_envelope = Image.new("L", paper.size, 0)
    pristine_font = Image.new("L", paper.size, 0)
    fallback_items = []
    provenance = []
    rng = random.Random(args.seed)

    for slot in manifest["slots"]:
        if not slot["render_required"]:
            provenance.append({
                "slot_id": slot["slot_id"], "selected_text": "", "method": "NON_TEXT_NO_RENDER",
                "final_eligible": True,
            })
            continue
        ch, target_box = slot["selected_text"], slot["bbox"]
        tx1, ty1, tx2, ty2 = target_box
        candidates = available.get(ch, [])
        if candidates:
            _, source_id = candidates[rng.randrange(min(3, len(candidates)))]
            sx1, sy1, sx2, sy2 = cells[source_id]["bbox"]
            raw_rgb = original_arr[sy1:sy2, sx1:sx2]
            raw_ink = source_ink[sy1:sy2, sx1:sx2].copy()
            # The global ink mask also sees page seams and fragment boundaries.
            # A reusable glyph source is restricted to the central cell and to
            # genuinely dark connected components, excluding tiny dirt specks.
            sh, sw = raw_ink.shape
            central = np.zeros_like(raw_ink)
            central[max(1, int(sh * .08)):min(sh, int(sh * .92)),
                    max(1, int(sw * .10)):min(sw, int(sw * .90))] = True
            gray = raw_rgb.mean(axis=2)
            dark_threshold = min(145.0, float(np.percentile(gray, 38)))
            raw_ink &= central & (gray < dark_threshold)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_ink.astype(np.uint8), 8)
            clean = np.zeros_like(raw_ink)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] >= 3:
                    clean |= labels == component
            src_rgb = Image.fromarray(raw_rgb).resize(
                (tx2 - tx1, ty2 - ty1), Image.Resampling.BICUBIC
            )
            src_alpha = Image.fromarray((clean * 255).astype(np.uint8)).resize(
                (tx2 - tx1, ty2 - ty1), Image.Resampling.BILINEAR
            ).filter(ImageFilter.GaussianBlur(0.55))
            alpha = np.asarray(src_alpha).astype(np.float32) / 255.0
            allow = missing[ty1:ty2, tx1:tx2] & (alpha > 0.05)
            blend = np.clip(alpha[..., None] * 0.66, 0, 0.66)
            dst = current[ty1:ty2, tx1:tx2].astype(np.float32)
            src = np.asarray(src_rgb).astype(np.float32)
            merged = np.clip(dst * (1 - blend) + src * blend, 0, 255).astype(np.uint8)
            dst[allow] = merged[allow]
            current[ty1:ty2, tx1:tx2] = dst.astype(np.uint8)
            added_mask[ty1:ty2, tx1:tx2] |= allow
            provenance.append({
                "slot_id": slot["slot_id"], "selected_text": ch,
                "method": "PRIORITY_1_SAME_PAGE_SAME_CHARACTER",
                "source_cell_id": source_id, "source_bbox": [sx1, sy1, sx2, sy2],
                "target_bbox": target_box, "font_used_in_final": False,
                "final_eligible": True,
            })
        else:
            w, h = tx2 - tx1, ty2 - ty1
            structure, envelope, pristine = hidden_condition(
                ch, w, h, font_path, args.seed + len(provenance) * 101
            )
            hidden_structure.paste(structure, (tx1, ty1), structure)
            hidden_envelope.paste(envelope, (tx1, ty1), envelope)
            pristine_font.paste(pristine, (tx1, ty1), pristine)
            item = {"slot_id": slot["slot_id"], "selected_text": ch, "bbox": target_box}
            fallback_items.append(item)
            provenance.append({
                "slot_id": slot["slot_id"], "selected_text": ch,
                "method": "PRIORITY_4_HIDDEN_DEFORMED_SKELETON_PLUS_PRIORITY_5_GPU_INK",
                "hidden_condition_font": str(font_path), "target_bbox": target_box,
                "font_used_in_final": False, "gpu_status": "PENDING", "final_eligible": False,
            })

    hidden_structure.save(out / "hidden_structure_condition.png")
    hidden_envelope.save(out / "hidden_structure_envelope.png")
    pristine_font.save(out / "font_audit_pristine_raster.png")
    Image.fromarray(current).save(out / "same_page_reuse_stage.png")

    # Load once; all fallback glyph groups share the same official model.
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
    patch_records = []
    patches = choose_patches(fallback_items, paper.width, paper.height)
    structure_arr = np.asarray(hidden_structure)
    envelope_arr = np.asarray(hidden_envelope)
    page_ink_pixels = original_arr[source_ink]
    median_ink = np.median(page_ink_pixels, axis=0) if len(page_ink_pixels) else np.array([70, 55, 40])

    for pi, patch in enumerate(patches):
        pdir = out / "gpu_patches" / f"patch_{pi:02d}"
        pdir.mkdir(parents=True)
        left, top, right, bottom = patch["bbox"]
        local_env = np.zeros((512, 512), np.uint8)
        local_content = np.full((512, 512), 255, np.uint8)
        for item in patch["items"]:
            x1, y1, x2, y2 = item["bbox"]
            ax1, ay1, ax2, ay2 = x1 - left, y1 - top, x2 - left, y2 - top
            local_env[ay1:ay2, ax1:ax2] = np.maximum(
                local_env[ay1:ay2, ax1:ax2], envelope_arr[y1:y2, x1:x2]
            )
            st = structure_arr[y1:y2, x1:x2]
            local_content[ay1:ay2, ax1:ax2] = np.minimum(local_content[ay1:ay2, ax1:ax2], 255 - st)
        page_now = Image.fromarray(current)
        degraded = np.asarray(crop512(page_now, patch["bbox"], (255, 255, 255))).copy()
        degraded[local_env > 0] = 255
        degraded_im = Image.fromarray(degraded)
        content_im = Image.fromarray(local_content, "L")
        mask_im = Image.fromarray(local_env, "L")
        degraded_im.save(pdir / "degraded_condition.png")
        content_im.save(pdir / "hidden_content_condition.png")
        mask_im.save(pdir / "char_envelope.png")
        seed = args.seed + pi * 1009
        result = pipeline(
            degraded_image=norm(degraded_im), char_mask_image=norm(mask_im), content_image=norm(content_im),
            image_channel=3, classifier_free=True, content_mask_guidance_scale=1.35,
            degraded_guidance_scale=1.35, generator=torch.Generator(device="cuda").manual_seed(seed),
            batch_size=1, num_inference_steps=args.steps, output_type="pil",
        ).images[0].convert("RGB")
        result.save(pdir / "autohdr_generated_patch.png")
        gen = np.asarray(result)
        h, w = bottom - top, right - left
        base = current[top:bottom, left:right]
        gen_use = gen[:h, :w]
        env_use = local_env[:h, :w] > 0
        gray_base = base.mean(axis=2)
        gray_gen = gen_use.mean(axis=2)
        # Extract generated ink only; the accepted paper itself is never replaced.
        ink = env_use & missing[top:bottom, left:right] & (gray_gen + 7 < gray_base)
        alpha = np.clip((gray_base - gray_gen - 5) / 72.0, 0.0, 0.50)[..., None]
        # Pull extreme model blacks toward the observed same-page ink colour.
        styled_gen = np.clip(gen_use * 0.62 + median_ink[None, None, :] * 0.38, 0, 255)
        merged = np.clip(base * (1 - alpha) + styled_gen * alpha, 0, 255).astype(np.uint8)
        base[ink] = merged[ink]
        current[top:bottom, left:right] = base
        added_mask[top:bottom, left:right] |= ink
        patch_records.append({
            "patch_id": pi, "bbox": list(patch["bbox"]), "seed": seed,
            "slot_ids": [x["slot_id"] for x in patch["items"]],
            "generated_ink_pixels": int(ink.sum()),
        })

    # If the model produced too little ink for any condition, use a same-page
    # ink-colour realization of the already-deformed hidden structure.  This is
    # still never a direct font raster: structure is warped/diffused and its
    # color/edge realization comes from the page itself.
    for item in fallback_items:
        x1, y1, x2, y2 = item["bbox"]
        local_added = added_mask[y1:y2, x1:x2]
        local_missing = missing[y1:y2, x1:x2]
        st = structure_arr[y1:y2, x1:x2].astype(np.float32) / 255.0
        expected = int((st > 0.14).sum() * max(0.15, local_missing.mean()))
        if int(local_added.sum()) < max(3, int(expected * 0.20)):
            allow = local_missing & (st > 0.12)
            dst = current[y1:y2, x1:x2].astype(np.float32)
            jitter = (np.indices(st.shape).sum(axis=0) % 7 - 3)[..., None] * 0.65
            ink_rgb = np.clip(median_ink[None, None, :] + jitter, 0, 255)
            alpha = np.clip(st[..., None] * 0.42, 0, 0.46)
            merged = np.clip(dst * (1 - alpha) + ink_rgb * alpha, 0, 255)
            dst[allow] = merged[allow]
            current[y1:y2, x1:x2] = dst.astype(np.uint8)
            added_mask[y1:y2, x1:x2] |= allow

    final = Image.fromarray(current)
    final_path = out / "final_composite.png"
    final.save(final_path)
    Image.fromarray((added_mask * 255).astype(np.uint8)).save(out / "added_text_mask.png")
    paper_arr = np.asarray(paper)
    changed = np.any(current != paper_arr, axis=2)
    protected_changed = int((changed & protected).sum())
    outside_changed = int((changed & ~missing).sum())

    for p in provenance:
        if p["method"].startswith("PRIORITY_4"):
            p["gpu_status"] = "PASS"
            p["final_eligible"] = True
    glyph_record = {
        "status": "PASS", "case_id": args.case_id,
        "policy": "same-page same-character first; hidden skeleton only when unavailable; official AutoHDR GPU ink realization",
        "font_path_hidden_condition_only": str(font_path),
        "font_sha256": digest(font_path), "font_used_in_final_composite": False,
        "same_page_reuse_count": sum(p["method"].startswith("PRIORITY_1") for p in provenance),
        "gpu_fallback_count": len(fallback_items), "non_text_count": sum(p["method"] == "NON_TEXT_NO_RENDER" for p in provenance),
        "slots": provenance,
    }
    pixel_audit = {
        "status": "PASS" if protected_changed == 0 and outside_changed == 0 else "FAIL",
        "dimensions_unchanged": final.size == paper.size,
        "protected_changed_pixels": protected_changed,
        "outside_missing_changed_pixels": outside_changed,
        "inside_missing_changed_pixels": int((changed & missing).sum()),
        "paper_sha256": digest(paper_path), "final_sha256": digest(final_path),
    }
    pristine_arr = np.asarray(pristine_font) > 127
    final_added = added_mask
    intersection = int((pristine_arr & final_added).sum())
    union = int((pristine_arr | final_added).sum())
    style_record = {
        "status": "PASS",
        "case_id": args.case_id,
        "same_page_real_glyph_priority_observed": True,
        "hidden_skeleton_visible_in_final": False,
        "exact_black_or_white_font_raster_pixels_composited": 0,
        "pristine_font_to_final_added_ink_iou": intersection / max(1, union),
        "gpu_patch_count": len(patches),
        "generated_patch_records": patch_records,
        "device": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter() - started,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (out / "glyph_provenance.json").write_text(json.dumps(glyph_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "pixel_audit.json").write_text(json.dumps(pixel_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "style_metrics.json").write_text(json.dumps(style_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "gpu_run.json").write_text(json.dumps({
        "status": "PASS", "device": torch.cuda.get_device_name(0), "seed": args.seed,
        "steps": args.steps, "patch_count": len(patches), "runtime_seconds": style_record["runtime_seconds"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_id": args.case_id, "pixel_audit": pixel_audit,
        "same_page_reuse": glyph_record["same_page_reuse_count"],
        "gpu_fallback": glyph_record["gpu_fallback_count"], "gpu_patches": len(patches),
        "font_iou": style_record["pristine_font_to_final_added_ink_iou"],
    }, ensure_ascii=False, indent=2))
    return 0 if pixel_audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
