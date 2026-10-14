from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from . import OFFICIAL_COMMIT
from .audit_writer import append_audit
from .common import (
    AdapterContractError,
    file_sha256,
    load_json,
    verify_official_tree,
    write_json,
)
from .plan_validator import validate_plan
from .safe_compositor import composite_approved


PATCH_SIZE = 448


def _phase(path: Path, name: str, detail: str = "") -> None:
    write_json(
        path,
        {
            "phase": name,
            "detail": detail,
            "updated_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        },
    )


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


def _patch_origin(bbox: list[int], width: int, height: int) -> tuple[int, int]:
    if width < PATCH_SIZE or height < PATCH_SIZE:
        raise AdapterContractError("IMAGE_SMALLER_THAN_OFFICIAL_PATCH")
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    return (
        max(0, min(center_x - PATCH_SIZE // 2, width - PATCH_SIZE)),
        max(0, min(center_y - PATCH_SIZE // 2, height - PATCH_SIZE)),
    )


def run_restore(
    image_path: Path,
    bundle_path: Path,
    plan_path: Path,
    case_dir: Path,
    official_source: Path,
    official_runtime: Path,
    *,
    expected_rag_version: str,
    expected_cbeta_commit: str,
    phase_file: Path,
) -> dict:
    _phase(phase_file, "LOADING", "Validating immutable input and loading models")
    tree = verify_official_tree(official_source, OFFICIAL_COMMIT)
    bundle = load_json(bundle_path)
    plan = load_json(plan_path)
    image_hash = file_sha256(image_path)
    validated = validate_plan(
        plan,
        bundle,
        expected_image_sha256=image_hash,
        expected_rag_version=expected_rag_version,
        expected_cbeta_commit=expected_cbeta_commit,
    )
    region_map = {
        item["region_id"]: item for item in bundle.get("damage_regions", [])
    }
    restore_dir = case_dir / "restore"
    patch_dir = restore_dir / "patches"
    condition_dir = restore_dir / "conditions"
    patch_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)
    audit_path = case_dir / "audit" / "adapter_events.jsonl"

    with official_import_context(official_source, official_runtime):
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        from diffusers import UNet2DModel
        from document.tools.build_HDR import build_pipeline
        from utils_pipeline import invert_image, render_char_with_font_T, restore_image

        if not torch.cuda.is_available():
            raise AdapterContractError("GPU_REQUIRED_FOR_OFFICIAL_RESTORE")
        device = torch.device("cuda")
        original = Image.open(image_path).convert("RGB")
        inverted = invert_image(original)
        generated_canvas = original.copy()
        approved_mask = Image.new("L", original.size, 0)

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
        _phase(phase_file, "RUNNING", "Official diffusion inference")
        patch_log: list[dict] = []
        try:
            for region_id in validated["approved_region_ids"]:
                region = region_map[region_id]
                text = validated["confirmed_texts"][region_id]
                if len(text.strip()) != 1:
                    raise AdapterContractError(
                        f"CONFIRMED_TEXT_MUST_BE_ONE_CHARACTER:{region_id}"
                    )
                bbox = [int(value) for value in region["bbox"]]
                x0, y0 = _patch_origin(bbox, *original.size)
                x1, y1 = x0 + PATCH_SIZE, y0 + PATCH_SIZE
                relative = [
                    bbox[0] - x0,
                    bbox[1] - y0,
                    bbox[2] - x0,
                    bbox[3] - y0,
                ]
                degraded = inverted.crop((x0, y0, x1, y1))
                mask = Image.new("L", (PATCH_SIZE, PATCH_SIZE), 0)
                mask.paste(255, tuple(relative))
                content = Image.new("L", (PATCH_SIZE, PATCH_SIZE), 255)
                glyph = render_char_with_font_T(text)
                if glyph.mode != "L":
                    glyph = glyph.convert("L")
                if glyph.getbbox() is None:
                    raise AdapterContractError(f"OFFICIAL_FONT_CANNOT_RENDER:{region_id}")
                glyph = glyph.resize(
                    (relative[2] - relative[0], relative[3] - relative[1]),
                    Image.Resampling.LANCZOS,
                )
                content.paste(glyph, (relative[0], relative[1]))
                degraded_array = np.asarray(degraded)
                mask_array = np.asarray(mask) > 0
                degraded_array = degraded_array.copy()
                degraded_array[mask_array] = 255
                degraded = Image.fromarray(degraded_array)

                condition_prefix = condition_dir / region_id
                degraded.save(condition_prefix.with_suffix(".degraded.png"), "PNG")
                content.save(condition_prefix.with_suffix(".content.png"), "PNG")
                mask.save(condition_prefix.with_suffix(".mask.png"), "PNG")

                to_tensor = lambda image: TF.normalize(
                    TF.to_tensor(image.resize((512, 512), Image.Resampling.LANCZOS)),
                    [0.5],
                    [0.5],
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    generated = pipeline(
                        degraded_image=to_tensor(degraded),
                        char_mask_image=to_tensor(mask),
                        content_image=to_tensor(content),
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
                patch_path = patch_dir / f"{region_id}.png"
                generated.save(patch_path, "PNG")
                generated_canvas.paste(generated, (x0, y0, x1, y1))
                approved_mask.paste(255, tuple(bbox))
                patch_log.append(
                    {
                        "region_id": region_id,
                        "confirmed_text": text,
                        "bbox": bbox,
                        "patch_bbox": [x0, y0, x1, y1],
                        "content_sha256": file_sha256(
                            condition_prefix.with_suffix(".content.png")
                        ),
                        "mask_sha256": file_sha256(
                            condition_prefix.with_suffix(".mask.png")
                        ),
                        "output_sha256": file_sha256(patch_path),
                    }
                )
        finally:
            del pipeline
            del unet
            del generator
            torch.cuda.empty_cache()

        generated_path = restore_dir / "generated_scratch.png"
        mask_path = restore_dir / "approved_union_mask.png"
        generated_canvas.save(generated_path, "PNG")
        approved_mask.save(mask_path, "PNG")
        peak = int(torch.cuda.max_memory_allocated())

    _phase(phase_file, "COMPOSITING", "Applying approved-region compositor")
    compositor = composite_approved(
        image_path, generated_path, mask_path, restore_dir
    )
    _phase(phase_file, "AUDITING", "Verifying constrained pixel boundaries")
    result = {
        "status": "OFFICIAL_RESTORE_WORKER_PASS",
        "plan_sha256": validated["plan_sha256"],
        "approved_region_ids": validated["approved_region_ids"],
        "confirmed_texts": validated["confirmed_texts"],
        "rejected_region_ids": validated["rejected_region_ids"],
        "abstained_region_ids": validated["abstained_region_ids"],
        "official_commit": tree["commit"],
        "patches": patch_log,
        "gpu_peak_bytes": peak,
        "exit_code": 0,
        "compositor": compositor,
        "human_review_status": "not_reviewed",
    }
    write_json(restore_dir / "official_restore_worker.json", result)
    append_audit(audit_path, "official_restore_completed", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--official-runtime", required=True, type=Path)
    parser.add_argument("--rag-version", required=True)
    parser.add_argument("--cbeta-commit", required=True)
    parser.add_argument("--phase-file", required=True, type=Path)
    args = parser.parse_args()
    run_restore(
        args.input,
        args.bundle,
        args.plan,
        args.case_dir,
        args.official_source,
        args.official_runtime,
        expected_rag_version=args.rag_version,
        expected_cbeta_commit=args.cbeta_commit,
        phase_file=args.phase_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
