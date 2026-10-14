#!/usr/bin/env python3
"""Run the unmodified official AutoHDR API in a disposable linked workspace.

The official API renders legacy fonts internally, so any produced image is
explicitly an E2E candidate, never a final no-font restoration deliverable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


LINKS = ("ckpt", "dist", "document", "mmdet", "models", "src", "utils",
         "infer_pipeline_api.py", "utils_pipeline.py")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--input-image", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    runtime, work = Path(args.runtime_root).resolve(), Path(args.workspace).resolve()
    if not (runtime / "infer_pipeline_api.py").is_file():
        raise FileNotFoundError("official AutoHDR API entrypoint absent")
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"workspace must be new or empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        target = runtime / name
        if not target.exists():
            raise FileNotFoundError(f"required runtime asset missing: {target}")
        (work / name).symlink_to(target, target_is_directory=target.is_dir())
    (work / "tmp_img").mkdir()
    (work / "results").mkdir()
    os.chdir(work)
    sys.path.insert(0, str(work))
    import torch
    from PIL import Image
    from infer_pipeline_api import main as official_main
    if not torch.cuda.is_available():
        raise RuntimeError("official E2E requires CUDA")
    opt = SimpleNamespace(
        vague_det_config="./ckpt/damage_detect.py", vague_det_weights="./ckpt/damage_detect.pth",
        model_name_or_path="./ckpt/AutoHDR-Qwen2-7B", ocr_det_weights="./ckpt/best.pt",
        det_batch_size=1, reg_batch_size=36, img_size=2048, conf_thres=0.45, iou_thres=0.2,
        device="0", seed=args.seed, image_channel=3, pipeline="DPM-Solver++", classifier_free=True,
        content_mask_guidance_scale=1.5, degraded_guidance_scale=1.2, solver_order=2,
        num_inference_steps=20, ddpm_num_steps=1000, ddpm_beta_schedule="linear", prediction_type="sample",
    )
    torch.cuda.empty_cache()
    started = time.perf_counter()
    stages, image = [], None
    for stage, maybe_image in official_main(Image.open(args.input_image).convert("RGB"), opt):
        stages.append(str(stage))
        if maybe_image is not None:
            image = maybe_image
    if image is not None:
        image.save(work / "official_font_rendered_candidate.png")
    record = {
        "status": "PASS", "case_id": args.case_id, "device": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter() - started, "stages": stages,
        "official_candidate_image": bool(image is not None),
        "final_image_policy": "NOT_FINAL; official API may render legacy font glyphs",
    }
    (work / "official_e2e_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
