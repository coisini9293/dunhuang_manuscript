#!/usr/bin/env python3
"""Download and audit a fixed public mirror of Stable Diffusion 2 Inpainting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "sd2-community/stable-diffusion-2-inpainting"
REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
ENDPOINT = "https://hf-mirror.com"
BASE = Path("/path/to/dunhuang-autohdr")
MODEL_DIR = BASE / "models" / f"sd2-inpainting-community-{REVISION[:12]}"
REPORT_DIR = BASE / "reports" / "paper_inpainting"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        endpoint=ENDPOINT,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "README.md",
            "LICENSE*",
            "model_index.json",
            "scheduler/*",
            "feature_extractor/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
        ],
    )
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "unet/config.json",
        "vae/config.json",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    files = []
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append(
                {
                    "path": path.relative_to(MODEL_DIR).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "status": "SD2_INPAINTING_MODEL_READY" if not missing else "SD2_INPAINTING_MODEL_INCOMPLETE",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "endpoint": ENDPOINT,
        "weight_variant": "fp16",
        "weight_format": "safetensors",
        "license": "CreativeML Open RAIL++-M",
        "model_dir": str(MODEL_DIR),
        "required_missing": missing,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "elapsed_seconds": time.time() - started,
        "files": files,
    }
    target = REPORT_DIR / "sd2_model_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
