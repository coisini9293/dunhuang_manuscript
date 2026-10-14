"""Official AutoHDR backend — subprocess mode (仅 Linux CUDA 可用)。"""
import json, subprocess, os, uuid
from pathlib import Path

OFFICIAL_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "AutoHDR"
PYTHON = os.environ.get("AUTOHDR_PYTHON", "python")


def official_detect(image_bytes: bytes = None) -> dict:
    """调用官方 AutoHDR 进行损坏检测和 OCR（Stage 1）。"""
    if not OFFICIAL_ROOT.exists():
        return {"error": "AutoHDR not cloned. Run: git clone ... third_party/AutoHDR"}
    return {"status": "autohdr_inference_not_executed", "note": "Requires Linux CUDA server."}


def official_predict(damages: list, ocr_chars: list) -> dict:
    return {"status": "autohdr_inference_not_executed"}


def official_restore(image_bytes: bytes = None, confirmed: list = None) -> dict:
    return {"status": "autohdr_inference_not_executed"}
