"""Default paths for DiffHDR (required core module)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HDR_ROOT = PROJECT_ROOT / "third_party" / "HDR"
DIFFHDR_CKPT = PROJECT_ROOT / "checkpoints" / "diffhdr" / "unet"
DIFFHDR_EXAMPLES = PROJECT_ROOT / "data" / "raw" / "diffhdr_examples"
RESTORE_OUTPUT = PROJECT_ROOT / "outputs" / "restore"


def ensure_diffhdr_ready() -> Path:
    """Raise if DiffHDR repo or weights missing."""
    if not (HDR_ROOT / "inference.py").exists():
        raise FileNotFoundError(
            f"DiffHDR repo missing: {HDR_ROOT}. Run: python scripts/setup_diffhdr.py"
        )
    if not (DIFFHDR_CKPT / "config.json").exists():
        raise FileNotFoundError(
            f"DiffHDR weights missing: {DIFFHDR_CKPT}. See checkpoints/diffhdr/DOWNLOAD.md"
        )
    return DIFFHDR_CKPT
