from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .common import AdapterContractError, file_sha256, write_json


def composite_approved(
    original_path: Path,
    generated_path: Path,
    approved_mask_path: Path,
    output_dir: Path,
) -> dict:
    """Copy generated pixels only where the approved binary mask is non-zero."""
    original = np.asarray(Image.open(original_path).convert("RGB"))
    generated = np.asarray(Image.open(generated_path).convert("RGB"))
    mask = np.asarray(Image.open(approved_mask_path).convert("L")) > 0
    if original.shape != generated.shape or original.shape[:2] != mask.shape:
        raise AdapterContractError("COMPOSITOR_DIMENSION_MISMATCH")
    if not bool(mask.any()):
        raise AdapterContractError("APPROVED_MASK_EMPTY")

    final = original.copy()
    final[mask] = generated[mask]
    outside_changed = np.any(final != original, axis=2) & ~mask
    outside_count = int(outside_changed.sum())
    if outside_count:
        raise AdapterContractError("OUTSIDE_APPROVED_MASK_CHANGED")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "constrained_restored.png"
    mask_path = output_dir / "approved_mask.png"
    difference_path = output_dir / "difference.png"
    comparison_path = output_dir / "comparison.png"
    Image.fromarray(final).save(final_path, format="PNG")
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
        mask_path, format="PNG"
    )
    difference = np.abs(final.astype(np.int16) - original.astype(np.int16))
    Image.fromarray(np.clip(difference, 0, 255).astype(np.uint8)).save(
        difference_path, format="PNG"
    )
    Image.fromarray(np.concatenate([original, final], axis=1)).save(
        comparison_path, format="PNG"
    )
    report = {
        "status": "CONSTRAINED_COMPOSITOR_PASS",
        "outside_mask_changed_pixels": outside_count,
        "approved_mask_pixels": int(mask.sum()),
        "changed_pixels_inside_mask": int(
            (np.any(final != original, axis=2) & mask).sum()
        ),
        "original_sha256": file_sha256(original_path),
        "generated_sha256": file_sha256(generated_path),
        "final_sha256": file_sha256(final_path),
        "human_review_status": "not_reviewed",
    }
    write_json(output_dir / "outside_mask_audit.json", report)
    return report

