"""Rejoin-specific evaluation: metrics + visual gallery."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def trim_light_margins(img_rgb: np.ndarray, threshold: int = 245, padding: int = 4) -> np.ndarray:
    """Remove scanner-white margins while preserving dark manuscript borders.

    ScienceDB fragments can be centred in a canvas several times wider than the
    actual fragment.  Keeping that canvas made the panorama look like Fragment B
    with a large empty gap.  This is intentionally only whitespace trimming; it
    does not claim geometric registration.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] < 3:
        return img_rgb
    content = np.any(img_rgb[:, :, :3] < threshold, axis=2)
    ys, xs = np.where(content)
    if not len(xs) or not len(ys):
        return img_rgb
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(img_rgb.shape[1], int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(img_rgb.shape[0], int(ys.max()) + padding + 1)
    return img_rgb[y0:y1, x0:x1]


def stitch_pair(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
    """Create a tight heuristic preview from two fragment silhouettes.

    The current pose labels are not geometric ground truth, so this preview uses
    scanner-margin trimming plus seam-profile matching.  It may choose B→A when
    that produces a more complementary torn edge.  It is not a replacement for
    a future registered panorama implementation.
    """
    img1_rgb = trim_light_margins(img1_rgb)
    img2_rgb = trim_light_margins(img2_rgb)
    h = max(img1_rgb.shape[0], img2_rgb.shape[0])

    def resize_h(img: np.ndarray) -> np.ndarray:
        scale = h / img.shape[0]
        w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (w, h))

    a, b = resize_h(img1_rgb), resize_h(img2_rgb)

    def seam_fit(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
        left_mask = np.any(left < 250, axis=2)
        right_mask = np.any(right < 250, axis=2)
        offsets: list[int] = []
        y0, y1 = int(h * 0.1), max(int(h * 0.9), int(h * 0.1) + 1)
        for y in range(y0, y1):
            left_x = np.flatnonzero(left_mask[y])
            right_x = np.flatnonzero(right_mask[y])
            if left_x.size and right_x.size:
                offsets.append(int(left_x[-1] - right_x[0] + 1))
        if not offsets:
            return float("inf"), left.shape[1]
        values = np.asarray(offsets, dtype=np.float32)
        offset = int(round(float(np.median(values))))
        mad = float(np.median(np.abs(values - offset)))
        return mad, offset

    score_ab, offset_ab = seam_fit(a, b)
    score_ba, offset_ba = seam_fit(b, a)
    if score_ba < score_ab:
        left, right, offset = b, a, offset_ba
    else:
        left, right, offset = a, b, offset_ab

    # Allow silhouettes to interlock. Scanner-white pixels are transparent;
    # actual manuscript pixels from the right fragment win in the overlap.
    min_x = min(0, offset)
    max_x = max(left.shape[1], offset + right.shape[1])
    canvas = np.full((h, max_x - min_x, 3), 255, dtype=np.uint8)
    left_x = -min_x
    right_x = offset - min_x
    left_roi = canvas[:, left_x:left_x + left.shape[1]]
    right_roi = canvas[:, right_x:right_x + right.shape[1]]
    left_mask = np.any(left < 250, axis=2)
    right_mask = np.any(right < 250, axis=2)
    left_roi[left_mask] = left[left_mask]
    right_roi[right_mask] = right[right_mask]
    return canvas


def edge_alignment_score(img1_rgb: np.ndarray, img2_rgb: np.ndarray, band: int = 24) -> float:
    """Boundary similarity between right edge of A and left edge of B (0-1)."""
    h = min(img1_rgb.shape[0], img2_rgb.shape[0])
    a = cv2.resize(img1_rgb, (int(img1_rgb.shape[1] * h / img1_rgb.shape[0]), h))
    b = cv2.resize(img2_rgb, (int(img2_rgb.shape[1] * h / img2_rgb.shape[0]), h))
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band = min(band, ga.shape[1], gb.shape[1])
    strip_a = ga[:, -band:].flatten()
    strip_b = gb[:, :band].flatten()
    if strip_a.std() < 1e-6 or strip_b.std() < 1e-6:
        return 0.0
    corr = np.corrcoef(strip_a, strip_b)[0, 1]
    return float(np.clip((corr + 1) / 2, 0, 1))


def find_reference_image(data_dir: Path, fragment_path: Path) -> Path | None:
    """Find full reference image (总图) in the same splice group."""
    from utils.rejoin_data import is_fragment

    group_dir = (data_dir / fragment_path).parent
    refs = [
        f
        for f in group_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"} and not is_fragment(f)
    ]
    return refs[0] if refs else None


def reference_ssim(stitched_rgb: np.ndarray, reference_path: Path) -> float:
    from skimage.metrics import structural_similarity as ssim

    ref = cv2.cvtColor(cv2.imread(str(reference_path)), cv2.COLOR_BGR2RGB)
    h = min(stitched_rgb.shape[0], ref.shape[0])
    w = min(stitched_rgb.shape[1], ref.shape[1])
    if h < 8 or w < 8:
        return 0.0
    s = cv2.resize(stitched_rgb, (w, h))
    r = cv2.resize(ref, (w, h))
    return float(ssim(s, r, channel_axis=2))


def summarize_rejoin_effect(records: list[dict]) -> dict[str, float]:
    """Aggregate rejoin-effect metrics from per-sample records."""
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    tp = [r for r in pos if r["pred"] == 1]
    fp = [r for r in neg if r["pred"] == 1]
    tn = [r for r in neg if r["pred"] == 0]
    fn = [r for r in pos if r["pred"] == 0]

    edge_scores = [r["edge_score"] for r in pos if r["edge_score"] is not None]
    ref_ssims = [r["ref_ssim"] for r in tp if r["ref_ssim"] is not None]

    return {
        "positive_pair_accuracy": len(tp) / max(len(pos), 1),
        "negative_pair_accuracy": len(tn) / max(len(neg), 1),
        "false_rejoin_rate": len(fp) / max(len(neg), 1),
        "missed_rejoin_rate": len(fn) / max(len(pos), 1),
        "mean_edge_alignment": float(np.mean(edge_scores)) if edge_scores else 0.0,
        "mean_reference_ssim": float(np.mean(ref_ssims)) if ref_ssims else 0.0,
        "reference_pairs_evaluated": len(ref_ssims),
    }
