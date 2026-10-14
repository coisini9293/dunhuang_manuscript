#!/usr/bin/env python3
"""Reference-guided, pixel-preserving Dunhuang fragment stitching.

The reference image supplies geometry only.  Source fragment pixels are never
restored or redrawn in this stage.  The output records every homography,
canvas transform, overlap and preservation measurement required to reproduce
the stitched manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"unable to read image: {path}")
    return image


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Keep manuscript material while excluding black/white scanner margins."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    raw = ((value > 24) & ~((saturation < 28) & (value > 238))).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2
    )
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        return raw
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    keep = np.zeros_like(raw)
    largest = int(stats[order[0], cv2.CC_STAT_AREA])
    for component in order[: min(4, len(order))]:
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= max(256, int(largest * 0.002)):
            keep[labels == component] = 255
    return keep


def align_to_reference(
    fragment_bgr: np.ndarray, reference_bgr: np.ndarray, ratio: float, ransac_threshold: float
) -> tuple[np.ndarray, dict]:
    gray_f = cv2.cvtColor(fragment_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    max_side = 1800.0
    sf = min(1.0, max_side / max(gray_f.shape))
    sr = min(1.0, max_side / max(gray_r.shape))
    small_f = cv2.resize(gray_f, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    small_r = cv2.resize(gray_r, None, fx=sr, fy=sr, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02)
    key_f, desc_f = sift.detectAndCompute(small_f, None)
    key_r, desc_r = sift.detectAndCompute(small_r, None)
    if desc_f is None or desc_r is None:
        raise RuntimeError("SIFT descriptors unavailable")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_f, desc_r, k=2)
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if len(good) < 8:
        raise RuntimeError(f"too few reference correspondences: {len(good)}")
    source = np.float32([key_f[m.queryIdx].pt for m in good]) / sf
    target = np.float32([key_r[m.trainIdx].pt for m in good]) / sr
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold)
    if homography is None or inlier_mask is None:
        raise RuntimeError("homography estimation failed")
    inliers = inlier_mask.reshape(-1).astype(bool)
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    inlier_errors = errors[inliers]
    meta = {
        "keypoints_fragment": len(key_f),
        "keypoints_reference": len(key_r),
        "good_matches": len(good),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "median_reprojection_error": float(np.median(inlier_errors)),
        "p95_reprojection_error": float(np.percentile(inlier_errors, 95)),
        "fragment_to_reference_homography": homography.tolist(),
    }
    if meta["inliers"] < 12 or meta["inlier_ratio"] < 0.20:
        raise RuntimeError(f"unstable reference geometry: {meta}")
    return homography.astype(np.float64), meta


def corners(shape: tuple[int, ...], transform: np.ndarray) -> np.ndarray:
    height, width = shape[:2]
    points = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
    return cv2.perspectiveTransform(points, transform)[0]


def local_scale(fragment_shape: tuple[int, ...], transform: np.ndarray) -> float:
    projected = corners(fragment_shape, transform)
    top = np.linalg.norm(projected[1] - projected[0])
    bottom = np.linalg.norm(projected[2] - projected[3])
    left = np.linalg.norm(projected[3] - projected[0])
    right = np.linalg.norm(projected[2] - projected[1])
    source_width, source_height = fragment_shape[1], fragment_shape[0]
    return float(np.median([source_width / max(top, 1e-6), source_width / max(bottom, 1e-6),
                            source_height / max(left, 1e-6), source_height / max(right, 1e-6)]))


def decompose(transform: np.ndarray) -> dict:
    a = transform[:2, :2]
    sx, sy = float(np.linalg.norm(a[:, 0])), float(np.linalg.norm(a[:, 1]))
    return {
        "translation_xy": [float(transform[0, 2]), float(transform[1, 2])],
        "rotation_deg": float(math.degrees(math.atan2(a[1, 0], a[0, 0]))),
        "scale_xy": [sx, sy],
        "perspective_terms": [float(transform[2, 0]), float(transform[2, 1])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--fragment-a", required=True)
    parser.add_argument("--fragment-b", required=True)
    parser.add_argument("--reference", required=True, help="same-group reference used only for geometry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-threshold", type=float, default=0.5)
    parser.add_argument("--textual-evidence", help="read-only OCR/CBETA evidence JSON for low-visual branch")
    parser.add_argument("--qwen-evidence", help="optional non-authoritative Qwen reranker evidence JSON")
    parser.add_argument("--ratio", type=float, default=0.72)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    args = parser.parse_args()

    paths = [Path(args.fragment_a).resolve(), Path(args.fragment_b).resolve()]
    reference_path = Path(args.reference).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fragments = [read_bgr(p) for p in paths]
    reference = read_bgr(reference_path)
    transforms_ref, fragment_meta = [], []
    for image in fragments:
        transform, meta = align_to_reference(image, reference, args.ratio, args.ransac_threshold)
        transforms_ref.append(transform)
        fragment_meta.append(meta)

    # Restore approximately native source resolution after using the low-resolution
    # printed reference as a coordinate anchor.
    upsample = float(np.median([local_scale(im.shape, h) for im, h in zip(fragments, transforms_ref)]))
    scale_matrix = np.array([[upsample, 0, 0], [0, upsample, 0], [0, 0, 1]], dtype=np.float64)
    scaled = [scale_matrix @ h for h in transforms_ref]
    all_corners = np.concatenate([corners(im.shape, h) for im, h in zip(fragments, scaled)])
    minimum = np.floor(all_corners.min(axis=0) - 12).astype(int)
    maximum = np.ceil(all_corners.max(axis=0) + 12).astype(int)
    translate = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    canvas_width, canvas_height = int(maximum[0] - minimum[0]), int(maximum[1] - minimum[1])
    canvas_size = (canvas_width, canvas_height)
    canvas_transforms = [translate @ h for h in scaled]

    warped_images, warped_masks, source_masks = [], [], []
    for index, (image, transform) in enumerate(zip(fragments, canvas_transforms)):
        source_mask = foreground_mask(image)
        warped_image = cv2.warpPerspective(image, transform, canvas_size, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpPerspective(source_mask, transform, canvas_size, flags=cv2.INTER_NEAREST)
        warped_mask = (warped_mask > 127).astype(np.uint8) * 255
        source_masks.append(source_mask)
        warped_images.append(warped_image)
        warped_masks.append(warped_mask)
        cv2.imwrite(str(output / f"fragment_{index}_warped.png"), warped_image)
        cv2.imwrite(str(output / f"fragment_{index}_mask.png"), warped_mask)

    mask_a, mask_b = warped_masks
    valid_a, valid_b = mask_a > 0, mask_b > 0
    overlap = valid_a & valid_b
    union = valid_a | valid_b
    result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    result[valid_a & ~valid_b] = warped_images[0][valid_a & ~valid_b]
    result[valid_b & ~valid_a] = warped_images[1][valid_b & ~valid_a]
    if overlap.any():
        dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 3)
        dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 3)
        weight_a = dist_a / np.maximum(dist_a + dist_b, 1e-6)
        blended = (warped_images[0].astype(np.float32) * weight_a[:, :, None]
                   + warped_images[1].astype(np.float32) * (1.0 - weight_a[:, :, None]))
        result[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)
    rgba = np.dstack([result, union.astype(np.uint8) * 255])
    cv2.imwrite(str(output / "stitched.png"), result)
    cv2.imwrite(str(output / "stitched_rgba.png"), rgba)
    cv2.imwrite(str(output / "stitch_mask.png"), union.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "overlap_mask.png"), overlap.astype(np.uint8) * 255)

    # GPU homology result is evidence for pairing; geometry remains independently
    # determined by SIFT/RANSAC and the same-group reference.
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    from pipelines.rejoin import RejoinPipeline
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else Path("/path/to/dunhuang-autohdr/MODELS/weights/rejoin.pth")
    homology = RejoinPipeline(checkpoint, device=args.device)
    confidence, learned_pose = homology.predict_pair(fragments[0], fragments[1])

    visual_homology_pass = float(confidence) >= args.visual_threshold
    textual_payload = None
    textual_homology_pass = False
    textual_score = None
    textual_path = None
    if args.textual_evidence:
        textual_path = Path(args.textual_evidence).resolve()
        textual_payload = json.loads(textual_path.read_text(encoding="utf-8"))
        textual_score = float(textual_payload.get("textual_homology_score", 0.0))
        textual_homology_pass = bool(
            textual_payload.get("status") == "PASS"
            and textual_payload.get("decision") == "TEXTUAL_HOMOLOGY_PASS"
            and textual_score >= float(textual_payload.get("threshold", 1.0))
        )
    qwen_path = Path(args.qwen_evidence).resolve() if args.qwen_evidence else None
    qwen_payload = json.loads(qwen_path.read_text(encoding="utf-8")) if qwen_path else None
    homology_pass = visual_homology_pass or textual_homology_pass
    homology_branch = (
        "VISUAL_GPU" if visual_homology_pass
        else "TEXTUAL_OCR_CBETA" if textual_homology_pass
        else "NONE"
    )

    difference = cv2.absdiff(warped_images[0], warped_images[1])
    seam_mean = float(difference[overlap].mean()) if overlap.any() else None
    overlap_ratio = float(overlap.sum() / max(1, min(valid_a.sum(), valid_b.sum())))
    duplicate_surviving_content = overlap_ratio > 0.80
    adjacency_pass = bool(overlap.any() and not duplicate_surviving_content)
    transformed_corners = [corners(im.shape, h) for im, h in zip(fragments, canvas_transforms)]
    bounds_ok = all(
        float(points[:, 0].min()) >= -1 and float(points[:, 1].min()) >= -1
        and float(points[:, 0].max()) <= canvas_width + 1 and float(points[:, 1].max()) <= canvas_height + 1
        for points in transformed_corners
    )
    records = []
    for index, (path, image, transform, ref_h, meta, src_mask, warped_mask) in enumerate(
        zip(paths, fragments, canvas_transforms, transforms_ref, fragment_meta, source_masks, warped_masks)
    ):
        scale_sq = abs(np.linalg.det(transform[:2, :2]))
        expected_area = max(1.0, float((src_mask > 0).sum()) * scale_sq)
        actual_area = int((warped_mask > 0).sum())
        records.append({
            "fragment": str(path), "sha256": sha256(path), "input_dimensions": [image.shape[1], image.shape[0]],
            **meta, "canvas_transform": transform.tolist(), "transform_summary": decompose(transform),
            "foreground_pixels_source": int((src_mask > 0).sum()), "foreground_pixels_warped": actual_area,
            "foreground_area_ratio_to_affine_expectation": float(actual_area / expected_area),
            "transformed_corners": transformed_corners[index].tolist(),
        })
    relative_b_to_a = np.linalg.inv(canvas_transforms[0]) @ canvas_transforms[1]
    relative_b_to_a /= relative_b_to_a[2, 2]
    geometry_pass = all(r["inliers"] >= 12 and r["inlier_ratio"] >= 0.20
                        and r["median_reprojection_error"] <= args.ransac_threshold for r in records)
    composition_pass = bool(bounds_ok and union.any() and adjacency_pass)
    stitch_pass = bool(homology_pass and geometry_pass and composition_pass)
    metadata = {
        "case_id": args.case_id,
        "status": "PASS" if stitch_pass else "FAIL",
        "homology_score": float(confidence),
        "visual_homology_threshold": args.visual_threshold,
        "visual_homology_decision": "MATCH" if visual_homology_pass else "LOW_CONFIDENCE_ROUTE_TO_TEXT",
        "homology_decision": "MATCH" if homology_pass else "NO_MATCH",
        "homology_branch_used": homology_branch,
        "learned_pose_raw": learned_pose.tolist(),
        "textual_homology_score": textual_score,
        "textual_homology_role": (
            "AUTHORITATIVE_LOW_VISUAL_FALLBACK" if homology_branch == "TEXTUAL_OCR_CBETA"
            else "NOT_REQUIRED_FOR_VISUAL_BRANCH" if visual_homology_pass
            else "UNAVAILABLE_OR_ABSTAIN"
        ),
        "textual_evidence_path": str(textual_path) if textual_path else None,
        "textual_evidence_sha256": sha256(textual_path) if textual_path else None,
        "qwen_reranker": {
            "path": str(qwen_path) if qwen_path else None,
            "sha256": sha256(qwen_path) if qwen_path else None,
            "status": qwen_payload.get("status") if qwen_payload else None,
            "agrees_with_cbeta": qwen_payload.get("agrees_with_deterministic_cbeta") if qwen_payload else None,
            "role": "NON_AUTHORITATIVE; ABSTAIN DOES_NOT_OVERRIDE_CBETA",
        },
        "adjacency_score": float(1.0 - min(1.0, abs(overlap_ratio - 0.05) / 0.75)) if adjacency_pass else 0.0,
        "adjacency_gate": "PASS" if adjacency_pass else "FAIL",
        "duplicate_surviving_content": duplicate_surviving_content,
        "transform_type": "REFERENCE_GUIDED_PROJECTIVE_HOMOGRAPHY",
        "relative_transform_b_to_a": relative_b_to_a.tolist(),
        "relative_transform_summary": decompose(relative_b_to_a),
        "reference_path": str(reference_path), "reference_sha256": sha256(reference_path),
        "reference_role": "GEOMETRY_ONLY",
        "native_resolution_scale": upsample,
        "canvas_dimensions": [canvas_width, canvas_height],
        "canvas_bounds_pass": bounds_ok,
        "overlap_pixels": int(overlap.sum()),
        "union_pixels": int(union.sum()),
        "overlap_ratio_of_smaller_fragment": overlap_ratio,
        "seam_mean_absolute_difference": seam_mean,
        "geometry_confidence": float(min(r["inlier_ratio"] for r in records)),
        "geometry_gate": "PASS" if geometry_pass else "FAIL",
        "composition_gate": "PASS" if composition_pass else "FAIL",
        "source_pixel_policy": "COORDINATE_TRANSFORM_AND_CONTROLLED_OVERLAP_BLEND_ONLY",
        "restoration_performed": False,
        "fragments": records,
        "stitched_path": str(output / "stitched.png"),
        "stitched_sha256": sha256(output / "stitched.png"),
        "runtime_seconds": time.perf_counter() - started,
        "stitch_pass": stitch_pass,
    }
    (output / "transform_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if stitch_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
