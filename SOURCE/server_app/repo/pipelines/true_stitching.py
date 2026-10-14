#!/usr/bin/env python3
"""Production fragment-to-fragment, pixel-preserving stitching.

No canonical stitched image, total image, or ground-truth reference is read by
production pose inference.  ``--reference`` is an optional post-output
comparison only and is kept outside the transform-estimation functions.
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"unable to read image: {path}")
    return image


def foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    raw = ((value > 24) & ~((saturation < 28) & (value > 238))).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        return raw
    keep = np.zeros_like(raw)
    ordered = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    largest = int(stats[ordered[0], cv2.CC_STAT_AREA])
    for component in ordered[: min(4, len(ordered))]:
        if int(stats[component, cv2.CC_STAT_AREA]) >= max(256, int(largest * .002)):
            keep[labels == component] = 255
    return keep


def corners(shape: tuple[int, ...], transform: np.ndarray) -> np.ndarray:
    height, width = shape[:2]
    points = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
    return cv2.perspectiveTransform(points, transform)[0]


def decompose(transform: np.ndarray) -> dict:
    matrix = transform[:2, :2]
    return {
        "translation_xy": [float(transform[0, 2]), float(transform[1, 2])],
        "rotation_deg": float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))),
        "scale_xy": [float(np.linalg.norm(matrix[:, 0])), float(np.linalg.norm(matrix[:, 1]))],
        "perspective_terms": [float(transform[2, 0]), float(transform[2, 1])],
    }


def direct_feature_ransac(a: np.ndarray, b: np.ndarray, ratio: float, threshold: float) -> tuple[np.ndarray | None, dict]:
    """Estimate B→A from only pairwise, mutually verified SIFT correspondences."""
    gray_a, gray_b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=.01)
    key_a, desc_a = sift.detectAndCompute(gray_a, None)
    key_b, desc_b = sift.detectAndCompute(gray_b, None)
    base = {"keypoints_a": len(key_a), "keypoints_b": len(key_b)}
    if desc_a is None or desc_b is None:
        return None, {**base, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = [m for m, n in matcher.knnMatch(desc_b, desc_a, k=2) if m.distance < ratio * n.distance]
    reverse = {(m.trainIdx, m.queryIdx) for m, n in matcher.knnMatch(desc_a, desc_b, k=2) if m.distance < ratio * n.distance}
    good = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
    base["good_matches"] = len(good)
    if len(good) < 12:
        return None, {**base, "inliers": 0, "inlier_ratio": 0.0}
    source = np.float32([key_b[m.queryIdx].pt for m in good])
    target = np.float32([key_a[m.trainIdx].pt for m in good])
    transform, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, threshold)
    if transform is None or inlier_mask is None:
        return None, {**base, "inliers": 0, "inlier_ratio": 0.0}
    inliers = inlier_mask.reshape(-1).astype(bool)
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), transform).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)[inliers]
    return transform.astype(np.float64), {
        **base, "inliers": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
        "median_reprojection_error": float(np.median(errors)),
        "p95_reprojection_error": float(np.percentile(errors, 95)),
    }


def plausible_feature_geometry(transform: np.ndarray | None, evidence: dict) -> bool:
    if transform is None or evidence.get("inliers", 0) < 12 or evidence.get("inlier_ratio", 0.0) < .20:
        return False
    summary = decompose(transform)
    return (
        all(.7 <= scale <= 1.4 for scale in summary["scale_xy"])
        and abs(summary["rotation_deg"]) <= 12
        and max(abs(term) for term in summary["perspective_terms"]) <= 1e-3
    )


def edge_profile(image: np.ndarray, mask: np.ndarray, side: str, width: int = 80) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = 255.0 - gray
    strip = ink[:, :width] if side == "left" else ink[:, -width:]
    valid = mask[:, :width] > 0 if side == "left" else mask[:, -width:] > 0
    profile = (strip * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 21), 0).ravel()
    return (profile - profile.mean()) / max(profile.std(), 1e-6)


def best_vertical_shift(first: np.ndarray, second: np.ndarray, limit: int = 80) -> tuple[int, float]:
    winner = (0, float("inf"))
    for shift in range(-limit, limit + 1):
        left, right = (first[shift:], second[: len(second) - shift]) if shift >= 0 else (first[: len(first) + shift], second[-shift:])
        if len(left) < 200:
            continue
        score = float(np.mean((left - right) ** 2))
        if score < winner[1]:
            winner = (shift, score)
    return winner


def pairwise_edge_geometry(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, dict]:
    """Non-overlap fallback: infer adjacency from the submitted pair's borders only."""
    hypotheses = []
    for name, a_side, b_side, b_left in (
        ("B_LEFT_OF_A", "left", "right", True),
        ("B_RIGHT_OF_A", "right", "left", False),
    ):
        dy, score = best_vertical_shift(edge_profile(a, mask_a, a_side), edge_profile(b, mask_b, b_side))
        overlap = min(96, max(24, min(a.shape[1], b.shape[1]) // 14))
        tx = -(b.shape[1] - overlap) if b_left else a.shape[1] - overlap
        hypotheses.append((score, name, tx, dy, overlap))
    score, orientation, tx, ty, overlap = min(hypotheses, key=lambda value: value[0])
    return np.array([[1., 0., float(tx)], [0., 1., float(ty)], [0., 0., 1.]], dtype=np.float64), {
        "method": "PAIRWISE_EDGE_PROFILE_SEARCH", "orientation": orientation,
        "edge_profile_mse": score, "horizontal_overlap_px": overlap, "vertical_shift_px": ty,
    }


def write_stitched(output: Path, fragments: list[np.ndarray], masks: list[np.ndarray], transforms: list[np.ndarray]) -> tuple[Path, dict]:
    all_corners = np.concatenate([corners(image.shape, transform) for image, transform in zip(fragments, transforms)])
    minimum = np.floor(all_corners.min(axis=0) - 12).astype(int)
    maximum = np.ceil(all_corners.max(axis=0) + 12).astype(int)
    width, height = int(maximum[0] - minimum[0]), int(maximum[1] - minimum[1])
    translate = np.array([[1., 0., -minimum[0]], [0., 1., -minimum[1]], [0., 0., 1.]], dtype=np.float64)
    canvas_transforms = [translate @ transform for transform in transforms]
    warped_images, warped_masks = [], []
    for index, (image, mask, transform) in enumerate(zip(fragments, masks, canvas_transforms)):
        warped = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpPerspective(mask, transform, (width, height), flags=cv2.INTER_NEAREST)
        warped_mask = (warped_mask > 127).astype(np.uint8) * 255
        warped_images.append(warped); warped_masks.append(warped_mask)
        cv2.imwrite(str(output / f"fragment_{index}_warped.png"), warped)
        cv2.imwrite(str(output / f"fragment_{index}_mask.png"), warped_mask)
    valid_a, valid_b = warped_masks[0] > 0, warped_masks[1] > 0
    overlap, union = valid_a & valid_b, valid_a | valid_b
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    composite[valid_a & ~valid_b] = warped_images[0][valid_a & ~valid_b]
    composite[valid_b & ~valid_a] = warped_images[1][valid_b & ~valid_a]
    if overlap.any():
        distance_a = cv2.distanceTransform(warped_masks[0], cv2.DIST_L2, 3)
        distance_b = cv2.distanceTransform(warped_masks[1], cv2.DIST_L2, 3)
        weight_a = distance_a / np.maximum(distance_a + distance_b, 1e-6)
        blend = warped_images[0].astype(np.float32) * weight_a[:, :, None] + warped_images[1].astype(np.float32) * (1. - weight_a[:, :, None])
        composite[overlap] = np.clip(blend[overlap], 0, 255).astype(np.uint8)
    stitched = output / "stitched.png"
    cv2.imwrite(str(stitched), composite)
    cv2.imwrite(str(output / "stitched_rgba.png"), np.dstack([composite, union.astype(np.uint8) * 255]))
    cv2.imwrite(str(output / "stitch_mask.png"), union.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "overlap_mask.png"), overlap.astype(np.uint8) * 255)
    return stitched, {"canvas_dimensions": [width, height], "canvas_transforms": canvas_transforms,
        "overlap_pixels": int(overlap.sum()), "union_pixels": int(union.sum()),
        "overlap_ratio_of_smaller_fragment": float(overlap.sum() / max(1, min(valid_a.sum(), valid_b.sum())))}


def evaluation_only(reference: Path | None, stitched: Path) -> dict:
    if reference is None:
        return {"reference_usage": "NOT_PROVIDED"}
    reference_image, result = read_bgr(reference), read_bgr(stitched)
    resized = cv2.resize(result, (reference_image.shape[1], reference_image.shape[0]), interpolation=cv2.INTER_AREA)
    return {"reference_usage": "EVALUATION_ONLY", "reference_path": str(reference), "reference_sha256": sha256(reference),
            "resized_mae": float(cv2.absdiff(reference_image, resized).mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True); parser.add_argument("--fragment-a", required=True); parser.add_argument("--fragment-b", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--repo-root", required=True); parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--visual-threshold", type=float, default=.5)
    parser.add_argument("--textual-evidence"); parser.add_argument("--qwen-evidence")
    parser.add_argument("--reference", help="optional evaluation input; excluded from production transform inference")
    parser.add_argument("--ratio", type=float, default=.72); parser.add_argument("--ransac-threshold", type=float, default=5.0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    paths = [Path(args.fragment_a).resolve(), Path(args.fragment_b).resolve()]
    fragments = [read_bgr(path) for path in paths]; masks = [foreground_mask(image) for image in fragments]
    direct, direct_evidence = direct_feature_ransac(fragments[0], fragments[1], args.ratio, args.ransac_threshold)
    if plausible_feature_geometry(direct, direct_evidence):
        relative_b_to_a, geometry = direct, {"method": "PAIRWISE_FEATURE_RANSAC", "feature_evidence": direct_evidence}
    else:
        relative_b_to_a, edge = pairwise_edge_geometry(fragments[0], fragments[1], masks[0], masks[1])
        geometry = {"method": "PAIRWISE_EDGE_AND_FEATURE_RANSAC", "feature_evidence": direct_evidence, **edge}
    stitched, composition = write_stitched(output, fragments, masks, [np.eye(3, dtype=np.float64), relative_b_to_a])
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    from pipelines.rejoin import RejoinPipeline
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    confidence, learned_pose = RejoinPipeline(checkpoint, device=args.device).predict_pair(*fragments)
    visual_pass = float(confidence) >= args.visual_threshold
    text = json.loads(Path(args.textual_evidence).read_text(encoding="utf-8")) if args.textual_evidence else None
    textual_pass = bool(text and text.get("status") == "PASS" and text.get("decision") == "TEXTUAL_HOMOLOGY_PASS" and float(text.get("textual_homology_score", 0)) >= float(text.get("threshold", 1)))
    adjacency_pass = bool(composition["overlap_pixels"] > 0 and composition["overlap_ratio_of_smaller_fragment"] < .80)
    feature = geometry["feature_evidence"]
    geometry_pass = bool(feature.get("inliers", 0) >= 12 and (geometry["method"] == "PAIRWISE_FEATURE_RANSAC" or geometry.get("edge_profile_mse", float("inf")) < 2.0))
    composition_pass = bool(composition["union_pixels"] > 0 and adjacency_pass)
    reference = Path(args.reference).resolve() if args.reference else None
    metadata = {
        "case_id": args.case_id, "status": "PASS" if (visual_pass or textual_pass) and geometry_pass and composition_pass else "FAIL",
        "PAIRWISE_TRANSFORM_SOURCE": "FRAGMENT_TO_FRAGMENT", "REFERENCE_USAGE": "EVALUATION_ONLY" if reference else "NOT_PROVIDED",
        "production_reference_read": False, "transform_type": geometry["method"], "pairwise_geometry": geometry,
        "relative_transform_b_to_a": relative_b_to_a.tolist(), "relative_transform_summary": decompose(relative_b_to_a),
        "homology_score": float(confidence), "visual_homology_decision": "MATCH" if visual_pass else "LOW_CONFIDENCE_ROUTE_TO_TEXT",
        "homology_branch_used": "VISUAL_GPU" if visual_pass else "TEXTUAL_OCR_CBETA" if textual_pass else "NONE", "learned_pose_raw": learned_pose.tolist(),
        "adjacency_gate": "PASS" if adjacency_pass else "FAIL", "geometry_gate": "PASS" if geometry_pass else "FAIL", "composition_gate": "PASS" if composition_pass else "FAIL",
        **{key: value for key, value in composition.items() if key != "canvas_transforms"},
        "fragments": [{"fragment": str(path), "sha256": sha256(path), "input_dimensions": [image.shape[1], image.shape[0]], "canvas_transform": transform.tolist()} for path, image, transform in zip(paths, fragments, composition["canvas_transforms"])],
        "stitched_path": str(stitched), "stitched_sha256": sha256(stitched), "evaluation": evaluation_only(reference, stitched),
        "source_pixel_policy": "COORDINATE_TRANSFORM_AND_CONTROLLED_OVERLAP_BLEND_ONLY", "restoration_performed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    metadata["stitch_pass"] = metadata["status"] == "PASS"
    (output / "transform_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if metadata["stitch_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
