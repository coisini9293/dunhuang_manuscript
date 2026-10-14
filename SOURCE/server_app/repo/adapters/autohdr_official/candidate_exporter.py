from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import ADAPTER_VERSION, OFFICIAL_COMMIT
from .common import AdapterContractError, canonical_sha256, write_json


REQUIRED_REGION_FIELDS = {
    "region_id",
    "bbox",
    "mask_path",
    "crop_path",
    "ocr_before",
    "ocr_after",
    "ocr_raw",
    "autohdr_candidate",
    "autohdr_confidence",
    "reading_order",
}


def finalize_candidate_bundle(payload: dict, output_path: Path | None = None) -> dict:
    core = deepcopy(payload)
    core.pop("candidate_bundle_sha256", None)
    core.setdefault("worker_version", ADAPTER_VERSION)
    core.setdefault("official_commit", OFFICIAL_COMMIT)
    required = {
        "case_id",
        "image_sha256",
        "official_commit",
        "worker_version",
        "damage_regions",
    }
    missing = sorted(required - set(core))
    if missing:
        raise AdapterContractError(f"Candidate bundle missing fields: {missing}")
    if core["official_commit"] != OFFICIAL_COMMIT:
        raise AdapterContractError("Candidate bundle Official commit mismatch")
    region_ids: set[str] = set()
    for region in core["damage_regions"]:
        absent = sorted(REQUIRED_REGION_FIELDS - set(region))
        if absent:
            raise AdapterContractError(
                f"Region is missing fields {absent}: {region.get('region_id')}"
            )
        if region["region_id"] in region_ids:
            raise AdapterContractError(f"Duplicate region_id: {region['region_id']}")
        region_ids.add(region["region_id"])
        bbox = region["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise AdapterContractError(f"Invalid xyxy bbox: {bbox}")
    bundle = {**core, "candidate_bundle_sha256": canonical_sha256(core)}
    if output_path is not None:
        write_json(output_path, bundle)
    return bundle

