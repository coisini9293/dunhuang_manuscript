from __future__ import annotations

from datetime import datetime, timezone

from .common import AdapterContractError, canonical_sha256


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AdapterContractError("Invalid confirmed_at timestamp") from exc
    if parsed.tzinfo is None:
        raise AdapterContractError("confirmed_at must include timezone")
    return parsed.astimezone(timezone.utc)


def validate_plan(
    plan: dict,
    bundle: dict,
    *,
    expected_image_sha256: str,
    expected_rag_version: str,
    expected_cbeta_commit: str,
    max_age_seconds: int = 86_400,
    now: datetime | None = None,
) -> dict:
    supplied_hash = plan.get("plan_sha256", "")
    core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if not supplied_hash or canonical_sha256(core) != supplied_hash:
        raise AdapterContractError("PLAN_SHA256_MISMATCH")
    if plan.get("allow_restore") is not True:
        raise AdapterContractError("PLAN_DOES_NOT_ALLOW_RESTORE")
    checks = {
        "case_id": bundle.get("case_id"),
        "image_sha256": expected_image_sha256,
        "candidate_bundle_sha256": bundle.get("candidate_bundle_sha256"),
        "rag_version": expected_rag_version,
        "cbeta_source_commit": expected_cbeta_commit,
    }
    for field, expected in checks.items():
        if plan.get(field) != expected:
            raise AdapterContractError(f"{field.upper()}_MISMATCH")

    region_map = {
        item["region_id"]: item for item in bundle.get("damage_regions", [])
    }
    approved = plan.get("approved_regions", [])
    rejected = plan.get("rejected_regions", [])
    abstained = plan.get("abstained_regions", [])
    if not approved:
        raise AdapterContractError("NO_APPROVED_REGIONS")
    all_decisions = approved + rejected + abstained
    decided_ids = [item.get("region_id") for item in all_decisions]
    if len(decided_ids) != len(set(decided_ids)):
        raise AdapterContractError("DUPLICATE_REGION_DECISION")
    if set(decided_ids) != set(region_map):
        raise AdapterContractError("INCOMPLETE_REGION_DECISIONS")
    for item in approved:
        if item.get("decision") not in {"accept", "manual_modify"}:
            raise AdapterContractError("INVALID_APPROVED_DECISION")
        text = item.get("selected_text")
        if not isinstance(text, str) or not text.strip():
            raise AdapterContractError("APPROVED_TEXT_EMPTY")
    for item in rejected:
        if item.get("decision") != "reject" or item.get("selected_text"):
            raise AdapterContractError("INVALID_REJECTED_DECISION")
    for item in abstained:
        if item.get("decision") != "abstain" or item.get("selected_text"):
            raise AdapterContractError("INVALID_ABSTAINED_DECISION")

    confirmed = _parse_time(plan.get("confirmed_at", ""))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (current - confirmed).total_seconds()
    if age < -300 or age > max_age_seconds:
        raise AdapterContractError("PLAN_EXPIRED")
    return {
        "valid": True,
        "plan_sha256": supplied_hash,
        "approved_region_ids": [item["region_id"] for item in approved],
        "rejected_region_ids": [item["region_id"] for item in rejected],
        "abstained_region_ids": [item["region_id"] for item in abstained],
        "confirmed_texts": {
            item["region_id"]: item["selected_text"] for item in approved
        },
    }

