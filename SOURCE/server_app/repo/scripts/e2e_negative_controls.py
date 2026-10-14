from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from adapters.autohdr_official.common import AdapterContractError, canonical_sha256
from adapters.autohdr_official.plan_validator import validate_plan


def expected_failure(name: str, function, expected: str) -> dict:
    try:
        function()
    except AdapterContractError as exc:
        actual = str(exc)
        if expected not in actual:
            raise AssertionError(f"{name}: expected {expected}, received {actual}")
        return {"name": name, "status": "PASS", "rejection": actual}
    raise AssertionError(f"{name}: mutation was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--worker-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text("utf-8"))
    plan = json.loads(args.plan.read_text("utf-8"))
    worker = json.loads(args.worker_result.read_text("utf-8"))

    validate_kwargs = {
        "expected_image_sha256": plan["image_sha256"],
        "expected_rag_version": plan["rag_version"],
        "expected_cbeta_commit": plan["cbeta_source_commit"],
    }
    baseline = validate_plan(plan, bundle, **validate_kwargs)
    tests = [{"name": "baseline_plan", "status": "PASS", **baseline}]

    unconfirmed = copy.deepcopy(plan)
    unconfirmed["allow_restore"] = False
    unconfirmed["plan_sha256"] = canonical_sha256(
        {key: value for key, value in unconfirmed.items() if key != "plan_sha256"}
    )
    tests.append(
        expected_failure(
            "unconfirmed_plan",
            lambda: validate_plan(unconfirmed, bundle, **validate_kwargs),
            "PLAN_DOES_NOT_ALLOW_RESTORE",
        )
    )

    forged = copy.deepcopy(plan)
    forged["manual_confirmed"] = True
    tests.append(
        expected_failure(
            "forged_manual_confirmed",
            lambda: validate_plan(forged, bundle, **validate_kwargs),
            "PLAN_SHA256_MISMATCH",
        )
    )

    tampered = copy.deepcopy(plan)
    tampered["approved_regions"][0]["selected_text"] = "偽"
    tests.append(
        expected_failure(
            "tampered_confirmed_text",
            lambda: validate_plan(tampered, bundle, **validate_kwargs),
            "PLAN_SHA256_MISMATCH",
        )
    )

    stale_bundle = copy.deepcopy(bundle)
    stale_bundle["candidate_bundle_sha256"] = "0" * 64
    tests.append(
        expected_failure(
            "stale_candidate_bundle",
            lambda: validate_plan(plan, stale_bundle, **validate_kwargs),
            "CANDIDATE_BUNDLE_SHA256_MISMATCH",
        )
    )

    incomplete = copy.deepcopy(plan)
    incomplete["abstained_regions"].pop()
    incomplete["plan_sha256"] = canonical_sha256(
        {key: value for key, value in incomplete.items() if key != "plan_sha256"}
    )
    tests.append(
        expected_failure(
            "missing_region_decision",
            lambda: validate_plan(incomplete, bundle, **validate_kwargs),
            "INCOMPLETE_REGION_DECISIONS",
        )
    )

    approved = {item["region_id"] for item in plan["approved_regions"]}
    patched = {item["region_id"] for item in worker["patches"]}
    abstained = {item["region_id"] for item in plan["abstained_regions"]}
    rejected = {item["region_id"] for item in plan["rejected_regions"]}
    if patched != approved:
        raise AssertionError("Patch set differs from approved set")
    if patched & abstained or patched & rejected:
        raise AssertionError("Negative-control region generated a patch")
    tests.extend(
        [
            {
                "name": "approved_patch_set_exact",
                "status": "PASS",
                "approved_count": len(approved),
                "patch_count": len(patched),
            },
            {
                "name": "abstained_regions_no_patch",
                "status": "PASS",
                "abstained_count": len(abstained),
            },
            {
                "name": "rejected_regions_no_patch",
                "status": "PASS",
                "rejected_count": len(rejected),
                "note": "No rejected or abstained region generated a repair patch.",
            },
        ]
    )

    # Exercise the HTTP gate in an isolated state directory. A client-supplied
    # manual_confirmed field cannot create a server plan.
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        os.environ["DUNHUANG_BASE"] = str(temp_root)
        os.environ["STATE_DB"] = str(temp_root / "state.sqlite3")
        os.environ["CASE_ROOT"] = str(temp_root / "cases")
        import inspect
        from fastapi import HTTPException
        from dunhuang_backend import app as backend

        backend.get_case = lambda case_id: {"case_id": case_id}
        backend.latest_plan = lambda case_id: None
        if "manual_confirmed" in inspect.signature(backend.restore).parameters:
            raise AssertionError("Restore endpoint accepts forged manual_confirmed")
        try:
            backend.restore("control", None)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise AssertionError(
                    f"Forged HTTP gate expected 409, received {exc.status_code}"
                )
            http_status = exc.status_code
            detail = exc.detail
        else:
            raise AssertionError("Restore without server plan was not rejected")
        tests.append(
            {
                "name": "http_forged_manual_confirmed",
                "status": "PASS",
                "http_status": http_status,
                "detail": detail,
                "note": "Endpoint has no manual_confirmed parameter.",
            }
        )

    report = {
        "status": "E2E_NEGATIVE_CONTROLS_PASS",
        "test_count": len(tests),
        "tests": tests,
        "human_review_status": "not_reviewed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
