from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from adapters.autohdr_official import OFFICIAL_COMMIT
from adapters.autohdr_official.candidate_exporter import finalize_candidate_bundle
from adapters.autohdr_official.common import AdapterContractError, canonical_sha256
from adapters.autohdr_official.plan_validator import validate_plan
from adapters.autohdr_official.rag_enricher import enrich_candidate_bundle
from adapters.autohdr_official.safe_compositor import composite_approved


def make_bundle() -> dict:
    return finalize_candidate_bundle(
        {
            "case_id": "case-unit",
            "image_sha256": "a" * 64,
            "official_commit": OFFICIAL_COMMIT,
            "worker_version": "unit",
            "damage_regions": [
                {
                    "region_id": name,
                    "bbox": bbox,
                    "mask_path": f"masks/{name}.png",
                    "crop_path": f"crops/{name}.png",
                    "ocr_before": "前",
                    "ocr_after": "後",
                    "ocr_raw": "",
                    "autohdr_candidate": "字",
                    "autohdr_confidence": None,
                    "reading_order": index,
                }
                for index, (name, bbox) in enumerate(
                    [
                        ("approved", [1, 1, 3, 3]),
                        ("rejected", [4, 1, 6, 3]),
                        ("abstained", [7, 1, 9, 3]),
                    ]
                )
            ],
        }
    )


def make_plan(bundle: dict) -> dict:
    core = {
        "case_id": bundle["case_id"],
        "image_sha256": bundle["image_sha256"],
        "analysis_run_id": "analysis-unit",
        "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
        "rag_version": "rag-unit",
        "cbeta_source_commit": "cbeta-unit",
        "plan_version": 1,
        "approved_regions": [
            {
                "region_id": "approved",
                "decision": "manual_modify",
                "selected_text": "寶",
                "selected_source": "human",
                "candidate_id": "",
                "citation_hash": "",
                "note": "unit",
            }
        ],
        "rejected_regions": [
            {"region_id": "rejected", "decision": "reject", "selected_text": ""}
        ],
        "abstained_regions": [
            {"region_id": "abstained", "decision": "abstain", "selected_text": ""}
        ],
        "manual_edits": [],
        "evidence_citations": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "actor": "unit-reviewer",
        "request_id": "unit-request",
        "request_payload_hash": "b" * 64,
        "allow_restore": True,
    }
    return {**core, "plan_sha256": canonical_sha256(core)}


class AdapterContractTests(unittest.TestCase):
    def test_mixed_decisions_validate_and_tamper_fails(self):
        bundle = make_bundle()
        plan = make_plan(bundle)
        result = validate_plan(
            plan,
            bundle,
            expected_image_sha256="a" * 64,
            expected_rag_version="rag-unit",
            expected_cbeta_commit="cbeta-unit",
        )
        self.assertEqual(result["approved_region_ids"], ["approved"])
        self.assertEqual(result["rejected_region_ids"], ["rejected"])
        self.assertEqual(result["abstained_region_ids"], ["abstained"])
        tampered = json.loads(json.dumps(plan))
        tampered["approved_regions"][0]["selected_text"] = "偽"
        with self.assertRaisesRegex(AdapterContractError, "PLAN_SHA256_MISMATCH"):
            validate_plan(
                tampered,
                bundle,
                expected_image_sha256="a" * 64,
                expected_rag_version="rag-unit",
                expected_cbeta_commit="cbeta-unit",
            )

    def test_rag_enrichment_never_selects(self):
        enriched = enrich_candidate_bundle(
            make_bundle(),
            lambda request: {
                "candidates": [{"candidate_text": "寶", "citation": "T0278"}],
                "decision_status": "candidate",
                "latency_ms": 1.0,
                "source_commit": "cbeta-unit",
            },
        )
        self.assertFalse(enriched["automatic_selection_performed"])
        self.assertTrue(
            all(
                region["decision_status"] == "pending_human_review"
                for region in enriched["damage_regions"]
            )
        )

    def test_compositor_changes_only_mask(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = np.zeros((8, 10, 3), dtype=np.uint8) + 20
            generated = np.zeros_like(original) + 240
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[2:5, 3:7] = 255
            Image.fromarray(original).save(root / "original.png")
            Image.fromarray(generated).save(root / "generated.png")
            Image.fromarray(mask).save(root / "mask.png")
            report = composite_approved(
                root / "original.png",
                root / "generated.png",
                root / "mask.png",
                root / "out",
            )
            final = np.asarray(Image.open(root / "out/constrained_restored.png"))
            self.assertEqual(report["outside_mask_changed_pixels"], 0)
            self.assertTrue(np.all(final[mask == 0] == original[mask == 0]))
            self.assertTrue(np.all(final[mask > 0] == generated[mask > 0]))
            self.assertEqual(report["human_review_status"], "not_reviewed")

    def test_duplicate_region_id_is_rejected(self):
        payload = make_bundle()
        payload.pop("candidate_bundle_sha256")
        payload["damage_regions"].append(dict(payload["damage_regions"][0]))
        with self.assertRaisesRegex(AdapterContractError, "Duplicate region_id"):
            finalize_candidate_bundle(payload)

    def test_server_allows_approved_region_with_negative_controls(self):
        try:
            from starlette.requests import Request
        except ImportError:
            self.skipTest("FastAPI is validated in the pinned server environment")
        import dunhuang_backend.app as backend
        from dunhuang_backend.db import StateStore, utcnow

        class RagStub:
            @staticmethod
            def health():
                return {
                    "index_version": "rag-unit",
                    "source_commit": "cbeta-unit",
                }

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            backend.store = StateStore(root / "state.sqlite3")
            backend.store.initialize()
            backend.CASE_ROOT = root / "cases"
            backend.CASE_ROOT.mkdir()
            backend._rag = RagStub()
            case_id = f"case-{uuid.uuid4()}"
            with backend.store.transaction() as db:
                db.execute(
                    "INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        case_id,
                        "EVIDENCE_READY",
                        utcnow(),
                        utcnow(),
                        "unit.png",
                        "a" * 64,
                        "image/png",
                        10,
                        8,
                        None,
                        None,
                    ),
                )
                db.execute(
                    "INSERT INTO analysis_runs VALUES(?,?,?,?,?,?,?)",
                    (
                        "analysis-unit",
                        case_id,
                        "COMPLETE",
                        utcnow(),
                        utcnow(),
                        "CONTRACT_FIXTURE",
                        "{}",
                    ),
                )
                for index, region_id in enumerate(
                    ["approved", "rejected", "abstained"]
                ):
                    db.execute(
                        "INSERT INTO damage_regions VALUES(?,?,?,?,?)",
                        (
                            region_id,
                            case_id,
                            json.dumps([index, 0, index + 1, 1]),
                            "CONTRACT_FIXTURE",
                            "[]",
                        ),
                    )
            bundle = backend.build_candidate_bundle(case_id)
            body = {
                "decisions": [
                    {
                        "region_id": "approved",
                        "decision": "manual_modify",
                        "selected_text": "寶",
                        "selected_source": "human",
                        "note": "explicit unit decision",
                    },
                    {"region_id": "rejected", "decision": "reject"},
                    {"region_id": "abstained", "decision": "abstain"},
                ],
                "reviewer": "unit-reviewer",
                "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
                "request_id": f"request-{uuid.uuid4()}",
            }
            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": f"/api/cases/{case_id}/decisions",
                    "headers": [],
                    "client": ("127.0.0.1", 12345),
                    "server": ("127.0.0.1", 8000),
                    "scheme": "http",
                    "query_string": b"",
                }
            )
            response = backend.decisions(
                case_id, backend.DecisionBatch.model_validate(body), request
            )
            self.assertTrue(response["allow_restore"])
            stored = backend.latest_plan(case_id)["plan"]
            self.assertEqual(len(stored["approved_regions"]), 1)
            self.assertEqual(len(stored["rejected_regions"]), 1)
            self.assertEqual(len(stored["abstained_regions"]), 1)


if __name__ == "__main__":
    unittest.main()
