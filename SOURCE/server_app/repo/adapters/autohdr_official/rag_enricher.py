from __future__ import annotations

from copy import deepcopy
from typing import Callable

from .common import AdapterContractError


def enrich_candidate_bundle(
    bundle: dict, query_region: Callable[[dict], dict]
) -> dict:
    """Attach RAG evidence without selecting or accepting any candidate."""
    enriched = deepcopy(bundle)
    for region in enriched.get("damage_regions", []):
        request = {
            "region_id": region["region_id"],
            "context_before": region.get("ocr_before", ""),
            "context_after": region.get("ocr_after", ""),
            "visible_fragment": region.get("ocr_raw", ""),
        }
        result = query_region(request)
        if not isinstance(result, dict):
            raise AdapterContractError("RAG callback returned a non-object")
        region["rag_evidence"] = {
            "candidates": result.get("candidates", []),
            "decision_status": result.get("decision_status", "abstain"),
            "latency_ms": result.get("latency_ms"),
            "corpus_version": result.get("corpus_version"),
            "index_version": result.get("index_version"),
            "source_commit": result.get("source_commit"),
            "retrieval_components": result.get("retrieval_components", {}),
            "abstention_reason": result.get("abstention_reason", ""),
        }
        region["decision_status"] = "pending_human_review"
    enriched["automatic_selection_performed"] = False
    return enriched

