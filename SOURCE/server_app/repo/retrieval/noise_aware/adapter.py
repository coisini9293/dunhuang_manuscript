from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .fusion import reciprocal_rank_fusion
from .query_variants import build_query_variants
from .rerank import rerank_candidates


class NoiseAwareRetriever:
    """Optional adapter; baseline mode delegates exactly once to RAG V2."""

    MODES = {"baseline", "noise_aware"}

    def __init__(self, retriever, *, confusion_map_path=None, max_variants: int = 12):
        self.retriever = retriever
        self.confusion_map_path = confusion_map_path
        self.max_variants = max_variants

    def query(
        self,
        context_before: str,
        context_after: str,
        *,
        retrieval_mode: str = "baseline",
        **kwargs: Any,
    ) -> dict:
        if retrieval_mode not in self.MODES:
            raise ValueError(
                f"retrieval_mode must be one of {sorted(self.MODES)}"
            )
        if retrieval_mode == "baseline":
            return self.retriever.query(
                context_before, context_after, **kwargs
            ).to_dict()

        started = time.perf_counter()
        variants, audit = build_query_variants(
            context_before,
            context_after,
            max_variants=self.max_variants,
            confusion_map_path=self.confusion_map_path,
        )
        result_sets: list[tuple[str, list[dict]]] = []
        executions = []
        statuses: list[str] = []
        for variant in variants:
            variant_started = time.perf_counter()
            result = self.retriever.query(
                variant.before, variant.after, **kwargs
            ).to_dict()
            candidates = result.get("candidates", [])
            result_sets.append((variant.variant_id, candidates))
            statuses.append(result.get("decision_status", "abstain"))
            executions.append(
                {
                    **variant.to_dict(),
                    "query_id": result.get("query_id", ""),
                    "decision_status": result.get("decision_status", "abstain"),
                    "candidate_count": len(candidates),
                    "latency_ms": round(
                        (time.perf_counter() - variant_started) * 1000, 3
                    ),
                }
            )
        top_k = int(kwargs.get("top_k", 5))
        fused = reciprocal_rank_fusion(result_sets)
        candidates = rerank_candidates(fused, len(variants))[:top_k]
        if not candidates:
            status = "abstain"
        elif context_before and context_after:
            status = (
                "retrieved_exact"
                if candidates[0].get("decision_status") == "retrieved_exact"
                else "retrieved_ambiguous"
            )
        else:
            status = "single_side_localization"
        query_id = hashlib.sha256(
            json.dumps(
                {
                    "mode": retrieval_mode,
                    "variants": [item.to_dict() for item in variants],
                    "options": kwargs,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        health = self.retriever.health()
        return {
            "query_id": query_id,
            "decision_status": status,
            "candidates": candidates,
            "normalized_before": audit["normalized_query"]["context_before"],
            "normalized_after": audit["normalized_query"]["context_after"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "vector_enabled": bool(health.get("vector_enabled", False)),
            "corpus_version": health.get("corpus_version", ""),
            "index_version": health.get("index_version", ""),
            "retrieval_mode": retrieval_mode,
            "raw_query": audit["raw_query"],
            "normalized_query": audit["normalized_query"],
            "applied_rules": audit["applied_rules"],
            "query_variants": executions,
            "fusion_method": {"name": "reciprocal_rank_fusion", "k": 60},
            "rerank_method": {
                "name": "explainable_continuity_rerank",
                "automatic_accept": False,
            },
            "confusion_map": audit["confusion_map"],
            "variant_count": len(variants),
            "variant_statuses": statuses,
        }
