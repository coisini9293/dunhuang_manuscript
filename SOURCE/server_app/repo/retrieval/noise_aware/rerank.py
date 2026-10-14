from __future__ import annotations

from .fusion import candidate_key


def rerank_candidates(candidates: list[dict], variant_count: int) -> list[dict]:
    if not candidates:
        return []
    maximum_rrf = max(float(item["_rrf_score"]) for item in candidates) or 1.0
    ranked = []
    for candidate in candidates:
        components = candidate.get("retrieval_components", {})
        continuity = float(components.get("continuity", 0.0))
        edit = float(components.get("edit", 0.0))
        citation = float(components.get("citation", 0.0))
        consensus = len(candidate["_origins"]) / max(variant_count, 1)
        baseline_ranks = [
            item["rank"]
            for item in candidate["_origins"]
            if item["variant_id"] == "qv-01"
        ]
        baseline_anchor = (
            1.0 / min(baseline_ranks) if baseline_ranks else 0.0
        )
        rrf_normalized = float(candidate["_rrf_score"]) / maximum_rrf
        final = (
            0.52 * rrf_normalized
            + 0.18 * continuity
            + 0.10 * edit
            + 0.05 * citation
            + 0.03 * consensus
            + 0.12 * baseline_anchor
        )
        audit = {
            "rrf": round(float(candidate.pop("_rrf_score")), 9),
            "rrf_normalized": round(rrf_normalized, 6),
            "continuity": round(continuity, 6),
            "edit_similarity": round(edit, 6),
            "citation_complete": round(citation, 6),
            "variant_consensus": round(consensus, 6),
            "baseline_anchor": round(baseline_anchor, 6),
            "formula": "0.52*rrf+0.18*continuity+0.10*edit+0.05*citation+0.03*consensus+0.12*baseline_anchor",
            "final": round(final, 6),
        }
        candidate["noise_aware_origins"] = candidate.pop("_origins")
        candidate["noise_aware_rerank"] = audit
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: (
            -item["noise_aware_rerank"]["final"],
            candidate_key(item),
        )
    )
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank
    return ranked
