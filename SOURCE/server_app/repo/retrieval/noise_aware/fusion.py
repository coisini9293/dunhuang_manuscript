from __future__ import annotations

from collections import defaultdict


RRF_K = 60


def candidate_key(candidate: dict) -> tuple:
    return (
        candidate.get("sutra_id", ""),
        candidate.get("volume", 0),
        candidate.get("chapter_name", ""),
        candidate.get("absolute_gap_start", 0),
        candidate.get("absolute_gap_end", 0),
        candidate.get("candidate_text", ""),
        candidate.get("text_hash", ""),
    )


def reciprocal_rank_fusion(
    result_sets: list[tuple[str, list[dict]]], *, rrf_k: int = RRF_K
) -> list[dict]:
    fused: dict[tuple, dict] = {}
    origins: dict[tuple, list[dict]] = defaultdict(list)
    scores: dict[tuple, float] = defaultdict(float)
    for variant_id, candidates in result_sets:
        for rank, candidate in enumerate(candidates, 1):
            key = candidate_key(candidate)
            fused.setdefault(key, dict(candidate))
            contribution = 1.0 / (rrf_k + rank)
            scores[key] += contribution
            origins[key].append(
                {
                    "variant_id": variant_id,
                    "rank": rank,
                    "rrf_contribution": round(contribution, 9),
                }
            )
    output = []
    for key, candidate in fused.items():
        candidate["_rrf_score"] = scores[key]
        candidate["_origins"] = origins[key]
        output.append(candidate)
    return sorted(
        output,
        key=lambda item: (
            -item["_rrf_score"],
            candidate_key(item),
        ),
    )
