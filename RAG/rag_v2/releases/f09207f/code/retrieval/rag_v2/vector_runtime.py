from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=2)
def _load_runtime(index_dir_text: str):
    import faiss
    from sentence_transformers import SentenceTransformer

    index_dir = Path(index_dir_text)
    manifest = json.loads(
        (index_dir / "vector_manifest.json").read_text(encoding="utf-8")
    )
    model_path = Path(manifest["model_local_path"])
    if not model_path.is_absolute():
        model_path = (index_dir / model_path).resolve()
    model = SentenceTransformer(
        str(model_path),
        device="cpu",
        local_files_only=True,
    )
    # FAISS's Windows C++ file API is not Unicode-path safe.
    serialized = np.frombuffer(
        (index_dir / "index.faiss").read_bytes(),
        dtype=np.uint8,
    )
    index = faiss.deserialize_index(serialized)
    index.hnsw.efSearch = int(manifest["faiss"]["ef_search"])
    chunk_ids = json.loads((index_dir / "vector_ids.json").read_text(encoding="utf-8"))
    return model, index, chunk_ids, manifest


def _query_variants(query: str) -> list[str]:
    if "[缺損]" not in query:
        return [query]
    before, after = query.split("[缺損]", 1)
    variants = [query]
    if before:
        variants.append(before)
    if after:
        variants.append(after)
    return variants


def search_faiss(
    index_dir: str | Path,
    query: str,
    limit: int = 100,
) -> dict[str, dict]:
    model, index, chunk_ids, _manifest = _load_runtime(str(Path(index_dir).resolve()))
    variants = _query_variants(query)
    vectors = model.encode(
        variants,
        batch_size=len(variants),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    per_variant_limit = min(max(limit, 1), index.ntotal)
    scores, ids = index.search(vectors, per_variant_limit)
    merged: dict[str, dict] = {}
    for variant_index, (variant_scores, variant_ids) in enumerate(zip(scores, ids)):
        for rank, (score, vector_id) in enumerate(
            zip(variant_scores.tolist(), variant_ids.tolist()), 1
        ):
            if vector_id < 0:
                continue
            chunk_id = chunk_ids[vector_id]
            current = merged.get(chunk_id)
            if current is None or float(score) > current["vector_similarity"]:
                merged[chunk_id] = {
                    "vector_similarity": max(0.0, min(1.0, float(score))),
                    "vector_raw_score": float(score),
                    "vector_variant": variant_index,
                    "vector_variant_rank": rank,
                }
    ordered = sorted(
        merged.items(),
        key=lambda item: item[1]["vector_similarity"],
        reverse=True,
    )[:limit]
    return {
        chunk_id: {**details, "vector_rank": rank}
        for rank, (chunk_id, details) in enumerate(ordered, 1)
    }
