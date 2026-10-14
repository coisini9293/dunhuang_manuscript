from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, REPORT_ROOT, T0278_ARTIFACT, T0278_DB, now_iso, require_disk, write_json


REQUESTED_MODEL_ID = "BAAI/bge-m3"
REQUESTED_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
MODEL_WEIGHT_FILE = "model.safetensors"
MODEL_WEIGHT_BYTES = 95_827_648
MODEL_WEIGHT_SHA256 = "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026"
MODEL_DIR = PROJECT_ROOT / "models" / "embeddings" / f"bge-small-zh-v1.5-{MODEL_REVISION[:12]}"
VECTOR_DIR = T0278_ARTIFACT / "faiss"
CACHE_DIR = PROJECT_ROOT / "cache" / "rag_v2" / "vector_build"
EVAL_SET = PROJECT_ROOT / "cache" / "rag_v2" / "t0278_eval_cases.jsonl"
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model() -> dict:
    import httpx
    from huggingface_hub import snapshot_download

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    weights_ready = (MODEL_DIR / MODEL_WEIGHT_FILE).exists()
    if not (MODEL_DIR / "modules.json").exists() or not weights_ready:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIR,
            allow_patterns=[
                "1_Pooling/*",
                "config.json",
                "config_sentence_transformers.json",
                "modules.json",
                "sentence_bert_config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            ],
        )
    weight_path = MODEL_DIR / MODEL_WEIGHT_FILE
    partial_path = MODEL_DIR / f"{MODEL_WEIGHT_FILE}.part"
    if not weight_path.exists():
        downloaded = partial_path.stat().st_size if partial_path.exists() else 0
        url = (
            f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"
            + MODEL_WEIGHT_FILE
        )
        # Fixed range requests prevent the local HTTP proxy from buffering a
        # multi-gigabyte response and make every completed block resumable.
        range_size = 32 * 1024 * 1024
        with httpx.Client(
            timeout=httpx.Timeout(60.0, read=120.0),
            follow_redirects=True,
        ) as client, partial_path.open("ab") as handle:
            while downloaded < MODEL_WEIGHT_BYTES:
                end = min(downloaded + range_size, MODEL_WEIGHT_BYTES) - 1
                last_error: Exception | None = None
                for attempt in range(1, 6):
                    try:
                        with client.stream(
                            "GET",
                            url,
                            headers={"Range": f"bytes={downloaded}-{end}"},
                        ) as response:
                            response.raise_for_status()
                            if response.status_code != 206:
                                raise RuntimeError(
                                    f"range request returned {response.status_code}"
                                )
                            for block in response.iter_bytes(8 * 1024 * 1024):
                                handle.write(block)
                                downloaded += len(block)
                            handle.flush()
                        last_error = None
                        break
                    except Exception as error:
                        last_error = error
                        print(
                            f"model range retry {attempt}/5 at {downloaded}: {error}",
                            flush=True,
                        )
                if last_error is not None:
                    raise last_error
                print(
                    f"model: {downloaded}/{MODEL_WEIGHT_BYTES} bytes",
                    flush=True,
                )
        if partial_path.stat().st_size != MODEL_WEIGHT_BYTES:
            raise RuntimeError(
                f"model size mismatch: {partial_path.stat().st_size} != {MODEL_WEIGHT_BYTES}"
            )
        if sha256_file(partial_path) != MODEL_WEIGHT_SHA256:
            raise RuntimeError("model SHA-256 mismatch")
        partial_path.replace(weight_path)
    files = []
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append(
                {
                    "path": path.relative_to(MODEL_DIR).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "requested_model_id": REQUESTED_MODEL_ID,
        "requested_model_revision": REQUESTED_MODEL_REVISION,
        "fallback": True,
        "fallback_reason": (
            "bge-m3 is not reasonable on this CPU-only host: its 2.27 GB weight "
            "download measured about 0.2-0.3 MiB/s and the planned body/metadata "
            "ablation requires 9,090 encodes with a 568M-parameter model."
        ),
        "local_path": str(MODEL_DIR),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }


def load_chunks() -> list[dict]:
    connection = sqlite3.connect(T0278_DB)
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT chunk_id, sutra_id, canonical_title, translator, volume,
                       page_start, column_start, line_start, normalized_text, text_hash
                FROM chunks WHERE scale='medium' ORDER BY chunk_id
                """
            )
        ]
    finally:
        connection.close()


def embedding_text(chunk: dict, mode: str) -> str:
    if mode == "body":
        return chunk["normalized_text"]
    return (
        f"經名：{chunk['canonical_title']}\n"
        f"譯者：{chunk['translator']}\n"
        f"正文：{chunk['normalized_text']}"
    )


def encode_mode(model, chunks: list[dict], mode: str, resume: bool) -> tuple[np.ndarray, dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vector_path = CACHE_DIR / f"t0278_{mode}.f32.npy"
    progress_path = CACHE_DIR / f"t0278_{mode}_progress.json"
    dimension = model.get_embedding_dimension()
    if vector_path.exists() and progress_path.exists() and resume:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete") and progress.get("count") == len(chunks):
            return np.load(vector_path, mmap_mode="r"), progress

    vectors = np.lib.format.open_memmap(
        vector_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(chunks), dimension),
    )
    started = time.perf_counter()
    for offset in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[offset : offset + BATCH_SIZE]
        encoded = model.encode(
            [embedding_text(chunk, mode) for chunk in batch],
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        vectors[offset : offset + len(batch)] = encoded
        vectors.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "count": len(chunks),
                    "completed": offset + len(batch),
                    "complete": offset + len(batch) == len(chunks),
                    "dimension": dimension,
                    "updated_at": now_iso(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if offset % (BATCH_SIZE * 25) == 0:
            print(f"{mode}: {offset + len(batch)}/{len(chunks)}", flush=True)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["seconds"] = round(time.perf_counter() - started, 3)
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return np.load(vector_path, mmap_mode="r"), progress


def load_eval_queries(limit: int = 160) -> list[dict]:
    cases = [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = [
        case
        for case in cases
        if case["expected"] != "abstain" and case["case_type"] != "filtered"
    ]
    # Fixed stratified prefix; no metric-based case selection.
    per_type: dict[str, list[dict]] = {}
    for case in eligible:
        per_type.setdefault(case["case_type"], []).append(case)
    selected: list[dict] = []
    while len(selected) < min(limit, len(eligible)):
        changed = False
        for case_type in sorted(per_type):
            if per_type[case_type]:
                selected.append(per_type[case_type].pop(0))
                changed = True
                if len(selected) >= limit:
                    break
        if not changed:
            break
    return selected


def vector_recall_score(
    model,
    vectors: np.ndarray,
    chunks: list[dict],
    cases: list[dict],
) -> dict:
    queries = [
        f"{case['query_before']}[缺損]{case['query_after']}"
        for case in cases
    ]
    query_vectors = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    matrix = np.asarray(vectors)
    hits = {1: 0, 3: 0, 5: 0, 100: 0}
    for case, query_vector in zip(cases, query_vectors):
        scores = matrix @ query_vector
        top = np.argpartition(scores, -min(100, len(scores)))[-min(100, len(scores)) :]
        top = top[np.argsort(scores[top])[::-1]]
        expected = [
            rank
            for rank, index in enumerate(top, 1)
            if chunks[index]["volume"] == case["expected_volume"]
            and chunks[index]["page_start"] <= case["expected_page"]
            and chunks[index]["page_start"] + 2 >= case["expected_page"]
        ]
        if expected:
            best = min(expected)
            for cutoff in hits:
                hits[cutoff] += int(best <= cutoff)
    return {
        "cases": len(cases),
        **{f"recall_at_{cutoff}": round(value / len(cases), 6) for cutoff, value in hits.items()},
    }


def build_faiss(vectors: np.ndarray, chunks: list[dict], selected_mode: str) -> dict:
    import faiss

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    dimension = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(np.asarray(vectors, dtype=np.float32))
    # FAISS's Windows C++ file API cannot open this project's Unicode path.
    # Serialize in memory and let Python's Unicode-aware Path handle I/O.
    serialized = faiss.serialize_index(index)
    (VECTOR_DIR / "index.faiss").write_bytes(serialized.tobytes())
    ids = [chunk["chunk_id"] for chunk in chunks]
    (VECTOR_DIR / "vector_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False),
        encoding="utf-8",
    )
    connection = sqlite3.connect(T0278_DB)
    try:
        connection.execute("DELETE FROM vector_map")
        connection.executemany(
            "INSERT INTO vector_map(vector_id, chunk_id, text_hash) VALUES(?, ?, ?)",
            [
                (vector_id, chunk["chunk_id"], chunk["text_hash"])
                for vector_id, chunk in enumerate(chunks)
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "type": "IndexHNSWFlat",
        "metric": "inner_product_cosine_normalized",
        "dimension": dimension,
        "vectors": index.ntotal,
        "m": 32,
        "ef_construction": 200,
        "ef_search": 128,
        "selected_embedding_input": selected_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_disk()
    if not T0278_DB.exists() or not EVAL_SET.exists():
        raise RuntimeError("Run T0278 migration, sparse build, and evaluation first")
    os.environ.setdefault("HF_HOME", str(MODEL_DIR.parent / ".hf_home"))
    # The local proxy is reliable with the Hub's resumable HTTP transport;
    # Xet opens many parallel proxy connections but makes no forward progress.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    started = time.perf_counter()
    tracemalloc.start()
    model_manifest = download_model()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )
    chunks = load_chunks()
    cases = load_eval_queries()
    body_vectors, body_progress = encode_mode(model, chunks, "body", args.resume)
    metadata_vectors, metadata_progress = encode_mode(model, chunks, "metadata", args.resume)
    body_score = vector_recall_score(model, body_vectors, chunks, cases)
    metadata_score = vector_recall_score(model, metadata_vectors, chunks, cases)
    selected_mode = (
        "metadata"
        if (
            metadata_score["recall_at_5"],
            metadata_score["recall_at_100"],
        )
        > (body_score["recall_at_5"], body_score["recall_at_100"])
        else "body"
    )
    selected_vectors = metadata_vectors if selected_mode == "metadata" else body_vectors
    faiss_manifest = build_faiss(selected_vectors, chunks, selected_mode)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    manifest = {
        "version": "rag-v2-vector-1.0.0",
        "built_at": now_iso(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "requested_model_id": REQUESTED_MODEL_ID,
        "requested_model_revision": REQUESTED_MODEL_REVISION,
        "model_fallback": True,
        "model_fallback_reason": model_manifest["fallback_reason"],
        "model_local_path": str(MODEL_DIR),
        "model_snapshot": model_manifest,
        "embedding_input_ablation": {
            "body": body_score,
            "metadata": metadata_score,
            "selection_rule": "recall@5, then recall@100; ties choose body",
            "selected": selected_mode,
        },
        "faiss": faiss_manifest,
        "progress": {"body": body_progress, "metadata": metadata_progress},
        "index_sha256": sha256_file(VECTOR_DIR / "index.faiss"),
        "vector_ids_sha256": sha256_file(VECTOR_DIR / "vector_ids.json"),
    }
    (VECTOR_DIR / "vector_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Remove transient Hugging Face cache only after the pinned local snapshot exists.
    shutil.rmtree(MODEL_DIR.parent / ".hf_home", ignore_errors=True)
    finalize_seconds = time.perf_counter() - started
    embedding_seconds = sum(
        float(progress.get("seconds", 0.0))
        for progress in (body_progress, metadata_progress)
    )
    statistics = {
        **manifest,
        "build_seconds_excluding_model_download": round(
            embedding_seconds + finalize_seconds, 3
        ),
        "embedding_seconds": round(embedding_seconds, 3),
        "finalize_seconds": round(finalize_seconds, 3),
        "peak_python_memory_bytes": peak,
        "faiss_index_bytes": (VECTOR_DIR / "index.faiss").stat().st_size,
        "vector_ids_bytes": (VECTOR_DIR / "vector_ids.json").stat().st_size,
        "model_bytes": model_manifest["total_bytes"],
    }
    write_json(REPORT_ROOT / "04_vector_index_statistics.json", statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
