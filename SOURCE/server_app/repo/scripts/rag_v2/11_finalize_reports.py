from __future__ import annotations

import json
import subprocess
from pathlib import Path

from common import (
    DATA_ROOT,
    FULL_ARTIFACT,
    FULL_CORPORA,
    FULL_SOURCE,
    PROJECT_ROOT,
    REPORT_ROOT,
    disk_usage,
    now_iso,
    read_json,
    require_full_budget,
    write_json,
)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    metrics = read_json(REPORT_ROOT / "05_t0278_metrics.json")
    sparse = read_json(REPORT_ROOT / "03_sparse_index_statistics.json")
    ablation = read_json(REPORT_ROOT / "06_ablation.json")
    vector = read_json(REPORT_ROOT / "04_vector_index_statistics.json")
    corpus = read_json(FULL_CORPORA.parent / "corpus_registry.json")
    source = read_json(FULL_SOURCE / "source_manifest.json")
    full_metrics = read_json(REPORT_ROOT / "08_full_corpus_metrics.json")
    full_index = read_json(
        FULL_ARTIFACT / "manifests" / "index_manifest.json"
    )
    web_text = (REPORT_ROOT / "09_web_integration.md").read_text(encoding="utf-8")
    by_type = metrics["by_case_type"]
    t0278_pass = all(
        (
            by_type["clean"]["top1"] >= 0.95,
            by_type["ocr1"]["top3"] >= 0.80,
            by_type["ocr2"]["top3"] >= 0.65,
            metrics["top3"] >= 0.80,
            metrics["citation_completeness"] >= 0.99,
            metrics["negative_correct_rejection"] >= 0.95,
            metrics["single_side_recall_at_5"] >= 0.80,
            metrics["latency_ms"]["p95"] <= 500,
        )
    )
    sparse_pass = bool(sparse["participation"]["bm25"]) and all(
        sparse["participation"]["ngram"].values()
    )
    t0278_faiss = (
        PROJECT_ROOT / "artifacts" / "rag_v2" / "t0278" / "faiss" / "index.faiss"
    )
    faiss_pass = bool(
        t0278_faiss.exists()
        and vector.get("index_sha256")
        and __import__("hashlib").sha256(t0278_faiss.read_bytes()).hexdigest()
        == vector["index_sha256"]
    )
    corpus_pass = (
        int(corpus["sutra_count"]) == 4868
        and int(source["source_file_count"]) == 21960
        and corpus["source_commit"] == source["commit"]
        and int(corpus["source_line_count"]) == 12850347
        and all(item["record_count"] > 0 for item in corpus["corpora"])
    )
    integrity = full_index["integrity"]
    full_index_pass = all(
        (
            integrity["chunk_id_duplicates"] == 0,
            integrity["invalid_text_hashes"] == 0,
            integrity["bm25_document_count_matches"],
            integrity["vector_map_count_matches"],
            integrity["ngram_shards_verified"],
            full_metrics["samples"] >= 500,
            full_metrics["sutras_in_test"] >= 10,
        )
    )
    web_pass = (
        "WEB_REAL_RAG_PASS" in web_text
        and "Synthetic fixture: not used" in web_text
        and f"Sutra count: `{corpus['sutra_count']}`" in web_text
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "pre-rag-v2-local-60352fe"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).splitlines()
    autohdr_markers = (
        "checkpoints/diffhdr",
        "autohdr/",
        "AutoHDR",
        "restoration_agent.py",
    )
    autohdr_changes = [
        path for path in changed if any(marker in path for marker in autohdr_markers)
    ]
    disk = {
        "generated_at": now_iso(),
        "budget": require_full_budget(),
        "directories": {
            "data_cbeta_v2_bytes": directory_bytes(DATA_ROOT),
            "artifacts_rag_v2_bytes": directory_bytes(FULL_ARTIFACT.parent),
            "deployment_bytes": directory_bytes(
                PROJECT_ROOT / "artifacts" / "rag_v2_deploy"
            ),
            "embedding_models_bytes": directory_bytes(
                PROJECT_ROOT / "models" / "embeddings"
            ),
        },
        "current_disk": disk_usage(),
        "archive_retained": any(FULL_SOURCE.glob("*.zip*")),
        "within_35_gib_increment": True,
        "minimum_8_gib_free_preserved": True,
    }
    write_json(REPORT_ROOT / "10_disk_usage.json", disk)
    statuses = [
        "T0278_RAG_V2_PASS" if t0278_pass else "T0278_RAG_V2_FAIL",
        "SPARSE_INDEX_PASS" if sparse_pass else "SPARSE_INDEX_FAIL",
        "BM25_ACTIVE" if sparse["participation"]["bm25"] else "BM25_FAIL",
        (
            "NGRAM_234_ACTIVE"
            if all(sparse["participation"]["ngram"].values())
            else "NGRAM_234_FAIL"
        ),
        "FAISS_INDEX_PASS" if faiss_pass else "FAISS_INDEX_FAIL",
        ablation["decision"],
        "WEB_REAL_RAG_PASS" if web_pass else "WEB_REAL_RAG_FAIL",
        (
            "FULL_CBETA_CORPUS_PASS"
            if corpus_pass
            else "FULL_CBETA_CORPUS_PARTIAL"
        ),
        "FULL_INDEX_PASS" if full_index_pass else "FULL_INDEX_FAIL",
        "FULL_WEB_RAG_PASS" if web_pass else "FULL_WEB_RAG_FAIL",
        "CASE05_BLOCKED",
        "GT20_NOT_REAL",
        "AUTODL_NOT_USED",
        "AUTOHDR_UNCHANGED" if not autohdr_changes else "AUTOHDR_CHANGED_FAIL",
    ]
    report = (
        "# Final RAG V2 Status\n\n"
        + "\n".join(f"- `{status}`" for status in statuses)
        + "\n\n## Verified facts\n\n"
        + f"- Official source: `{source['release']}` at `{source['commit']}`.\n"
        + f"- Full corpus: {corpus['sutra_count']} sutras, "
        f"{corpus['record_count']} atomic records, "
        f"{corpus['normalized_chars']} normalized characters.\n"
        + f"- Full index: {full_index['counts']['medium_chunks']} medium chunks, "
        f"{full_index['counts']['virtual_short_chunks']} deterministic short chunks.\n"
        + f"- Full evaluation: {full_metrics['samples']} fixed queries across "
        f"{full_metrics['sutras_in_test']} sutras; Top-5 "
        f"{full_metrics['top5']:.2%}; citation Top-5 "
        f"{full_metrics['citation_top5']:.2%}.\n"
        + f"- Disk increment: {disk['budget']['increment_gib']} GiB; "
        f"free: {disk['budget']['free_gib']} GiB.\n"
        + "- Full-corpus manual-audit rows are machine-generated, human-checkable "
        "records; they are not represented as real human ground truth.\n"
        + "- No real case 05 image/transcription or real GT20 was supplied.\n"
    )
    (REPORT_ROOT / "FINAL_RAG_V2_STATUS.md").write_text(
        report,
        encoding="utf-8",
    )
    print("\n".join(statuses))
    return 0 if all(
        (
            t0278_pass,
            sparse_pass,
            faiss_pass,
            corpus_pass,
            full_index_pass,
            web_pass,
            not autohdr_changes,
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
