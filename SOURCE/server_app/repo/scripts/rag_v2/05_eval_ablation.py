from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

from common import CACHE_ROOT, REPORT_ROOT, T0278_ARTIFACT, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_MODULE = runpy.run_path(str(SCRIPT_DIR / "04_eval_t0278.py"))
generate_cases = EVAL_MODULE["generate_cases"]
evaluate = EVAL_MODULE["evaluate"]
VECTOR_DIR = T0278_ARTIFACT / "faiss"


def ocr_top3(metrics: dict) -> float:
    groups = [metrics["by_case_type"]["ocr1"], metrics["by_case_type"]["ocr2"]]
    samples = sum(group["samples"] for group in groups)
    return sum(group["top3"] * group["samples"] for group in groups) / samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (VECTOR_DIR / "vector_manifest.json").exists():
        raise RuntimeError("Build the T0278 vector index first")
    cases_path = CACHE_ROOT / "t0278_eval_cases.jsonl"
    cases = generate_cases(cases_path)
    test_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    variants = {
        "A_ngram": {"use_ngram": True, "use_bm25": False, "vector_enabled": False},
        "B_bm25": {"use_ngram": False, "use_bm25": True, "vector_enabled": False},
        "C_sparse_hybrid": {
            "use_ngram": True,
            "use_bm25": True,
            "vector_enabled": False,
        },
        "D_sparse_plus_faiss": {
            "use_ngram": True,
            "use_bm25": True,
            "vector_enabled": True,
        },
    }
    results: dict[str, dict] = {}
    for name, options in variants.items():
        report_path = REPORT_ROOT / f"ablation_{name}.json"
        if args.resume and report_path.exists():
            result = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            metrics, rows = evaluate(
                cases,
                vector_dir=VECTOR_DIR,
                **options,
            )
            result = {
                "variant": name,
                "test_set_sha256": test_hash,
                "metrics": metrics,
                "errors": [
                    row
                    for row in rows
                    if (
                        row["expected"] == "fillable"
                        and not row["top3"]
                        or row["expected"] == "single_side_localization"
                        and not row["top5"]
                        or row["expected"] == "abstain"
                        and row["decision_status"] != "abstain"
                    )
                ],
            }
            write_json(report_path, result)
        results[name] = result
        print(name, json.dumps(result["metrics"], ensure_ascii=False), flush=True)

    c_metrics = results["C_sparse_hybrid"]["metrics"]
    d_metrics = results["D_sparse_plus_faiss"]["metrics"]
    ocr_gain = ocr_top3(d_metrics) - ocr_top3(c_metrics)
    citation_non_decrease = (
        d_metrics["citation_completeness"] >= c_metrics["citation_completeness"]
    )
    p95_acceptable = d_metrics["latency_ms"]["p95"] <= 5000
    enabled = ocr_gain >= 0.03 and citation_non_decrease and p95_acceptable
    decision = (
        "VECTOR_ASSIST_ENABLED"
        if enabled
        else "VECTOR_INDEX_BUILT_BUT_DISABLED_NO_GAIN"
    )
    summary = {
        "test_set_sha256": test_hash,
        "same_test_set_for_all_variants": True,
        "variants": {name: result["metrics"] for name, result in results.items()},
        "ocr_top3_c": round(ocr_top3(c_metrics), 6),
        "ocr_top3_d": round(ocr_top3(d_metrics), 6),
        "ocr_top3_absolute_gain": round(ocr_gain, 6),
        "citation_non_decrease": citation_non_decrease,
        "p95_acceptable": p95_acceptable,
        "enable_threshold": 0.03,
        "decision": decision,
        "default_vector_enabled": enabled,
    }
    write_json(REPORT_ROOT / "06_ablation.json", summary)
    lines = [
        "# T0278 Vector Ablation",
        "",
        f"- Fixed test set SHA-256: `{test_hash}`",
        "- All A/B/C/D variants used the same 380-case no-leak test set.",
        "",
        "| Variant | Top-1 | Top-3 | Top-5 | OCR Top-3 | Single-side R@5 | Negative reject | Citation | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variants:
        metrics = results[name]["metrics"]
        lines.append(
            "| {name} | {top1:.4f} | {top3:.4f} | {top5:.4f} | "
            "{ocr:.4f} | {single:.4f} | {negative:.4f} | {citation:.4f} | "
            "{p95:.3f} |".format(
                name=name,
                top1=metrics["top1"],
                top3=metrics["top3"],
                top5=metrics["top5"],
                ocr=ocr_top3(metrics),
                single=metrics["single_side_recall_at_5"],
                negative=metrics["negative_correct_rejection"],
                citation=metrics["citation_completeness"],
                p95=metrics["latency_ms"]["p95"],
            )
        )
    lines.extend(
        [
            "",
            f"- D − C OCR Top-3 absolute gain: `{ocr_gain:.4f}`",
            f"- Citation correctness non-decrease: `{citation_non_decrease}`",
            f"- D P95 acceptable (≤ 5000 ms): `{p95_acceptable}`",
            f"- Decision: `{decision}`",
            "",
            "FAISS is recall-only. Candidate text and citations always come from original CBETA chunks and pass character-continuity reranking.",
        ]
    )
    (REPORT_ROOT / "06_ablation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
