from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from retrieval.noise_aware import NoiseAwareRetriever
from retrieval.rag_v2.full_retriever import FullRagV2Retriever


CONFUSIONS = {
    "未": "末",
    "末": "未",
    "己": "已",
    "已": "巳",
    "人": "入",
    "入": "人",
    "土": "士",
    "士": "土",
    "日": "曰",
    "曰": "日",
    "目": "自",
    "自": "目",
    "佛": "彿",
    "彿": "佛",
    "眾": "衆",
    "衆": "眾",
    "華": "花",
    "花": "華",
}
NOISE_TYPES = (
    "delete_one",
    "replace_one",
    "insert_punctuation",
    "insert_space",
    "truncate_context",
    "trusted_variant",
)


def mutate(before: str, after: str, kind: str) -> tuple[str, str, dict]:
    left, right = before, after
    detail: dict[str, object] = {"type": kind}
    target = "before" if left else "after"
    text = left if left else right
    if kind == "delete_one" and text:
        index = len(text) // 2
        changed = text[:index] + text[index + 1 :]
        detail.update({"side": target, "index": index, "original": text[index]})
    elif kind == "replace_one" and text:
        index = next(
            (i for i, char in enumerate(text) if char in CONFUSIONS),
            len(text) // 2,
        )
        replacement = CONFUSIONS.get(text[index], "日" if text[index] != "日" else "曰")
        changed = text[:index] + replacement + text[index + 1 :]
        detail.update(
            {
                "side": target,
                "index": index,
                "original": text[index],
                "replacement": replacement,
            }
        )
    elif kind == "insert_punctuation" and text:
        index = len(text) // 2
        changed = text[:index] + "，" + text[index:]
        detail.update({"side": target, "index": index, "inserted": "，"})
    elif kind == "insert_space" and text:
        changed = " ".join(text)
        detail.update({"side": target})
    elif kind == "truncate_context":
        left = left[-max(1, len(left) // 2) :] if left else ""
        right = right[: max(1, len(right) // 2)] if right else ""
        detail.update({"side": "both"})
        return left, right, detail
    elif kind == "trusted_variant" and text:
        index = next(
            (i for i, char in enumerate(text) if char in CONFUSIONS),
            None,
        )
        if index is None:
            detail["skipped_no_mapping"] = True
            changed = text
        else:
            replacement = CONFUSIONS[text[index]]
            changed = text[:index] + replacement + text[index + 1 :]
            detail.update(
                {
                    "side": target,
                    "index": index,
                    "original": text[index],
                    "replacement": replacement,
                }
            )
    else:
        changed = text
    if target == "before":
        left = changed
    else:
        right = changed
    return left, right, detail


def hits(result: dict, truth: str, sutra_id: str | None) -> dict[str, bool]:
    values = [
        (
            item.get("candidate_text") == truth
            and (not sutra_id or item.get("sutra_id") == sutra_id)
        )
        for item in result.get("candidates", [])
    ]
    return {
        "top_1": any(values[:1]),
        "top_3": any(values[:3]),
        "top_5": any(values[:5]),
    }


def summarize(rows: list[dict], mode: str) -> dict:
    mode_rows = [row for row in rows if row["mode"] == mode]
    count = len(mode_rows)
    return {
        "sample_count": count,
        "top_1": round(sum(row["hits"]["top_1"] for row in mode_rows) / count, 6),
        "top_3": round(sum(row["hits"]["top_3"] for row in mode_rows) / count, 6),
        "top_5": round(sum(row["hits"]["top_5"] for row in mode_rows) / count, 6),
        "average_latency_ms": round(
            statistics.fmean(row["latency_ms"] for row in mode_rows), 3
        ),
        "p95_latency_ms": round(
            sorted(row["latency_ms"] for row in mode_rows)[
                min(count - 1, int(count * 0.95))
            ],
            3,
        ),
        "failures": [
            {
                "case_id": row["case_id"],
                "noise_type": row["noise"]["type"],
                "decision_status": row["decision_status"],
                "returned": row["returned"],
            }
            for row in mode_rows
            if not row["hits"]["top_5"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=18)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fillable = [
        item
        for item in cases
        if item.get("expected") == "fillable" and item.get("ground_truth")
    ]
    # Fixed stride spans the 570-case manifest without random state.
    stride = max(1, len(fillable) // args.sample_count)
    selected = fillable[::stride][: args.sample_count]
    engine = FullRagV2Retriever(
        args.runtime_root / "metadata.sqlite3",
        args.runtime_root / "ngram",
        vector_dir=args.runtime_root / "faiss",
        vector_enabled=False,
    )
    adapter = NoiseAwareRetriever(engine)
    rows = []
    for index, case in enumerate(selected):
        noise_type = NOISE_TYPES[index % len(NOISE_TYPES)]
        before, after, noise = mutate(
            case["query_before"], case["query_after"], noise_type
        )
        common = {
            "sutra_ids": case.get("sutra_ids"),
            "translators": case.get("translators"),
            "editions": case.get("editions"),
            "expected_missing_length": case.get("missing_length"),
            "top_k": 5,
        }
        for mode in ("baseline", "noise_aware"):
            started = time.perf_counter()
            result = adapter.query(
                before,
                after,
                retrieval_mode=mode,
                **common,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case.get("case_type"),
                    "mode": mode,
                    "noise": noise,
                    "query_before": before,
                    "query_after": after,
                    "ground_truth": case["ground_truth"],
                    "expected_sutra_id": case.get("expected_sutra_id"),
                    "decision_status": result["decision_status"],
                    "hits": hits(
                        result,
                        case["ground_truth"],
                        case.get("expected_sutra_id"),
                    ),
                    "latency_ms": round(latency_ms, 3),
                    "returned": [
                        {
                            "rank": item["rank"],
                            "candidate_text": item.get("candidate_text", ""),
                            "sutra_id": item.get("sutra_id", ""),
                            "citation": item.get("citation", ""),
                        }
                        for item in result.get("candidates", [])
                    ],
                    "variant_count": result.get("variant_count", 1),
                }
            )
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(selected),
                    "case_id": case["case_id"],
                    "noise_type": noise_type,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    baseline = summarize(rows, "baseline")
    noise_aware = summarize(rows, "noise_aware")
    payload = {
        "schema_version": "1.0.0",
        "evaluation_kind": "fixed_manifest_deterministic_synthetic_ocr_noise",
        "source_manifest": str(args.cases),
        "source_manifest_case_count": len(cases),
        "sample_selection": {
            "method": "fixed_stride",
            "sample_count": len(selected),
            "case_ids": [item["case_id"] for item in selected],
            "noise_types": list(NOISE_TYPES),
        },
        "index_health": engine.health(),
        "baseline": baseline,
        "noise_aware": noise_aware,
        "delta": {
            key: round(noise_aware[key] - baseline[key], 6)
            for key in ("top_1", "top_3", "top_5")
        },
        "recommendation": (
            "NOISE_AWARE_AVAILABLE_DEFAULT_BASELINE"
            if noise_aware["top_5"] <= baseline["top_5"]
            else "NOISE_AWARE_OPTIONAL_MEASURED_GAIN_DEFAULT_BASELINE"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["delta"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
