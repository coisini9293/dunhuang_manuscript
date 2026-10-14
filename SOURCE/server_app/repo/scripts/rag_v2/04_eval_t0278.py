from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import CACHE_ROOT, REPORT_ROOT, T0278_DB, now_iso, write_json
from retrieval.rag_v2.normalize import confusion_map, simplified_alias
from retrieval.rag_v2.hybrid_retriever import RagV2Retriever


SEED = 20260723
CONTEXT = 12
MISSING_LENGTHS = [1, 2, 4, 8]
NOISE_FALLBACK = "未末己已巳人入土士口日目自一二佛彿眾衆華花"


def load_groups() -> list[dict]:
    connection = sqlite3.connect(T0278_DB)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT record_id, volume, chapter_name, page, column_name, line,
               normalized_text
        FROM records
        ORDER BY volume, chapter_name, page, column_name, line
        """
    ).fetchall()
    connection.close()
    groups: dict[tuple[int, str], dict] = {}
    for row in rows:
        key = (row["volume"], row["chapter_name"])
        group = groups.setdefault(
            key,
            {
                "volume": row["volume"],
                "chapter_name": row["chapter_name"],
                "text": "",
                "segments": [],
            },
        )
        start = len(group["text"])
        group["text"] += row["normalized_text"]
        group["segments"].append(
            {
                "start": start,
                "end": len(group["text"]),
                "page": row["page"],
                "column": row["column_name"],
                "line": row["line"],
            }
        )
    return [group for group in groups.values() if len(group["text"]) >= 80]


def mutate(text: str, errors: int, rng: random.Random) -> str:
    values = list(text)
    indexes = rng.sample(range(len(values)), min(errors, len(values)))
    mapping = confusion_map()
    for index in indexes:
        alternatives = sorted(mapping.get(values[index], set()) - {values[index]})
        values[index] = rng.choice(alternatives or list(NOISE_FALLBACK))
    return "".join(values)


def location_for(group: dict, offset: int) -> dict:
    for segment in group["segments"]:
        if segment["start"] <= offset < segment["end"]:
            return {
                "page": segment["page"],
                "column": segment["column"],
                "line": segment["line"],
            }
    return {"page": None, "column": None, "line": None}


def generate_cases(path: Path) -> list[dict]:
    if path.exists():
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    rng = random.Random(SEED)
    groups = load_groups()
    plan = (
        [("clean", 80)]
        + [("ocr1", 60)]
        + [("ocr2", 60)]
        + [("simplified", 40)]
        + [("variant", 20)]
        + [("left_only", 30)]
        + [("right_only", 30)]
        + [("filtered", 20)]
        + [("repeated", 20)]
    )
    cases: list[dict] = []
    attempts = 0
    for case_type, target in plan:
        made = 0
        while made < target and attempts < 100_000:
            attempts += 1
            group = rng.choice(groups)
            text = group["text"]
            missing_length = MISSING_LENGTHS[(len(cases) + made) % len(MISSING_LENGTHS)]
            if len(text) < 2 * CONTEXT + missing_length + 2:
                continue
            position = rng.randint(CONTEXT, len(text) - CONTEXT - missing_length)
            before = text[position - CONTEXT : position]
            ground_truth = text[position : position + missing_length]
            after = text[position + missing_length : position + missing_length + CONTEXT]
            if not ground_truth or ground_truth in before or ground_truth in after:
                continue
            query_before, query_after = before, after
            if case_type == "ocr1":
                query_before = mutate(before, 1, rng)
            elif case_type == "ocr2":
                query_before = mutate(before, 1, rng)
                query_after = mutate(after, 1, rng)
            elif case_type == "simplified":
                query_before = simplified_alias(before)
                query_after = simplified_alias(after)
            elif case_type == "variant":
                query_before = mutate(before, 1, rng)
            elif case_type == "left_only":
                query_after = ""
            elif case_type == "right_only":
                query_before = ""
            elif case_type == "repeated":
                query_before = before[-6:]
                query_after = after[:6]
            if ground_truth in query_before or ground_truth in query_after:
                continue
            location = location_for(group, position)
            cases.append(
                {
                    "case_id": f"t0278_v2_{len(cases):04d}",
                    "case_type": case_type,
                    "query_before": query_before,
                    "query_after": query_after,
                    "ground_truth": ground_truth,
                    "missing_length": missing_length,
                    "expected_volume": group["volume"],
                    "expected_chapter": group["chapter_name"],
                    "expected_page": location["page"],
                    "expected_column": location["column"],
                    "expected_line": location["line"],
                    "sutra_ids": ["T0278"] if case_type == "filtered" else None,
                    "translators": ["佛馱跋陀羅"] if case_type == "filtered" else None,
                    "expected": (
                        "single_side_localization"
                        if case_type in ("left_only", "right_only")
                        else "fillable"
                    ),
                }
            )
            made += 1
        if made != target:
            raise RuntimeError(f"Could not generate {case_type}: {made}/{target}")
    for index in range(20):
        cases.append(
            {
                "case_id": f"t0278_v2_negative_{index:03d}",
                "case_type": "negative",
                "query_before": f"甲乙丙丁戊己庚辛壬癸{index}",
                "query_after": f"天地玄黃宇宙洪荒日月{index}",
                "ground_truth": "",
                "missing_length": 4,
                "expected_volume": None,
                "expected_chapter": None,
                "expected_page": None,
                "expected_column": None,
                "expected_line": None,
                "sutra_ids": ["T0278"],
                "translators": None,
                "expected": "abstain",
            }
        )
    rng.shuffle(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return cases


def evaluate(
    cases: list[dict],
    use_ngram: bool = True,
    use_bm25: bool = True,
    vector_enabled: bool = False,
    vector_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    retriever = RagV2Retriever(
        T0278_DB,
        vector_dir=vector_dir,
        vector_enabled=vector_enabled,
        use_ngram=use_ngram,
        use_bm25=use_bm25,
    )
    rows: list[dict] = []
    for case in cases:
        result = retriever.query(
            case["query_before"],
            case["query_after"],
            sutra_ids=case.get("sutra_ids"),
            translators=case.get("translators"),
            expected_missing_length=case["missing_length"],
            top_k=5,
        )
        candidates = result.candidates
        candidate_texts = [candidate.candidate_text for candidate in candidates]
        gt = case["ground_truth"]
        if case["expected"] == "single_side_localization":
            hits = [
                candidate.volume == case["expected_volume"]
                and candidate.chapter_name == case["expected_chapter"]
                for candidate in candidates
            ]
            reciprocal_rank = next(
                (1 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0
            )
            top1 = bool(hits[:1] and hits[0])
            top3 = any(hits[:3])
            top5 = any(hits[:5])
            exact = False
        elif case["expected"] == "abstain":
            reciprocal_rank = 0.0
            top1 = top3 = top5 = exact = False
        else:
            reciprocal_rank = next(
                (
                    1 / (index + 1)
                    for index, candidate in enumerate(candidate_texts)
                    if candidate == gt
                ),
                0.0,
            )
            top1 = bool(candidate_texts[:1] and candidate_texts[0] == gt)
            top3 = gt in candidate_texts[:3]
            top5 = gt in candidate_texts[:5]
            exact = top1
        citation_complete = (
            all(
                all(
                    [
                        candidate.sutra_id,
                        candidate.canonical_title,
                        candidate.translator,
                        candidate.edition,
                        candidate.volume,
                        candidate.page_start,
                        candidate.column_start,
                        candidate.line_start,
                        candidate.page_end,
                        candidate.column_end,
                        candidate.line_end,
                        len(candidate.text_hash) == 64,
                    ]
                )
                for candidate in candidates
            )
            if candidates
            else False
        )
        rows.append(
            {
                **case,
                "decision_status": result.decision_status,
                "candidate_texts": candidate_texts,
                "candidate_citations": [candidate.citation for candidate in candidates],
                "exact_match": exact,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "reciprocal_rank": reciprocal_rank,
                "citation_complete": citation_complete,
                "latency_ms": result.latency_ms,
                "components": [
                    candidate.retrieval_components for candidate in candidates
                ],
            }
        )
    metrics = compute_metrics(rows)
    return metrics, rows


def compute_metrics(rows: list[dict]) -> dict:
    fillable = [row for row in rows if row["expected"] == "fillable"]
    single = [
        row for row in rows if row["expected"] == "single_side_localization"
    ]
    negatives = [row for row in rows if row["expected"] == "abstain"]
    evaluated = fillable + single
    total_chars = sum(len(row["ground_truth"]) for row in fillable)
    correct_chars = 0
    for row in fillable:
        prediction = (row["candidate_texts"] or [""])[0]
        correct_chars += sum(
            left == right for left, right in zip(row["ground_truth"], prediction)
        )
    by_type = {}
    for case_type in sorted({row["case_type"] for row in rows}):
        group = [row for row in rows if row["case_type"] == case_type]
        by_type[case_type] = {
            "samples": len(group),
            "top1": round(sum(row["top1"] for row in group) / len(group), 4),
            "top3": round(sum(row["top3"] for row in group) / len(group), 4),
            "top5": round(sum(row["top5"] for row in group) / len(group), 4),
            "abstain_rate": round(
                sum(row["decision_status"] == "abstain" for row in group) / len(group),
                4,
            ),
        }
    latencies = sorted(row["latency_ms"] for row in rows)
    statuses = Counter(row["decision_status"] for row in rows)
    return {
        "generated_at": now_iso(),
        "samples": len(rows),
        "fillable_samples": len(fillable),
        "single_side_samples": len(single),
        "negative_samples": len(negatives),
        "exact_match": round(sum(row["exact_match"] for row in fillable) / len(fillable), 4),
        "character_accuracy": round(correct_chars / max(total_chars, 1), 4),
        "top1": round(sum(row["top1"] for row in evaluated) / len(evaluated), 4),
        "top3": round(sum(row["top3"] for row in evaluated) / len(evaluated), 4),
        "top5": round(sum(row["top5"] for row in evaluated) / len(evaluated), 4),
        "mrr": round(sum(row["reciprocal_rank"] for row in evaluated) / len(evaluated), 4),
        "recall_at_5": round(sum(row["top5"] for row in evaluated) / len(evaluated), 4),
        "abstain_rate": round(statuses["abstain"] / len(rows), 4),
        "ambiguous_rate": round(statuses["retrieved_ambiguous"] / len(rows), 4),
        "citation_completeness": round(
            sum(row["citation_complete"] for row in evaluated) / len(evaluated), 4
        ),
        "negative_correct_rejection": round(
            sum(row["decision_status"] == "abstain" for row in negatives)
            / max(len(negatives), 1),
            4,
        ),
        "single_side_recall_at_5": round(
            sum(row["top5"] for row in single) / max(len(single), 1), 4
        ),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 3),
            "mean": round(statistics.mean(latencies), 3),
        },
        "status_counts": dict(statuses),
        "by_case_type": by_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    cases_path = CACHE_ROOT / "t0278_eval_cases.jsonl"
    cases = generate_cases(cases_path)
    if arguments.limit:
        cases = cases[: arguments.limit]
    metrics, rows = evaluate(cases)
    metrics["round"] = arguments.round
    metrics["test_set_sha256"] = hashlib.sha256(
        cases_path.read_bytes()
    ).hexdigest()
    write_json(REPORT_ROOT / f"05_t0278_metrics_round{arguments.round}.json", metrics)
    write_json(REPORT_ROOT / "05_t0278_metrics.json", metrics)
    errors = [
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
    ]
    error_path = REPORT_ROOT / "error_cases.jsonl"
    with error_path.open("w", encoding="utf-8") as stream:
        for row in errors:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
