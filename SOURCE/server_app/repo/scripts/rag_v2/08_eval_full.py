from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path

from common import (
    CACHE_ROOT,
    FULL_ARTIFACT,
    FULL_DB,
    REPORT_ROOT,
    now_iso,
    process_memory,
    write_json,
)
from retrieval.rag_v2.full_retriever import FullRagV2Retriever
from retrieval.rag_v2.normalize import confusion_map, simplified_alias


SEED = 20260723
CONTEXT = 12
MISSING_LENGTHS = (1, 2, 4, 8)
NOISE_FALLBACK = "未末己已巳人入土士口日目自一二佛彿眾衆華花"


def mutate(text: str, errors: int, rng: random.Random) -> str:
    values = list(text)
    mapping = confusion_map()
    for index in rng.sample(range(len(values)), min(errors, len(values))):
        alternatives = sorted(mapping.get(values[index], set()) - {values[index]})
        values[index] = rng.choice(alternatives or list(NOISE_FALLBACK))
    return "".join(values)


def source_location(db: sqlite3.Connection, group_id: int, offset: int) -> dict:
    row = db.execute(
        """
        SELECT page,column_name,line,line_head
        FROM records
        WHERE group_id=? AND norm_start<=? AND norm_end>?
        ORDER BY norm_start DESC LIMIT 1
        """,
        (group_id, offset, offset),
    ).fetchone()
    return dict(row) if row else {}


def generate_cases(path: Path) -> list[dict]:
    if path.exists():
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    rng = random.Random(SEED)
    connection = sqlite3.connect(FULL_DB)
    connection.row_factory = sqlite3.Row
    groups = [
        dict(row)
        for row in connection.execute(
            """
            SELECT group_id,sutra_id,canonical_title,translator,edition,volume,
                   chapter_name,normalized_text
            FROM groups WHERE length(normalized_text)>=100
            ORDER BY sutra_id,volume,group_id
            """
        )
    ]
    by_sutra: dict[str, list[dict]] = {}
    for group in groups:
        by_sutra.setdefault(group["sutra_id"], []).append(group)
    sutras = sorted(by_sutra)
    if len(sutras) < 10:
        raise RuntimeError("Full evaluation requires at least 10 sutras")
    plan = [
        ("clean", 120),
        ("ocr1", 90),
        ("ocr2", 70),
        ("simplified", 50),
        ("filtered", 40),
        ("edition_filtered", 20),
        ("translator_filtered", 20),
        ("left_only", 40),
        ("right_only", 40),
        ("repeated", 30),
    ]
    cases: list[dict] = []
    sutra_cursor = 0
    for case_type, target in plan:
        made = 0
        attempts = 0
        while made < target and attempts < target * 100:
            attempts += 1
            sutra_id = sutras[sutra_cursor % len(sutras)]
            sutra_cursor += 1
            group = rng.choice(by_sutra[sutra_id])
            text = group["normalized_text"]
            missing_length = MISSING_LENGTHS[(len(cases) + made) % 4]
            if len(text) <= 2 * CONTEXT + missing_length:
                continue
            position = rng.randint(CONTEXT, len(text) - CONTEXT - missing_length)
            before = text[position - CONTEXT : position]
            ground_truth = text[position : position + missing_length]
            after = text[position + missing_length : position + missing_length + CONTEXT]
            if not ground_truth or ground_truth in before or ground_truth in after:
                continue
            if (
                case_type == "translator_filtered"
                and group["translator"] in ("", "不詳")
            ):
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
            elif case_type == "left_only":
                query_after = ""
            elif case_type == "right_only":
                query_before = ""
            elif case_type == "repeated":
                query_before = before[-6:]
                query_after = after[:6]
            if ground_truth in query_before or ground_truth in query_after:
                continue
            location = source_location(connection, group["group_id"], position)
            cases.append(
                {
                    "case_id": f"full_v2_{len(cases):04d}",
                    "case_type": case_type,
                    "query_before": query_before,
                    "query_after": query_after,
                    "ground_truth": ground_truth,
                    "missing_length": missing_length,
                    "expected_sutra_id": sutra_id,
                    "expected_title": group["canonical_title"],
                    "expected_translator": group["translator"],
                    "expected_edition": group["edition"],
                    "expected_volume": group["volume"],
                    "expected_chapter": group["chapter_name"],
                    "expected_page": location.get("page"),
                    "expected_column": location.get("column_name"),
                    "expected_line": location.get("line"),
                    "expected_line_head": location.get("line_head"),
                    "sutra_ids": [sutra_id] if case_type == "filtered" else None,
                    "translators": (
                        [group["translator"]]
                        if case_type == "translator_filtered"
                        else None
                    ),
                    "editions": (
                        [group["edition"]]
                        if case_type == "edition_filtered"
                        else None
                    ),
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
    # Explicitly cover parallel/similar passages occurring in different
    # sutras, editions, or translator attributions. Exact parallel windows
    # make the expected gap auditable while allowing multiple valid citations.
    parallel_seen: dict[str, tuple[dict, int]] = {}
    parallel_pairs: list[tuple[dict, int, dict, int]] = []
    parallel_missing = 4
    parallel_width = CONTEXT * 2 + parallel_missing
    for group in groups:
        text = group["normalized_text"]
        if len(text) < parallel_width:
            continue
        starts = {
            0,
            max(0, (len(text) - parallel_width) // 2),
            len(text) - parallel_width,
        }
        for start in sorted(starts):
            window = text[start : start + parallel_width]
            previous = parallel_seen.get(window)
            if previous:
                other, other_start = previous
                if (
                    other["sutra_id"] != group["sutra_id"]
                    and (
                        other["translator"] != group["translator"]
                        or other["edition"] != group["edition"]
                    )
                ):
                    parallel_pairs.append((other, other_start, group, start))
            else:
                parallel_seen[window] = (group, start)
    rng.shuffle(parallel_pairs)
    parallel_made = 0
    for first, start, parallel, _parallel_start in parallel_pairs:
        if parallel_made >= 30:
            break
        text = first["normalized_text"]
        position = start + CONTEXT
        before = text[start:position]
        ground_truth = text[position : position + parallel_missing]
        after = text[
            position + parallel_missing : position + parallel_missing + CONTEXT
        ]
        if (
            not before
            or not after
            or not ground_truth
            or ground_truth in before
            or ground_truth in after
        ):
            continue
        location = source_location(connection, first["group_id"], position)
        cases.append(
            {
                "case_id": f"full_v2_{len(cases):04d}",
                "case_type": "cross_translation",
                "query_before": before,
                "query_after": after,
                "ground_truth": ground_truth,
                "missing_length": parallel_missing,
                "expected_sutra_id": first["sutra_id"],
                "expected_title": first["canonical_title"],
                "expected_translator": first["translator"],
                "expected_edition": first["edition"],
                "expected_volume": first["volume"],
                "expected_chapter": first["chapter_name"],
                "expected_page": location.get("page"),
                "expected_column": location.get("column_name"),
                "expected_line": location.get("line"),
                "expected_line_head": location.get("line_head"),
                "parallel_sutra_id": parallel["sutra_id"],
                "parallel_translator": parallel["translator"],
                "parallel_edition": parallel["edition"],
                "sutra_ids": None,
                "translators": None,
                "editions": None,
                "expected": "fillable",
            }
        )
        parallel_made += 1
    if sum(case["case_type"] == "cross_translation" for case in cases) < 20:
        raise RuntimeError(
            "Could not build at least 20 cross-translation parallel cases"
        )
    for index in range(20):
        cases.append(
            {
                "case_id": f"full_v2_negative_{index:03d}",
                "case_type": "negative",
                "query_before": f"甲乙丙丁戊己庚辛壬癸全庫{index}",
                "query_after": f"天地玄黃宇宙洪荒日月全庫{index}",
                "ground_truth": "",
                "missing_length": 4,
                "expected_sutra_id": None,
                "expected": "abstain",
                "sutra_ids": None,
                "translators": None,
                "editions": None,
            }
        )
    connection.close()
    rng.shuffle(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return cases


def evaluate(cases: list[dict]) -> tuple[dict, list[dict]]:
    cold_started = time.perf_counter()
    retriever = FullRagV2Retriever(
        FULL_DB,
        FULL_ARTIFACT / "ngram",
        vector_dir=FULL_ARTIFACT / "faiss",
        vector_enabled=False,
    )
    cold_load_ms = round((time.perf_counter() - cold_started) * 1000, 3)
    rows = []
    for index, case in enumerate(cases, 1):
        result = retriever.query(
            case["query_before"],
            case["query_after"],
            sutra_ids=case.get("sutra_ids"),
            translators=case.get("translators"),
            editions=case.get("editions"),
            expected_missing_length=case["missing_length"],
            top_k=5,
        )
        candidates = result.candidates
        if case["expected"] == "fillable":
            hits = [
                candidate.candidate_text == case["ground_truth"]
                for candidate in candidates
            ]
        elif case["expected"] == "single_side_localization":
            hits = [
                candidate.sutra_id == case["expected_sutra_id"]
                and candidate.volume == case["expected_volume"]
                for candidate in candidates
            ]
        else:
            hits = []
        citation_hits = [
            candidate.sutra_id == case.get("expected_sutra_id")
            and candidate.volume == case.get("expected_volume")
            and candidate.page_start <= (case.get("expected_page") or -1) <= candidate.page_end
            for candidate in candidates
        ]
        row = {
            **case,
            "decision_status": result.decision_status,
            "candidate_texts": [candidate.candidate_text for candidate in candidates],
            "candidate_citations": [candidate.citation for candidate in candidates],
            "top1": bool(hits[:1] and hits[0]),
            "top3": any(hits[:3]),
            "top5": any(hits[:5]),
            "exact_match": bool(hits[:1] and hits[0]),
            "reciprocal_rank": next(
                (1.0 / rank for rank, hit in enumerate(hits, 1) if hit),
                0.0,
            ),
            "citation_top5": any(citation_hits[:5]),
            "citation_complete": all(
                candidate.citation
                and candidate.sutra_id
                and candidate.volume > 0
                and candidate.page_start > 0
                and candidate.column_start
                and candidate.line_start > 0
                and len(candidate.text_hash) == 64
                for candidate in candidates
            )
            if candidates
            else case["expected"] == "abstain",
            "latency_ms": result.latency_ms,
        }
        rows.append(row)
        if index % 50 == 0:
            print(f"full eval: {index}/{len(cases)}", flush=True)
    positives = [row for row in rows if row["expected"] != "abstain"]
    negatives = [row for row in rows if row["expected"] == "abstain"]
    latencies = sorted(row["latency_ms"] for row in rows)
    memory = process_memory()
    by_type = {}
    for case_type in sorted({row["case_type"] for row in rows}):
        group = [row for row in rows if row["case_type"] == case_type]
        by_type[case_type] = {
            "samples": len(group),
            "top1": round(sum(row["top1"] for row in group) / len(group), 4),
            "top3": round(sum(row["top3"] for row in group) / len(group), 4),
            "top5": round(sum(row["top5"] for row in group) / len(group), 4),
            "citation_top5": round(
                sum(row["citation_top5"] for row in group) / len(group), 4
            ),
        }
    metrics = {
        "generated_at": now_iso(),
        "samples": len(rows),
        "sutras_in_test": len(
            {row.get("expected_sutra_id") for row in positives if row.get("expected_sutra_id")}
        ),
        "top1": round(sum(row["top1"] for row in positives) / len(positives), 4),
        "top3": round(sum(row["top3"] for row in positives) / len(positives), 4),
        "top5": round(sum(row["top5"] for row in positives) / len(positives), 4),
        "exact_match": round(
            sum(row["exact_match"] for row in positives) / len(positives),
            4,
        ),
        "character_accuracy": round(
            sum(
                sum(
                    left == right
                    for left, right in zip(
                        row["ground_truth"],
                        row["candidate_texts"][0] if row["candidate_texts"] else "",
                    )
                )
                for row in positives
                if row["expected"] == "fillable"
            )
            / max(
                sum(
                    len(row["ground_truth"])
                    for row in positives
                    if row["expected"] == "fillable"
                ),
                1,
            ),
            4,
        ),
        "mrr": round(
            sum(row["reciprocal_rank"] for row in positives) / len(positives),
            4,
        ),
        "recall_at_5": round(
            sum(row["top5"] for row in positives) / len(positives),
            4,
        ),
        "citation_top5": round(
            sum(row["citation_top5"] for row in positives) / len(positives), 4
        ),
        "citation_completeness": round(
            sum(row["citation_complete"] for row in rows) / len(rows),
            4,
        ),
        "abstain_rate": round(
            sum(row["decision_status"] == "abstain" for row in rows) / len(rows),
            4,
        ),
        "ambiguous_rate": round(
            sum(
                row["decision_status"] == "retrieved_ambiguous" for row in rows
            )
            / len(rows),
            4,
        ),
        "negative_correct_rejection": round(
            sum(row["decision_status"] == "abstain" for row in negatives)
            / len(negatives),
            4,
        ),
        "latency_ms": {
            "cold_load": cold_load_ms,
            "p50": round(statistics.median(latencies), 3),
            "p95": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 3),
            "mean": round(statistics.mean(latencies), 3),
        },
        "memory": memory,
        "status_counts": dict(Counter(row["decision_status"] for row in rows)),
        "by_case_type": by_type,
        "case05_status": "CASE05_BLOCKED",
        "gt20_status": "GT20_NOT_REAL",
    }
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    case_path = CACHE_ROOT / "full_eval_cases.jsonl"
    cases = generate_cases(case_path)
    metrics, rows = evaluate(cases)
    metrics["test_set_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    write_json(REPORT_ROOT / "08_full_corpus_metrics.json", metrics)
    errors = [
        row
        for row in rows
        if (
            row["expected"] == "abstain"
            and row["decision_status"] != "abstain"
            or row["expected"] != "abstain"
            and not row["top5"]
        )
    ]
    with (REPORT_ROOT / "full_corpus_error_cases.jsonl").open(
        "w",
        encoding="utf-8",
    ) as stream:
        for row in errors:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manual = []
    seen_sutras = set()
    for row in rows:
        sutra_id = row.get("expected_sutra_id")
        if row["expected"] == "abstain" or not sutra_id or sutra_id in seen_sutras:
            continue
        seen_sutras.add(sutra_id)
        manual.append(
            {
                **row,
                "is_correct": row["top5"],
                "citation_source_match": row["citation_top5"],
                "review_note": (
                    "Top-5 contains the held-out source text and source citation."
                    if row["top5"] and row["citation_top5"]
                    else "Requires review: text or expected source citation is absent."
                ),
                "reviewer": "Codex local audit",
                "real_human_ground_truth": False,
            }
        )
        if len(manual) == 50:
            break
    with (REPORT_ROOT / "full_corpus_manual_audit.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in manual:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics["manual_audit_rows"] = len(manual)
    metrics["error_rows"] = len(errors)
    write_json(REPORT_ROOT / "08_full_corpus_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
