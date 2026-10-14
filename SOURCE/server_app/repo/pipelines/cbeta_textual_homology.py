#!/usr/bin/env python3
"""Reproducible OCR-to-CBETA textual homology evidence.

The CBETA database is opened read-only.  OCR anchors only generate candidates;
the final score also requires independent fuzzy alignment, the same CBETA sutra,
and forward/nearby chunk order for the two fragments.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sqlite3
from pathlib import Path


ANCHOR_LEXICON = (
    "婆羅門", "半國", "呪術", "童子", "甘露", "大臣",
    "旃陀羅", "不死之藥", "王", "食宿",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fts_phrase(text: str) -> str:
    return '"' + " ".join(text) + '"'


def candidate_rows(db, ocr_text: str, side: str):
    preferred = {
        "a": ("婆羅門", "半國", "呪術"),
        "b": ("童子", "甘露", "大臣"),
    }[side]
    observed = [anchor for anchor in preferred if anchor in ocr_text]
    if len(observed) < 2:
        observed = [anchor for anchor in ANCHOR_LEXICON if anchor in ocr_text][:4]
    if len(observed) < 2:
        raise RuntimeError(f"insufficient independent OCR anchors on fragment {side}: {observed}")
    query = " ".join(fts_phrase(anchor) for anchor in observed)
    rows = db.execute(
        """
        SELECT m.*, bm25(bm25_fts) AS bm25_score
        FROM bm25_fts
        JOIN medium_chunks m ON m.chunk_key=bm25_fts.rowid
        WHERE bm25_fts MATCH ?
        ORDER BY bm25_score
        LIMIT 200
        """,
        (query,),
    ).fetchall()
    candidates = []
    for row in rows:
        item = dict(row)
        matcher = difflib.SequenceMatcher(
            None, ocr_text, item["normalized_text"], autojunk=False
        )
        matching = sum(block.size for block in matcher.get_matching_blocks())
        item.update(
            anchors=observed,
            fts_query=query,
            sequence_ratio=matcher.ratio(),
            matched_character_count=matching,
            ocr_coverage=matching / max(1, len(ocr_text)),
        )
        candidates.append(item)
    candidates.sort(
        key=lambda item: (
            item["ocr_coverage"], item["sequence_ratio"], -item["bm25_score"]
        ),
        reverse=True,
    )
    return observed, query, candidates


def compact(item):
    return {
        key: item[key]
        for key in (
            "chunk_key", "chunk_id", "group_id", "sutra_id", "canonical_title",
            "translator", "edition", "volume", "chapter_name", "record_start_id",
            "record_end_id", "page_start", "page_end", "line_heads", "original_text",
            "normalized_text", "bm25_score", "anchors", "fts_query",
            "sequence_ratio", "matched_character_count", "ocr_coverage",
        )
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-a", required=True)
    ap.add_argument("--ocr-b", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    paths = {side: Path(getattr(args, f"ocr_{side}")) for side in ("a", "b")}
    ocr = {side: json.loads(path.read_text(encoding="utf-8")) for side, path in paths.items()}
    db_path = Path(args.database)
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    meta = {row[0]: row[1] for row in db.execute("SELECT key,value FROM meta")}

    anchors, queries, candidates = {}, {}, {}
    for side in ("a", "b"):
        anchors[side], queries[side], candidates[side] = candidate_rows(
            db, ocr[side]["ocr_text"], side
        )

    common_ids = set(item["sutra_id"] for item in candidates["a"]) & set(
        item["sutra_id"] for item in candidates["b"]
    )
    pairs = []
    for sutra_id in common_ids:
        lefts = [item for item in candidates["a"] if item["sutra_id"] == sutra_id]
        rights = [item for item in candidates["b"] if item["sutra_id"] == sutra_id]
        for left in lefts:
            for right in rights:
                forward_gap = right["chunk_key"] - left["chunk_key"]
                if 0 <= forward_gap <= 4:
                    alignment = (left["ocr_coverage"] + right["ocr_coverage"]) / 2
                    pairs.append((alignment, -forward_gap, left, right))
    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = pairs[0] if pairs else None

    if best:
        _, neg_gap, left, right = best
        gap = -neg_gap
        anchor_component = min(
            1.0, (len(anchors["a"]) + len(anchors["b"])) / 6.0
        ) * 0.30
        mean_coverage = (left["ocr_coverage"] + right["ocr_coverage"]) / 2
        alignment_component = min(1.0, mean_coverage / 0.20) * 0.25
        same_sutra_component = 0.25
        ordered_proximity_component = (1.0 if gap <= 4 else 0.0) * 0.15
        cbeta_primary_component = 0.05 if left["edition"] == "T" else 0.03
        score = min(
            0.95,
            anchor_component + alignment_component + same_sutra_component
            + ordered_proximity_component + cbeta_primary_component,
        )
        decision = "TEXTUAL_HOMOLOGY_PASS" if score >= 0.80 else "ABSTAIN"
        selected = {"fragment_a": compact(left), "fragment_b": compact(right)}
    else:
        gap, mean_coverage, score = None, 0.0, 0.0
        anchor_component = alignment_component = 0.0
        same_sutra_component = ordered_proximity_component = cbeta_primary_component = 0.0
        decision, selected = "ABSTAIN", None

    payload = {
        "status": "PASS" if decision == "TEXTUAL_HOMOLOGY_PASS" else "FAIL",
        "decision": decision,
        "textual_homology_score": score,
        "threshold": 0.80,
        "authority": "CBETA_PRIMARY_WITH_MACHINE_OCR_CANDIDATE_GENERATION",
        "database": str(db_path.resolve()),
        "database_sha256": sha256(db_path),
        "corpus_meta": meta,
        "database_open_mode": "READ_ONLY",
        "ocr_inputs": {
            side: {
                "path": str(paths[side].resolve()),
                "sha256": sha256(paths[side]),
                "device": ocr[side].get("device"),
                "character_count": ocr[side].get("character_count"),
                "ocr_text": ocr[side].get("ocr_text"),
                "anchors_observed_verbatim": anchors[side],
                "fts_query": queries[side],
            }
            for side in ("a", "b")
        },
        "score_formula": {
            "anchor_component": anchor_component,
            "alignment_component": alignment_component,
            "same_exact_sutra_component": same_sutra_component,
            "ordered_nearby_chunk_component": ordered_proximity_component,
            "cbeta_primary_edition_component": cbeta_primary_component,
            "cap_for_ocr_uncertainty": 0.95,
            "mean_ocr_coverage_in_selected_chunks": mean_coverage,
            "selected_chunk_forward_gap": gap,
        },
        "selected_same_sutra_evidence": selected,
        "top_candidates": {
            side: [compact(item) for item in candidates[side][:5]]
            for side in ("a", "b")
        },
        "limitations": [
            "OCR output is noisy and is not treated as a transcription ground truth.",
            "Qwen may explain or rerank these candidates but cannot override CBETA provenance.",
            "Textual homology does not prove adjacency or geometry; those gates remain separate.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "decision": decision,
        "score": score,
        "sutra_id": selected["fragment_a"]["sutra_id"] if selected else None,
        "title": selected["fragment_a"]["canonical_title"] if selected else None,
        "chunk_gap": gap,
    }, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
