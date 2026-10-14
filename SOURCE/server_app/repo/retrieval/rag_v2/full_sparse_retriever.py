from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from .compact_ngram import lookup_positions
from .full_storage import FullRagStorage


class FullSparseRetriever:
    def __init__(self, database: str | Path, ngram_root: str | Path):
        self.database = Path(database)
        self.ngram_root = Path(ngram_root)
        self.storage = FullRagStorage(self.database, readonly=True)
        self.edition_filter: ContextVar[list[str] | None] = ContextVar(
            "full_rag_editions",
            default=None,
        )
        self._load_group_boundaries()

    def _load_group_boundaries(self) -> None:
        with self.storage.connect() as db:
            rows = db.execute(
                """
                SELECT group_id,global_start,global_end,sutra_id,translator,edition
                FROM groups ORDER BY global_start
                """
            ).fetchall()
        self.group_ids = np.asarray([row["group_id"] for row in rows], dtype=np.int64)
        self.group_starts = np.asarray(
            [row["global_start"] for row in rows], dtype=np.uint32
        )
        self.group_ends = np.asarray(
            [row["global_end"] for row in rows], dtype=np.uint32
        )
        self.group_filters = {
            int(row["group_id"]): (
                row["sutra_id"],
                row["translator"],
                row["edition"],
            )
            for row in rows
        }

    def _position_groups(
        self,
        positions: np.ndarray,
        sutra_ids: list[str] | None,
        translators: list[str] | None,
        editions: list[str] | None,
    ) -> list[tuple[int, int]]:
        if not len(positions):
            return []
        indexes = np.searchsorted(self.group_ends, positions, side="right")
        result: list[tuple[int, int]] = []
        sutra_filter = set(sutra_ids or [])
        translator_filter = set(translators or [])
        edition_filter = set(editions or [])
        for position, index in zip(positions.tolist(), indexes.tolist()):
            if index >= len(self.group_ids):
                continue
            group_id = int(self.group_ids[index])
            sutra_id, translator, edition = self.group_filters[group_id]
            if sutra_filter and sutra_id not in sutra_filter:
                continue
            if translator_filter and translator not in translator_filter:
                continue
            if edition_filter and edition not in edition_filter:
                continue
            result.append((group_id, int(position) - int(self.group_starts[index])))
        return result

    def ngram_recall(
        self,
        before: str,
        after: str,
        sutra_ids: list[str] | None = None,
        translators: list[str] | None = None,
        editions: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, dict]:
        editions = editions if editions is not None else self.edition_filter.get()
        selected: list[tuple[str, np.ndarray]] = []
        for n in (2, 3, 4):
            for side in (before, after):
                grams = list(
                    dict.fromkeys(
                        side[index : index + n]
                        for index in range(max(0, len(side) - n + 1))
                    )
                )
                if not grams:
                    continue
                postings = [
                    (gram, lookup_positions(self.ngram_root, gram, limit=5001))
                    for gram in grams
                ]
                postings.sort(key=lambda item: len(item[1]))
                selected.extend(postings[:2])
        group_hits: dict[int, dict] = defaultdict(
            lambda: {"grams": set(), "positions": [], "hits": 0}
        )
        position_hits: list[tuple[str, int, int]] = []
        for gram, positions in selected:
            for group_id, local_position in self._position_groups(
                positions,
                sutra_ids,
                translators,
                editions,
            ):
                position_hits.append((gram, group_id, local_position))
                hit = group_hits[group_id]
                hit["grams"].add(f"{len(gram)}:{gram}")
                hit["positions"].append(local_position)
                hit["hits"] += 1
        if not group_hits:
            return {}
        # The compact index stores 32-bit hashes. Rank a slightly wider
        # shortlist, then verify every retained posting against canonical text
        # so a hash collision can never become a retrieval candidate.
        verification_groups = sorted(
            group_hits,
            key=lambda group_id: (
                len(group_hits[group_id]["grams"]),
                group_hits[group_id]["hits"],
            ),
            reverse=True,
        )[: min(600, len(group_hits))]
        verified: dict[int, dict] = defaultdict(
            lambda: {"grams": set(), "positions": [], "hits": 0}
        )
        group_texts: dict[int, str] = {}
        with self.storage.connect() as db:
            for offset in range(0, len(verification_groups), 400):
                batch = verification_groups[offset : offset + 400]
                for row in db.execute(
                    f"SELECT group_id,normalized_text FROM groups WHERE group_id IN "
                    f"({','.join('?' for _ in batch)})",
                    batch,
                ):
                    group_texts[int(row["group_id"])] = row["normalized_text"]
        allowed = set(verification_groups)
        for gram, group_id, local_position in position_hits:
            if group_id not in allowed:
                continue
            text = group_texts.get(group_id, "")
            if text[local_position : local_position + len(gram)] != gram:
                continue
            hit = verified[group_id]
            hit["grams"].add(f"{len(gram)}:{gram}")
            hit["positions"].append(local_position)
            hit["hits"] += 1
        best_groups = sorted(
            verified,
            key=lambda group_id: (
                len(verified[group_id]["grams"]),
                verified[group_id]["hits"],
            ),
            reverse=True,
        )[: min(300, len(verified))]
        results: dict[str, dict] = {}
        with self.storage.connect() as db:
            for offset in range(0, len(best_groups), 400):
                batch = best_groups[offset : offset + 400]
                rows = db.execute(
                    f"SELECT chunk_id,group_id,char_start,char_end "
                    f"FROM medium_chunks WHERE group_id IN "
                    f"({','.join('?' for _ in batch)})",
                    batch,
                ).fetchall()
                by_group: dict[int, list] = defaultdict(list)
                for row in rows:
                    by_group[int(row["group_id"])].append(row)
                for group_id in batch:
                    details = verified[group_id]
                    positions = details["positions"]
                    chunks = [
                        row
                        for row in by_group[group_id]
                        if any(
                            int(row["char_start"]) <= position < int(row["char_end"])
                            for position in positions
                        )
                    ]
                    if not chunks:
                        chunks = by_group[group_id][:2]
                    score = min(
                        1.0,
                        len(details["grams"]) / max(len(selected), 1)
                        + min(details["hits"], 10) * 0.02,
                    )
                    for row in chunks:
                        chunk_id = row["chunk_id"]
                        current = results.get(chunk_id)
                        if current is None or score > current["ngram_score"]:
                            results[chunk_id] = {
                                "ngram_score": score,
                                "matched_grams": sorted(details["grams"]),
                            }
        ordered = sorted(
            results.items(),
            key=lambda item: item[1]["ngram_score"],
            reverse=True,
        )[:limit]
        return {
            chunk_id: {**details, "ngram_rank": rank}
            for rank, (chunk_id, details) in enumerate(ordered, 1)
        }

    @staticmethod
    def _bm25_terms(before: str, after: str) -> list[str]:
        text = before + after
        bigrams = [
            text[index : index + 2]
            for index in range(max(0, len(text) - 1))
        ]
        chars = list(text) if len(bigrams) < 2 else []
        return list(dict.fromkeys(bigrams + chars))

    def bm25_recall(
        self,
        before: str,
        after: str,
        sutra_ids: list[str] | None = None,
        translators: list[str] | None = None,
        editions: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, dict]:
        editions = editions if editions is not None else self.edition_filter.get()
        terms = self._bm25_terms(before, after)
        if not terms:
            return {}
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        clauses = ["bm25_fts MATCH ?"]
        parameters: list[object] = [expression]
        if sutra_ids:
            clauses.append(
                f"m.sutra_id IN ({','.join('?' for _ in sutra_ids)})"
            )
            parameters.extend(sutra_ids)
        if translators:
            clauses.append(
                f"m.translator IN ({','.join('?' for _ in translators)})"
            )
            parameters.extend(translators)
        if editions:
            clauses.append(
                f"m.edition IN ({','.join('?' for _ in editions)})"
            )
            parameters.extend(editions)
        parameters.append(limit)
        with self.storage.connect() as db:
            rows = db.execute(
                f"""
                SELECT m.chunk_id,bm25(bm25_fts) AS raw_score
                FROM bm25_fts
                JOIN medium_chunks m ON m.chunk_key=bm25_fts.rowid
                WHERE {' AND '.join(clauses)}
                ORDER BY raw_score
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        if not rows:
            return {}
        raw_values = [abs(float(row["raw_score"])) for row in rows]
        maximum = max(raw_values) or 1.0
        return {
            row["chunk_id"]: {
                "bm25_score": abs(float(row["raw_score"])) / maximum,
                "bm25_raw_score": float(row["raw_score"]),
                "bm25_rank": rank,
                "matched_terms": terms,
            }
            for rank, row in enumerate(rows, 1)
        }
