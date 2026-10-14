from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .bm25_index import B, K1, tokenize
from .normalize import query_aliases
from .storage import RagStorage


class SparseRetriever:
    def __init__(self, database_path: str | Path):
        self.storage = RagStorage(database_path, readonly=True)

    @staticmethod
    def _filters(
        sutra_ids: list[str] | None, translators: list[str] | None
    ) -> tuple[str, list[str]]:
        clauses: list[str] = []
        values: list[str] = []
        if sutra_ids:
            clauses.append(f"c.sutra_id IN ({','.join('?' for _ in sutra_ids)})")
            values.extend(sutra_ids)
        if translators:
            clauses.append(f"c.translator IN ({','.join('?' for _ in translators)})")
            values.extend(translators)
        return (" AND " + " AND ".join(clauses)) if clauses else "", values

    def ngram_recall(
        self,
        before: str,
        after: str,
        sutra_ids: list[str] | None,
        translators: list[str] | None,
        limit: int = 250,
    ) -> dict[str, dict]:
        scores: dict[str, dict] = defaultdict(
            lambda: {"hits": 0, "query_grams": 0, "matched_grams": set(), "positions": {}}
        )
        filter_sql, filter_values = self._filters(sutra_ids, translators)
        with self.storage.connect() as db:
            short_count = db.execute(
                "SELECT COUNT(*) FROM chunks WHERE scale='short'"
            ).fetchone()[0]
            grams_by_n: dict[int, set[tuple[str, str]]] = defaultdict(set)
            for side, value in (("left", before), ("right", after)):
                if not value:
                    continue
                for alias in query_aliases(value):
                    edge = alias[-16:] if side == "left" else alias[:16]
                    for n in (2, 3, 4):
                        for index in range(max(0, len(edge) - n + 1)):
                            grams_by_n[n].add((side, edge[index : index + n]))
            total_query_grams = sum(len(values) for values in grams_by_n.values())
            all_query_grams = sorted(
                {(n, gram) for n, values in grams_by_n.items() for _, gram in values}
            )
            if not all_query_grams:
                return {}
            df_values_sql = ",".join("(?,?)" for _ in all_query_grams)
            df_values: list[object] = []
            for n, gram in all_query_grams:
                df_values.extend([n, gram])
            df_rows = db.execute(
                f"""
                WITH q(n, gram) AS (VALUES {df_values_sql})
                SELECT q.n, q.gram, t.document_frequency
                FROM q JOIN ngram_terms t ON t.n=q.n AND t.gram=q.gram
                """,
                df_values,
            ).fetchall()
            df_lookup = {
                (row["n"], row["gram"]): row["document_frequency"] for row in df_rows
            }
            # Query only the rarest grams per n and side. All 2/3/4-gram
            # channels still participate, while high-frequency fan-out stays bounded.
            selected: set[tuple[int, str]] = set()
            for n, side_grams in grams_by_n.items():
                for side in ("left", "right"):
                    options = [
                        (n, gram)
                        for gram_side, gram in side_grams
                        if gram_side == side and (n, gram) in df_lookup
                    ]
                    options.sort(key=lambda item: (df_lookup[item], item[1]))
                    selected.update(options[:4])
            query_grams = sorted(selected)
            if not query_grams:
                return {}
            values_sql = ",".join("(?,?,?)" for _ in query_grams)
            query_values: list[object] = []
            for n, gram in query_grams:
                query_values.extend([n, gram, n])
            rows = db.execute(
                f"""
                WITH q(n, gram, weight) AS (VALUES {values_sql})
                SELECT p.chunk_id, SUM(q.weight) AS hits
                FROM q
                JOIN ngram_postings p ON p.n=q.n AND p.gram=q.gram
                JOIN ngram_terms t ON t.n=p.n AND t.gram=p.gram
                JOIN chunks c ON c.chunk_id=p.chunk_id
                WHERE t.document_frequency <= ? {filter_sql}
                GROUP BY p.chunk_id
                ORDER BY hits DESC
                LIMIT ?
                """,
                [*query_values, max(100, int(short_count * 0.20)), *filter_values, limit],
            ).fetchall()
            ranked_ids = [row["chunk_id"] for row in rows]
            if ranked_ids:
                id_placeholders = ",".join("?" for _ in ranked_ids)
                gram_conditions = " OR ".join("(p.n=? AND p.gram=?)" for _ in query_grams)
                detail_values: list[object] = []
                for n, gram in query_grams:
                    detail_values.extend([n, gram])
                details = db.execute(
                    f"""
                    SELECT p.chunk_id, p.n, p.gram
                    FROM ngram_postings p
                    WHERE p.chunk_id IN ({id_placeholders})
                      AND ({gram_conditions})
                    """,
                    [*ranked_ids, *detail_values],
                ).fetchall()
                for detail in details:
                    scores[detail["chunk_id"]]["matched_grams"].add(
                        f"{detail['n']}:{detail['gram']}"
                    )
            for row in rows:
                item = scores[row["chunk_id"]]
                item["hits"] = row["hits"]
                item["query_grams"] = total_query_grams
        ranked = sorted(scores, key=lambda key: scores[key]["hits"], reverse=True)[:limit]
        result: dict[str, dict] = {}
        maximum = max([scores[key]["hits"] for key in ranked] or [1])
        for rank, chunk_id in enumerate(ranked, 1):
            item = scores[chunk_id]
            result[chunk_id] = {
                "ngram_score": item["hits"] / maximum,
                "ngram_rank": rank,
                "matched_grams": sorted(item["matched_grams"]),
                "positions": item["positions"],
            }
        return result

    def bm25_recall(
        self,
        before: str,
        after: str,
        sutra_ids: list[str] | None,
        translators: list[str] | None,
        limit: int = 250,
    ) -> dict[str, dict]:
        query_terms = tokenize((before or "") + (after or ""))
        if not query_terms:
            return {}
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, set[str]] = defaultdict(set)
        filter_sql, filter_values = self._filters(sutra_ids, translators)
        with self.storage.connect() as db:
            meta = self.storage.get_meta()
            statistics = json.loads(meta.get("bm25_statistics", "{}"))
            document_count = max(1, int(statistics.get("documents", 1)))
            average_length = max(1.0, float(statistics.get("avgdl", 1.0)))
            terms = sorted(set(query_terms))
            placeholders = ",".join("?" for _ in terms)
            term_rows = db.execute(
                f"SELECT term, document_frequency FROM bm25_terms WHERE term IN ({placeholders})",
                terms,
            ).fetchall()
            query_terms_with_idf = [
                (
                    row["term"],
                    math.log(
                        1
                        + (document_count - row["document_frequency"] + 0.5)
                        / (row["document_frequency"] + 0.5)
                    ),
                )
                for row in term_rows
            ]
            if not query_terms_with_idf:
                return {}
            values_sql = ",".join("(?,?)" for _ in query_terms_with_idf)
            query_values: list[object] = []
            for term, idf in query_terms_with_idf:
                query_values.extend([term, idf])
            rows = db.execute(
                f"""
                WITH q(term, idf) AS (VALUES {values_sql})
                SELECT p.chunk_id,
                       SUM(q.idf * p.term_frequency * ?
                           / (p.term_frequency + ? * (1 - ? + ? * d.doc_len / ?)))
                           AS score
                FROM q
                JOIN bm25_postings p ON p.term=q.term
                JOIN bm25_docs d ON d.chunk_id=p.chunk_id
                JOIN chunks c ON c.chunk_id=p.chunk_id
                WHERE 1=1 {filter_sql}
                GROUP BY p.chunk_id
                ORDER BY score DESC
                LIMIT ?
                """,
                [
                    *query_values,
                    K1 + 1,
                    K1,
                    B,
                    B,
                    average_length,
                    *filter_values,
                    limit,
                ],
            ).fetchall()
            for row in rows:
                scores[row["chunk_id"]] = row["score"]
            ranked_ids = [row["chunk_id"] for row in rows]
            if ranked_ids:
                id_placeholders = ",".join("?" for _ in ranked_ids)
                details = db.execute(
                    f"""
                    SELECT chunk_id, term FROM bm25_postings
                    WHERE chunk_id IN ({id_placeholders})
                      AND term IN ({placeholders})
                    """,
                    [*ranked_ids, *terms],
                ).fetchall()
                for detail in details:
                    matched[detail["chunk_id"]].add(detail["term"])
        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
        maximum = max([scores[key] for key in ranked] or [1.0])
        return {
            chunk_id: {
                "bm25_score": scores[chunk_id] / maximum,
                "bm25_raw_score": scores[chunk_id],
                "bm25_rank": rank,
                "matched_terms": sorted(matched[chunk_id]),
            }
            for rank, chunk_id in enumerate(ranked, 1)
        }
