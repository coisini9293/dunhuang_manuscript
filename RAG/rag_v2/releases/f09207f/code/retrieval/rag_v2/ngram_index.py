from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from .storage import RagStorage


def _positions(text: str, gram: str) -> list[int]:
    found: list[int] = []
    start = 0
    while True:
        index = text.find(gram, start)
        if index < 0:
            return found
        found.append(index)
        start = index + 1


def build_ngram_indexes(storage: RagStorage, batch_size: int = 25_000) -> dict:
    totals: dict[int, dict[str, int]] = {}
    with storage.connect() as db:
        db.execute("DELETE FROM ngram_postings")
        db.execute("DELETE FROM ngram_terms")
        chunks = db.execute(
            "SELECT chunk_id, normalized_text, simplified_alias FROM chunks WHERE scale='short'"
        ).fetchall()
        for n in (2, 3, 4):
            batch: list[tuple[int, str, str, str]] = []
            document_frequency: dict[str, int] = defaultdict(int)
            posting_count = 0
            for chunk in chunks:
                variants = {chunk["normalized_text"], chunk["simplified_alias"]}
                by_gram: dict[str, set[int]] = defaultdict(set)
                for text in variants:
                    for index in range(max(0, len(text) - n + 1)):
                        by_gram[text[index : index + n]].add(index)
                for gram, positions in by_gram.items():
                    document_frequency[gram] += 1
                    batch.append(
                        (n, gram, chunk["chunk_id"], json.dumps(sorted(positions)))
                    )
                    posting_count += 1
                    if len(batch) >= batch_size:
                        db.executemany(
                            """
                            INSERT OR REPLACE INTO ngram_postings
                            (n, gram, chunk_id, positions) VALUES (?, ?, ?, ?)
                            """,
                            batch,
                        )
                        db.commit()
                        batch.clear()
            if batch:
                db.executemany(
                    """
                    INSERT OR REPLACE INTO ngram_postings
                    (n, gram, chunk_id, positions) VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()
            db.executemany(
                "INSERT OR REPLACE INTO ngram_terms(n, gram, document_frequency) VALUES(?, ?, ?)",
                [(n, gram, count) for gram, count in document_frequency.items()],
            )
            db.commit()
            totals[n] = {
                "terms": len(document_frequency),
                "postings": posting_count,
            }
    storage.set_meta("ngram_ready", "true")
    storage.set_meta("ngram_statistics", totals)
    return {"by_n": totals, "active": True}


def prove_ngram_participation(storage: RagStorage) -> dict[str, bool]:
    with storage.connect() as db:
        return {
            str(n): db.execute(
                "SELECT EXISTS(SELECT 1 FROM ngram_postings WHERE n=? LIMIT 1)", (n,)
            ).fetchone()[0]
            == 1
            for n in (2, 3, 4)
        }
