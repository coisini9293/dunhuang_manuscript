from __future__ import annotations

from collections import Counter, defaultdict

from .storage import RagStorage


K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    characters = list(text)
    bigrams = [text[index : index + 2] for index in range(max(0, len(text) - 1))]
    return [f"c:{char}" for char in characters] + [f"b:{gram}" for gram in bigrams]


def build_bm25_index(storage: RagStorage, batch_size: int = 25_000) -> dict:
    document_frequency: dict[str, int] = defaultdict(int)
    doc_count = 0
    total_length = 0
    posting_count = 0
    with storage.connect() as db:
        db.execute("DELETE FROM bm25_postings")
        db.execute("DELETE FROM bm25_terms")
        db.execute("DELETE FROM bm25_docs")
        chunks = db.execute(
            "SELECT chunk_id, normalized_text FROM chunks WHERE scale='medium'"
        ).fetchall()
        posting_batch: list[tuple[str, str, int]] = []
        doc_batch: list[tuple[str, int]] = []
        for chunk in chunks:
            terms = tokenize(chunk["normalized_text"])
            frequencies = Counter(terms)
            doc_count += 1
            total_length += len(terms)
            doc_batch.append((chunk["chunk_id"], len(terms)))
            for term, frequency in frequencies.items():
                document_frequency[term] += 1
                posting_batch.append((term, chunk["chunk_id"], frequency))
                posting_count += 1
            if len(posting_batch) >= batch_size:
                db.executemany(
                    """
                    INSERT OR REPLACE INTO bm25_postings
                    (term, chunk_id, term_frequency) VALUES (?, ?, ?)
                    """,
                    posting_batch,
                )
                db.executemany(
                    "INSERT OR REPLACE INTO bm25_docs(chunk_id, doc_len) VALUES(?, ?)",
                    doc_batch,
                )
                db.commit()
                posting_batch.clear()
                doc_batch.clear()
        if posting_batch:
            db.executemany(
                """
                INSERT OR REPLACE INTO bm25_postings
                (term, chunk_id, term_frequency) VALUES (?, ?, ?)
                """,
                posting_batch,
            )
        if doc_batch:
            db.executemany(
                "INSERT OR REPLACE INTO bm25_docs(chunk_id, doc_len) VALUES(?, ?)",
                doc_batch,
            )
        db.executemany(
            "INSERT OR REPLACE INTO bm25_terms(term, document_frequency) VALUES(?, ?)",
            list(document_frequency.items()),
        )
        db.commit()
    average_length = total_length / max(doc_count, 1)
    statistics = {
        "active": True,
        "documents": doc_count,
        "terms": len(document_frequency),
        "postings": posting_count,
        "avgdl": average_length,
        "k1": K1,
        "b": B,
        "tokenization": "character + character_bigram",
    }
    storage.set_meta("bm25_ready", "true")
    storage.set_meta("bm25_statistics", statistics)
    return statistics


def prove_bm25_participation(storage: RagStorage) -> bool:
    with storage.connect() as db:
        return (
            db.execute("SELECT EXISTS(SELECT 1 FROM bm25_postings LIMIT 1)").fetchone()[0]
            == 1
        )
