from __future__ import annotations

import json
import sqlite3


def row_to_chunk(row: sqlite3.Row) -> dict:
    chunk = dict(row)
    chunk["line_heads"] = json.loads(chunk["line_heads"])
    return chunk


def citation_text(chunk: dict) -> str:
    return (
        f"{chunk['sutra_id']}《{chunk['canonical_title']}》"
        f"卷{chunk['volume']} "
        f"p{chunk['page_start']:04d}{chunk['column_start']}{chunk['line_start']:02d}"
        f"–p{chunk['page_end']:04d}{chunk['column_end']}{chunk['line_end']:02d}"
        f"（{chunk['translator']}，{chunk['edition']}）"
    )


def citation_complete(chunk: dict) -> bool:
    required = [
        "sutra_id",
        "canonical_title",
        "translator",
        "edition",
        "volume",
        "page_start",
        "column_start",
        "line_start",
        "page_end",
        "column_end",
        "line_end",
        "source_version",
        "source_commit",
        "text_hash",
    ]
    return all(chunk.get(key) not in (None, "") for key in required) and len(
        chunk.get("text_hash", "")
    ) == 64
