from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from .schema import AtomicRecord, TextChunk


SHORT_SIZE = 32
SHORT_OVERLAP = 16
MEDIUM_TARGET = 160
MEDIUM_OVERLAP = 64
MEDIUM_MIN = 128
MEDIUM_MAX = 192


def _hash64(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _group_records(records: Iterable[AtomicRecord]) -> list[list[AtomicRecord]]:
    grouped: dict[tuple[str, int, str], list[AtomicRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.sutra_id, record.volume, record.chapter_name)].append(record)
    groups = []
    for key in sorted(grouped):
        groups.append(
            sorted(
                grouped[key],
                key=lambda item: (item.page, item.column, item.line, item.record_id),
            )
        )
    return groups


def _group_offsets(records: list[AtomicRecord]) -> tuple[str, list[tuple[int, int, AtomicRecord]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int, AtomicRecord]] = []
    cursor = 0
    for record in records:
        text = record.normalized_text
        if not text:
            continue
        pieces.append(text)
        offsets.append((cursor, cursor + len(text), record))
        cursor += len(text)
    return "".join(pieces), offsets


def _covering_records(
    offsets: list[tuple[int, int, AtomicRecord]], start: int, end: int
) -> list[AtomicRecord]:
    return [record for left, right, record in offsets if right > start and left < end]


def _make_chunk(
    scale: str,
    records: list[AtomicRecord],
    normalized_text: str,
    char_start: int,
    char_end: int,
    ordinal: int,
) -> TextChunk:
    first, last = records[0], records[-1]
    identity = (
        f"{first.sutra_id}|{first.volume}|{first.chapter_name}|{scale}|"
        f"{first.record_id}|{last.record_id}|{char_start}|{char_end}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    chunk_id = f"{first.sutra_id}_v{first.volume:03d}_{scale}_{digest}"
    original = "\n".join(record.original_text for record in records)
    return TextChunk(
        chunk_id=chunk_id,
        scale=scale,
        sutra_id=first.sutra_id,
        canonical_title=first.canonical_title,
        translator=first.translator,
        edition=first.edition,
        volume=first.volume,
        chapter_name=first.chapter_name,
        record_start_id=first.record_id,
        record_end_id=last.record_id,
        page_start=first.page,
        column_start=first.column,
        line_start=first.line,
        page_end=last.page,
        column_end=last.column,
        line_end=last.line,
        line_heads=[record.line_head for record in records],
        original_text=original,
        normalized_text=normalized_text,
        simplified_alias="".join(record.simplified_alias for record in records),
        char_start=char_start,
        char_end=char_end,
        source_version=first.source_version,
        source_commit=first.source_commit,
        text_hash=_hash64(normalized_text),
    )


def build_short_chunks(records: Iterable[AtomicRecord]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    step = SHORT_SIZE - SHORT_OVERLAP
    for group in _group_records(records):
        text, offsets = _group_offsets(group)
        if not text:
            continue
        starts = list(range(0, max(len(text) - SHORT_SIZE + 1, 1), step))
        tail_start = max(0, len(text) - SHORT_SIZE)
        if tail_start not in starts:
            starts.append(tail_start)
        for ordinal, start in enumerate(sorted(set(starts))):
            end = min(start + SHORT_SIZE, len(text))
            covered = _covering_records(offsets, start, end)
            chunks.append(_make_chunk("short", covered, text[start:end], start, end, ordinal))
    return chunks


def build_medium_chunks(records: Iterable[AtomicRecord]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for group in _group_records(records):
        valid = [record for record in group if record.normalized_text]
        if not valid:
            continue
        prefix = [0]
        for record in valid:
            prefix.append(prefix[-1] + len(record.normalized_text))
        start_index = 0
        ordinal = 0
        while start_index < len(valid):
            end_index = start_index
            while end_index < len(valid) and prefix[end_index + 1] - prefix[start_index] < MEDIUM_TARGET:
                end_index += 1
            end_index = min(end_index, len(valid) - 1)
            while (
                end_index + 1 < len(valid)
                and prefix[end_index + 2] - prefix[start_index] <= MEDIUM_MAX
                and prefix[end_index + 1] - prefix[start_index] < MEDIUM_MIN
            ):
                end_index += 1
            selected = valid[start_index : end_index + 1]
            normalized = "".join(record.normalized_text for record in selected)
            chunks.append(
                _make_chunk(
                    "medium",
                    selected,
                    normalized,
                    prefix[start_index],
                    prefix[end_index + 1],
                    ordinal,
                )
            )
            if end_index == len(valid) - 1:
                break
            desired_next = max(prefix[start_index], prefix[end_index + 1] - MEDIUM_OVERLAP)
            next_index = start_index + 1
            while next_index < end_index and prefix[next_index] < desired_next:
                next_index += 1
            start_index = max(start_index + 1, next_index)
            ordinal += 1
    return chunks
