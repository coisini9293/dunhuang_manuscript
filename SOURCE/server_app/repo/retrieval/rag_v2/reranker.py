from __future__ import annotations

from dataclasses import dataclass

from Levenshtein import ratio

from .normalize import traditional_alias


DEFAULT_WEIGHTS = {
    "continuity": 0.35,
    "edit": 0.25,
    "ngram": 0.15,
    "bm25": 0.10,
    "vector": 0.10,
    "citation": 0.05,
}


def edit_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return max(
        ratio(left, right),
        ratio(traditional_alias(left), traditional_alias(right)),
    )


@dataclass(slots=True)
class BreakpointMatch:
    gap_start: int
    gap_end: int
    gap: str
    matched_before: str
    matched_after: str
    left_edit: float
    right_edit: float
    continuity: float
    gap_consistency: float


def best_breakpoint(
    text: str,
    before: str,
    after: str,
    max_missing: int,
    expected_missing_length: int | None,
) -> BreakpointMatch | None:
    if not before or not after or not text:
        return None
    before_window = before[-min(12, len(before)) :]
    after_window = after[: min(12, len(after))]
    lengths = (
        [expected_missing_length]
        if expected_missing_length is not None
        else list(range(1, max_missing + 1))
    )
    best: tuple[float, BreakpointMatch] | None = None
    for gap_start in range(1, len(text)):
        available_before = text[max(0, gap_start - len(before_window)) : gap_start]
        if len(available_before) < min(4, len(before_window)):
            continue
        left_score = edit_similarity(available_before, before_window[-len(available_before) :])
        if left_score < 0.55:
            continue
        for gap_length in lengths:
            if gap_length is None or gap_length < 1:
                continue
            gap_end = gap_start + gap_length
            if gap_end >= len(text):
                continue
            available_after = text[gap_end : gap_end + len(after_window)]
            if len(available_after) < min(4, len(after_window)):
                continue
            right_score = edit_similarity(available_after, after_window[: len(available_after)])
            if right_score < 0.55:
                continue
            continuity = min(left_score, right_score)
            consistency = (
                1.0
                if expected_missing_length is None or gap_length == expected_missing_length
                else 0.0
            )
            score = 0.45 * continuity + 0.45 * ((left_score + right_score) / 2) + 0.10 * consistency
            match = BreakpointMatch(
                gap_start=gap_start,
                gap_end=gap_end,
                gap=text[gap_start:gap_end],
                matched_before=available_before,
                matched_after=available_after,
                left_edit=left_score,
                right_edit=right_score,
                continuity=continuity,
                gap_consistency=consistency,
            )
            if best is None or score > best[0]:
                best = (score, match)
    return best[1] if best else None


def single_side_similarity(text: str, context: str) -> tuple[float, int]:
    if not text or not context:
        return 0.0, -1
    window = context[-min(16, len(context)) :]
    best_score, best_position = 0.0, -1
    for position in range(0, max(1, len(text) - len(window) + 1)):
        candidate = text[position : position + len(window)]
        score = edit_similarity(candidate, window[: len(candidate)])
        if score > best_score:
            best_score, best_position = score, position
    return best_score, best_position
