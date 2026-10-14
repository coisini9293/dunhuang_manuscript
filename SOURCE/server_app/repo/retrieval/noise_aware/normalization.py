from __future__ import annotations

import re
import unicodedata


_SPACE = re.compile(r"\s+")
_CONTROL_CATEGORIES = {"Cc", "Cf"}


def normalize_conservatively(text: str) -> tuple[str, list[dict[str, str]]]:
    """Normalize transport/OCR noise without rewriting visible CJK glyphs."""
    raw = text or ""
    rules: list[dict[str, str]] = []
    normalized = unicodedata.normalize("NFKC", raw)
    if normalized != raw:
        rules.append({"rule": "unicode_nfkc", "reason": "canonical width/form"})
    without_controls = "".join(
        char
        for char in normalized
        if unicodedata.category(char) not in _CONTROL_CATEGORIES
    )
    if without_controls != normalized:
        rules.append({"rule": "remove_controls", "reason": "non-visible OCR control"})
    collapsed = _SPACE.sub("", without_controls)
    if collapsed != without_controls:
        rules.append({"rule": "remove_whitespace", "reason": "layout whitespace"})
    return collapsed, rules
