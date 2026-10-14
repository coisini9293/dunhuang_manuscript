from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from zhconv import convert as zh_convert


PROFILE = "cbeta_rag_v2"
VERSION = "2.0.0"
_CJK_OR_GAIJI = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|CB\d{5}")


@lru_cache(maxsize=1)
def confusion_map() -> dict[str, set[str]]:
    path = Path(__file__).resolve().parent / "config" / "ocr_confusions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, set[str]] = {}
    for group in payload["groups"]:
        values = set(group)
        for char in group:
            mapping[char] = values
    return mapping


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize for retrieval while preserving a normalized→original index map."""
    original = text or ""
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for original_index, char in enumerate(original):
        folded = unicodedata.normalize("NFKC", char)
        for folded_char in folded:
            if _CJK_OR_GAIJI.fullmatch(folded_char):
                normalized_chars.append(folded_char)
                index_map.append(original_index)
    return "".join(normalized_chars), index_map


def normalize_query(text: str) -> str:
    return normalize_with_map(text)[0]


def simplified_alias(text: str) -> str:
    return zh_convert(text or "", "zh-hans")


def traditional_alias(text: str) -> str:
    return zh_convert(text or "", "zh-hant")


def query_aliases(text: str) -> list[str]:
    normalized = normalize_query(text)
    aliases = [normalized, traditional_alias(normalized), simplified_alias(normalized)]
    return list(dict.fromkeys(value for value in aliases if value))


def chars_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if traditional_alias(left) == traditional_alias(right):
        return True
    return right in confusion_map().get(left, set())
