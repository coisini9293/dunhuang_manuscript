from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalization import normalize_conservatively


@dataclass(frozen=True, slots=True)
class QueryVariant:
    variant_id: str
    before: str
    after: str
    strategy: str
    reason: str
    applied_rules: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["applied_rules"] = list(self.applied_rules)
        return value


def load_confusion_map(path: str | Path | None = None) -> dict:
    source = Path(path) if path else Path(__file__).with_name("confusion_map.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["_source"] = str(source)
    return payload


def _single_substitutions(text: str, groups: list[list[str]]):
    for index, char in enumerate(text):
        for group in groups:
            if char not in group:
                continue
            for replacement in group:
                if replacement != char:
                    yield (
                        text[:index] + replacement + text[index + 1 :],
                        f"{char}->{replacement}@{index}",
                    )


def build_query_variants(
    context_before: str,
    context_after: str,
    *,
    max_variants: int = 12,
    confusion_map_path: str | Path | None = None,
) -> tuple[list[QueryVariant], dict]:
    if not 1 <= max_variants <= 12:
        raise ValueError("max_variants must be between 1 and 12")
    normalized_before, before_rules = normalize_conservatively(context_before)
    normalized_after, after_rules = normalize_conservatively(context_after)
    confusion = load_confusion_map(confusion_map_path)
    groups = confusion.get("groups", [])
    variants: list[QueryVariant] = []
    seen: set[tuple[str, str]] = set()

    def add(before: str, after: str, strategy: str, reason: str, rules=()) -> None:
        key = (before, after)
        if len(variants) >= max_variants or key in seen or not (before or after):
            return
        seen.add(key)
        variants.append(
            QueryVariant(
                variant_id=f"qv-{len(variants) + 1:02d}",
                before=before,
                after=after,
                strategy=strategy,
                reason=reason,
                applied_rules=tuple(rules),
            )
        )

    add(context_before or "", context_after or "", "raw", "unaltered baseline query")
    rule_names = tuple(
        item["rule"] for item in [*before_rules, *after_rules]
    )
    add(
        normalized_before,
        normalized_after,
        "conservative_normalization",
        "NFKC/control/whitespace cleanup only",
        rule_names,
    )
    if len(normalized_before) > 1 or len(normalized_after) > 1:
        before_mid = max(1, len(normalized_before) // 2)
        after_mid = max(1, len(normalized_after) // 2)
        add(
            normalized_before[:before_mid],
            normalized_after[:after_mid],
            "front_half",
            "bounded half-context probe",
        )
        add(
            normalized_before[before_mid:],
            normalized_after[after_mid:],
            "back_half",
            "bounded half-context probe",
        )
    if len(normalized_before) > 12:
        add(
            normalized_before[-12:],
            normalized_after,
            "nearest_left_context",
            "retain the 12 glyphs nearest the gap",
        )
    if len(normalized_after) > 12:
        add(
            normalized_before,
            normalized_after[:12],
            "nearest_right_context",
            "retain the 12 glyphs nearest the gap",
        )
    if len(normalized_before) > 6:
        add(
            normalized_before[-8:],
            normalized_after,
            "short_left_context",
            "reduce distant OCR noise on the left",
        )
    if len(normalized_after) > 6:
        add(
            normalized_before,
            normalized_after[:8],
            "short_right_context",
            "reduce distant OCR noise on the right",
        )
    if normalized_before:
        add(
            normalized_before[:-1],
            normalized_after,
            "drop_boundary_left",
            "remove one unscored glyph nearest the gap; no confidence claimed",
        )
    if normalized_after:
        add(
            normalized_before,
            normalized_after[1:],
            "drop_boundary_right",
            "remove one unscored glyph nearest the gap; no confidence claimed",
        )
    for replacement, rule in _single_substitutions(normalized_before, groups):
        add(
            replacement,
            normalized_after,
            "confusion_substitution_left",
            "one configured OCR-confusion substitution",
            (rule,),
        )
    for replacement, rule in _single_substitutions(normalized_after, groups):
        add(
            normalized_before,
            replacement,
            "confusion_substitution_right",
            "one configured OCR-confusion substitution",
            (rule,),
        )
    audit = {
        "raw_query": {
            "context_before": context_before or "",
            "context_after": context_after or "",
        },
        "normalized_query": {
            "context_before": normalized_before,
            "context_after": normalized_after,
        },
        "applied_rules": {
            "context_before": before_rules,
            "context_after": after_rules,
        },
        "confusion_map": {
            "version": confusion.get("version", ""),
            "provenance": confusion.get("provenance", {}),
            "source": confusion["_source"],
        },
    }
    return variants, audit
