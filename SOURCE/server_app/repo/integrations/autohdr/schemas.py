"""AutoHDR 数据契约 schema。"""
from typing import Optional


class DamageRegion:
    def __init__(self, damage_id: str, bbox: list, **kwargs):
        self.damage_id = damage_id
        self.bbox = bbox
        self.polygon = kwargs.get("polygon")
        self.damage_type = kwargs.get("damage_type")
        self.damage_grade = kwargs.get("damage_grade")
        self.source = kwargs.get("source", "autohdr")
        self.reading_order = kwargs.get("reading_order")
        self.left_context = kwargs.get("left_context", "")
        self.right_context = kwargs.get("right_context", "")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class OCRCharacter:
    def __init__(self, char_id: str, bbox: list, text: str = "", **kwargs):
        self.char_id = char_id
        self.bbox = bbox
        self.text = text
        self.topk = kwargs.get("topk", [])
        self.score = kwargs.get("score")
        self.column_id = kwargs.get("column_id")
        self.reading_order = kwargs.get("reading_order")
        self.review_status = kwargs.get("review_status", "unreviewed")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class TextCandidate:
    def __init__(self, text: str, rank: int = 1, source: str = "autohdr", **kwargs):
        self.text = text
        self.rank = rank
        self.score = kwargs.get("score")
        self.source = source
        self.citation = kwargs.get("citation")
        self.allow_restore = kwargs.get("allow_restore", False)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class FusionDecision:
    def __init__(self, damage_id: str, status: str, **kwargs):
        self.damage_id = damage_id
        self.status = status
        self.ui_color = kwargs.get("ui_color", "gray")
        self.autohdr_candidates = kwargs.get("autohdr_candidates", [])
        self.cbeta_candidates = kwargs.get("cbeta_candidates", [])
        self.cbeta_citation = kwargs.get("cbeta_citation", "")
        self.reasoning = kwargs.get("reasoning", "")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class HumanDecision:
    def __init__(self, damage_id: str, selected_text: str = "", selected_source: str = "abstain",
                 status: str = "abstain", **kwargs):
        self.damage_id = damage_id
        self.selected_text = selected_text
        self.selected_source = selected_source
        self.status = status
        self.reviewer_note = kwargs.get("reviewer_note", "")
        self.timestamp = kwargs.get("timestamp", "")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
