"""AutoHDR Worker Schemas — 阶段结果数据结构。"""

from pydantic import BaseModel
from typing import Optional


class CharResult(BaseModel):
    char_id: str
    bbox: list[int]  # [x, y, w, h]
    ocr_topk: list[dict] = []
    ocr_top1: str = ""
    ocr_score: Optional[float] = None
    damage_grade: Optional[str] = None
    is_damaged: bool = False
    reading_order: int = 0


class DamageResult(BaseModel):
    damage_id: str
    bbox: list[int]
    placeholder: str = "<|extra_0|>"
    left_context: str = ""
    right_context: str = ""
    column_context: str = ""
    neighbor_char_ids: list[str] = []


class Stage1Result(BaseModel):
    run_id: str
    image_size: list[int]
    chars: list[CharResult] = []
    damages: list[DamageResult] = []
    reading_order: str = "vertical_r2l"
    ocr_full_text: str = ""


class TextCandidate(BaseModel):
    text: str
    score: Optional[float] = None
    rank: int = 1
    source: str = "autohdr"


class Stage2Result(BaseModel):
    run_id: str
    damage_predictions: list[dict] = []


class Stage3Result(BaseModel):
    run_id: str
    restored_path: str = ""
    confirmed_characters: dict = {}
    runtime_sec: float = 0
