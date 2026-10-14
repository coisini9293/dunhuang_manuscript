"""决策策略：AutoHDR + CBETA 证据融合规则。"""
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class FusionDecision:
    damage_id: str
    status: str  # evidence_agreement|evidence_conflict|cbeta_ambiguous|autohdr_only|abstain|manual_confirmed
    ui_color: str
    selected_text: Optional[str] = None
    selected_source: Optional[str] = None
    allow_restore: bool = False
    autohdr_candidates: list = field(default_factory=list)
    cbeta_candidates: list = field(default_factory=list)
    cbeta_citation: str = ""
    reasoning: str = ""

    def to_dict(self):
        return {k:v for k,v in self.__dict__.items()}


def decide_fusion(
    damage_id: str,
    autohdr_candidates: list,
    cbeta_candidates: list,
    human_override: Optional[dict] = None,
) -> FusionDecision:
    """固定融合规则（不可自动选择分数高的一方）。"""
    if human_override and human_override.get("status") == "manual_confirmed":
        return FusionDecision(
            damage_id=damage_id,
            status="manual_confirmed",
            ui_color="blue",
            selected_text=human_override.get("selected_text"),
            selected_source=human_override.get("selected_source", "manual"),
            allow_restore=True,
            autohdr_candidates=autohdr_candidates,
            cbeta_candidates=cbeta_candidates,
            reasoning="人工确认",
        )

    if human_override and human_override.get("status") == "abstain":
        return FusionDecision(
            damage_id=damage_id, status="abstain", ui_color="gray",
            autohdr_candidates=autohdr_candidates,
            cbeta_candidates=cbeta_candidates,
            reasoning="人工选择不修复",
        )

    a_top = autohdr_candidates[0]["text"] if autohdr_candidates else ""
    c_top = cbeta_candidates[0]["missing_chars"] if cbeta_candidates else ""
    cbeta_citation = (
        cbeta_candidates[0].get("citation", "") if cbeta_candidates else ""
    )

    # A: 一致
    if a_top and c_top and a_top == c_top:
        return FusionDecision(damage_id=damage_id, status="evidence_agreement",
                             ui_color="blue", allow_restore=False,
                             autohdr_candidates=autohdr_candidates,
                             cbeta_candidates=cbeta_candidates,
                             cbeta_citation=cbeta_citation,
                             reasoning="AutoHDR与CBETA一致，需人工确认")

    # B: 冲突
    if a_top and c_top and a_top != c_top:
        return FusionDecision(damage_id=damage_id, status="evidence_conflict",
                             ui_color="yellow", allow_restore=False,
                             autohdr_candidates=autohdr_candidates,
                             cbeta_candidates=cbeta_candidates,
                             cbeta_citation=cbeta_citation,
                             reasoning="AutoHDR与CBETA不一致")

    # C: CBETA 多个不同文本候选
    distinct_cbeta_texts = {
        candidate.get("missing_chars", "")
        for candidate in cbeta_candidates
        if candidate.get("missing_chars", "")
    }
    if len(distinct_cbeta_texts) > 1:
        return FusionDecision(damage_id=damage_id, status="cbeta_ambiguous",
                             ui_color="yellow", allow_restore=False,
                             autohdr_candidates=autohdr_candidates,
                             cbeta_candidates=cbeta_candidates,
                             cbeta_citation=cbeta_citation,
                             reasoning="CBETA多候选歧义")

    # D: 仅 CBETA（一个文本，可保留多个重复位置引用）
    if c_top and not a_top:
        return FusionDecision(
            damage_id=damage_id,
            status="cbeta_only",
            ui_color="cyan",
            selected_text=c_top,
            selected_source="cbeta",
            allow_restore=False,
            autohdr_candidates=autohdr_candidates,
            cbeta_candidates=cbeta_candidates,
            cbeta_citation=cbeta_citation,
            reasoning="仅CBETA有原文候选，必须人工确认",
        )

    # E: CBETA 单侧只定位，不提供缺字
    if cbeta_candidates and not c_top:
        return FusionDecision(
            damage_id=damage_id,
            status="single_side_localization",
            ui_color="cyan",
            allow_restore=False,
            autohdr_candidates=autohdr_candidates,
            cbeta_candidates=cbeta_candidates,
            cbeta_citation=cbeta_citation,
            reasoning="单侧上下文只能定位候选，不宣称精确补字",
        )

    # F: 仅 AutoHDR
    if a_top and not c_top:
        return FusionDecision(damage_id=damage_id, status="autohdr_only",
                             ui_color="yellow", allow_restore=False,
                             autohdr_candidates=autohdr_candidates,
                             reasoning="仅AutoHDR有候选")

    # G: 两者均无
    return FusionDecision(damage_id=damage_id, status="abstain", ui_color="gray",
                         reasoning="无可靠候选")
