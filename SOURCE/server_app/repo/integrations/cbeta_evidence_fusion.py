"""CBETA 证据融合：AutoHDR Stage 2 → CBETA 检索 → 融合决策。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.rag_agent import RagAgent
from integrations.decision_policy import decide_fusion, FusionDecision


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = RagAgent()
    return _agent


def fuse_for_damage(damage: dict, sutra_hint: str = "大方广佛华严经") -> FusionDecision:
    """对单个缺损执行 CBETA 检索 + 融合决策。"""
    agent = get_agent()
    ctx_before = damage.get("left_context", "")
    ctx_after = damage.get("right_context", "")

    # CBETA 检索
    decision = agent.decide(
        ctx_before,
        ctx_after,
        sutra_hint=sutra_hint,
        expected_missing_length=damage.get("expected_missing_length"),
        sutra_ids=damage.get("sutra_ids"),
        translators=damage.get("translators"),
    )
    cbeta_candidates = []
    if decision.source == "cbeta_v2" and decision.candidates:
        cbeta_candidates = [
            {
                **candidate,
                "missing_chars": candidate.get("candidate_text", ""),
            }
            for candidate in decision.candidates
        ]
    elif decision.char:
        cbeta_candidates = [{
            "missing_chars": decision.char,
            "citation": decision.citation,
            "retrieval_score": decision.retrieval_score,
            "decision_status": decision.decision_status,
        }]

    # AutoHDR 候选
    autohdr_candidates = damage.get("autohdr_candidates", [])

    return decide_fusion(
        damage_id=damage.get("damage_id", ""),
        autohdr_candidates=autohdr_candidates,
        cbeta_candidates=cbeta_candidates,
    )


def fuse_batch(damages: list[dict], sutra_hint: str = "大方广佛华严经") -> list[FusionDecision]:
    return [fuse_for_damage(d, sutra_hint) for d in damages]
