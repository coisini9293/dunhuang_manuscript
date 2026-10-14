"""RAG Agent：四层 fallback 决策核心（多子维度置信度）。

L1: CBETA 本地检索（retrieved_exact / retrieved_ambiguous / abstain）
L2: 同文异本残卷库（暂空实现）
L3: LLM API 兜底（LLM_ENABLED=false 时走 mock，true 时调通义千问）
    L3 只能提供 llm_suggestion 黄色候选，不能把模型自报概率当作真实置信度。
L4: 标灰跳过（abstain，永不炸锅）

评分字段（不允许仅凭字符串精确命中就宣称补字可信度 100%）：
- retrieval_score: 检索命中质量
- context_match_score: 上下文匹配度
- source_authority: 来源权威度字符串（cbeta_primary / llm_secondary / mock / none）
- source_authority_value: 数值（用于合成 confidence，不用于决策）
- visual_alignment_score: 视觉对齐评分（null = 未评估）
- decision_status: retrieved_exact / retrieved_ambiguous / llm_suggestion / manual_confirmed / abstain / error_fallback
- evidence_type: cbeta_exact / cbeta_ambiguous / llm_suggestion / manual / none
- allow_auto_render: false（即使 retrieved_exact 也不自动渲染，需人工确认）

颜色：
- blue:   L1 retrieved_exact
- cyan:   L1 retrieved_ambiguous
- yellow: L3 llm_suggestion
- orange: L3 mock（不可信，仅流程演示）
- gray:   L4 abstain
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cbeta_index import Cbetaindex, Cbetamatch
from utils.multi_scale_index import MultiScaleIndex, ChunkMatch
from utils.llm_client import predict_missing_char, is_api_ready, is_mock_mode


# 来源权威度
AUTHORITY_LLM_STR = "llm_secondary"
AUTHORITY_LLM_VAL = 0.5
AUTHORITY_CBETA_STR = "cbeta_primary"
AUTHORITY_CBETA_VAL = 0.95
AUTHORITY_MOCK_STR = "mock"
AUTHORITY_MOCK_VAL = 0.0
AUTHORITY_NONE_STR = "none"
AUTHORITY_NONE_VAL = 0.0

# L3 候选的置信度阈值
L3_FILL_THRESHOLD = 0.6


@dataclass
class RestoreDecision:
    """单字补全决策（多子维度置信度）。"""
    char: Optional[str]
    source: str              # cbeta / collation / llm_qwen_max / mock_llm / none
    # 评分字段
    retrieval_score: float
    context_match_score: float
    source_authority: str              # 字符串
    source_authority_value: float      # 数值
    visual_alignment_score: float | None
    decision_status: str               # retrieved_exact / retrieved_ambiguous / llm_suggestion / manual_confirmed / abstain / error_fallback
    evidence_type: str                 # cbeta_exact / cbeta_ambiguous / llm_suggestion / manual / none
    allow_auto_render: bool
    # 元信息
    citation: str
    color: str               # blue / cyan / yellow / orange / gray
    layer: str               # L1 / L2 / L3 / L4
    candidates: list = field(default_factory=list)
    before_matched_len: int = 0
    after_matched_len: int = 0
    missing_len: int = 0

    @property
    def confidence(self) -> float:
        """合成置信度（仅用于展示，不用于决策）。"""
        return round(
            self.retrieval_score * 0.35
            + self.context_match_score * 0.35
            + self.source_authority_value * 0.30,
            3,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = self.confidence
        return d


class RagAgent:
    """RAG 四层 fallback agent。"""

    def __init__(
        self,
        cbeta_dir: str | Path | None = None,
        use_multiscale: bool = True,
        use_rag_v2: bool = True,
    ):
        if cbeta_dir is None:
            cbeta_dir = Path(__file__).resolve().parent.parent / "data" / "cbeta"
        self.cbeta = Cbetaindex(cbeta_dir)  # 保留兼容
        self.use_rag_v2 = use_rag_v2
        self.rag_v2 = None
        if use_rag_v2:
            from services.rag_v2_service import get_rag_v2

            self.rag_v2 = get_rag_v2()
            if self.rag_v2 is not None:
                print("[rag_agent] 使用 CBETA RAG V2")
        # RAG V2 is the primary local index. Avoid rebuilding the legacy in-memory
        # MultiScaleIndex when the persistent V2 database is available.
        self.use_multiscale = use_multiscale and self.rag_v2 is None
        if self.use_multiscale:
            parsed_dir = Path(cbeta_dir) / "parsed"
            if parsed_dir.exists():
                print(f"[rag_agent] 使用多尺度索引 (MultiScaleIndex)")
                self.ms_index = MultiScaleIndex(parsed_dir)
            else:
                print("[rag_agent] parsed/ 不存在，回退旧索引")
                self.use_multiscale = False
                self.ms_index = None
        else:
            self.ms_index = None
        self.llm_ready = is_api_ready()
        self.llm_mock = is_mock_mode()
        mode = "mock" if self.llm_mock else ("ready" if self.llm_ready else "disabled")
        print(f"[rag_agent] LLM 模式: {mode}")

    def decide(
        self,
        context_before: str,
        context_after: str,
        sutra_hint: str = "",
        expected_missing_length: int | None = None,
        sutra_ids: list[str] | None = None,
        translators: list[str] | None = None,
    ) -> RestoreDecision:
        """四层 fallback 决策。"""
        # L1: CBETA 本地检索
        decision = self._l1_cbeta(
            context_before,
            context_after,
            expected_missing_length=expected_missing_length,
            sutra_ids=sutra_ids,
            translators=translators,
        )
        if decision is not None and decision.decision_status != "abstain":
            return decision

        l1_fallback = decision

        # L2: 同文异本残卷库（暂空实现）
        decision = self._l2_collation(context_before, context_after)
        if decision is not None and decision.decision_status != "abstain":
            return decision

        # L3: LLM API 兜底（mock 或真实）
        decision = self._l3_llm(context_before, context_after, sutra_hint)
        if decision is not None and decision.decision_status != "abstain":
            return decision

        # 如果 L1 有 retrieved_ambiguous 命中，返回它（优于完全 abstain）
        if l1_fallback is not None and l1_fallback.decision_status == "retrieved_ambiguous":
            return l1_fallback

        # L4: 标灰跳过
        return RestoreDecision(
            char=None,
            source="none",
            retrieval_score=0.0,
            context_match_score=0.0,
            source_authority=AUTHORITY_NONE_STR,
            source_authority_value=AUTHORITY_NONE_VAL,
            visual_alignment_score=None,
            decision_status="abstain",
            evidence_type="none",
            allow_auto_render=False,
            citation="",
            color="gray",
            layer="L4",
        )

    def _l1_cbeta(
        self,
        before: str,
        after: str,
        expected_missing_length: int | None = None,
        sutra_ids: list[str] | None = None,
        translators: list[str] | None = None,
    ) -> Optional[RestoreDecision]:
        """L1: CBETA 多尺度检索（优先 MultiScaleIndex）。"""
        if self.use_rag_v2 and self.rag_v2 is not None:
            result = self.rag_v2.query(
                before,
                after,
                sutra_ids=sutra_ids,
                translators=translators,
                expected_missing_length=expected_missing_length,
                top_k=5,
            )
            if result.candidates:
                top = result.candidates[0]
                return RestoreDecision(
                    char=top.candidate_text or None,
                    source="cbeta_v2",
                    retrieval_score=float(
                        top.retrieval_components.get("final", 0.0)
                    ),
                    context_match_score=float(
                        top.retrieval_components.get("continuity", 0.0)
                    ),
                    source_authority=AUTHORITY_CBETA_STR,
                    source_authority_value=AUTHORITY_CBETA_VAL,
                    visual_alignment_score=None,
                    decision_status=result.decision_status,
                    evidence_type=(
                        "cbeta_exact"
                        if result.decision_status == "retrieved_exact"
                        else "cbeta_ambiguous"
                    ),
                    allow_auto_render=False,
                    citation=top.citation,
                    color=(
                        "blue"
                        if result.decision_status == "retrieved_exact"
                        else "cyan"
                    ),
                    layer="L1",
                    candidates=[candidate.to_dict() for candidate in result.candidates],
                    before_matched_len=len(top.matched_before),
                    after_matched_len=len(top.matched_after),
                    missing_len=len(top.candidate_text),
                )
            if result.decision_status == "abstain":
                return None
        if self.use_multiscale and self.ms_index:
            import re
            b_norm = re.sub(r"[^\u4e00-\u9fff]", "", before)
            a_norm = re.sub(r"[^\u4e00-\u9fff]", "", after)

            # 双侧搜索
            if b_norm and a_norm:
                matches = self.ms_index.search(b_norm, a_norm, max_missing=10, top_k=5)
                if matches:
                    m = matches[0]
                    if m.missing_chars and m.decision_status != "abstain":
                        return self._build_l1_decision(m)

            # 单侧 fallback：至少返回定位信息
            side = b_norm if b_norm else a_norm
            if side:
                matches = self.ms_index.search_single_side(side, top_k=5)
                if matches:
                    m = matches[0]
                    return self._build_l1_decision(m)

        # 回退旧索引
        match = self.cbeta.search_by_context(before, after)
        if match is None:
            return None
        if not match.missing_chars:
            return None

        color = "blue" if match.decision_status == "retrieved_exact" else "cyan"
        return RestoreDecision(
            char=match.missing_chars,
            source="cbeta",
            retrieval_score=match.retrieval_score,
            context_match_score=match.context_match_score,
            source_authority=match.source_authority,
            source_authority_value=match.source_authority_value,
            visual_alignment_score=match.visual_alignment_score,
            decision_status=match.decision_status,
            evidence_type=match.evidence_type,
            allow_auto_render=match.allow_auto_render,
            citation=match.citation,
            color=color,
            layer="L1",
            before_matched_len=match.before_matched_len,
            after_matched_len=match.after_matched_len,
            missing_len=match.missing_len,
        )

    def _build_l1_decision(self, m) -> RestoreDecision:
        """从 ChunkMatch 构建 RestoreDecision。"""
        color = "blue" if m.decision_status == "retrieved_exact" else "cyan"
        return RestoreDecision(
            char=m.missing_chars if m.missing_chars else None,
            source="cbeta",
            retrieval_score=m.retrieval_score,
            context_match_score=m.context_match_score,
            source_authority=m.source_authority,
            source_authority_value=m.source_authority_value,
            visual_alignment_score=None,
            decision_status=m.decision_status,
            evidence_type=m.evidence_type,
            allow_auto_render=False,
            citation=m.citation,
            color=color,
            layer="L1",
            before_matched_len=m.before_matched_len,
            after_matched_len=m.after_matched_len,
            missing_len=m.missing_len,
        )

    def _l2_collation(self, before: str, after: str) -> Optional[RestoreDecision]:
        """L2: 同文异本残卷库（暂空实现）。"""
        return None

    def _l3_llm(self, before: str, after: str, sutra_hint: str) -> Optional[RestoreDecision]:
        """L3: LLM API 上下文预测（mock 或真实）。

        L3 只能提供 llm_suggestion 黄色候选，不能把模型自报概率当作真实置信度。
        """
        result = predict_missing_char(before, after, sutra_hint, top_k=5)
        if result is None:
            return None

        is_mock = result.get("mock", False)
        candidates = result.get("candidates", [])
        source = result.get("source", "llm_qwen_max")

        if not candidates:
            return None

        top1 = candidates[0]
        char = top1.get("char", "")
        raw_conf = float(top1.get("confidence", 0))

        if is_mock:
            # mock 模式：强制 abstain，颜色 orange，不可信
            return RestoreDecision(
                char=None,
                source="mock_llm",
                retrieval_score=0.0,
                context_match_score=0.0,
                source_authority=AUTHORITY_MOCK_STR,
                source_authority_value=AUTHORITY_MOCK_VAL,
                visual_alignment_score=None,
                decision_status="abstain",
                evidence_type="none",
                allow_auto_render=False,
                citation=f"mock L3 (LLM_ENABLED=false), top1={char}",
                color="orange",
                layer="L3",
                candidates=candidates,
            )

        # 真实 LLM：只能是 llm_suggestion，allow_auto_render=false
        if raw_conf < L3_FILL_THRESHOLD:
            return RestoreDecision(
                char=None,
                source=source,
                retrieval_score=raw_conf,
                context_match_score=raw_conf,
                source_authority=AUTHORITY_LLM_STR,
                source_authority_value=AUTHORITY_LLM_VAL,
                visual_alignment_score=None,
                decision_status="abstain",
                evidence_type="none",
                allow_auto_render=False,
                citation=f"LLM top1={char} conf={raw_conf:.2f} < {L3_FILL_THRESHOLD}",
                color="gray",
                layer="L3",
                candidates=candidates,
            )

        return RestoreDecision(
            char=char,
            source=source,
            retrieval_score=raw_conf,
            context_match_score=raw_conf,
            source_authority=AUTHORITY_LLM_STR,
            source_authority_value=AUTHORITY_LLM_VAL,
            visual_alignment_score=None,
            decision_status="llm_suggestion",
            evidence_type="llm_suggestion",
            allow_auto_render=False,  # LLM 候选必须人工确认
            citation=f"LLM 上下文预测 (top-1 of {len(candidates)})",
            color="yellow",
            layer="L3",
            candidates=candidates,
        )


if __name__ == "__main__":
    agent = RagAgent()
    print(f"CBETA 卷数: {len(agent.cbeta.texts)}")
    print()

    # 索引单元测试 1：CBETA 命中（4 字缺损）
    print("--- 索引单元测试 1: CBETA 命中（4 字） ---")
    before = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after = "以为庄严上妙宝轮"
    decision = agent.decide(before, after, sutra_hint="大方广佛华严经")
    print(f"  层: {decision.layer}")
    print(f"  字: {decision.char}")
    print(f"  retrieval_score:     {decision.retrieval_score}")
    print(f"  context_match_score: {decision.context_match_score}")
    print(f"  source_authority:    {decision.source_authority}")
    print(f"  decision_status:     {decision.decision_status}")
    print(f"  evidence_type:       {decision.evidence_type}")
    print(f"  allow_auto_render:   {decision.allow_auto_render}")
    print(f"  颜色: {decision.color}")

    # 索引单元测试 2：非佛经内容，走 L3 mock -> L4
    print("\n--- 索引单元测试 2: 非佛经内容（L3 mock -> L4） ---")
    before2 = "今天天气真好"
    after2 = "我们去公园玩"
    decision2 = agent.decide(before2, after2, sutra_hint="")
    print(f"  层: {decision2.layer}")
    print(f"  decision_status: {decision2.decision_status}")
    print(f"  source: {decision2.source}")
