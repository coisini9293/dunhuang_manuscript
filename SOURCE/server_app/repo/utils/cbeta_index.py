"""CBETA 本地索引器：佛经对校检索（RAG agent L1）。

读取 data/cbeta/ 下的经文文件，建立简易索引，支持：
1. 上下文检索：给定一段文字，找到它在经文中的位置
2. 缺损补全：给定缺损前后的文字，在经文中匹配原文，返回缺损处应有的字
3. Top-k 检索：返回多个候选命中

文件命名约定：{CBETA编号}_{卷次:03d}.txt，如 T0278_001.txt
文件首行格式：{经名} 卷第{X} {品名}

置信度模型（4 子维度，不允许仅凭字符串精确命中就宣称 100%）：
- retrieval_score: 检索命中质量（基于前后文匹配长度）
- context_match_score: 上下文匹配度（基于匹配窗口内字符重合度）
- source_authority: 来源权威度（CBETA 大正藏 = 0.95，非 1.0，因分卷差异风险）
- decision_status: verified / conditional / abstain
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 来源权威度
AUTHORITY_CBETA = 0.95  # CBETA 大正藏，非 1.0 因五十卷本 vs 六十卷本分卷差异风险

# decision_status 阈值
VERIFIED_RETRIEVAL = 0.7     # retrieval_score >= 此值才 verified
VERIFIED_CONTEXT = 0.7       # context_match_score >= 此值才 verified


@dataclass
class Cbetamatch:
    """CBETA 检索命中结果（多子维度置信度）。

    评分字段（不允许仅凭字符串精确命中就宣称补字可信度 100%）：
    - retrieval_score: 检索命中质量（0-1）
    - context_match_score: 上下文匹配度（0-1）
    - source_authority: 来源权威度字符串（"cbeta_primary" / "collation_secondary" / "llm_secondary" / "mock" / "none"）
    - source_authority_value: 来源权威度数值（用于合成 confidence，不直接用于决策）
    - visual_alignment_score: 视觉对齐评分（null = 未评估）
    - decision_status: retrieved_exact / retrieved_ambiguous / llm_suggestion / manual_confirmed / abstain / error_fallback
    - evidence_type: cbeta_exact / cbeta_ambiguous / llm_suggestion / manual / none
    - allow_auto_render: 是否允许自动渲染（false = 需人工确认）
    """
    sutra_id: str
    sutra_name: str
    juan: int
    pin: str
    matched_text: str
    missing_chars: str
    # 评分字段
    retrieval_score: float
    context_match_score: float
    source_authority: str          # 字符串："cbeta_primary" 等
    source_authority_value: float  # 数值，用于合成 confidence
    visual_alignment_score: float | None  # null = 未评估
    decision_status: str           # retrieved_exact / retrieved_ambiguous / abstain
    evidence_type: str             # cbeta_exact / cbeta_ambiguous
    allow_auto_render: bool        # false = 需人工确认
    citation: str
    # 详细信息
    before_matched_len: int = 0
    after_matched_len: int = 0
    missing_len: int = 0

    @property
    def confidence(self) -> float:
        """兼容旧代码的合成置信度（仅用于展示，不用于决策）。"""
        return round(
            self.retrieval_score * 0.35
            + self.context_match_score * 0.35
            + self.source_authority_value * 0.30,
            3,
        )


class Cbetaindex:
    """CBETA 本地索引。"""

    def __init__(self, cbeta_dir: str | Path):
        self.cbeta_dir = Path(cbeta_dir)
        self.texts: list[dict] = []
        self._load()

    def _load(self):
        """加载经文：优先 parsed/ JSONL，降级到旧 txt。"""
        parsed_dir = self.cbeta_dir / "parsed"
        if parsed_dir.exists() and list(parsed_dir.glob("T0278_*.jsonl")):
            self._load_jsonl(parsed_dir)
            return
        self._load_txt()

    def _load_jsonl(self, parsed_dir: Path):
        """从 parsed/ JSONL 加载全量数据。"""
        import json
        for f in sorted(parsed_dir.glob("T0278_*.jsonl")):
            lines = f.read_text(encoding="utf-8").splitlines()
            records = [json.loads(l) for l in lines if l.strip()]
            if not records:
                continue
            # 按卷分组（一个 JSONL = 一卷）
            vol = records[0].get("volume_number", 1)
            pin = records[0].get("chapter_name") or ""
            sutra_name = records[0].get("canonical_title") or "大方广佛华严经"
            # 合并所有 normalized_text
            normalized = "".join(
                r.get("normalized_text", "") for r in records
                if r.get("normalized_text")
            )
            self.texts.append({
                "id": "T0278",
                "name": sutra_name,
                "juan": vol,
                "pin": pin,
                "content": "",  # JSONL 不保留原文
                "clean": normalized,
                "file": f.name,
            })

    def _load_txt(self):
        """降级：从旧 txt 格式加载。"""
        for f in sorted(self.cbeta_dir.glob("T*_*.txt")):
            raw = f.read_text(encoding="utf-8")
            lines = raw.strip().split("\n", 2)
            if len(lines) < 3:
                continue
            header = lines[0].strip()
            m = re.match(r"(.+?)\s+卷第(.+?)\s+(.+)", header)
            if m:
                sutra_name = m.group(1)
                pin = m.group(3)
            else:
                sutra_name = header
                pin = ""

            parts = f.stem.split("_")
            sutra_id = parts[0]
            juan = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

            content = lines[2] if len(lines) > 2 else ""
            clean = re.sub(r"[^\u4e00-\u9fff]", "", content)

            self.texts.append({
                "id": sutra_id,
                "name": sutra_name,
                "juan": juan,
                "pin": pin,
                "content": content,
                "clean": clean,
                "file": f.name,
            })

    def search_by_context(
        self,
        context_before: str,
        context_after: str,
        max_missing: int = 10,
    ) -> Optional[Cbetamatch]:
        """根据缺损前后的文字，在经文中查找匹配，返回最佳命中。"""
        matches = self.search_topk(context_before, context_after, max_missing, top_k=1)
        return matches[0] if matches else None

    def search_topk(
        self,
        context_before: str,
        context_after: str,
        max_missing: int = 10,
        top_k: int = 5,
    ) -> list[Cbetamatch]:
        """Top-k 检索：返回多个候选命中，按 retrieval_score 降序。"""
        before = re.sub(r"[^\u4e00-\u9fff]", "", context_before)
        after = re.sub(r"[^\u4e00-\u9fff]", "", context_after)

        before_window = before[-15:] if len(before) >= 15 else before
        after_window = after[:15] if len(after) >= 15 else after

        if len(before_window) < 3 or len(after_window) < 3:
            return []

        candidates: list[Cbetamatch] = []

        for text in self.texts:
            clean = text["clean"]
            idx = clean.find(before_window)
            while idx != -1 and len(candidates) < top_k * 3:
                after_start = idx + len(before_window)
                for skip in range(max_missing + 1):
                    check_pos = after_start + skip
                    if check_pos + len(after_window) > len(clean):
                        break
                    if clean[check_pos:check_pos + len(after_window)] == after_window:
                        missing = clean[after_start:check_pos]
                        match = self._build_match(
                            text, idx, before_window, after_window, missing, skip
                        )
                        if match is not None:
                            candidates.append(match)
                        break
                idx = clean.find(before_window, idx + 1)

        # 按 retrieval_score 降序，取 top_k
        candidates.sort(key=lambda m: m.retrieval_score, reverse=True)
        return candidates[:top_k]

    def _build_match(
        self,
        text: dict,
        idx: int,
        before_window: str,
        after_window: str,
        missing: str,
        skip: int,
    ) -> Optional[Cbetamatch]:
        """构建单个命中结果，计算 4 子维度置信度。"""
        before_len = len(before_window)
        after_len = len(after_window)
        missing_len = len(missing)

        # retrieval_score: 检索命中质量（前后文越长越可信）
        # 用前后文匹配长度的对数压缩，避免线性饱和
        retrieval = min(1.0, (before_len + after_len) / 20.0)
        if before_len < 5 or after_len < 5:
            retrieval *= 0.6  # 短上下文降权

        # context_match_score: 上下文匹配度
        # 基于匹配窗口内字符重合度 + 缺损长度合理性
        overlap_ratio = (before_len + after_len) / (before_len + after_len + missing_len + 1e-6)
        context_match = min(1.0, overlap_ratio)
        if missing_len == 0 and skip == 0:
            # 前后文直接相连，可能误匹配
            context_match *= 0.7
        # missing_len > max_missing 不可能发生（search_topk 已限制 skip 范围）

        # source_authority: CBETA 大正藏 = "cbeta_primary"（数值 0.95，非 1.0）
        authority_str = "cbeta_primary"
        authority_val = AUTHORITY_CBETA

        # decision_status: retrieved_exact / retrieved_ambiguous / abstain
        # evidence_type: cbeta_exact / cbeta_ambiguous
        # allow_auto_render: false（即使 retrieved_exact 也不自动渲染，需人工确认）
        if retrieval >= VERIFIED_RETRIEVAL and context_match >= VERIFIED_CONTEXT and missing_len > 0:
            status = "retrieved_exact"
            evidence = "cbeta_exact"
        elif missing_len > 0 and retrieval >= 0.4:
            status = "retrieved_ambiguous"
            evidence = "cbeta_ambiguous"
        else:
            status = "abstain"
            evidence = "none"

        return Cbetamatch(
            sutra_id=text["id"],
            sutra_name=text["name"],
            juan=text["juan"],
            pin=text["pin"],
            matched_text=text["clean"][idx:idx + before_len + missing_len + after_len],
            missing_chars=missing,
            retrieval_score=round(retrieval, 3),
            context_match_score=round(context_match, 3),
            source_authority=authority_str,
            source_authority_value=authority_val,
            visual_alignment_score=None,  # 未评估
            decision_status=status,
            evidence_type=evidence,
            allow_auto_render=False,  # 即使 retrieved_exact 也不自动渲染
            citation=f"{text['id']} 卷{text['juan']} {text['pin']}",
            before_matched_len=before_len,
            after_matched_len=after_len,
            missing_len=missing_len,
        )

    def list_sutras(self) -> list[dict]:
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "juan": t["juan"],
                "pin": t["pin"],
                "chars": len(t["clean"]),
                "file": t["file"],
            }
            for t in self.texts
        ]


if __name__ == "__main__":
    idx = Cbetaindex(Path(__file__).resolve().parent.parent / "data" / "cbeta")
    print(f"已加载 {len(idx.texts)} 卷经文：")
    for s in idx.list_sutras():
        print(f"  {s['id']} 卷{s['juan']} {s['name']} - {s['pin']} ({s['chars']} 字)")

    print("\n--- 索引单元测试 ---")
    # 测试 1：挖掉 4 字
    print("\n[测试 1] 挖掉 众宝杂华（4 字）")
    before = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after = "以为庄严上妙宝轮圆满清净"
    result = idx.search_by_context(before, after, max_missing=5)
    if result:
        print(f"  命中: {result.citation}")
        print(f"  缺损字: {result.missing_chars} (len={result.missing_len})")
        print(f"  retrieval_score:     {result.retrieval_score}")
        print(f"  context_match_score: {result.context_match_score}")
        print(f"  source_authority:    {result.source_authority}")
        print(f"  decision_status:     {result.decision_status}")
        print(f"  合成 confidence:     {result.confidence}")
    else:
        print("  未命中")

    # 测试 2：挖掉 8 字
    print("\n[测试 2] 挖掉 众宝杂华以为庄严（8 字）")
    before2 = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after2 = "上妙宝轮圆满清净无量妙色"
    result2 = idx.search_by_context(before2, after2, max_missing=10)
    if result2:
        print(f"  命中: {result2.citation}")
        print(f"  缺损字: {result2.missing_chars} (len={result2.missing_len})")
        print(f"  retrieval_score:     {result2.retrieval_score}")
        print(f"  context_match_score: {result2.context_match_score}")
        print(f"  source_authority:    {result2.source_authority}")
        print(f"  decision_status:     {result2.decision_status}")
