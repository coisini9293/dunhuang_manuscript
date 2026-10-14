"""多尺度文本分块索引 — 3C 检索核心。

实现三尺度滑动窗口分块：
- short: 32 字, overlap 16
- medium: 96 字, overlap 48
- long: 256 字, overlap 128

两阶段检索：
- 阶段 1: short/medium bigram 召回
- 阶段 2: long 上下文验证 + 编辑距离重排

每个 chunk 可回溯到卷号、品名、行号和页栏行。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from zhconv import convert as zh_convert
from Levenshtein import distance as lev_distance


# 来源权威度
AUTHORITY_CBETA = "cbeta_primary"
AUTHORITY_CBETA_VAL = 0.95


@dataclass
class ChunkMatch:
    """单个 chunk 检索命中。"""
    chunk_id: str
    volume: int
    pin: str
    scale: str              # short / medium / long
    matched_text: str
    missing_chars: str
    retrieval_score: float
    context_match_score: float
    source_authority: str
    source_authority_value: float
    decision_status: str
    evidence_type: str
    allow_auto_render: bool
    citation: str
    before_matched_len: int = 0
    after_matched_len: int = 0
    missing_len: int = 0
    edit_similarity: float = 1.0


@dataclass
class TextChunk:
    """文本分块。"""
    chunk_id: str
    volume: int
    pin: str
    scale: str
    normalized_traditional: str    # 繁体（用于检索匹配）
    normalized_simplified: str     # 简体（用于搜索 bigram）
    original_text: str             # 原始文本（含标点，用于证据引用）
    start_line: str
    end_line: str
    char_offset: int = 0


def _to_traditional(text: str) -> str:
    """简体→繁体 (zhconv)。"""
    return zh_convert(text, "zh-hant")


def _to_simple(text: str) -> str:
    """繁体→简体 (zhconv)。"""
    return zh_convert(text, "zh-hans")


def _edit_similarity(s1: str, s2: str) -> float:
    """Levenshtein 编辑距离相似度 (0-1)。"""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return 1.0 - lev_distance(s1, s2) / max_len


class MultiScaleIndex:
    """多尺度文本分块索引。"""

    def __init__(self, parsed_dir: str | Path, cbeta_dir: str | Path | None = None):
        self.parsed_dir = Path(parsed_dir)
        self.chunks: dict[str, list[TextChunk]] = {"short": [], "medium": [], "long": []}
        self.chunk_by_id: dict[str, TextChunk] = {}
        self.bigram_index: dict[str, set[str]] = {}
        self._build()

    # ---- 分块参数 ----
    SHORT_SIZE = 32
    SHORT_OVERLAP = 16
    MEDIUM_SIZE = 96
    MEDIUM_OVERLAP = 48
    LONG_SIZE = 256
    LONG_OVERLAP = 128

    # L1 阈值
    MIN_RETRIEVAL = 0.4
    MIN_CONTEXT = 0.3

    def _build(self):
        """加载 JSONL 并构建多尺度 chunks。"""
        records_all = []
        for f in sorted(self.parsed_dir.glob("T0278_*.jsonl")):
            vol = int(f.stem.split("_")[1])
            lines = f.read_text(encoding="utf-8").splitlines()
            records = [json.loads(l) for l in lines if l.strip()]
            records_all.append((vol, records))

        for vol, records in records_all:
            pin = records[0].get("chapter_name") or ""
            normalized = "".join(
                r.get("normalized_text", "") for r in records
                if r.get("normalized_text")
            )

            # 保留 original_text 用于证据展示
            # 同时保留行号映射
            line_offsets = []  # [(char_offset, line_head)]
            offset = 0
            for r in records:
                nt = r.get("normalized_text", "")
                if nt:
                    line_offsets.append((offset, r.get("line_head", "")))
                offset += len(nt or "")

            # 三种尺度分块
            self._chunkify(normalized, vol, pin, "short",
                           self.SHORT_SIZE, self.SHORT_OVERLAP, line_offsets)
            self._chunkify(normalized, vol, pin, "medium",
                           self.MEDIUM_SIZE, self.MEDIUM_OVERLAP, line_offsets)
            self._chunkify(normalized, vol, pin, "long",
                           self.LONG_SIZE, self.LONG_OVERLAP, line_offsets)

        # 构建 bigram 索引（short + medium）
        for scale in ["short", "medium"]:
            for ch in self.chunks[scale]:
                text = ch.normalized_traditional
                for i in range(len(text) - 1):
                    bg = text[i:i + 2]
                    if bg not in self.bigram_index:
                        self.bigram_index[bg] = set()
                    self.bigram_index[bg].add(ch.chunk_id)

        print(f"[MultiScaleIndex] chunks: "
              f"short={len(self.chunks['short'])}, "
              f"medium={len(self.chunks['medium'])}, "
              f"long={len(self.chunks['long'])}, "
              f"bigrams={len(self.bigram_index)}")

    def _chunkify(self, text: str, vol: int, pin: str, scale: str,
                  size: int, overlap: int, line_offsets: list):
        """滑动窗口分块。分别保存繁体/简体/原文。"""
        step = size - overlap
        text_simple = _to_simple(text)
        n = max(1, (len(text) - size) // step + 1)
        for i in range(n):
            start = i * step
            end = min(start + size, len(text))
            chunk_t = text[start:end]         # 繁体
            chunk_s = text_simple[start:end]  # 简体
            if len(chunk_t) < 4:
                continue

            start_line, end_line = "", ""
            for off, lh in line_offsets:
                if off <= start: start_line = lh
                if off >= end: end_line = lh; break
            if not end_line and line_offsets:
                end_line = line_offsets[-1][1]

            cid = f"T0278_v{vol:03d}_{scale}_s{start:06d}"
            ch = TextChunk(
                chunk_id=cid, volume=vol, pin=pin, scale=scale,
                normalized_traditional=chunk_t,
                normalized_simplified=chunk_s,
                original_text="",
                start_line=start_line, end_line=end_line,
                char_offset=start,
            )
            self.chunks[scale].append(ch)
            self.chunk_by_id[cid] = ch

            # bigram 用简体版
            if scale in ("short", "medium"):
                txt = chunk_s if chunk_s else chunk_t
                for j in range(len(txt) - 1):
                    bg = txt[j:j + 2]
                    self.bigram_index.setdefault(bg, set()).add(cid)

    # ---- 检索 ----

    def search(self, context_before: str, context_after: str,
               max_missing: int = 10, top_k: int = 10) -> list[ChunkMatch]:
        """两阶段检索。查询文本保持原样（支持简繁混合输入）。"""
        before = re.sub(r"[^\u4e00-\u9fff]", "", context_before)
        after = re.sub(r"[^\u4e00-\u9fff]", "", context_after)

        # 阶段 1: bigram 召回（short + medium）
        candidate_ids = self._recall(before, after)
        if not candidate_ids:
            return []

        # 阶段 2: 验证 + 重排
        matches = []
        for cid in candidate_ids:
            ch = self._get_chunk(cid)
            if ch is None:
                continue
            match = self._verify(ch, before, after, max_missing)
            if match:
                matches.append(match)

        # 按 retrieval_score 降序
        matches.sort(key=lambda m: m.retrieval_score, reverse=True)
        return matches[:top_k]

    def search_single_side(self, context: str, top_k: int = 10) -> list[ChunkMatch]:
        """单侧上下文搜索（left_only/right_only fallback）。"""
        text = re.sub(r"[^\u4e00-\u9fff]", "", context)
        if len(text) < 8:
            return []

        # 在 long chunks 中搜索包含该文本的所有位置
        matches = []
        for ch in self.chunks["long"]:
            idx = ch.normalized_traditional.find(text)
            if idx == -1 and ch.normalized_simplified:
                idx = ch.original_text.find(text)
            if idx >= 0:
                matches.append(ChunkMatch(
                    chunk_id=ch.chunk_id, volume=ch.volume, pin=ch.pin,
                    scale="long", matched_text=text, missing_chars="",
                    retrieval_score=0.3, context_match_score=0.3,
                    source_authority=AUTHORITY_CBETA,
                    source_authority_value=AUTHORITY_CBETA_VAL * 0.7,
                    decision_status="retrieved_ambiguous",
                    evidence_type="cbeta_ambiguous",
                    allow_auto_render=False,
                    citation=f"T0278 卷{ch.volume} {ch.pin} (single-side)",
                ))
        matches.sort(key=lambda m: m.retrieval_score, reverse=True)
        return matches[:top_k]

    def _recall(self, before: str, after: str) -> list[str]:
        """bigram 召回 + 中长文直接搜索 fallback。"""
        # 用 before 后 12 字和 after 前 12 字做 bigram 匹配
        b_key = before[-12:] if len(before) >= 12 else before
        a_key = after[:12] if len(after) >= 12 else after

        scores: dict[str, float] = {}
        for key in [b_key, a_key]:
            for i in range(len(key) - 1):
                bg = key[i:i + 2]
                if bg in self.bigram_index:
                    for cid in self.bigram_index[bg]:
                        scores[cid] = scores.get(cid, 0) + 1

        # 如果 recall 太少，在 long chunks 中直接搜索 before 的最后 10 字
        if len(scores) < 5:
            b_fallback = before[-10:] if len(before) >= 10 else before
            for ch in self.chunks["long"]:
                if b_fallback in ch.normalized_traditional:
                    scores[ch.chunk_id] = scores.get(ch.chunk_id, 0) + 5

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return sorted_ids[:50]

    def _get_chunk(self, chunk_id: str) -> TextChunk | None:
        for scale in ["short", "medium", "long"]:
            for ch in self.chunks[scale]:
                if ch.chunk_id == chunk_id:
                    return ch
        return None

    def _verify(self, ch: TextChunk, before: str, after: str,
                max_missing: int) -> ChunkMatch | None:
        """在候选 chunk 中验证（优先原 chunk 的简繁双版，再尝试同卷 long chunk）。"""
        # 1. 先尝试候选 chunk 本身（short/medium 有简繁双版）
        m = self._verify_in_text(ch.normalized_traditional, ch, before, after, max_missing)
        if m:
            return m

        # 2. 再在同卷 long chunk 中搜索
        vol = ch.volume
        for lc in self.chunks["long"]:
            if lc.volume == vol:
                # 尝试繁体版
                m = self._try_match(lc.normalized_traditional, lc, before[-12:] if len(before)>=12 else before,
                                    after[:12] if len(after)>=12 else after, max_missing)
                if m:
                    return m
                # 尝试简化版（如果有）
                if lc.original_text and lc.original_text != lc.normalized_traditional:
                    m = self._try_match(lc.original_text, lc, before[-12:] if len(before)>=12 else before,
                                        after[:12] if len(after)>=12 else after, max_missing)
                    if m:
                        return m
        return None

    def _verify_in_text(self, text: str, ch: TextChunk, before: str, after: str,
                        max_missing: int) -> ChunkMatch | None:
        """在给定文本中验证 before/after 匹配。"""
        text = ch.normalized_traditional

    def _verify_in_text(self, text: str, ch: TextChunk, before: str, after: str,
                        max_missing: int) -> ChunkMatch | None:
        """多路匹配：查询简繁 × chunk 简繁。"""
        b_win = before[-12:] if len(before) >= 12 else before
        a_win = after[:12] if len(after) >= 12 else after
        if len(b_win) < 4 or len(a_win) < 4:
            return None

        # 查询双版本
        q_before = [b_win, _to_traditional(b_win)]
        q_after = [a_win, _to_traditional(a_win)]

        # chunk 双版本
        c_texts = [text]
        if ch.original_text and ch.original_text != text:
            c_texts.append(ch.original_text)

        for c_txt in c_texts:
            for qb in q_before:
                for qa in q_after:
                    m = self._try_match(c_txt, ch, qb, qa, max_missing)
                    if m:
                        return m
        return None

    def _try_match(self, text: str, ch: TextChunk, b_win: str, a_win: str,
                   max_missing: int) -> ChunkMatch | None:
        """在特定文本中尝试匹配。"""
        idx = text.find(b_win)
        if idx == -1:
            return None
        after_start = idx + len(b_win)
        for skip in range(max_missing + 1):
            check_pos = after_start + skip
            if text[check_pos:check_pos + len(a_win)] == a_win:
                missing = text[after_start:check_pos]
                return self._build_match(ch, text, b_win, a_win, missing, idx, skip)
        return None

    def _build_match(self, ch: TextChunk, text: str, b_win: str, a_win: str,
                     missing: str, idx: int, skip: int) -> ChunkMatch:
        b_len = len(b_win)
        a_len = len(a_win)
        m_len = len(missing)

        retrieval = min(1.0, (b_len + a_len) / 16.0)
        if b_len < 5 or a_len < 5:
            retrieval *= 0.6

        overlap_ratio = (b_len + a_len) / (b_len + a_len + m_len + 1e-6)
        context_match = min(1.0, overlap_ratio)

        if retrieval >= 0.5 and context_match >= 0.5 and m_len > 0:
            status = "retrieved_exact"
            evidence = "cbeta_exact"
        elif m_len > 0 and retrieval >= 0.3:
            status = "retrieved_ambiguous"
            evidence = "cbeta_ambiguous"
        else:
            status = "abstain"
            evidence = "none"

        # 编辑距离相似度
        edit_sim = 1.0  # 精确匹配时 = 1

        return ChunkMatch(
            chunk_id=ch.chunk_id,
            volume=ch.volume,
            pin=ch.pin,
            scale=ch.scale,
            matched_text=text[idx:idx + b_len + m_len + a_len],
            missing_chars=missing,
            retrieval_score=round(retrieval, 3),
            context_match_score=round(context_match, 3),
            source_authority=AUTHORITY_CBETA,
            source_authority_value=AUTHORITY_CBETA_VAL,
            decision_status=status,
            evidence_type=evidence,
            allow_auto_render=False,
            citation=f"T0278 卷{ch.volume} {ch.pin} (chunk {ch.chunk_id})",
            before_matched_len=b_len,
            after_matched_len=a_len,
            missing_len=m_len,
            edit_similarity=edit_sim,
        )

    def sample_chunk(self) -> dict:
        """返回一个示例 chunk。"""
        for ch in self.chunks["short"][:1]:
            return {
                "chunk_id": ch.chunk_id,
                "volume": ch.volume,
                "pin": ch.pin,
                "scale": ch.scale,
                "normalized_traditional": ch.normalized_traditional,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "char_offset": ch.char_offset,
            }
        return {}
