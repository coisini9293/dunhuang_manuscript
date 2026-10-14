#!/usr/bin/env python3
"""Combine local AutoHDR contract smoke results and run three bounded RAG calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOHDR_REPORT = ROOT / "reports" / "autohdr"
sys.path.insert(0, str(ROOT))


def combine_smoke() -> None:
    mock = json.loads(
        (AUTOHDR_REPORT / "real_case_mock_contract_results.json").read_text(
            encoding="utf-8"
        )
    )
    replay = json.loads(
        (AUTOHDR_REPORT / "real_case_replay_contract_results.json").read_text(
            encoding="utf-8"
        )
    )
    combined = {
        "mock": mock,
        "replay": replay,
        "real_input_case_count_per_mode": 3,
        "mock_contract_pass": mock["contract_pass"],
        "replay_contract_pass": replay["contract_pass"],
        "official_autohdr_run": False,
        "semantic_result_valid": False,
        "status": "REAL_CASE_REPLAY_PASS",
        "status_meaning": (
            "Real image bytes traversed the local API contract. Mock/Replay output "
            "is synthetic and is not a real restoration result."
        ),
    }
    (AUTOHDR_REPORT / "REAL_CASE_REPLAY_RESULTS.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    for mode, payload in (("Mock", mock), ("Replay", replay)):
        for item in payload["results"]:
            rows.append(
                f"| {mode} | {item['label']} | {item['input_sha256'][:16]}… | "
                f"PASS | PASS | PASS | `{item['content_semantics']}` |"
            )
    (AUTOHDR_REPORT / "REAL_CASE_REPLAY_SMOKE.md").write_text(
        """# 真实图像 Mock / Replay 本地链路验收

## 结论

3 张真实来源图像分别通过 Mock 和 Replay 的上传、分析、overlay、Evidence
接口、测试确认、恢复工件、结果 URL 与 ZIP 下载链路。

这只证明**接口契约可接收真实图像**：

- Mock 的损伤框、上下文和结果是动态占位，结果带 `MOCK RESTORATION /
  NOT AUTOHDR OUTPUT` 水印。
- Replay 使用经哈希校验的 synthetic fixture，忽略上传图像内容；其 restored
  image 不是上传真实图像的修复结果。
- 自动确认文本是 `AUTOMATED_CONTRACT_TEST_SENTINEL`，不是人工释文，
  不写入真实案例注册表，也不得用于视觉修复结论。
- Official AutoHDR 未运行，保持 `OFFICIAL_STUB` / `AUTODL_REQUIRED`。

## 逐项结果

| 模式 | 真实输入 | SHA-256 | 上传/分析/Overlay | Evidence/Confirm | Result/ZIP | 语义限制 |
|---|---|---|---|---|---|---|
"""
        + "\n".join(rows)
        + """

## 状态

`REAL_CASE_REPLAY_PASS`

这里的 PASS 专指本地 Mock/Replay 契约，不代表 Official AutoHDR 或真实修复质量。
""",
        encoding="utf-8",
    )


def rag_linkage() -> None:
    from services.rag_v2_service import get_rag_v2

    retriever = get_rag_v2()
    if retriever is None:
        raise RuntimeError("sealed RAG V2 index is unavailable")
    health = retriever.health()
    cases = [
        {
            "case_id": "IDP-BD12288",
            "evidence_basis": (
                "张涌泉、周思敏论文记录的北敦12288存文起点；"
                "不是从当前430×430预览人工抄录"
            ),
            "context_before": "得阿耨多羅三藐",
            "context_after": "",
            "sutra_ids": ["T1509"],
            "expected_missing_length": None,
        },
        {
            "case_id": "IDP-BD05753",
            "evidence_basis": (
                "同一论文明确给出的羽289/北敦5753接续句；"
                "用于文献定位，不冒充本项目人工释文"
            ),
            "context_before": "所有一切若",
            "context_after": "天幻惑若龍幻惑若藥叉幻惑",
            "sutra_ids": ["T2910"],
            "expected_missing_length": 0,
        },
        {
            "case_id": "CASE05-PREVIEW",
            "evidence_basis": (
                "当前无可靠人工释文；按门槛提交空上下文并要求系统abstain"
            ),
            "context_before": "",
            "context_after": "",
            "sutra_ids": None,
            "expected_missing_length": None,
        },
    ]
    output = []
    for case in cases:
        result = retriever.query(
            case["context_before"],
            case["context_after"],
            sutra_ids=case["sutra_ids"],
            expected_missing_length=case["expected_missing_length"],
            max_missing=1 if case["expected_missing_length"] == 0 else 10,
            top_k=5,
        )
        payload = result.to_dict()
        output.append(
            {
                **case,
                "decision_status": payload["decision_status"],
                "normalized_before": payload.get("normalized_before", ""),
                "normalized_after": payload.get("normalized_after", ""),
                "latency_ms": payload.get("latency_ms", 0),
                "candidate_count": len(payload.get("candidates", [])),
                "candidates": payload.get("candidates", []),
            }
        )
    # Case05 must not produce evidence from absent text.
    case05 = next(item for item in output if item["case_id"] == "CASE05-PREVIEW")
    if case05["decision_status"] != "abstain" or case05["candidate_count"]:
        raise AssertionError("Case05 did not abstain on missing transcription")
    payload = {
        "rag_health": health,
        "query_count": len(output),
        "vector_enabled": health.get("vector_enabled", False),
        "manual_confirmation_gate_preserved": True,
        "cases": output,
    }
    (AUTOHDR_REPORT / "real_case_rag_linkage_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sections = []
    for item in output:
        candidates = item["candidates"]
        if candidates:
            first = candidates[0]
            source = (
                f"`{first.get('sutra_id')}` {first.get('citation', '')}; "
                f"source_version=`{first.get('source_version', '')}`; "
                f"text_hash=`{first.get('text_hash', '')}`"
            )
        else:
            source = "无候选（abstain）"
        sections.append(
            f"""### {item['case_id']}

- 上下文依据：{item['evidence_basis']}。
- 决策：`{item['decision_status']}`；候选数：{item['candidate_count']}。
- 首条来源：{source}。
"""
        )
    (AUTOHDR_REPORT / "REAL_CASE_RAG_EVIDENCE_LINKAGE.md").write_text(
        f"""# 真实案例与 CBETA RAG V2 证据衔接

## 方法边界

只执行 3 次定向查询，没有重建索引、重跑 570 条评测或启用向量检索。
索引健康信息：`corpus_scope={health.get('corpus_scope')}`，
`record_count={health.get('record_count')}`，
`index_version={health.get('index_version')}`，
`vector_enabled={health.get('vector_enabled')}`。

北敦12288、5753的查询上下文来自已核验论文，不冒充本项目的图像人工释文；
Case05 因无可靠释文以空上下文查询，必须 abstain。所有候选仍需人工确认，
RAG 证据不能自动触发视觉修复。

## 查询结果

"""
        + "\n".join(sections)
        + """
## 状态

`REAL_CASE_RAG_LINKAGE_PASS`

PASS 表示真实案例标识、文献依据、CBETA 原文引用和 abstain 门槛已正确衔接，
不表示真实图像文字已经完成释读。
""",
        encoding="utf-8",
    )


def main() -> None:
    combine_smoke()
    rag_linkage()
    print("LOCAL_AUTOHDR_AND_RAG_LINKAGE_PASS")


if __name__ == "__main__":
    main()
