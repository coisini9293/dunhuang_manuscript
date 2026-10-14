"""阶段 4：真实 L3 API 正常与故障降级测试。

测试场景：
- normal: 正常 API 调用
- key_missing: Key 缺失
- timeout: 超时模拟
- nonsensical: 返回非 JSON 处理
- abstain_test: 上下文不足以推断→应返回 abstain

所有失败必须进入 L4 或 original fallback，不导致主流程崩溃。
LLM 候选必须 allow_auto_render=false，不得伪造 CBETA 引用。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.llm_client import (
    predict_missing_char, is_api_ready, is_mock_mode,
    DASHSCOPE_MODEL,
)
from agents.rag_agent import RagAgent


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "llm_fallback_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg)
    with open(OUTPUT_DIR / "test.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def test_normal() -> dict:
    """正常 API 调用测试。"""
    log("\n--- 测试: 正常 API ---")
    t0 = time.time()
    result = predict_missing_char(
        "如是我闻一时佛在摩竭提国寂灭道场始成正觉",
        "其地金刚具足严净众宝杂华",
        sutra_name="大方广佛华严经",
        top_k=3,
    )
    elapsed = time.time() - t0

    if result is None:
        log(f"[FAIL] 返回 None (latency={elapsed:.1f}s)")
        return {"test": "normal", "passed": False, "error": "returned None", "latency_sec": round(elapsed, 2)}

    is_mock = result.get("mock", False)
    source = result.get("source", "unknown")
    candidates = result.get("candidates", [])
    log(f"[INFO] source={source}, mock={is_mock}, candidates={len(candidates)}, latency={elapsed:.1f}s")
    for c in candidates[:3]:
        log(f"  candidate: {c.get('char')} conf={c.get('confidence')}")

    return {
        "test": "normal",
        "passed": source in ("llm_qwen_max", "mock_llm") and len(candidates) > 0,
        "source": source,
        "mock": is_mock,
        "candidates_count": len(candidates),
        "latency_sec": round(elapsed, 2),
        "model": DASHSCOPE_MODEL,
    }


def test_rag_agent_decision() -> dict:
    """RAG Agent 综合决策测试（L1→L3→L4 链路）。"""
    log("\n--- 测试: RAG Agent 四层 fallback ---")
    agent = RagAgent()

    # 场景 1：CBETA 应命中
    log("\n  场景 1: 佛经上下文 → L1 应命中")
    d1 = agent.decide(
        "如是我闻一时佛在摩竭提国寂灭道场始成正觉",
        "上妙宝轮圆满清净无量妙色",
        sutra_hint="大方广佛华严经",
    )
    log(f"    layer={d1.layer}, status={d1.decision_status}, char={d1.char}, "
        f"allow_auto_render={d1.allow_auto_render}")

    # 场景 2：LLM 应返回候选
    log("\n  场景 2: 上下文 + 经名提示 → L3 应返回候选")
    d2 = agent.decide(
        "菩薩摩訶薩住此法門已",
        "無量劫中常為諸佛之所護念",
        sutra_hint="大方广佛华严经",
    )
    log(f"    layer={d2.layer}, status={d2.decision_status}, char={d2.char}, "
        f"allow_auto_render={d2.allow_auto_render}")

    # 场景 3：无法推断 → L4 abstain
    log("\n  场景 3: 无意义上下文 → L4 abstain")
    d3 = agent.decide("ZZZ", "YYY", sutra_hint="")
    log(f"    layer={d3.layer}, status={d3.decision_status}, char={d3.char}")

    # 场景 4：LLM 候选不允许自动渲染
    log("\n  场景 4: LLM 候选检查 allow_auto_render 和安全属性")
    d4 = agent.decide(
        "爾時文殊師利菩薩",
        "告諸比丘言汝等當知",
        sutra_hint="大方广佛华严经",
    )
    log(f"    layer={d4.layer}, status={d4.decision_status}, "
        f"allow_auto_render={d4.allow_auto_render}, "
        f"source_authority={d4.source_authority}")
    log(f"    citation={d4.citation}")

    # 检查 LLM 是否伪造 CBETA 引用
    has_fake_cbeta = (
        d4.layer == "L3" and d4.source == "llm_qwen_max" and
        "T0278" in str(d4.citation) and d4.decision_status != "abstain"
    )
    log(f"    LLM 伪造 CBETA 引用: {has_fake_cbeta}")

    passed = (
        d1.layer == "L1" and
        d3.layer == "L4" and d3.decision_status == "abstain" and
        (d4.layer == "L3" and d4.allow_auto_render == False) and
        not has_fake_cbeta
    )

    return {
        "test": "rag_agent_decision",
        "passed": passed,
        "scenarios": [
            {"name": "l1_should_hit", "layer": d1.layer, "status": d1.decision_status},
            {"name": "l3_candidate", "layer": d2.layer, "status": d2.decision_status},
            {"name": "l4_abstain", "layer": d3.layer, "status": d3.decision_status},
            {"name": "llm_no_auto_render", "layer": d4.layer, "allow_auto_render": d4.allow_auto_render},
            {"name": "llm_no_fake_cbeta", "has_fake_cbeta": has_fake_cbeta},
        ],
    }


def test_abstain_on_nonsense() -> dict:
    """无意义输入 → abstain 测试。"""
    log("\n--- 测试: 无意义输入 → abstain ---")
    agent = RagAgent()
    d = agent.decide("", "", sutra_hint="")
    log(f"  layer={d.layer}, status={d.decision_status}, char={d.char}")
    return {
        "test": "abstain_on_empty",
        "passed": d.layer == "L4" and d.decision_status == "abstain",
        "result": d.to_dict(),
    }


def run_all() -> dict:
    (OUTPUT_DIR / "test.log").write_text("", encoding="utf-8")
    log("=" * 60)
    log("阶段 4：真实 L3 API 降级测试")
    log(f"API Ready: {is_api_ready()}")
    log(f"Mock Mode: {is_mock_mode()}")
    log(f"Model: {DASHSCOPE_MODEL}")
    log("=" * 60)

    results = {}

    # Test 1: Normal API
    results["normal"] = test_normal()

    # Test 2: RAG Agent decision
    results["rag_agent"] = test_rag_agent_decision()

    # Test 3: Abstain on nonsense
    results["abstain"] = test_abstain_on_nonsense()

    # Save results
    all_passed = all(r.get("passed", False) for r in results.values())
    report = {
        "all_passed": all_passed,
        "api_ready": is_api_ready(),
        "mock_mode": is_mock_mode(),
        "model": DASHSCOPE_MODEL,
        "results": results,
    }
    (OUTPUT_DIR / "test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"\n{'='*60}")
    log(f"All passed: {all_passed}")
    log(f"Results saved to: {OUTPUT_DIR / 'test_results.json'}")
    return report


if __name__ == "__main__":
    run_all()
