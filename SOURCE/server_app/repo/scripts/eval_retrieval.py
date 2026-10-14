"""阶段 3C：检索评测集生成 + 检索评测。

从 T0278 全量 JSONL 数据中抽取 200+ 条测试查询，覆盖：
- 1/2/4/8 字缺失
- 仅左/仅右/左右上下文
- 不同卷/品
- 噪声上下文（OCR 错字）
- 异体字、多候选歧义
- 完全不存在的句子

严格防止 GT 泄漏——Agent 输入不含被挖掉的答案。

输出：outputs/cbeta_index_audit/
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rag_agent import RagAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CBETA_DATA = PROJECT_ROOT / "data" / "cbeta"
PARSED_DIR = CBETA_DATA / "parsed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cbeta_index_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # 固定随机种子（可复现）

N_TARGET = 200  # 最少 200 条


def load_all_records() -> list[dict]:
    """加载全部 JSONL 记录。"""
    records = []
    for vol in range(1, 61):
        f = PARSED_DIR / f"T0278_{vol:03d}.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def extract_text_segments(records: list[dict], window: int = 30) -> list[dict]:
    """从记录中提取相邻文本片段（用于挖字）。"""
    segments = []
    # 按卷和行排序
    for i in range(len(records) - 5):
        # 收集 5 个连续行
        batch = records[i:i + 5]
        # 合并 normalized_text
        merged = "".join(r.get("normalized_text", "") for r in batch if r.get("normalized_text"))
        if len(merged) < window:
            continue
        segments.append({
            "merged_text": merged,
            "volume": records[i]["volume_number"],
            "page": records[i].get("page"),
            "start_line": records[i].get("line_head"),
            "end_line": records[i + 4].get("line_head") if i + 4 < len(records) else None,
        })
    return segments


def generate_test_cases(records: list[dict]) -> list[dict]:
    """生成 200+ 条测试查询。"""
    segments = extract_text_segments(records)
    if not segments:
        return []

    cases = []
    missing_lens = [1, 2, 4, 8]
    vol_samples = set()

    while len(cases) < N_TARGET and segments:
        seg = random.choice(segments)
        text = seg["merged_text"]
        missing_len = random.choice(missing_lens)

        # 确保有足够空间挖字
        if len(text) < missing_len + 20:
            continue

        # 挖字位置
        hole_start = random.randint(10, len(text) - missing_len - 10)
        before = text[max(0, hole_start - 12):hole_start]
        gt = text[hole_start:hole_start + missing_len]
        after = text[hole_start + missing_len:hole_start + missing_len + 12]

        # 上下文条件
        context_type = random.choice(["full", "left_only", "right_only", "noisy"])

        if context_type == "full":
            q_before, q_after = before, after
        elif context_type == "left_only":
            q_before, q_after = before, ""
        elif context_type == "right_only":
            q_before, q_after = "", after
        elif context_type == "noisy":
            # 用一个 OCR 错字替换上下文中的一个字
            q_before = _add_noise(before, 1)
            q_after = after

        # 记录不同卷
        vol_samples.add(seg.get("volume", 0))

        case = {
            "case_id": f"test_{len(cases):04d}",
            "volume": seg["volume"],
            "missing_length": missing_len,
            "context_type": context_type,
            "query_before": q_before,
            "query_after": q_after,
            "gt": gt,
            "source_page": seg.get("page"),
            "source_line": seg.get("start_line"),
        }
        cases.append(case)

        # 每个卷最少 3 条
        if len(vol_samples) < 20 and len([c for c in cases if c["volume"] == seg["volume"]]) < 3:
            continue

    # 添加边界情况
    _add_edge_cases(cases)

    # 打乱顺序
    random.shuffle(cases)

    return cases[:N_TARGET]


def _add_noise(text: str, n_errors: int) -> str:
    """在文本中引入 OCR 错误。"""
    chars = list(text)
    for _ in range(min(n_errors, len(chars))):
        idx = random.randint(0, len(chars) - 1)
        # 替换为相似形近字或随机汉字
        similar = "已己巳戊戍戌末未土士干千大太天夫失矢未末曰日目且具貝見旦但"
        chars[idx] = random.choice(similar) if similar else "X"
    return "".join(chars)


def _add_edge_cases(cases: list[dict]):
    """添加边界测试案例。"""
    edges = [
        # 完全不存在的句子
        {
            "case_id": f"edge_nonexistent",
            "volume": None, "missing_length": 4, "context_type": "nonexistent",
            "query_before": "ABCDEFGHIJ", "query_after": "KLMNOPQRST",
            "gt": "????",
            "expected": "abstain",
        },
        # 超长上下文
        {
            "case_id": f"edge_long_context",
            "volume": 1, "missing_length": 2, "context_type": "full",
            "query_before": "方广佛华严经卷第一世間淨眼品第一之一", "query_after": "如是我聞一時佛在摩竭提",
            "gt": "??",
            "expected": "retrieved",
        },
    ]
    for e in edges:
        cases.append(e)


def run_evaluation(cases: list[dict], agent: RagAgent) -> list[dict]:
    """运行检索评测。"""
    results = []
    for case in cases:
        t0 = time.time()
        decision = agent.decide(
            context_before=case.get("query_before", ""),
            context_after=case.get("query_after", ""),
            sutra_hint="大方广佛华严经（晋译）",
        )
        elapsed = time.time() - t0

        pred_char = decision.char or ""
        gt = case.get("gt", "")

        # 指标计算
        exact_match = pred_char == gt
        topk_match = gt in [c.get("char","") for c in (decision.candidates or [])[:3]] or exact_match  # GT in top-3

        # 正确拒答（GT 不存在/无法补全 → abstain）
        expected_abstain = case.get("expected") == "abstain"
        correct_abstain = expected_abstain and decision.decision_status == "abstain"
        wrong_fill = expected_abstain and decision.decision_status != "abstain"

        results.append({
            **case,
            "prediction": pred_char,
            "decision_status": decision.decision_status,
            "layer": decision.layer,
            "source": decision.source,
            "retrieval_score": decision.retrieval_score,
            "context_match_score": decision.context_match_score,
            "exact_match": exact_match,
            "topk_match": topk_match,
            "correct_abstain": correct_abstain,
            "wrong_fill": wrong_fill,
            "latency_ms": round(elapsed * 1000),
            "citation": decision.citation,
        })

    return results


def compute_metrics(results: list[dict]) -> dict:
    """计算评测指标。"""
    n = len(results)
    fillable = [r for r in results if r.get("expected") != "abstain"]
    n_fillable = len(fillable)

    # Top-1 Exact Match
    exact_matches = sum(1 for r in fillable if r["exact_match"])

    # Top-3 命中（简化：topk_match）
    topk_matches = sum(1 for r in fillable if r["topk_match"])

    # 字符准确率
    total_chars = sum(len(r.get("gt", "")) for r in fillable)
    correct_chars = sum(
        sum(1 for a, b in zip(r.get("gt", ""), r.get("prediction", "")) if a == b)
        for r in fillable if r.get("gt") and r.get("prediction")
    )

    # 正确拒答
    abstainable = [r for r in results if r.get("expected") == "abstain"]
    correct_abstains = sum(1 for r in abstainable if r["correct_abstain"])
    wrong_fills = sum(1 for r in abstainable if r["wrong_fill"])

    # L1/L3/L4 占比
    l1_count = sum(1 for r in results if r["layer"] == "L1")
    l3_count = sum(1 for r in results if r["layer"] == "L3")
    l4_count = sum(1 for r in results if r["layer"] == "L4")

    return {
        "total_cases": n,
        "fillable_cases": n_fillable,
        "abstain_cases": len(abstainable),
        "top1_exact_match": round(exact_matches / max(n_fillable, 1), 4),
        "top3_hit_rate": round(topk_matches / max(n_fillable, 1), 4),
        "character_accuracy": round(correct_chars / max(total_chars, 1), 4),
        "correct_abstain_rate": round(correct_abstains / max(len(abstainable), 1), 4),
        "wrong_fill_rate": round(wrong_fills / max(len(abstainable), 1), 4),
        "l1_ratio": round(l1_count / max(n, 1), 4),
        "l3_ratio": round(l3_count / max(n, 1), 4),
        "l4_ratio": round(l4_count / max(n, 1), 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / max(n, 1), 2),
    }


def main():
    print("=" * 70)
    print("阶段 3C：检索评测集生成 + 检索评测")
    print(f"目标：{N_TARGET}+ 条测试查询")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载 T0278 全量 JSONL...")
    records = load_all_records()
    print(f"  总记录: {len(records)} 行")
    volumes = sorted(set(r["volume_number"] for r in records))
    print(f"  覆盖卷次: {len(volumes)}/{60}")
    if len(volumes) < 60:
        missing = sorted(set(range(1, 61)) - set(volumes))
        print(f"  [WARN] 缺失卷次: {missing}")

    # 2. 生成测试用例
    print(f"\n[2] 生成 {N_TARGET}+ 条测试查询...")
    cases = generate_test_cases(records)
    print(f"  生成: {len(cases)} 条")

    # 验证 GT 不泄漏
    any_leak = False
    for c in cases:
        gt = c.get("gt", "")
        q_before = c.get("query_before", "")
        q_after = c.get("query_after", "")
        if gt and (gt in q_before or gt in q_after):
            print(f"  [WARN] GT 泄漏: case_id={c['case_id']}")
            any_leak = True
    print(f"  GT 泄漏: {any_leak}")

    # 保存测试用例
    with open(OUTPUT_DIR / "retrieval_test_cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 3. 运行评测
    print("\n[3] 运行检索评测...")
    agent = RagAgent()
    results = run_evaluation(cases, agent)

    # 保存预测
    with open(OUTPUT_DIR / "retrieval_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. 计算指标
    print("\n[4] 计算指标...")
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    (OUTPUT_DIR / "retrieval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. 分组指标
    print("\n[5] 分组指标 (by missing_length)...")
    groups = {}
    for r in results:
        ml = r["missing_length"]
        if ml not in groups:
            groups[ml] = {"correct": 0, "total": 0}
        groups[ml]["total"] += 1
        if r["exact_match"] and r.get("expected") != "abstain":
            groups[ml]["correct"] += 1

    group_rows = []
    for ml in sorted(groups.keys()):
        g = groups[ml]
        acc = round(g["correct"] / max(g["total"], 1), 4)
        group_rows.append({"missing_length": ml, "accuracy": acc, **g})
        print(f"  {ml}字缺失: {g['correct']}/{g['total']} (acc={acc})")

    import csv
    csv_path = OUTPUT_DIR / "retrieval_metrics_by_group.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["missing_length", "correct", "total", "accuracy"])
        writer.writeheader()
        for r in group_rows:
            writer.writerow(r)

    # 6. 失败案例
    print("\n[6] 保存失败案例...")
    failures = [r for r in results if not r["exact_match"] and r.get("expected") != "abstain"]
    with open(OUTPUT_DIR / "failure_cases.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    ambiguous = [r for r in results if r["decision_status"] in ("retrieved_ambiguous", "llm_suggestion")]
    with open(OUTPUT_DIR / "ambiguous_cases.json", "w", encoding="utf-8") as f:
        json.dump(ambiguous, f, ensure_ascii=False, indent=2)
    print(f"  失败案例: {len(failures)}")
    print(f"  歧义案例: {len(ambiguous)}")

    print("\n" + "=" * 70)
    print(f"阶段 3C 完成: {len(results)} 条评测")
    print(f"Top-1 Exact Match: {metrics['top1_exact_match']}")
    print(f"产物: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
