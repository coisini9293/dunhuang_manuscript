"""05 号案例验收脚本（阶段 1）。

严格约束：
1. 05 号只能标记为"晋译五十华严文本系统"，cbeta_reference_candidate=T0278，
   scroll_mapping=null，scroll_mapping_verified=false。
2. "众宝杂华"测试标记为 synthetic_masked_text_unit_test，real_case_validation=false。
3. 评分字段：retrieval_score / context_match_score / source_authority /
   visual_alignment_score / decision_status / evidence_type / allow_auto_render。
4. 三路径独立运行：original / manual / rag。
5. 默认 original。

输出：outputs/mvp_05/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rag_agent import RagAgent
from utils.cbeta_index import Cbetaindex
from utils.mask_gen import extract_content_image
from utils.text_renderer import render_content_image, render_overlay


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUNHUANG_IIIF_DIR = PROJECT_ROOT.parent / "data" / "data" / "raw" / "dunhuang_iiif"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mvp_05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dunhuang_05_image() -> tuple[np.ndarray, str]:
    case_dir = DUNHUANG_IIIF_DIR / "05_0100bd6eba9c42f5b62dbcebffc928c0"
    img_path = case_dir / "preview.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"05 号案例图不存在: {img_path}")
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        rgb = np.array(Image.open(img_path).convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
    return bgr, str(img_path)


def build_case_metadata(img_path: str, img_size: tuple[int, int]) -> dict:
    """构建 05 号案例元数据（严格约束）。"""
    return {
        "case_id": "05",
        "title": "北敦15770号(局部)+北敦15632号+北敦14919号(局部)",
        "source": "数字藏经洞 cave17.e-dunhuang.com",
        "source_url": "https://cave17.e-dunhuang.com/patching-up/preview/0100bd6eba9c42f5b62dbcebffc928c0.html",
        "academic_reference": "张小艳,傅及斯.敦煌本'晋译五十华严'残卷缀合研究[J].浙江师范大学学报(社会科学版),2014,39(06):13-26.",
        "image_path": img_path,
        "image_size": list(img_size),
        # 关键约束
        "identified_text_family": "晋译五十华严文本系统",
        "cbeta_reference_candidate": "T0278",
        "scroll_mapping": None,
        "scroll_mapping_verified": False,
        "real_case_text_location_verified": False,
        "caveat": "五十卷本与通行六十卷本存在分卷差异，未做实际释文比对前不得写死卷次",
    }


def build_visible_transcription() -> str:
    """可见文字人工转录（待人工校对）。"""
    return """# 05 号案例可见文字人工转录

## 状态
mock_pending_manual_verification

## 重要说明
这是 mock 转录，仅用于流程演示，不代表 05 号案例的真实释文。
真实释文需由古文字学背景人员人工转录。

## 残片信息
- 北敦15770号（局部）
- 北敦15632号
- 北敦14919号（局部）

## 转录（待人工校对）
（MVP 阶段未做真实 OCR，转录留空待人工校对）

## 学术参考
张小艳,傅及斯.敦煌本"晋译五十华严"残卷缀合研究[J].浙江师范大学学报(社会科学版),2014,39(06):13-26.
"""


def run_unit_tests(agent: RagAgent) -> dict:
    """跑合成缺字单元测试（不冒充真实案例）。"""
    tests = []

    before1 = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after1 = "以为庄严上妙宝轮圆满清净"
    d1 = agent.decide(before1, after1, sutra_hint="大方广佛华严经")
    tests.append({
        "test_id": "unit_1",
        "test_type": "synthetic_masked_text_unit_test",
        "real_case_validation": False,
        "description": "合成缺字单元测试：从 T0278 卷一挖掉 4 字，验证 CBETA 检索",
        "input": {"before": before1, "after": after1, "expected_missing": "众宝杂华"},
        "result": d1.to_dict(),
        "pass": d1.char == "众宝杂华" and d1.decision_status == "retrieved_exact",
        "note": "合成单元测试，不是 05 号真实案例验证",
    })

    before2 = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after2 = "上妙宝轮圆满清净无量妙色"
    d2 = agent.decide(before2, after2, sutra_hint="大方广佛华严经")
    tests.append({
        "test_id": "unit_2",
        "test_type": "synthetic_masked_text_unit_test",
        "real_case_validation": False,
        "description": "合成缺字单元测试：从 T0278 卷一挖掉 8 字，验证 CBETA 检索",
        "input": {"before": before2, "after": after2, "expected_missing": "众宝杂华以为庄严"},
        "result": d2.to_dict(),
        "pass": d2.char == "众宝杂华以为庄严" and d2.decision_status == "retrieved_exact",
        "note": "合成单元测试，不是 05 号真实案例验证",
    })

    before3 = "今天天气真好"
    after3 = "我们去公园玩"
    d3 = agent.decide(before3, after3, sutra_hint="")
    tests.append({
        "test_id": "unit_3",
        "test_type": "synthetic_masked_text_unit_test",
        "real_case_validation": False,
        "description": "合成缺字单元测试：非佛经内容，验证 L4 标灰降级",
        "input": {"before": before3, "after": after3},
        "result": d3.to_dict(),
        "pass": d3.layer == "L4" and d3.decision_status == "abstain",
        "note": "验证永不炸锅机制",
    })

    return {
        "test_type": "synthetic_masked_text_unit_test",
        "real_case_validation": False,
        "all_pass": all(t["pass"] for t in tests),
        "tests": tests,
    }


def generate_content_image(
    content_source: str,
    degraded_bgr: np.ndarray,
    mask_gray: np.ndarray,
    decisions: list[dict],
    manual_text: str,
) -> np.ndarray:
    """根据 content_source 生成 content_image。"""
    h, w = degraded_bgr.shape[:2]

    if content_source == "original":
        content_gray = extract_content_image(degraded_bgr)
        return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR)

    if content_source == "manual":
        ys, xs = np.where(mask_gray > 0)
        if len(ys) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        else:
            bbox = (0, 0, w, h)
        manual_decisions = [{
            "char": manual_text,
            "bbox": bbox,
            "color": "blue",
            "source": "manual",
        }]
        return render_content_image(degraded_bgr, mask_gray, manual_decisions)

    if content_source == "rag":
        return render_content_image(degraded_bgr, mask_gray, decisions)

    raise ValueError(f"未知 content_source: {content_source}")


def run_mvp(content_source: str = "original") -> dict:
    """跑 MVP 验收。"""
    print("=" * 70)
    print(f"05 号案例验收（阶段 1）— content_source={content_source}")
    print("重要：合成单元测试，不是 05 号真实案例验证")
    print("=" * 70)

    # 1. 加载图
    print("\n[1/7] 加载 05 号案例图...")
    degraded, img_path = load_dunhuang_05_image()
    h, w = degraded.shape[:2]
    print(f"  加载成功：{w}x{h}")

    # 2. 案例元数据
    print("\n[2/7] 构建案例元数据...")
    case_meta = build_case_metadata(img_path, (w, h))
    print(f"  identified_text_family: {case_meta['identified_text_family']}")
    print(f"  cbeta_reference_candidate: {case_meta['cbeta_reference_candidate']}")
    print(f"  scroll_mapping: {case_meta['scroll_mapping']}")
    print(f"  scroll_mapping_verified: {case_meta['scroll_mapping_verified']}")
    print(f"  real_case_text_location_verified: {case_meta['real_case_text_location_verified']}")

    # 3. 可见文字转录
    print("\n[3/7] 构建可见文字转录（mock）...")
    transcription = build_visible_transcription()

    # 4. 跑合成单元测试
    print("\n[4/7] 跑合成缺字单元测试...")
    agent = RagAgent()
    unit_report = run_unit_tests(agent)
    for t in unit_report["tests"]:
        status = "PASS" if t["pass"] else "FAIL"
        print(f"  {t['test_id']}: {status} - {t['description']}")
    print(f"  全部通过: {unit_report['all_pass']}")
    print(f"  test_type: {unit_report['test_type']}")
    print(f"  real_case_validation: {unit_report['real_case_validation']}")

    # 5. CBETA Top-k 检索
    print("\n[5/7] CBETA Top-k 检索...")
    before = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after = "以为庄严上妙宝轮圆满清净"
    topk_matches = agent.cbeta.search_topk(before, after, max_missing=10, top_k=5)
    retrieval_topk = []
    for m in topk_matches:
        retrieval_topk.append({
            "sutra_id": m.sutra_id,
            "sutra_name": m.sutra_name,
            "juan": m.juan,
            "pin": m.pin,
            "missing_chars": m.missing_chars,
            "retrieval_score": m.retrieval_score,
            "context_match_score": m.context_match_score,
            "source_authority": m.source_authority,
            "visual_alignment_score": m.visual_alignment_score,
            "decision_status": m.decision_status,
            "evidence_type": m.evidence_type,
            "allow_auto_render": m.allow_auto_render,
            "citation": m.citation,
            "test_type": "synthetic_masked_text_unit_test",
            "real_case_validation": False,
        })
    print(f"  Top-k 命中数: {len(retrieval_topk)}")

    # RAG 决策
    print("\n  RAG 决策...")
    test_damages = [
        {
            "bbox": (200, 100, 60, 200),
            "context_before": before,
            "context_after": after,
            "sutra_hint": "大方广佛华严经（晋译）",
        },
    ]
    decisions = []
    for d in test_damages:
        decision = agent.decide(
            context_before=d["context_before"],
            context_after=d["context_after"],
            sutra_hint=d["sutra_hint"],
        )
        d_dict = decision.to_dict()
        d_dict["bbox"] = list(d["bbox"])
        d_dict["test_type"] = "synthetic_masked_text_unit_test"
        d_dict["real_case_validation"] = False
        decisions.append(d_dict)
        print(f"    层={decision.layer}, 字={decision.char}, status={decision.decision_status}")

    # 6. 生成 mask + 三种 content image
    print(f"\n[6/7] 生成 content_image (source={content_source})...")
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in test_damages:
        x, y, bw, bh = d["bbox"]
        mask[y:y + bh, x:x + bw] = 255

    manual_text = "众宝杂华"  # 人工指定（假设已知，仅用于 manual 路径演示）
    content_img = generate_content_image(content_source, degraded, mask, decisions, manual_text)

    # 保存当前路径的 content image
    content_path = OUTPUT_DIR / f"content_{content_source}.png"
    Image.fromarray(content_img[:, :, ::-1]).save(str(content_path))
    print(f"  content_{content_source}.png 已保存")

    # 如果是 original 路径，同时生成三种 content image（用于对比）
    if content_source == "original":
        for src in ["original", "manual", "rag"]:
            c = generate_content_image(src, degraded, mask, decisions, manual_text)
            p = OUTPUT_DIR / f"content_{src}.png"
            Image.fromarray(c[:, :, ::-1]).save(str(p))
            print(f"  content_{src}.png 已保存")

    # mask overlay
    overlay = render_overlay(degraded, decisions)
    overlay_path = OUTPUT_DIR / "mask_overlay.png"
    Image.fromarray(overlay[:, :, ::-1]).save(str(overlay_path))
    print(f"  mask_overlay.png 已保存")

    # 7. 保存所有 JSON
    print("\n[7/7] 保存 JSON 产物...")
    (OUTPUT_DIR / "case_metadata.json").write_text(
        json.dumps(case_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "visible_transcription.txt").write_text(transcription, encoding="utf-8")
    (OUTPUT_DIR / "retrieval_topk.json").write_text(
        json.dumps(retrieval_topk, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "rag_decision.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    validation_summary = {
        "content_source": content_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_metadata": str(OUTPUT_DIR / "case_metadata.json"),
        "visible_transcription": str(OUTPUT_DIR / "visible_transcription.txt"),
        "retrieval_topk": str(OUTPUT_DIR / "retrieval_topk.json"),
        "rag_decision": str(OUTPUT_DIR / "rag_decision.json"),
        "content_image": str(content_path),
        "mask_overlay": str(overlay_path),
        "unit_tests": {
            "test_type": "synthetic_masked_text_unit_test",
            "real_case_validation": False,
            "all_pass": unit_report["all_pass"],
            "tests_count": len(unit_report["tests"]),
        },
        "scroll_mapping_verified": False,
        "real_case_text_location_verified": False,
        "caveat": "合成单元测试通过，05 号真实案例尚未完成释文定位",
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  case_metadata.json")
    print(f"  visible_transcription.txt")
    print(f"  retrieval_topk.json")
    print(f"  rag_decision.json")
    print(f"  content_{content_source}.png")
    print(f"  mask_overlay.png")
    print(f"  validation_summary.json")

    print("\n" + "=" * 70)
    print(f"验收完成（content_source={content_source}）")
    print(f"合成单元测试全部通过: {unit_report['all_pass']}")
    print(f"05 号真实案例验证: False（尚未完成释文定位）")
    print(f"产物目录: {OUTPUT_DIR}")
    print("=" * 70)
    return validation_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="05 号案例验收")
    parser.add_argument(
        "--content-source",
        choices=["original", "manual", "rag"],
        default="original",
        help="content_image 来源（默认 original）",
    )
    args = parser.parse_args()
    run_mvp(content_source=args.content_source)
