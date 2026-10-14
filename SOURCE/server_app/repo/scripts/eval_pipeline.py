"""阶段 8：完整实验评测框架。

评测维度：
- 检索：Top-1 EM, Top-3, Char Accuracy, L1/L3/L4 占比
- 修复：PSNR, SSIM, stroke_iou, mask 外一致性（在 GT 可用时）
- 消融：original/manual_correct/manual_wrong/blank/rag_correct
"""
import json, sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def build_report():
    """汇总已有结果生成评测报告。"""
    sections = {}

    # 检索指标
    audit = PROJECT_ROOT / "outputs" / "cbeta_index_audit"
    if (audit / "retrieval_metrics.json").exists():
        sections["retrieval"] = json.loads((audit / "retrieval_metrics.json").read_text(encoding="utf-8"))

    # 方向性验证
    cgv = PROJECT_ROOT / "outputs" / "content_guidance_validation"
    if (cgv / "run_metadata.json").exists():
        sections["content_guidance"] = json.loads((cgv / "run_metadata.json").read_text(encoding="utf-8"))

    # DiffHDR
    dabc = PROJECT_ROOT / "outputs" / "mvp_05" / "diffhdr_abc"
    if (dabc / "pixel_integrity.json").exists():
        sections["pixel_integrity"] = json.loads((dabc / "pixel_integrity.json").read_text(encoding="utf-8"))

    # 05 号案例
    rc05 = PROJECT_ROOT / "outputs" / "real_case_05"
    if (rc05 / "t0278_location_results.json").exists():
        sections["case_05"] = json.loads((rc05 / "t0278_location_results.json").read_text(encoding="utf-8"))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "experimental",
        "note": "评测数据基于当前实验状态。阶段3优化完成后部分指标将更新。",
        "sections": sections,
    }
    (OUTPUT / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测报告: {OUTPUT / 'evaluation_report.json'}")
    return report


if __name__ == "__main__":
    build_report()
