"""阶段 5：05 号真实案例文本定位（修复版）。

使用修复后的 MultiScaleIndex 进行定位。
当前使用 mock 转录，标记 pending_transcription_verification。
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.multi_scale_index import MultiScaleIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_case_05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# T0278 真实文本片段（模拟 05 号残卷中可能出现的文字）
REAL_SEGMENTS = [
    ("如是我聞一時佛在摩竭提國寂滅道場始成正覺", "其地金剛具足嚴淨眾寶雜華以為莊飾"),
    ("大方廣佛華嚴經卷第一", "世間淨眼品第一之一"),
    ("東晉天竺三藏佛馱跋陀羅譯", ""),
    ("盧舍那佛宿世善友一切成就功德大海", "諸波羅蜜周滿普照慧眼清淨"),
    ("菩薩摩訶薩住此法門已", "能於十方一切世界轉大法輪"),
]

def main():
    print("=" * 60)
    print("阶段 5：05 号真实案例文本定位")
    print("=" * 60)

    idx = MultiScaleIndex(PROJECT_ROOT / "data" / "cbeta" / "parsed")

    results = []
    for seg_id, (before, after) in enumerate(REAL_SEGMENTS, 1):
        print(f"\nSegment {seg_id}: \"{before[:30]}...\"")

        if after:
            ms = idx.search(before, after, max_missing=15, top_k=10)
        else:
            # 单侧搜索
            ms = idx.search_single_side(before, top_k=10) if hasattr(idx, 'search_single_side') else []

        candidates = []
        for m in ms:
            candidates.append({
                "volume": m.volume, "pin": m.pin,
                "scale": m.scale,
                "missing_chars": m.missing_chars,
                "retrieval_score": m.retrieval_score,
                "context_match_score": m.context_match_score,
                "decision_status": m.decision_status,
                "citation": m.citation,
                "chunk_id": m.chunk_id,
            })

        best_vol = ms[0].volume if ms else None
        status = "retrieved" if ms else "no_match"
        results.append({
            "segment_id": seg_id,
            "query_before": before[:50],
            "query_after": after[:50] if after else None,
            "top_match_volume": best_vol,
            "status": status,
            "candidates": candidates[:10],
        })
        print(f"  -> vol={best_vol}, status={status}, candidates={len(candidates)}")

    report = {
        "case_id": "05",
        "identified_text_family": "晋译五十华严文本系统",
        "cbeta_reference_candidate": "T0278",
        "scroll_mapping": None,
        "scroll_mapping_verified": False,
        "real_case_text_location_verified": False,
        "note": "使用修复后的 MultiScaleIndex (zhconv + 多尺度分块)。文本片段为 T0278 真实内容，非 05 号残卷真实释文。待人工转录后更新。",
        "segments": results,
        "verdict": {
            "retrieved": sum(1 for r in results if r["status"] == "retrieved"),
            "no_match": sum(1 for r in results if r["status"] == "no_match"),
        },
    }

    (OUTPUT_DIR / "t0278_location_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 更新 case_metadata
    case_meta = {
        "case_id": "05",
        "identified_text_family": "晋译五十华严文本系统",
        "cbeta_reference_candidate": "T0278",
        "scroll_mapping": None,
        "scroll_mapping_verified": False,
        "real_case_text_location_verified": False,
        "location_status": "partial_mock_retrieval",
        "caveat": "使用 T0278 真实文本片段模拟定位。真实 05 号残卷释文待人工转录。五十卷本与六十卷本存在分卷差异。",
    }
    (OUTPUT_DIR / "case_metadata.json").write_text(
        json.dumps(case_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nResult: {report['verdict']}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
