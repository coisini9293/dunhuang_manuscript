#!/usr/bin/env python3
"""Build the real-case registry and expert transcription work queue."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "real_cases"
REGISTRY_DIR = ROOT / "data" / "real_cases" / "registry"
TRANSCRIPTION_DIR = REPORT / "transcription"
TEMPLATE_DIR = TRANSCRIPTION_DIR / "visible_text_templates"

REGISTRY_FIELDS = [
    "case_id",
    "source_type",
    "institution",
    "collection",
    "shelfmark",
    "local_image_path",
    "group_id",
    "is_pair_or_group",
    "suitable_for_rejoin",
    "suitable_for_autohdr",
    "li_shengduo_evidence_level",
    "cbeta_relation_status",
    "case05_candidate",
    "needs_manual_transcription",
    "needs_expert_review",
    "download_source_url",
    "image_status",
    "notes",
]

TODO_FIELDS = [
    "case_id",
    "image_path",
    "recommended_priority",
    "visible_left_context",
    "visible_right_context",
    "unclear_region",
    "transcription_status",
    "expert_review_status",
    "remarks",
]


def bool_value(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def source_details(group_type: str) -> tuple[str, str, str]:
    if group_type == "dunhuang_iiif":
        return (
            "local_iiif",
            "数字藏经洞（敦煌研究院等合作平台）",
            "缀合预览",
        )
    if group_type == "idp_li_shengduo":
        return (
            "other",
            "中国国家图书馆 / 国际敦煌项目（IDP）",
            "北敦",
        )
    if group_type == "sciencedb_real_rejoin":
        return (
            "local_sciencedb",
            "Science Data Bank（原图馆藏机构随组而异）",
            "面向残片缀合的敦煌文献图像数据集 / 真实缀合组",
        )
    if group_type == "sciencedb_artificial":
        return (
            "local_sciencedb",
            "Science Data Bank（人工切割数据）",
            "面向残片缀合的敦煌文献图像数据集 / 人工数据组",
        )
    return ("other", "项目既有示例", group_type)


def cbeta_status(row: dict) -> str:
    shelfmarks = row["shelfmarks"]
    title = row["title_or_description"]
    if "BD12288" in shelfmarks:
        return "candidate_identified_大智度论卷八_not_line_confirmed"
    if "BD05753" in shelfmarks:
        return "candidate_identified_金有陀罗尼经_not_line_confirmed"
    if row["case05_candidate"].lower() == "true":
        return "unknown_real_transcription_missing"
    if any(
        marker in title
        for marker in (
            "经",
            "論",
            "论",
            "陀罗尼",
            "华严",
            "法华",
            "佛名",
            "瑜伽师地",
        )
    ):
        return "title_only_unverified"
    return "unknown"


def registry_from_local() -> list[dict]:
    with (REPORT / "local_image_inventory.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fh:
        rows = list(csv.DictReader(fh))
    records = []
    for row in rows:
        source_type, institution, collection = source_details(row["group_type"])
        needs_future_transcription = row["group_type"] in {
            "dunhuang_iiif",
            "idp_li_shengduo",
            "sciencedb_real_rejoin",
        }
        image_status = (
            "unavailable_placeholder"
            if row["image_role"] == "unavailable_placeholder"
            else "available"
        )
        evidence = row["li_shengduo_evidence_level"]
        if "BD12288" in row["shelfmarks"] or "BD05753" in row["shelfmarks"]:
            evidence = "confirmed"
        notes = row["notes"]
        if row["group_type"] == "sciencedb_artificial":
            notes = (
                "人工切割数据；保留在总表用于来源区分，不得作为真实残卷成果。"
            )
        elif row["group_type"] == "diffhdr_examples":
            notes = "DiffHDR 示例资产；不是本轮收集的真实残卷。"
        records.append(
            {
                "case_id": f"LC-{row['image_id']}",
                "source_type": source_type,
                "institution": institution,
                "collection": collection,
                "shelfmark": row["shelfmarks"],
                "local_image_path": row["local_path"],
                "group_id": row["group_id"],
                "is_pair_or_group": bool_value(row["suitable_for_rejoin"]),
                "suitable_for_rejoin": bool_value(row["suitable_for_rejoin"]),
                "suitable_for_autohdr": bool_value(row["suitable_for_autohdr"]),
                "li_shengduo_evidence_level": evidence,
                "cbeta_relation_status": cbeta_status(row),
                "case05_candidate": bool_value(row["case05_candidate"]),
                "needs_manual_transcription": needs_future_transcription,
                "needs_expert_review": needs_future_transcription,
                "download_source_url": row["source_url"],
                "image_status": image_status,
                "notes": notes,
                "_group_type": row["group_type"],
                "_width": int(row["width"]),
                "_height": int(row["height"]),
                "_title": row["title_or_description"],
                "_image_role": row["image_role"],
            }
        )
    return records


def registry_from_downloads() -> list[dict]:
    manifest = json.loads(
        (REPORT / "download_manifest.json").read_text(encoding="utf-8")
    )
    result = []
    for row in manifest["downloads"]:
        metadata = json.loads(
            (ROOT / row["local_image_path"]).parent.joinpath("iiif_info.json").read_text(
                encoding="utf-8"
            )
        )
        title = row["shelfmark"]
        case05 = "北敦15770" in title and "北敦15632" in title
        result.append(
            {
                "case_id": f"DL-{row['candidate_id']}",
                "source_type": "downloaded",
                "institution": row["institution"],
                "collection": row["collection"],
                "shelfmark": title,
                "local_image_path": row["local_image_path"],
                "group_id": row["candidate_id"].removeprefix("EDH-"),
                "is_pair_or_group": True,
                "suitable_for_rejoin": True,
                "suitable_for_autohdr": True,
                "li_shengduo_evidence_level": row[
                    "li_shengduo_evidence_level"
                ],
                "cbeta_relation_status": (
                    "unknown_real_transcription_missing"
                    if case05
                    else "title_only_or_unknown"
                ),
                "case05_candidate": case05,
                "needs_manual_transcription": True,
                "needs_expert_review": True,
                "download_source_url": row["source_page_url"],
                "image_status": "available",
                "notes": (
                    "公开 IIIF 下载原图；无明确再分发许可，仅限本地研究。"
                    + (
                        " 该图是拼合预览，不是三个独立馆藏原片。"
                        if case05
                        else ""
                    )
                ),
                "_group_type": "downloaded_iiif",
                "_width": int(metadata["width"]),
                "_height": int(metadata["height"]),
                "_title": title,
                "_image_role": "stitched_preview",
            }
        )
    return result


def write_registry(records: list[dict]) -> None:
    public = [{key: row[key] for key in REGISTRY_FIELDS} for row in records]
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with (REGISTRY_DIR / "real_case_registry.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(public)
    (REGISTRY_DIR / "real_case_registry.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def todo_priority(row: dict) -> str:
    if row["case05_candidate"]:
        return "P0"
    if row["li_shengduo_evidence_level"] == "confirmed":
        return "P0"
    if row["_group_type"] in {"downloaded_iiif", "dunhuang_iiif"}:
        return "P1"
    return "P2"


def make_todos(records: list[dict]) -> list[dict]:
    # Use downloaded IIIF originals instead of duplicate local previews.
    todos = [
        row
        for row in records
        if row["needs_manual_transcription"]
        and row["_group_type"] != "dunhuang_iiif"
        and row["image_status"] == "available"
    ]
    output = []
    for row in todos:
        output.append(
            {
                "case_id": row["case_id"],
                "image_path": row["local_image_path"],
                "recommended_priority": todo_priority(row),
                "visible_left_context": "",
                "visible_right_context": "",
                "unclear_region": "",
                "transcription_status": "not_started",
                "expert_review_status": "not_reviewed",
                "remarks": (
                    f"{row['_title']}；原图 {row['_width']}×{row['_height']}。"
                    "不得用 OCR 直接填充人工释文。"
                ),
            }
        )
    return sorted(
        output,
        key=lambda item: (item["recommended_priority"], item["case_id"]),
    )


def safe_template_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def write_transcription(todos: list[dict]) -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    expected_templates = {
        f"{safe_template_name(row['case_id'])}.md" for row in todos
    }
    for existing in TEMPLATE_DIR.glob("*.md"):
        if existing.name not in expected_templates:
            existing.unlink()
    with (TRANSCRIPTION_DIR / "EXPERT_TRANSCRIPTION_TODO.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=TODO_FIELDS)
        writer.writeheader()
        writer.writerows(todos)
    for row in todos:
        template = f"""# 可见文字抄录：{row['case_id']}

- 图像：`{row['image_path']}`
- 优先级：`{row['recommended_priority']}`
- 抄录状态：`not_started`
- 专家复核：`not_reviewed`

## 行序与可见文字

> 每行使用 `L01`、`L02`……编号；严格按图像可见顺序记录，不补写经文。

| 行号 | 左侧可见上下文 | 缺损/不清区域 | 右侧可见上下文 | 备注 |
|---|---|---|---|---|
| L01 |  |  |  |  |

## 疑字记录

| 行号 | 图上字形 | 暂拟字 | 置信说明 | 专家意见 |
|---|---|---|---|---|
|  |  |  |  |  |

## RAG 查询准备

- 左上下文：
- 右上下文：
- 经号/经名候选（若人工可确认）：
- 禁止事项：不得把 OCR、RAG 候选或模型生成内容回填为“图上可见文字”。
"""
        (TEMPLATE_DIR / f"{safe_template_name(row['case_id'])}.md").write_text(
            template, encoding="utf-8"
        )

    (TRANSCRIPTION_DIR / "TRANSCRIPTION_GUIDELINES.md").write_text(
        """# 真实残卷人工释文规范

## 基本原则

只抄录图像中实际可见的文字。OCR、CBETA 检索和模型候选都只能帮助定位，
不能直接成为最终人工释文；最终记录必须能回指到具体图像、区域和行号。

## 可见文字

1. 依图像阅读顺序编号为 `L01`、`L02`……，不要按现代标点重排。
2. 保留原字形能确认的繁体、异体和俗字；规范字可另写在备注，不能覆盖原录。
3. 左右断口分别填入“左侧可见上下文”“右侧可见上下文”，不要跨缺口自行连句。
4. 只把肉眼确实可见的字录入正文，墨迹、纸纹和后加标记须在备注区分。

## 疑字

- 可见字形但不能唯一识别：使用 `〔疑：甲/乙〕`，同时记录图上字形特征。
- 能大致判定部件但不能定字：使用 `〔疑字：氵+?〕` 等描述。
- 不得因 CBETA 中出现某字就把疑字改成确定字；须由专家在“专家意见”栏确认。

## 缺损与看不清

- `□`：纸面存在、字位可见但字迹无法辨清；一个可确定字位使用一个方框。
- `[…缺…]`：载体物理缺失、断裂或孔洞，无法确认原有多少字。
- `[…约缺3字…]`：只有在版式/行格提供可靠依据时才估计字数，并写明依据。
- “看不清”表示载体仍在但成像或墨迹不足；“确实缺失”表示纸张或文字承载区不存在。
  两者不得互换。

## 行序与区域

- 每个 ROI 必须记录原图路径、像素坐标或可复现的区域说明。
- 跨两片的接缝分别记录 A 片末行和 B 片首行，不预先宣称连续。
- 拼合总图只能用于定位；独立原片存在时，最终释文必须回到原片核验。

## 复核门槛

1. `transcription_status=completed` 只表示初录完成。
2. 专家逐字核对后才可设 `expert_review_status=confirmed`。
3. 只有 `manual_confirmed` 后，文字候选才允许进入视觉修复确认步骤。
4. 单侧上下文只能定位，不得据此伪造缺字。
""",
        encoding="utf-8",
    )

    summary = {
        "todo_count": len(todos),
        "p0_count": sum(item["recommended_priority"] == "P0" for item in todos),
        "p1_count": sum(item["recommended_priority"] == "P1" for item in todos),
        "p2_count": sum(item["recommended_priority"] == "P2" for item in todos),
        "existing_completed_transcriptions": 0,
        "templates_count": len(todos),
        "status": "TRANSCRIPTION_TODO_READY",
    }
    (TRANSCRIPTION_DIR / "transcription_queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    records = registry_from_local() + registry_from_downloads()
    case_ids = [row["case_id"] for row in records]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("duplicate case_id")
    write_registry(records)
    todos = make_todos(records)
    write_transcription(todos)
    print(
        f"REAL_CASE_REGISTRY_PASS records={len(records)} "
        f"transcription_todos={len(todos)}"
    )


if __name__ == "__main__":
    main()
