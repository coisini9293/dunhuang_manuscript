#!/usr/bin/env python3
"""Prepare non-destructive AutoHDR/rejoin input queues from the real registry."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "real_cases" / "registry" / "real_case_registry.json"
INVENTORY = ROOT / "reports" / "real_cases" / "local_image_inventory.csv"
GROUPS = ROOT / "reports" / "real_cases" / "local_group_inventory.json"
QUEUE_JSON = ROOT / "data" / "real_cases" / "autohdr_queue.json"
QUEUE_CSV = ROOT / "data" / "real_cases" / "autohdr_queue.csv"
REPORT = ROOT / "reports" / "autohdr" / "REAL_CASE_AUTOHDR_INPUTS.md"

QUEUE_FIELDS = [
    "queue_id",
    "case_id",
    "image_path",
    "source_type",
    "group_id",
    "shelfmark",
    "task_type",
    "priority",
    "roi_status",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "mask_status",
    "preprocessing_recommendation",
    "manual_transcription_status",
    "official_autohdr_status",
    "notes",
]


def load_inventory() -> dict[str, dict]:
    with INVENTORY.open(encoding="utf-8-sig", newline="") as fh:
        return {
            row["local_path"]: row
            for row in csv.DictReader(fh)
        }


def select_queue() -> list[dict]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = load_inventory()
    selected: list[dict] = []

    def add(row: dict, priority: str, reason: str) -> None:
        path = row["local_image_path"]
        metadata = inventory.get(path, {})
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        file_bytes = (ROOT / path).stat().st_size
        preprocess = []
        if max(width, height) > 4000 or file_bytes > 20 * 1024 * 1024:
            preprocess.append(
                "保留原图；为当前20MB上传接口另建无损/高质量工作副本"
            )
        if "总图" in Path(path).name:
            preprocess.append("优先回到独立原片选ROI；总图只作定位")
        if row["source_type"] == "downloaded":
            preprocess.append("核对IIIF方向与尺寸，不覆盖原图")
        if not preprocess:
            preprocess.append("保持原方向；由人工在副本上选择ROI，不改原图")
        selected.append(
            {
                "queue_id": f"AH-{len(selected)+1:03d}",
                "case_id": row["case_id"],
                "image_path": path,
                "source_type": row["source_type"],
                "group_id": row["group_id"],
                "shelfmark": row["shelfmark"],
                "task_type": "single_image_damage_restore",
                "priority": priority,
                "roi_status": "pending_manual_selection",
                "roi_x": "",
                "roi_y": "",
                "roi_width": "",
                "roi_height": "",
                "mask_status": "optional_pending_advanced_manual_mask",
                "preprocessing_recommendation": "；".join(preprocess),
                "manual_transcription_status": "not_started",
                "official_autohdr_status": "OFFICIAL_STUB",
                "notes": reason,
            }
        )

    # P0: Case05 and the two academically confirmed Li Shengduo links.
    for row in registry:
        if row["case05_candidate"] and row["source_type"] == "downloaded":
            add(row, "P0", "Case05候选；仅有拼合预览，不能当独立原片。")
    for row in registry:
        if (
            row["li_shengduo_evidence_level"] == "confirmed"
            and row["suitable_for_autohdr"]
            and row.get("image_status") == "available"
        ):
            add(
                row,
                "P0",
                "缀合关系有论文证据；当前图为低分辨率预览，待人工下载高清原图。",
            )

    # P1: remaining downloaded IIIF sources.
    for row in registry:
        if row["source_type"] == "downloaded" and not row["case05_candidate"]:
            add(row, "P1", "官方公开IIIF拼合图；用于真实输入结构与ROI准备。")

    # P1/P2: highest-resolution independent ScienceDB real fragments.
    science_rows = []
    for row in registry:
        path = row["local_image_path"]
        metadata = inventory.get(path)
        if (
            row["source_type"] == "local_sciencedb"
            and row["suitable_for_autohdr"]
            and metadata
            and metadata["group_type"] == "sciencedb_real_rejoin"
            and "总图" not in Path(path).name
        ):
            science_rows.append(
                (int(metadata["pixels"]), row)
            )
    for _, row in sorted(
        science_rows, key=lambda item: item[0], reverse=True
    )[:10]:
        add(row, "P1", "ScienceDB真实缀合组独立残片；适合人工选择损伤ROI。")
    return selected


def main() -> None:
    queue = select_queue()
    QUEUE_JSON.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_JSON.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with QUEUE_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(queue)

    groups = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    rejoin = sorted(
        (
            group
            for group in groups
            if group["group_type"] in {"sciencedb_real_rejoin", "dunhuang_iiif"}
            and group["suitable_for_rejoin"]
        ),
        key=lambda group: (
            group["fragment_count"],
            group["image_count"],
            group["title"],
        ),
        reverse=True,
    )[:10]
    rejoin_lines = [
        f"{idx}. `{group['title']}` — {group['fragment_count']} 个独立残片，"
        f"group_id=`{group['group_id']}`。"
        for idx, group in enumerate(rejoin, 1)
    ]
    autohdr_lines = [
        f"{idx}. `{item['image_path']}` — {item['priority']}，"
        f"{item['preprocessing_recommendation']}。"
        for idx, item in enumerate(queue, 1)
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        """# 真实案例 AutoHDR 输入准备

## 当前 AutoHDR 模式审计

- `mock`：动态生成与输入尺寸相关的占位损伤框和带水印结果；不是模型推理。
- `replay`：读取带 SHA-256 校验的 synthetic contract fixture；忽略真实输入内容，
  只能验证接口契约，不能宣称对真实图像完成修复。
- `official`：当前实现返回 `autohdr_inference_not_executed`，依赖 Linux/CUDA
  与官方工程，真实状态为 `OFFICIAL_STUB` / `AUTODL_REQUIRED`。
- 恢复接口服务端要求至少一条 `manual_confirmed` 决策；没有确认时返回 abstain。

## 适合缀合的输入组

"""
        + "\n".join(rejoin_lines)
        + """

## AutoHDR 单图/ROI 队列

"""
        + "\n".join(autohdr_lines)
        + """

## ROI、Mask 与预处理约束

- 所有 ROI 坐标当前为空，必须由人工在原图或工作副本上确认。
- Mask 为高级可选项；自动损伤框可以先行展示，但不得自动采纳候选。
- 旋转、裁剪、缩放只允许产生工作副本，原始文件和 SHA-256 不变。
- 拼合总图适合定位，不应替代可获得的独立原片。
- 队列中的 `official_autohdr_status` 全部保持 `OFFICIAL_STUB`。
""",
        encoding="utf-8",
    )
    print(f"AUTOHDR_QUEUE_PASS items={len(queue)} rejoin_groups={len(rejoin)}")


if __name__ == "__main__":
    main()
