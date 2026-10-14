from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "real_cases"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".jp2"}
SHELFMARK_RE = re.compile(
    r"(?:S\.\d+|P\.\d+|BD\d+|Дx\.\d+|D\d{3,}|北敦\d+号?|津图\d+号?|"
    r"羽\d+号?|Pt\d+|Dy\d+[A-Z]?(?:\(\d+\))?)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> tuple[str, str, str]:
    text = path.as_posix()
    if "/sciencedb/" in text:
        if "/人工数据组/" in text:
            return "local_sciencedb", "sciencedb_artificial", path.parent.name
        return "local_sciencedb", "sciencedb_real_rejoin", path.parent.name
    if "/dunhuang_iiif/" in text:
        return "local_iiif", "dunhuang_iiif", path.parent.name
    if "/idp_li_shengduo/" in text:
        return "other", "idp_li_shengduo", path.stem
    return "other", "diffhdr_examples", path.stem.split("_", 1)[0]


def role(path: Path, group_type: str) -> str:
    stem = path.stem
    if group_type == "diffhdr_examples":
        if stem.endswith("_damaged_img"):
            return "damaged_input"
        if stem.endswith("_mask_img"):
            return "mask"
        if stem.endswith("_content_img"):
            return "content"
        return "reference"
    if stem == "总图" or (group_type == "sciencedb_artificial" and "v1_" not in stem):
        return "composite_or_reference"
    if group_type == "dunhuang_iiif":
        return "stitched_preview"
    return "fragment"


def metadata_for(path: Path, group_type: str) -> tuple[str, str, str]:
    if group_type == "dunhuang_iiif":
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("title", ""), metadata.get("preview_page", ""), str(metadata_path)
    if group_type == "idp_li_shengduo":
        manifest = json.loads(
            (RAW / "idp_li_shengduo" / "manifest.json").read_text(encoding="utf-8")
        )
        item = next((row for row in manifest if row["id"] == path.stem), {})
        return item.get("note", ""), item.get("page", ""), str(
            RAW / "idp_li_shengduo" / "manifest.json"
        )
    if group_type.startswith("sciencedb_"):
        return path.parent.name, "", str(RAW / "sciencedb" / "source.json")
    return "DiffHDR example asset", "", str(RAW / "README.md")


def image_row(path: Path) -> dict:
    source_type, group_type, group_id = classify(path)
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or path.suffix.lstrip(".").upper()
        mode = image.mode
    file_sha256 = sha256(path)
    is_unavailable_placeholder = (
        group_type == "idp_li_shengduo"
        and file_sha256
        == "bf34679bb416414bb2295964e109f621d8760c4bbb2806038bf4c32fe56cbb70"
    )
    title, source_url, metadata_path = metadata_for(path, group_type)
    search_text = " ".join((path.name, path.parent.name, title))
    shelfmarks = sorted(set(SHELFMARK_RE.findall(search_text)), key=str.casefold)
    image_role = (
        "unavailable_placeholder"
        if is_unavailable_placeholder
        else role(path, group_type)
    )
    is_real_candidate = group_type in {
        "sciencedb_real_rejoin",
        "dunhuang_iiif",
        "idp_li_shengduo",
    }
    is_fragment = image_role == "fragment"
    suitable_autohdr = bool(
        is_real_candidate
        and image_role not in {"composite_or_reference", "stitched_preview"}
        and min(width, height) >= 400
    )
    if group_type == "dunhuang_iiif":
        suitable_autohdr = min(width, height) >= 700
    if is_unavailable_placeholder:
        is_real_candidate = False
        suitable_autohdr = False
    case05 = "0100bd6e" in path.as_posix() or bool(
        {"北敦15770号", "北敦15632号", "北敦14919号"} & set(shelfmarks)
    )
    return {
        "image_id": hashlib.sha1(path.relative_to(ROOT).as_posix().encode()).hexdigest()[:12],
        "source_type": source_type,
        "group_type": group_type,
        "group_id": group_id,
        "title_or_description": title,
        "local_path": path.relative_to(ROOT).as_posix(),
        "directory_name": path.parent.name,
        "filename": path.name,
        "extension": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256,
        "width": width,
        "height": height,
        "pixels": width * height,
        "mode": mode,
        "format": image_format,
        "shelfmarks": " | ".join(shelfmarks),
        "image_role": image_role,
        "is_real_candidate": is_real_candidate,
        "is_fragment": is_fragment,
        "suitable_for_rejoin": (
            group_type
            in {"sciencedb_artificial", "sciencedb_real_rejoin", "dunhuang_iiif"}
            and not is_unavailable_placeholder
        ),
        "suitable_for_autohdr": suitable_autohdr,
        "case05_candidate": case05,
        "has_metadata_or_description": bool(metadata_path),
        "metadata_path": Path(metadata_path).relative_to(ROOT).as_posix(),
        "source_url": source_url,
        "li_shengduo_evidence_level": (
            "probable" if group_type == "idp_li_shengduo" else "unknown"
        ),
        "notes": (
            "IMAGE TEMPORARILY UNAVAILABLE placeholder; no manuscript pixels."
            if is_unavailable_placeholder
            else
            "Official stitched preview; component images are not separately present."
            if group_type == "dunhuang_iiif"
            else "ScienceDB artificial split; not a real-fragment demo."
            if group_type == "sciencedb_artificial"
            else ""
        ),
    }


def group_rows(images: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for image in images:
        grouped[(image["group_type"], image["group_id"])].append(image)
    groups = []
    for (group_type, group_id), rows in sorted(grouped.items()):
        fragments = [row for row in rows if row["image_role"] == "fragment"]
        case05 = any(row["case05_candidate"] for row in rows)
        shelfmarks = sorted(
            {
                mark.strip()
                for row in rows
                for mark in row["shelfmarks"].split("|")
                if mark.strip()
            },
            key=str.casefold,
        )
        groups.append(
            {
                "group_id": group_id,
                "group_type": group_type,
                "title": rows[0]["title_or_description"],
                "image_count": len(rows),
                "fragment_count": len(fragments),
                "shelfmarks": shelfmarks,
                "image_paths": [row["local_path"] for row in rows],
                "source_urls": sorted({row["source_url"] for row in rows if row["source_url"]}),
                "has_composite": any(
                    row["image_role"] in {"composite_or_reference", "stitched_preview"}
                    for row in rows
                ),
                "is_pair_or_group": len(fragments) >= 2
                or group_type == "dunhuang_iiif",
                "suitable_for_rejoin": any(row["suitable_for_rejoin"] for row in rows),
                "suitable_for_autohdr": any(row["suitable_for_autohdr"] for row in rows),
                "case05_candidate": case05,
                "li_shengduo_evidence_level": (
                    "probable" if group_type == "idp_li_shengduo" else "unknown"
                ),
                "needs_manual_transcription": group_type
                in {"sciencedb_real_rejoin", "dunhuang_iiif", "idp_li_shengduo"},
                "notes": (
                    "Case05 stitched preview exists, but separate high-resolution component "
                    "images and a real manual transcription are absent."
                    if case05
                    else "Local file is an IMAGE TEMPORARILY UNAVAILABLE placeholder."
                    if any(row["image_role"] == "unavailable_placeholder" for row in rows)
                    else ""
                ),
            }
        )
    return groups


def contact_sheet(images: list[dict], groups: list[dict]) -> str:
    by_group = defaultdict(list)
    for row in images:
        by_group[(row["group_type"], row["group_id"])].append(row)
    sections = []
    for group in groups:
        rows = by_group[(group["group_type"], group["group_id"])]
        cards = []
        for row in rows:
            image_url = "../../" + row["local_path"]
            cards.append(
                f"""
                <figure>
                  <a href="{html.escape(image_url)}"><img loading="lazy"
                     src="{html.escape(image_url)}" alt="{html.escape(row['filename'])}"></a>
                  <figcaption><b>{html.escape(row['filename'])}</b><br>
                    {row['width']}×{row['height']} · {row['bytes'] / 1024:.1f} KiB<br>
                    {html.escape(row['image_role'])} · {html.escape(row['shelfmarks'] or 'no shelfmark')}
                  </figcaption>
                </figure>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(group['group_id'])}</h2>
              <p>{html.escape(group['group_type'])} · {group['image_count']} images ·
                 shelfmarks: {html.escape(', '.join(group['shelfmarks']) or 'unknown')}
                 {' · <strong>CASE05 candidate</strong>' if group['case05_candidate'] else ''}</p>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Local real-case contact sheet</title>
<style>
body{{font:14px/1.5 system-ui;margin:24px;background:#f5efe4;color:#2b241d}}
header,section{{max-width:1500px;margin:0 auto 28px;background:#fff;padding:20px;border-radius:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px}}
figure{{margin:0;border:1px solid #ddd5c8;padding:8px;border-radius:9px;background:#fbfaf7}}
img{{width:100%;height:180px;object-fit:contain;background:#eee8dc}}
figcaption{{font-size:12px;word-break:break-all}} strong{{color:#9b542c}}
</style></head><body>
<header><h1>本地真实残卷联系表</h1>
<p>共 {len(images)} 张图、{len(groups)} 个分组。人工数据组仅作算法结构参考，不计入真实残卷 Demo。</p></header>
{''.join(sections)}
</body></html>"""


def recommendations(images: list[dict], groups: list[dict]) -> str:
    real_groups = [
        group
        for group in groups
        if group["group_type"] in {"sciencedb_real_rejoin", "dunhuang_iiif"}
    ]
    rejoin = sorted(
        real_groups,
        key=lambda row: (row["fragment_count"], row["image_count"]),
        reverse=True,
    )[:8]
    autohdr = sorted(
        [row for row in images if row["suitable_for_autohdr"]],
        key=lambda row: row["pixels"],
        reverse=True,
    )[:12]
    return (
        "# 本地真实 Demo 候选\n\n"
        "## 结论\n\n"
        "- 现有素材足够制作第一轮真实“缀合输入/来源展示”和本地 mock/replay Demo。\n"
        "- 现有素材不足以宣称 Official AutoHDR 已验证，也不足以解除 Case05：05 号只有官方缀合预览，"
        "没有三个独立高清原片和经过专家确认的真实释文。\n"
        "- ScienceDB 人工数据组不作为真实残卷成果；DiffHDR 示例也不冒充真实案例。\n\n"
        "- 两个 IDP 本地 JPG 是 `IMAGE TEMPORARILY UNAVAILABLE` 占位图，不含残卷像素，"
        "不能进入 AutoHDR；其馆藏/论文证据仍保留。\n\n"
        "## 最适合同源/缀合展示的本地组\n\n"
        + "\n".join(
            f"{index}. `{row['group_id']}`：{row['fragment_count']} 个独立残片，"
            f"{'有总图/预览' if row['has_composite'] else '无总图'}；"
            f"馆藏号 {', '.join(row['shelfmarks']) or '待补'}。"
            for index, row in enumerate(rejoin, 1)
        )
        + "\n\n## 最适合 AutoHDR 输入准备的本地图\n\n"
        + "\n".join(
            f"{index}. `{row['local_path']}` — {row['width']}×{row['height']}，"
            f"来源类型 `{row['group_type']}`。"
            for index, row in enumerate(autohdr, 1)
        )
        + "\n\n## 李盛铎旧藏线索\n\n"
        "- `BD12288`（北敦12288）与羽1号《大智度论》缀合：本地只有 IDP 不可用占位图与既有说明，"
        "暂列 `probable`，需网络阶段核验论文/官方页面。\n"
        "- `BD05753`（北敦5753）与羽289号《金有陀罗尼经》缀合：同样只有不可用占位图，暂列 `probable`。\n"
        "- 其他仅含 BD/北敦编号的图片不能仅凭编号宣称属于李盛铎旧藏，统一列 `unknown`。\n"
    )


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path
        for path in RAW.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    images = [image_row(path) for path in image_paths]
    groups = group_rows(images)
    csv_path = REPORT / "local_image_inventory.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(images)
    summary = {
        "generated_from": "data/raw",
        "image_count": len(images),
        "bytes": sum(row["bytes"] for row in images),
        "group_count": len(groups),
        "counts_by_group_type": dict(Counter(row["group_type"] for row in images)),
        "unavailable_placeholder_count": sum(
            row["image_role"] == "unavailable_placeholder" for row in images
        ),
        "case05_candidate_images": [
            row["local_path"] for row in images if row["case05_candidate"]
        ],
        "groups": groups,
    }
    (REPORT / "local_group_inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT / "local_contact_sheet.html").write_text(
        contact_sheet(images, groups),
        encoding="utf-8",
    )
    (REPORT / "REAL_DEMO_CANDIDATES_LOCAL.md").write_text(
        recommendations(images, groups),
        encoding="utf-8",
    )
    (REPORT / "CASE05_LOCAL_AUDIT.md").write_text(
        """# Case05 本地审计

## 找到的真实输入

- 官方数字藏经洞缀合预览：
  `data/raw/dunhuang_iiif/05_0100bd6eba9c42f5b62dbcebffc928c0/preview.jpg`
- 题名：北敦15770号（局部）+北敦15632号+北敦14919号（局部）
- 本地尺寸：3000×1264。
- 元数据包含官方预览页和 IIIF Image API 地址。

## 仍缺内容

- 三个馆藏号各自的独立高清原图；
- 可核验的真实人工释文与行序；
- 具体 CBETA 卷次/页栏行的人工确认；
- 专家确认；
- Official AutoHDR 真实推理结果。

既有 `outputs/real_case_05/visible_transcription.txt` 与 `outputs/mvp_05/` 明确标为
mock/合成流程产物，不能作为真实释文或 Official 结果。

## 状态

`CASE05 still blocked`

本轮把该预览登记为 `case05_candidate=true`，但不将 CASE05 改为 UNBLOCKED。
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "images": len(images),
                "groups": len(groups),
                "case05_candidates": summary["case05_candidate_images"],
                "reports": str(REPORT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
