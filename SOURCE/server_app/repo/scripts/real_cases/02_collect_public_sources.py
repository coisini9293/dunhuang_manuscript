#!/usr/bin/env python3
"""Collect the first bounded batch of openly accessible real-case sources.

Only the six public Digital Dunhuang IIIF endpoints are downloaded
automatically.  The two legacy IDP/NLC pages are recorded for manual action;
this script does not attempt to bypass their browser/download workflow.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_IIIF = ROOT / "data" / "raw" / "dunhuang_iiif"
RAW_IDP = ROOT / "data" / "raw" / "idp_li_shengduo"
INCOMING = ROOT / "data" / "real_cases" / "incoming"
REPORT = ROOT / "reports" / "real_cases"
USER_AGENT = "DunhuangRealCasesAudit/1.0 (bounded public-source audit)"

ACADEMIC_EVIDENCE_URL = "https://www.sohu.com/a/658599611_723484"
ACADEMIC_EVIDENCE_TITLE = "张涌泉、周思敏：《李盛铎旧藏敦煌写卷残断原因新探》"
MDPI_BD12288_URL = "https://www.mdpi.com/2077-1444/16/11/1387"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> tuple[int, str]:
    """Download a public URL with a recoverable .part file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        content_type = response.headers.get("Content-Type", "")
        mode = "ab" if start and status == 206 else "wb"
        with part.open(mode) as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
    part.replace(destination)
    return status, content_type


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def public_iiif_candidates() -> list[dict]:
    candidates = []
    for metadata_file in sorted(RAW_IIIF.glob("*/metadata.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        uuid = metadata["uuid"]
        candidates.append(
            {
                "candidate_id": f"EDH-{uuid}",
                "institution": "数字藏经洞（敦煌研究院等合作平台）",
                "collection": "缀合预览",
                "shelfmark": metadata["title"],
                "source_page_url": metadata["preview_page"],
                "image_url": metadata["full"],
                "iiif_info_url": metadata["info"],
                "source_authenticity": "confirmed_official_page",
                "li_shengduo_evidence_level": "unknown",
                "download_policy": "public_iiif_no_login",
                "notes": (
                    "官方缀合预览及 IIIF Image API 可直接访问；未发现该组属于李盛铎"
                    "旧藏的证据。公开可访问不等于可再分发，下载仅供本项目研究核验。"
                ),
                "metadata": metadata,
            }
        )
    return candidates


def idp_candidates() -> list[dict]:
    manifest = json.loads(
        (RAW_IDP / "manifest.json").read_text(encoding="utf-8")
    )
    result = []
    for item in manifest:
        is_bd12288 = item["id"] == "BD12288"
        result.append(
            {
                "candidate_id": f"IDP-{item['id']}",
                "institution": "中国国家图书馆 / 国际敦煌项目（IDP）",
                "collection": "北敦",
                "shelfmark": item["beidun"],
                "source_page_url": item["page"],
                "image_url": "",
                "iiif_info_url": "",
                "source_authenticity": "confirmed_official_catalog_url",
                "li_shengduo_evidence_level": "confirmed",
                "download_policy": "human_action_required_legacy_page",
                "notes": item["note"],
                "evidence_urls": [
                    ACADEMIC_EVIDENCE_URL,
                    *([MDPI_BD12288_URL] if is_bd12288 else []),
                ],
                "local_preview": str(
                    (RAW_IDP / item["image"].split("/")[-1]).relative_to(ROOT)
                ).replace("\\", "/"),
            }
        )
    return result


def collect_iiif(candidate: dict) -> dict:
    destination = INCOMING / "digital_dunhuang" / candidate["candidate_id"][4:]
    image_path = destination / "original.jpg"
    info_path = destination / "iiif_info.json"
    page_path = destination / "source_page.html"
    errors: list[str] = []

    for url, path in (
        (candidate["image_url"], image_path),
        (candidate["iiif_info_url"], info_path),
        (candidate["source_page_url"], page_path),
    ):
        try:
            fetch(url, path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    downloaded = image_path.exists() and image_path.stat().st_size > 0
    image_hash = sha256(image_path) if downloaded else ""
    image_size = image_path.stat().st_size if downloaded else 0
    source_record = {
        key: value
        for key, value in candidate.items()
        if key not in {"metadata"}
    }
    source_record.update(
        {
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "local_image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
            "image_sha256": image_hash,
            "image_bytes": image_size,
            "download_status": "downloaded" if downloaded and not errors else "partial",
            "errors": errors,
            "rights_note": (
                "The endpoint is publicly readable. No redistribution licence was "
                "identified in this audit; retain for research verification only."
            ),
        }
    )
    write_json(destination / "source.json", source_record)
    write_text(
        destination / "README.md",
        "\n".join(
            [
                f"# {candidate['shelfmark']}",
                "",
                f"- 官方页面：{candidate['source_page_url']}",
                f"- IIIF Image API：{candidate['image_url']}",
                f"- 下载状态：{source_record['download_status']}",
                f"- SHA-256：`{image_hash or 'not available'}`",
                "",
                "用途限制：本地研究、来源核验与模型输入准备。未核实再分发许可，"
                "不得把原图提交到 Git 或部署包。",
                "",
            ]
        ),
    )
    write_text(
        destination / "SHA256SUMS.txt",
        f"{image_hash}  original.jpg\n" if image_hash else "",
    )
    return source_record


def write_reports(candidates: list[dict], downloads: list[dict]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    download_by_id = {item["candidate_id"]: item for item in downloads}
    for candidate in candidates:
        downloaded = download_by_id.get(candidate["candidate_id"], {})
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "institution": candidate["institution"],
                "collection": candidate["collection"],
                "shelfmark": candidate["shelfmark"],
                "source_page_url": candidate["source_page_url"],
                "source_authenticity": candidate["source_authenticity"],
                "li_shengduo_evidence_level": candidate[
                    "li_shengduo_evidence_level"
                ],
                "download_policy": candidate["download_policy"],
                "download_status": downloaded.get(
                    "download_status",
                    "human_action_required"
                    if candidate["download_policy"].startswith("human_")
                    else "not_downloaded",
                ),
                "local_image_path": downloaded.get(
                    "local_image_path", candidate.get("local_preview", "")
                ),
                "notes": candidate["notes"],
            }
        )
    fieldnames = list(rows[0])
    with (REPORT / "online_source_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evidence_lines = []
    for candidate in candidates:
        evidence_lines.append(
            json.dumps(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_authenticity": candidate["source_authenticity"],
                    "li_shengduo_evidence_level": candidate[
                        "li_shengduo_evidence_level"
                    ],
                    "source_page_url": candidate["source_page_url"],
                    "evidence_urls": candidate.get("evidence_urls", []),
                    "claim": candidate["notes"],
                },
                ensure_ascii=False,
            )
        )
    write_text(
        REPORT / "online_source_evidence.jsonl",
        "\n".join(evidence_lines) + "\n",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_round_limit": 20,
        "candidate_count": len(candidates),
        "automatic_download_count": sum(
            item["download_status"] == "downloaded" for item in downloads
        ),
        "human_action_count": sum(
            row["download_status"] == "human_action_required" for row in rows
        ),
        "total_downloaded_bytes": sum(item["image_bytes"] for item in downloads),
        "downloads": downloads,
    }
    write_json(REPORT / "download_manifest.json", manifest)

    human_actions = [
        {
            "case_id": "IDP-BD12288",
            "institution": "中国国家图书馆 / 国际敦煌项目（IDP）",
            "shelfmark": "北敦12288",
            "current_page": (
                "http://idp.nlc.cn/database/oo_loader.a4d?pm=BD12288;img=1"
            ),
            "only_action": (
                "在该馆藏详情页中手动选择可用的最大尺寸全图下载；当前自动浏览器"
                "连接停留在 about:blank，无法可靠确认按钮文字，禁止猜测或绕过。"
            ),
            "download_spec": "页面提供的最大尺寸原图（不要截图缩略图）",
            "expected_type": "JPEG/TIFF/JP2（以页面实际提供为准）",
            "expected_identity": "馆藏号 BD12288 / 北敦12288",
            "save_to": "data/real_cases/incoming/idp_nlc/BD12288/original.<ext>",
            "status": "HUMAN_ACTION_REQUIRED",
        },
        {
            "case_id": "IDP-BD05753",
            "institution": "中国国家图书馆 / 国际敦煌项目（IDP）",
            "shelfmark": "北敦5753",
            "current_page": (
                "http://idp.nlc.cn/database/oo_loader.a4d?pm=BD05753;img=1"
            ),
            "only_action": (
                "在该馆藏详情页中手动选择可用的最大尺寸全图下载；当前自动网页"
                "读取返回错误，无法可靠确认按钮文字，禁止猜测或绕过。"
            ),
            "download_spec": "页面提供的最大尺寸原图（不要截图缩略图）",
            "expected_type": "JPEG/TIFF/JP2（以页面实际提供为准）",
            "expected_identity": "馆藏号 BD05753 / 北敦5753",
            "save_to": "data/real_cases/incoming/idp_nlc/BD05753/original.<ext>",
            "status": "HUMAN_ACTION_REQUIRED",
        },
    ]
    write_json(REPORT / "HUMAN_ACTION_QUEUE.json", human_actions)
    action_sections = ["# 人工操作队列", ""]
    for item in human_actions:
        action_sections.extend(
            [
                "```text",
                "HUMAN_ACTION_REQUIRED",
                "",
                f"案例编号：{item['case_id']}",
                f"馆藏机构：{item['institution']}",
                f"馆藏编号：{item['shelfmark']}",
                f"当前页面：{item['current_page']}",
                f"需要用户执行的唯一动作：{item['only_action']}",
                f"应选择的下载规格：{item['download_spec']}",
                f"预期文件类型：{item['expected_type']}",
                f"预期文件名或识别特征：{item['expected_identity']}",
                f"下载后应保存到：{item['save_to']}",
                "完成后请回复：已完成，继续",
                "```",
                "",
            ]
        )
    write_text(REPORT / "HUMAN_ACTION_QUEUE.md", "\n".join(action_sections))

    write_text(
        REPORT / "LI_SHENGDUO_EVIDENCE_AUDIT.md",
        f"""# 李盛铎旧藏证据审计

## 已确认

### 北敦12288号 ↔ 羽1号《大智度论》

- {ACADEMIC_EVIDENCE_TITLE}明确逐项比对内容、行款、字体和纸幅，并判定可缀合。
- 文章给出对应《大正藏》范围 `T25/115A8—121B13`。
- 2025 年同行评议论文另有 BD12288 与 Yu1 的缀合图。
- 证据等级：`confirmed`。

### 北敦5753号 ↔ 羽289号《金有陀罗尼经》

- 同一研究明确说明二号内容连续、接口边缘及乌丝栏吻合、纸高均为
  26.9 厘米，并可合成完整一卷。
- 文章给出对应《大正藏》范围 `T85/1455C16—1456C10`。
- 证据等级：`confirmed`。

## 来源

- 学术文章公开转载：{ACADEMIC_EVIDENCE_URL}
- BD12288 补充同行评议来源：{MDPI_BD12288_URL}
- 馆藏详情页见 `HUMAN_ACTION_QUEUE.md`。

## 限制

- 当前只有两个北敦缩略预览，没有羽1、羽289的合法本地高清原图。
- 学术证据可以确认缀合关系，但不能替代原图下载许可、人工释文或专家复核。
- 其余数字藏经洞与 ScienceDB 组未发现李盛铎归属证据，不得凭图像相似宣称。
""",
    )

    write_text(
        REPORT / "REAL_CASE_DOWNLOAD_REPORT.md",
        f"""# 第一轮真实案例公开来源下载报告

## 范围

- 候选组：{len(candidates)}（上限 20，未超限）。
- 自动下载：{manifest['automatic_download_count']} 组数字藏经洞公开 IIIF 原图。
- 自动下载原图合计：{manifest['total_downloaded_bytes']:,} bytes。
- 需人工操作：{manifest['human_action_count']} 组 IDP/NLC 旧式馆藏页。
- 未连接 AutoDL，未运行 Official AutoHDR，未改动 RAG V2。

## 合法性与安全边界

- 自动下载仅使用无需登录、验证码或条款确认的公开 IIIF URL。
- 数字藏经洞页面未在本次审计中识别出明确再分发许可证；原图仅保留在
  Git 忽略目录，用于本地研究核验，不进入仓库或部署包。
- IDP/NLC 页面无法稳定自动加载，因此没有猜测下载链接、抓取隐藏接口或
  绕过网页限制。

## 结果

- 6 个 IIIF 案例均保存 `original.jpg`、`source.json`、`README.md`、
  `SHA256SUMS.txt`、`source_page.html`、`iiif_info.json`。
- 二进制原图由 `.gitignore` 排除；来源、校验和与报告可审计。
- Case05 下载的是官方**拼合预览原图**，不是北敦15770、15632、14919
  三张独立高清馆藏原图，因此仍为 `CASE05_BLOCKED`。

## 状态

`REAL_CASE_DOWNLOAD_PARTIAL`
""",
    )


def main() -> int:
    iiif = public_iiif_candidates()
    idp = idp_candidates()
    candidates = iiif + idp
    if len(candidates) > 20:
        raise RuntimeError("first-round candidate limit exceeded")
    downloads = [collect_iiif(candidate) for candidate in iiif]
    write_reports(candidates, downloads)
    completed = sum(item["download_status"] == "downloaded" for item in downloads)
    print(f"PUBLIC_SOURCE_COLLECTION_PASS candidates={len(candidates)} downloaded={completed}")
    return 0 if completed == len(iiif) else 1


if __name__ == "__main__":
    sys.exit(main())
