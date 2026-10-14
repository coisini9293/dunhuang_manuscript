#!/usr/bin/env python3
"""Download project datasets into data/raw."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
sys.path.insert(0, str(PROJECT_ROOT))
from utils.rejoin_data import SCIENCEDB_DATASET_DIRNAME

SCIENCEDB_FILE_ID = "ebfd249ff3b8724ca4806ee1866601a0"
SCIENCEDB_URL = f"https://china.scidb.cn/download?fileId={SCIENCEDB_FILE_ID}"
SCIENCEDB_MD5 = "09fe690c09adf6fd72214127e5b602ea"
SCIENCEDB_SIZE = 361_544_222

IDP_BASE = "http://idp.nlc.cn"

# 张涌泉论文核心案例：羽1 ↔ 北敦12288；羽289 ↔ 北敦5753（IDP 编号 BD05753）
IDP_CASES = [
    {"id": "BD12288", "beidun": "北敦12288", "note": "羽1号《大智度论》可缀合"},
    {"id": "BD05753", "beidun": "北敦5753", "note": "羽289号《金有陀罗尼经》可缀合"},
]

DUNHUANG_PATCHING_UP_URL = "https://cave17.e-dunhuang.com/patching-up/"
DUNHUANG_IIIF_PREFIX = "469ebd88c24a4ff1905d6aaa922b07d1"
DUNHUANG_SAMPLE_COUNT = 8
DUNHUANG_PREVIEW_SIZE = "3000,"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path, *, referer: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def download_sciencedb() -> Path:
    out_dir = RAW_DIR / "sciencedb"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "面向残片缀合的敦煌文献图像数据集.zip"
    meta_path = out_dir / "source.json"

    if zip_path.exists() and zip_path.stat().st_size == SCIENCEDB_SIZE:
        if md5_file(zip_path) == SCIENCEDB_MD5:
            print(f"[sciencedb] 已存在且 MD5 校验通过: {zip_path}")
        else:
            print(f"[sciencedb] 文件大小匹配但 MD5 不符，重新下载")
            zip_path.unlink()
    if not zip_path.exists():
        print(f"[sciencedb] 下载中 ({SCIENCEDB_SIZE / 1024 / 1024:.1f} MB)...")
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--continue-at",
            "-",
            "-o",
            str(zip_path),
            SCIENCEDB_URL,
        ]
        subprocess.run(cmd, check=True)

    digest = md5_file(zip_path)
    if digest != SCIENCEDB_MD5:
        raise RuntimeError(f"ScienceDB MD5 不匹配: {digest}")

    extract_dir = out_dir / SCIENCEDB_DATASET_DIRNAME
    if not extract_dir.exists():
        print("[sciencedb] 解压中...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # zip 内中文路径需按 UTF-8 解码，避免解压成乱码目录名
                name = member.filename
                if not (member.flag_bits & 0x800):
                    try:
                        name = name.encode("cp437").decode("gbk")
                    except UnicodeError:
                        name = name.encode("cp437").decode("utf-8", errors="replace")
                target = extract_dir / name
                if member.is_dir() or name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    meta = {
        "source": "Science Data Bank",
        "page": "https://www.scidb.cn/detail?dataSetId=5d08723f70d14b7b8ee76ee2bf57654b",
        "doi": "10.57760/sciencedb.j00001.01083",
        "license": "CC BY-NC-SA 4.0",
        "file_id": SCIENCEDB_FILE_ID,
        "download_url": SCIENCEDB_URL,
        "md5": digest,
        "size_bytes": zip_path.stat().st_size,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sciencedb] 完成: {zip_path}")
    return zip_path


def download_diffhdr_examples() -> None:
    out_dir = RAW_DIR / "diffhdr_examples"
    repo_examples = PROJECT_ROOT / "third_party" / "HDR" / "examples"
    if repo_examples.exists():
        if not out_dir.exists() or not any(out_dir.iterdir()):
            shutil.copytree(repo_examples, out_dir, dirs_exist_ok=True)
        print(f"[diffhdr] 已复制 examples -> {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    third_party = PROJECT_ROOT / "third_party"
    third_party.mkdir(parents=True, exist_ok=True)
    hdr_dir = third_party / "HDR"
    if not hdr_dir.exists():
        print("[diffhdr] 克隆 HDR 仓库...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/yeungchenwa/HDR.git", str(hdr_dir)],
            check=True,
        )
    shutil.copytree(hdr_dir / "examples", out_dir, dirs_exist_ok=True)
    print(f"[diffhdr] 完成: {out_dir}")


def idp_record_url(manuscript_id: str) -> str:
    return f"{IDP_BASE}/database/oo_loader.a4d?pm={quote(manuscript_id)};img=1"


def idp_extract_recnum(html: str) -> str | None:
    for pattern in (r"recnum=(\d+)", r"showLargeImage\((\d+)\)"):
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None


def download_idp_cases() -> None:
    out_dir = RAW_DIR / "idp_li_shengduo"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for case in IDP_CASES:
        manuscript_id = case["id"]
        page_url = idp_record_url(manuscript_id)
        dest = out_dir / f"{manuscript_id}.jpg"
        saved = None
        recnum = None
        full_image_available = None

        try:
            html = subprocess.check_output(
                ["curl", "-sL", "--max-time", "60", "-A", "Mozilla/5.0", page_url],
                text=True,
            )
            if "未找到" in html or "itemNotFound" in html:
                raise RuntimeError("IDP 记录不存在")
            recnum = idp_extract_recnum(html)
            if not recnum:
                raise RuntimeError("无法解析 recnum")
            full_image_available = "該寫卷暫無全圖" not in html

            referer = f"{IDP_BASE}/database/oo_scroll_h.a4d?recnum={recnum}"
            image_url = (
                f"{IDP_BASE}/image_IDP.a4d?type=loadRotatedMainImage;"
                f"recnum={recnum};rotate=0;imageType=_M"
            )
            result = subprocess.run(
                [
                    "curl",
                    "-sL",
                    "--fail",
                    "--max-time",
                    "60",
                    "-A",
                    "Mozilla/5.0",
                    "-e",
                    referer,
                    "-o",
                    str(dest),
                    image_url,
                ],
                capture_output=True,
                text=True,
            )
            if (
                result.returncode == 0
                and dest.exists()
                and dest.stat().st_size > 5_000
                and dest.read_bytes()[:2] == b"\xff\xd8"
            ):
                saved = str(dest.relative_to(RAW_DIR))
                print(f"[idp] 成功: {manuscript_id} -> {dest}")
            else:
                dest.unlink(missing_ok=True)
                raise RuntimeError(result.stderr.strip() or "图像下载失败")
        except Exception as exc:  # noqa: BLE001
            print(f"[idp] 失败 {manuscript_id}: {exc}（页面: {page_url}）")

        manifest.append(
            {
                **case,
                "page": page_url,
                "recnum": recnum,
                "full_image_available": full_image_available,
                "image": saved,
                "note_api": "勿用 /manuscripts/ 或 https；正确入口为 oo_loader.a4d",
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_text(url: str, *, referer: str | None = None) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


def scrape_dunhuang_catalog(limit: int) -> list[dict[str, str]]:
    html = fetch_text(DUNHUANG_PATCHING_UP_URL, referer="https://cave17.e-dunhuang.com/")
    pattern = re.compile(
        r'href="(/patching-up/preview/([a-f0-9]{32})\.html)">\s*'
        r'<div class="collection-name">([^<]+)</div>',
        re.MULTILINE,
    )
    records: list[dict[str, str]] = []
    for match in pattern.finditer(html):
        records.append(
            {
                "uuid": match.group(2),
                "title": match.group(3).strip(),
                "preview_page": urljoin(DUNHUANG_PATCHING_UP_URL, match.group(1)),
            }
        )
        if len(records) >= limit:
            break
    return records


def dunhuang_iiif_urls(uuid: str) -> dict[str, str]:
    base = (
        f"https://cave17.e-dunhuang.com/iiif/{DUNHUANG_IIIF_PREFIX}/v3/{uuid}"
    )
    return {
        "info": f"{base}/info.json",
        "preview": f"{base}/full/{DUNHUANG_PREVIEW_SIZE}/0/default.jpg",
        "full": f"{base}/full/max/0/default.jpg",
    }


def download_dunhuang_samples() -> None:
    out_dir = RAW_DIR / "dunhuang_iiif"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = scrape_dunhuang_catalog(DUNHUANG_SAMPLE_COUNT)
    if not records:
        raise RuntimeError("[dunhuang] 未能从缀合目录页解析到条目")

    saved_images = 0
    manifest: list[dict] = []
    for i, rec in enumerate(records, start=1):
        urls = dunhuang_iiif_urls(rec["uuid"])
        item_dir = out_dir / f"{i:02d}_{rec['uuid']}"
        item_dir.mkdir(parents=True, exist_ok=True)

        meta = {**rec, **urls}
        try:
            info_text = fetch_text(urls["info"], referer=rec["preview_page"])
            meta["iiif_info"] = json.loads(info_text)
        except Exception as exc:  # noqa: BLE001
            meta["iiif_info_error"] = str(exc)

        meta_path = item_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        preview_path = item_dir / "preview.jpg"
        if preview_path.exists() and preview_path.stat().st_size > 50_000:
            print(f"[dunhuang] 已存在: {rec['title']} -> {preview_path}")
            saved_images += 1
        else:
            try:
                download_url(urls["preview"], preview_path, referer=rec["preview_page"])
                if preview_path.stat().st_size > 50_000:
                    saved_images += 1
                    print(f"[dunhuang] 样例图: {rec['title']} -> {preview_path}")
                else:
                    preview_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[dunhuang] 样例图失败 {rec['title']}: {exc}")

        manifest.append(
            {
                **rec,
                "dir": str(item_dir.relative_to(RAW_DIR)),
                "preview_image": str(preview_path.relative_to(RAW_DIR))
                if preview_path.exists()
                else None,
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = out_dir / "README.txt"
    readme.write_text(
        "数字藏经洞缀合目录样例（首页前若干条）。\n"
        f"来源: {DUNHUANG_PATCHING_UP_URL}\n"
        "每条目录含 metadata.json（含 IIIF info）与 preview.jpg（宽 3000px）。\n"
        "完整分辨率可改用 metadata 中的 full URL 单独下载。\n",
        encoding="utf-8",
    )
    print(f"[dunhuang] 完成，{saved_images}/{len(records)} 张预览图 -> {out_dir}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {RAW_DIR}")
    download_sciencedb()
    download_diffhdr_examples()
    download_idp_cases()
    download_dunhuang_samples()
    print("全部下载任务完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
