"""阶段 3B：T0278 全量本地语料构建。

从 CBETA Normal Text GitHub 下载 T0278 60 卷，解析为 JSONL，
建 BM25 + bigram + trigram 索引。

数据来源：
- repository: DILA-edu/cbeta-normal-text
- path: T/T09/T09n0278/
- 文件: 001.txt ~ 060.txt（卷 1-60）
- raw URL: https://raw.githubusercontent.com/DILA-edu/cbeta-normal-text/main/T/T09/T09n0278/

输出目录：
  data/cbeta/
    source_manifest.json
    raw/normal_text/T0278_*.txt
    parsed/T0278_*.jsonl
    index/{metadata, bm25, bigram, trigram}
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CBETA_DATA = PROJECT_ROOT / "data" / "cbeta"
CBETA_DATA.mkdir(parents=True, exist_ok=True)

RAW_DIR = CBETA_DATA / "raw" / "normal_text"
PARSED_DIR = CBETA_DATA / "parsed"
INDEX_DIR = CBETA_DATA / "index"
for d in [RAW_DIR, PARSED_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPO = "https://github.com/DILA-edu/cbeta-normal-text"
RAW_BASE = "https://raw.githubusercontent.com/DILA-edu/cbeta-normal-text/main/T/T09/T09n0278"
COMMIT = "2ffcfdb5011e97637295c6589061b9779273ddda"  # 2025 R3, Dec 29 2025
N_VOLUMES = 60


def download_volume(vol: int) -> str:
    """下载单卷，返回文件路径。"""
    fname = f"{vol:03d}.txt"
    url = f"{RAW_BASE}/{fname}"
    local_path = RAW_DIR / f"T0278_{fname}"
    if local_path.exists():
        return str(local_path)

    import urllib.request
    print(f"  下载 T0278 卷{vol}...")
    try:
        urllib.request.urlretrieve(url, str(local_path))
        time.sleep(0.5)  # github rate limit
    except Exception as e:
        print(f"  [FAIL] 卷{vol}: {e}")
        return None
    return str(local_path)


def parse_volume(vol: int) -> list[dict]:
    """解析单卷为 JSONL 记录列表。"""
    txt_path = RAW_DIR / f"T0278_{vol:03d}.txt"
    if not txt_path.exists():
        return []

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    records = []
    pin = None
    juan_title = None
    current_page = None
    current_col = None

    for line in lines:
        # 解析页栏行信息
        m = re.match(r"(T\d+n\d+)_p(\d+)([a-c])(\d+)║(.*)", line)
        if m:
            sutra_id = m.group(1)
            page = int(m.group(2))
            col = m.group(3)
            row = int(m.group(4))
            text = m.group(5).strip()

            if current_page is None:
                current_page = page
            if current_col is None:
                current_col = col

            # 提取品名（如 "大方廣佛華嚴經卷第一"、"世間淨眼品第一之一"）
            if "卷" in text and "第" in text and ("方" in text or "佛" in text):
                juan_title = text
                pin = None
            elif "品" in text and "第" in text:
                pin = text

            # 跳过纯标识行（空行、卷次行、译者行等）
            if (not text or
                text.startswith("No.") or
                text == juan_title or
                (pin and text == pin) or
                "三藏" in text and "譯" in text):
                continue

            # 构建记录
            normalized = re.sub(r"[^\u4e00-\u9fff]", "", text)

            records.append({
                "sutra_id": "T0278",
                "canonical_title": juan_title,
                "volume_number": vol,
                "chapter_name": pin,
                "page": page,
                "column": col,
                "line": row,
                "line_head": f"{sutra_id}_p{page:04d}{col}{row:02d}",
                "original_text": text if text else None,
                "normalized_text": normalized if normalized else None,
                "gaiji": [],
                "variant_notes": [],
                "source_file": f"T0278_{vol:03d}.txt",
                "source_version": "CBETA Normal Text 2025 R3",
                "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16] if text else None,
            })

    return records


def compute_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def build_bm25_index(records: list[dict]):
    """构建 BM25 索引（内存）。"""
    from math import log

    # 收集所有文档
    docs = [r["normalized_text"] for r in records if r["normalized_text"]]
    n_docs = len(docs)

    # 词频统计
    df = {}  # document frequency
    doc_terms = []
    for doc in docs:
        terms = list(doc)  # 字符级
        doc_terms.append(terms)
        for t in set(terms):
            df[t] = df.get(t, 0) + 1

    # BM25 参数
    k1, b = 1.5, 0.75
    avgdl = sum(len(t) for t in doc_terms) / max(n_docs, 1)

    # 保存索引
    index = {"n_docs": n_docs, "avgdl": avgdl, "k1": k1, "b": b,
             "df": df, "doc_terms": doc_terms, "records": records}
    return index


def build_ngram_index(records: list[dict], n: int):
    """构建字符 n-gram 索引。"""
    index = {}
    for i, r in enumerate(records):
        text = r.get("normalized_text", "")
        if not text:
            continue
        for j in range(len(text) - n + 1):
            gram = text[j:j + n]
            if gram not in index:
                index[gram] = []
            if not index[gram] or index[gram][-1] != i:
                index[gram].append(i)
    return index


def main():
    print("=" * 70)
    print("阶段 3B：T0278 全量本地语料构建")
    print(f"数据源：{REPO}")
    print(f"Commit：{COMMIT}")
    print(f"卷数：{N_VOLUMES}")
    print("=" * 70)

    # 1. 下载
    print("\n[1] 下载 T0278 60 卷...")
    hashes = {}
    for vol in range(1, N_VOLUMES + 1):
        path = download_volume(vol)
        if path:
            hashes[vol] = compute_sha256(Path(path))
    print(f"  下载完成：{len(hashes)}/{N_VOLUMES} 卷")

    # 2. 解析
    print("\n[2] 解析为 JSONL...")
    all_records = []
    volume_stats = []
    for vol in range(1, N_VOLUMES + 1):
        records = parse_volume(vol)
        if records:
            # 保存 JSONL
            jsonl_path = PARSED_DIR / f"T0278_{vol:03d}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            all_records.extend(records)

        chars = sum(len(r.get("normalized_text", "") or "") for r in records)
        volume_stats.append({
            "volume": vol, "records": len(records),
            "chars": chars,
            "sha256": hashes.get(vol, ""),
        })
        print(f"  卷{vol}: {len(records)} 行, {chars} 字 (normalized)")

    # 3. 保存 source_manifest
    print("\n[3] 保存 source_manifest.json...")
    manifest = {
        "repository": REPO,
        "branch": "main",
        "commit": COMMIT,
        "release": "CBETA Normal Text 2025 R3",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "sutra_id": "T0278",
        "canonical_title": "大方廣佛華嚴經（東晉佛馱跋陀羅譯）",
        "total_volumes": N_VOLUMES,
        "downloaded_volumes": len(hashes),
        "total_records": len(all_records),
        "total_chars_normalized": sum(s["chars"] for s in volume_stats),
        "volume_sha256": {str(v): h for v, h in hashes.items() if h},
        "parser_version": "1.0.0",
        "normalization_version": "1.0.0",
        "note": "Normal Text 用于检索；XML P5 用于证据定位。当前仅下载 Normal Text。",
    }
    (CBETA_DATA / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. 构建检索索引
    print("\n[4] 构建检索索引...")
    # BM25
    bm25_index = build_bm25_index(all_records)
    # 保存 BM25（简化版，存为 JSON）
    bm25_out = {
        "n_docs": bm25_index["n_docs"],
        "avgdl": bm25_index["avgdl"],
        "k1": bm25_index["k1"],
        "b": bm25_index["b"],
        "df": bm25_index["df"],
    }
    (INDEX_DIR / "bm25_index.json").write_text(
        json.dumps(bm25_out, ensure_ascii=False, indent=2))

    # bigram + trigram
    for n in [2, 3]:
        idx = build_ngram_index(all_records, n)
        # 只存键（值太大）
        keys_out = {"n": n, "keys_count": len(idx),
                    "sample_keys": list(idx.keys())[:100]}
        (INDEX_DIR / f"char_{n}gram_index.json").write_text(
            json.dumps(keys_out, ensure_ascii=False, indent=2))

    # corpus metadata
    corpus_meta = {
        "sutra_id": "T0278",
        "total_records": len(all_records),
        "total_chars_normalized": sum(s["chars"] for s in volume_stats),
        "volumes": volume_stats,
    }
    (INDEX_DIR / "corpus_metadata.json").write_text(
        json.dumps(corpus_meta, ensure_ascii=False, indent=2))

    # 5. 保存 volume_statistics
    print("\n[5] 保存统计...")
    import csv
    csv_path = CBETA_DATA / "parsed" / "volume_statistics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["volume", "records", "chars", "sha256"])
        writer.writeheader()
        for s in volume_stats:
            writer.writerow(s)

    print(f"\n=== 完成 ===")
    print(f"下载: {len(hashes)}/{N_VOLUMES} 卷")
    print(f"解析: {len(all_records)} 行")
    print(f"总字数(normalized): {manifest['total_chars_normalized']}")
    print(f"source_manifest: {CBETA_DATA / 'source_manifest.json'}")
    print(f"corpus_metadata: {INDEX_DIR / 'corpus_metadata.json'}")


if __name__ == "__main__":
    main()
