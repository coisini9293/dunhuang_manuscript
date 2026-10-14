#!/usr/bin/env python3
"""从 ScienceDB 解压目录生成 train/val/test 配对文件。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.rejoin_data import build_pairs, collect_groups, write_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/sciencedb")
    parser.add_argument("--output-dir", default="data/raw/sciencedb")
    parser.add_argument("--neg-ratio", type=float, default=1.0, help="负样本数 = 正样本数 × 该比例")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    groups = collect_groups(raw_dir)
    if not groups:
        print(f"错误: {raw_dir} 下未找到有效缀合组")
        return 1

    pairs = build_pairs(groups, raw_dir, neg_ratio=args.neg_ratio)
    counts = write_splits(pairs, Path(args.output_dir))
    pos = sum(1 for p in pairs if p[2] == 1)
    neg = sum(1 for p in pairs if p[2] == 0)
    print(f"组数: {len(groups)} | 正样本: {pos} | 负样本: {neg}")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
