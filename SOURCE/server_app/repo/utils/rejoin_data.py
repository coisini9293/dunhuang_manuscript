"""缀合数据与预处理：训练/评测/推理共用。"""
from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ScienceDB 解压后的标准目录名（避免 zip 解压乱码）
SCIENCEDB_DATASET_DIRNAME = "面向残片缀合的敦煌文献图像数据集"

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 训练增强：不做水平翻转（敦煌竖排文字），仅轻微颜色/几何扰动
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=3),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.12, contrast=0.12),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def is_fragment(path: Path) -> bool:
    """排除总图/完整参考图，只保留残片。"""
    name = path.name
    if "总图" in name:
        return False
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return False
    stem = path.stem
    # ponytail: 人工组 Dy054(2).png 是完整图，v1_* 才是残片
    if stem.endswith(")") and "v1" not in stem and path.suffix.lower() == ".png":
        return False
    return True


def find_dataset_root(raw_dir: Path) -> Path:
    """Locate ScienceDB image root (supports correct UTF-8 folder names)."""
    candidates = [
        raw_dir / SCIENCEDB_DATASET_DIRNAME,
        raw_dir / "extracted" / SCIENCEDB_DATASET_DIRNAME,
    ]
    for path in candidates:
        if path.is_dir():
            return path

    extracted = raw_dir / "extracted"
    if extracted.exists():
        for child in sorted(extracted.iterdir()):
            if child.is_dir():
                return child

    raise FileNotFoundError(
        f"未找到数据集目录，请解压到: {raw_dir / SCIENCEDB_DATASET_DIRNAME}"
    )


def collect_groups(raw_dir: Path) -> list[dict]:
    dataset_root = find_dataset_root(raw_dir)
    groups: list[dict] = []
    for category in sorted(dataset_root.iterdir()):
        if not category.is_dir():
            continue
        for group_dir in sorted(category.iterdir()):
            if not group_dir.is_dir():
                continue
            frags = sorted(f for f in group_dir.iterdir() if f.is_file() and is_fragment(f))
            if len(frags) >= 2:
                groups.append({"category": category.name, "name": group_dir.name, "fragments": frags})
    return groups


def build_pairs(groups: list[dict], raw_dir: Path, neg_ratio: float = 1.0) -> list[tuple]:
    """返回 (rel_path1, rel_path2, label, dx, dy, rot)。"""
    pairs: list[tuple] = []
    rel = lambda p: str(p.relative_to(raw_dir))

    for group in groups:
        frags = group["fragments"]
        for a, b in combinations(frags, 2):
            pairs.append((rel(a), rel(b), 1, 0.0, 0.0, 0.0))

    positives = len(pairs)
    neg_target = int(positives * neg_ratio)
    all_groups = list(groups)
    rng = random.Random(42)
    while len([p for p in pairs if p[2] == 0]) < neg_target and len(all_groups) >= 2:
        g1, g2 = rng.sample(all_groups, 2)
        a = rng.choice(g1["fragments"])
        b = rng.choice(g2["fragments"])
        pairs.append((rel(a), rel(b), 0, 0.0, 0.0, 0.0))

    rng.shuffle(pairs)
    return pairs


def write_splits(pairs: list[tuple], output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    splits = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}
    counts: dict[str, int] = {}
    for name, rows in splits.items():
        path = output_dir / f"{name}.txt"
        path.write_text(
            "\n".join(f"{a},{b},{label},{dx},{dy},{rot}" for a, b, label, dx, dy, rot in rows),
            encoding="utf-8",
        )
        counts[name] = len(rows)
    return counts


def load_pair_file(data_dir: Path, split: str) -> list[tuple]:
    split_file = data_dir / f"{split}.txt"
    if not split_file.exists():
        return []
    pairs: list[tuple] = []
    for line in split_file.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        pairs.append((parts[0], parts[1], int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
    return pairs


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(image_rgb: np.ndarray, train: bool = False) -> torch.Tensor:
    transform = TRAIN_TRANSFORM if train else TRANSFORM
    return transform(image_rgb)


def demo() -> None:
    raw = Path("data/raw/sciencedb")
    groups = collect_groups(raw)
    assert len(groups) == 95, f"expected 95 groups, got {len(groups)}"
    pairs = build_pairs(groups, raw)
    assert any(p[2] == 1 for p in pairs)
    assert any(p[2] == 0 for p in pairs)
    print(f"ok: {len(groups)} groups, {len(pairs)} pairs")


if __name__ == "__main__":
    demo()
