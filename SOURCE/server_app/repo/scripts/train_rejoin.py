#!/usr/bin/env python3
"""缀合模型训练。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.rejoin.load import DEFAULT_MODEL_CONFIG, build_rejoin_models
from utils.device import pick_device
from utils.plots import plot_training_curves
from utils.rejoin_data import load_pair_file, read_image_rgb, to_tensor


class RejoinDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.pairs = load_pair_file(self.data_dir, split)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, label, dx, dy, rot = self.pairs[idx]
        img1 = to_tensor(read_image_rgb(self.data_dir / p1), train=self.augment)
        img2 = to_tensor(read_image_rgb(self.data_dir / p2), train=self.augment)
        return (
            img1,
            img2,
            torch.tensor([label], dtype=torch.float32),
            torch.tensor([dx, dy, rot], dtype=torch.float32),
        )


def run_epoch(
    models,
    loader,
    optimizer,
    scheduler,
    device,
    train: bool,
    pose_weight: float,
) -> dict[str, float]:
    backbone, classifier, pose_regressor = models
    cls_criterion = nn.BCELoss()

    if train:
        backbone.train()
        classifier.train()
        pose_regressor.train()
    else:
        backbone.eval()
        classifier.eval()
        pose_regressor.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for img1, img2, label, _pose_gt in loader:
        img1, img2 = img1.to(device), img2.to(device)
        label = label.to(device)

        feat1 = backbone(img1)
        feat2 = backbone(img2)
        pred_label = classifier(feat1, feat2)

        loss = cls_criterion(pred_label, label)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(classifier.parameters()),
                max_norm=2.0,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        pred_bin = (pred_label > 0.5).float()
        correct += (pred_bin == label).sum().item()
        total += label.numel()

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": correct / max(total, 1),
    }


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -1.0
        self.counter = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train(
    data_dir: Path,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 16,
    ckpt_dir: Path | None = None,
    plot_dir: Path | None = None,
    device_name: str = "auto",
    patience: int = 12,
    min_delta: float = 1e-4,
    model_config: dict | None = None,
) -> Path:
    device = pick_device(device_name)
    ckpt_dir = ckpt_dir or Path("checkpoints")
    plot_dir = plot_dir or Path("outputs/plots")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        cfg.update(model_config)

    print(f"Device: {device}")
    print(f"Model: {cfg}")

    backbone, classifier, pose_regressor, cfg = build_rejoin_models(
        cfg, pretrained=True, device=device
    )

    train_set = RejoinDataset(data_dir, "train", augment=True)
    val_set = RejoinDataset(data_dir, "val", augment=False)
    if len(train_set) == 0:
        raise RuntimeError("Training set is empty. Run: python scripts/prepare_pairs.py")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": lr * 0.1, "weight_decay": 1e-4},
            {"params": classifier.parameters(), "lr": lr, "weight_decay": 1e-4},
            {"params": pose_regressor.parameters(), "lr": lr, "weight_decay": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr * 0.1, lr, lr],
        epochs=epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1,
    )

    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_path = ckpt_dir / "rejoin.pth"

    for epoch in range(epochs):
        train_metrics = run_epoch(
            (backbone, classifier, pose_regressor),
            train_loader,
            optimizer,
            scheduler,
            device,
            train=True,
            pose_weight=0.0,
        )
        val_metrics = run_epoch(
            (backbone, classifier, pose_regressor),
            val_loader,
            None,
            None,
            device,
            train=False,
            pose_weight=0.0,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

        improved = early_stop.step(val_metrics["acc"])
        if improved:
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "classifier": classifier.state_dict(),
                    "pose_regressor": pose_regressor.state_dict(),
                    "model_config": cfg,
                    "val_acc": val_metrics["acc"],
                    "epoch": epoch + 1,
                    "device": str(device),
                },
                best_path,
            )
            print(f"  -> Best checkpoint updated (val_acc={val_metrics['acc']:.4f})")

        plot_training_curves(history, plot_dir)
        (plot_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    print(f"Best model saved: {best_path} (val_acc={early_stop.best:.4f})")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/sciencedb")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (epochs)")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34"])
    parser.add_argument("--fusion", default="diff_mul", choices=["concat", "diff_mul"])
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()
    train(
        Path(args.data_dir),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ckpt_dir=Path(args.ckpt_dir),
        plot_dir=Path(args.plot_dir),
        device_name=args.device,
        patience=args.patience,
        min_delta=args.min_delta,
        model_config={"arch": args.arch, "fusion": args.fusion, "hidden": args.hidden},
    )
