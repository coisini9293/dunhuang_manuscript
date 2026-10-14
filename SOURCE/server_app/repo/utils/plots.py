"""Training / evaluation plots (English labels only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from utils.matplotlib_fonts import configure_matplotlib_font

configure_matplotlib_font()
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_acc_curves.png", dpi=150)
    plt.close(fig)


def plot_eval_metrics(metrics: dict, split: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    f1 = 2 * metrics["precision"] * metrics["recall"] / max(metrics["precision"] + metrics["recall"], 1e-8)

    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [metrics["acc"], metrics["precision"], metrics["recall"], f1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Rejoin Evaluation Metrics ({split.capitalize()} Set)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"eval_metrics_{split}.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, split: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred: No", "Pred: Yes"])
    ax.set_yticks([0, 1], labels=["GT: No", "GT: Yes"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix ({split.capitalize()} Set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{split}.png", dpi=150)
    plt.close(fig)


def plot_rejoin_effect_metrics(metrics: dict, split: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "Positive Pair Acc",
        "Negative Pair Acc",
        "Mean Edge Alignment",
        "Mean Reference SSIM",
    ]
    values = [
        metrics["positive_pair_accuracy"],
        metrics["negative_pair_accuracy"],
        metrics["mean_edge_alignment"],
        metrics["mean_reference_ssim"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Rejoin Effect Metrics ({split.capitalize()} Set)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"rejoin_effect_{split}.png", dpi=150)
    plt.close(fig)


def plot_test_gallery(records: list[dict], out_dir: Path, split: str, max_rows: int = 6) -> None:
    """Each row: Fragment A | Fragment B | Stitched Preview | label text."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    # ponytail: pick diverse cases — TP, TN, FP, FN first
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}
    for r in records:
        key = ("T" if r["pred"] == r["label"] else "F") + ("P" if r["pred"] == 1 else "N")
        buckets[key].append(r)
    selected: list[dict] = []
    for key in ("TP", "FP", "FN", "TN"):
        selected.extend(buckets[key][: max(1, max_rows // 4)])
    selected = selected[:max_rows]

    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    for row, rec in enumerate(selected):
        img1 = cv2.cvtColor(cv2.imread(str(rec["path1"])), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(str(rec["path2"])), cv2.COLOR_BGR2RGB)
        stitch = rec["stitch"]

        for col, img in enumerate((img1, img2, stitch)):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                titles = ["Fragment A", "Fragment B", "Stitched Preview"]
                axes[row, col].set_title(titles[col], fontsize=10)

        status = "TP" if rec["pred"] == rec["label"] == 1 else (
            "TN" if rec["pred"] == rec["label"] == 0 else ("FP" if rec["pred"] == 1 else "FN")
        )
        axes[row, 0].set_ylabel(
            f"{status}\nGT={rec['label']} Pred={rec['pred']}\nConf={rec['confidence']:.2f}",
            fontsize=8,
            rotation=0,
            labelpad=36,
        )

    fig.suptitle(f"Test Set Rejoin Results ({split.capitalize()} Set)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"test_gallery_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
