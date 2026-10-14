#!/usr/bin/env python3
"""DiffHDR restoration evaluation on bundled examples (required module)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib

matplotlib.use("Agg")
from utils.matplotlib_fonts import configure_matplotlib_font

configure_matplotlib_font()
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from pipelines.restore import DiffHDRRestorer
from utils.device import pick_device
from utils.restore_paths import DIFFHDR_CKPT, DIFFHDR_EXAMPLES, HDR_ROOT, RESTORE_OUTPUT, ensure_diffhdr_ready


def find_examples(examples_dir: Path) -> list[dict]:
    samples = []
    for degraded in sorted(examples_dir.glob("*_damaged_img.png")):
        prefix = degraded.name.replace("_damaged_img.png", "")
        mask = examples_dir / f"{prefix}_mask_img.png"
        content = examples_dir / f"{prefix}_content_img.png"
        gt = examples_dir / f"{prefix}.png"
        if mask.exists() and content.exists() and gt.exists():
            samples.append(
                {
                    "id": prefix,
                    "degraded": degraded,
                    "mask": mask,
                    "content": content,
                    "gt": gt,
                }
            )
    return samples


def plot_restore_metrics(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in results]
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(ids, psnrs, color="#4C78A8")
    axes[0].set_title("Restoration PSNR per Sample")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(ids, ssims, color="#54A24B")
    axes[1].set_title("Restoration SSIM per Sample")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "restore_metrics.png", dpi=150)
    plt.close(fig)


def plot_restore_gallery(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = np.array([axes])
    titles = ["Degraded", "Restored", "Ground Truth", "Mask"]

    for row, r in enumerate(results):
        imgs = [
            cv2.cvtColor(cv2.imread(str(r["degraded"])), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(str(r["restored"])), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(str(r["gt"])), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(str(r["mask"])), cv2.COLOR_BGR2RGB),
        ]
        for col, img in enumerate(imgs):
            axes[row, col].imshow(img, cmap="gray" if col == 3 else None)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=10)
        axes[row, 0].set_ylabel(
            f"{r['id']}\nPSNR={r['psnr']:.1f}\nSSIM={r['ssim']:.3f}",
            fontsize=8,
            rotation=0,
            labelpad=40,
        )

    fig.suptitle("DiffHDR Restoration Results (Test Examples)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "restore_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_before_after_compare(result: dict, out_path: Path) -> Path:
    """Save side-by-side comparison: degraded | restored [| ground truth]."""
    degraded = cv2.cvtColor(cv2.imread(str(result["degraded"])), cv2.COLOR_BGR2RGB)
    restored = cv2.cvtColor(cv2.imread(str(result["restored"])), cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(cv2.imread(str(result["gt"])), cv2.COLOR_BGR2RGB)
    if degraded.shape != restored.shape:
        restored = cv2.resize(restored, (degraded.shape[1], degraded.shape[0]))
    if degraded.shape != gt.shape:
        gt = cv2.resize(gt, (degraded.shape[1], degraded.shape[0]))

    panels = [
        (degraded, "修复前 (退化图)"),
        (restored, f"修复后 (修复结果)\nPSNR={result['psnr']:.1f} dB  SSIM={result['ssim']:.3f}"),
        (gt, "真值 (原图)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"样本 {result['id']} 修复前后对比", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_all_comparisons(results: list[dict], output_dir: Path, plot_dir: Path) -> list[Path]:
    """Save per-sample comparisons under output_dir and plot_dir."""
    paths: list[Path] = []
    for r in results:
        sample_path = output_dir / r["id"] / f"{r['id']}_compare.png"
        plot_path = plot_dir / f"restore_compare_{r['id']}.png"
        save_before_after_compare(r, sample_path)
        save_before_after_compare(r, plot_path)
        paths.extend([sample_path, plot_path])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-dir", default=str(DIFFHDR_EXAMPLES))
    parser.add_argument("--output-dir", default=str(RESTORE_OUTPUT))
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    ensure_diffhdr_ready()
    examples_dir = Path(args.examples_dir)
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = find_examples(examples_dir)
    if not samples:
        print(f"Error: no DiffHDR examples in {examples_dir}")
        return 1

    device = pick_device(args.device)
    restore_device = "mps" if str(device) == "mps" else ("cuda:0" if str(device) == "cuda" else "cpu")
    restorer = DiffHDRRestorer(HDR_ROOT, DIFFHDR_CKPT)

    results = []
    for sample in samples:
        sample_out = output_dir / sample["id"]
        sample_out.mkdir(parents=True, exist_ok=True)
        restored_path = restorer.restore(
            sample["degraded"],
            sample["mask"],
            sample["content"],
            sample_out,
            steps=args.steps,
            content_scale=1.5,
            degraded_scale=1.2,
            device=restore_device,
        )
        gt = cv2.imread(str(sample["gt"]))
        restored = cv2.imread(str(restored_path))
        if gt.shape != restored.shape:
            restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]))
        p = float(psnr(gt, restored))
        s = float(ssim(gt, restored, channel_axis=2))
        results.append({**sample, "restored": restored_path, "psnr": p, "ssim": s})
        print(f"[{sample['id']}] PSNR={p:.2f} SSIM={s:.4f}")

    summary = {
        "mean_psnr": float(np.mean([r["psnr"] for r in results])),
        "mean_ssim": float(np.mean([r["ssim"] for r in results])),
        "samples": [
            {
                "id": r["id"],
                "psnr": r["psnr"],
                "ssim": r["ssim"],
                "restored": str(r["restored"]),
                "compare": str(output_dir / r["id"] / f"{r['id']}_compare.png"),
            }
            for r in results
        ],
    }
    (plot_dir / "restore_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_restore_metrics(results, plot_dir)
    plot_restore_gallery(results, plot_dir)
    compare_paths = save_all_comparisons(results, output_dir, plot_dir)

    print(f"Mean PSNR={summary['mean_psnr']:.2f} SSIM={summary['mean_ssim']:.4f}")
    print(f"Saved: {plot_dir / 'restore_metrics.png'}")
    print(f"Saved: {plot_dir / 'restore_gallery.png'}")
    for p in compare_paths[::2]:
        print(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
