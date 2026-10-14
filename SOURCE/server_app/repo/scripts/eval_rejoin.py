#!/usr/bin/env python3
"""Rejoin evaluation: classification + rejoin effect + test gallery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.rejoin.load import load_rejoin_checkpoint
from scripts.train_rejoin import RejoinDataset
from utils.device import pick_device
from utils.plots import (
    plot_confusion_matrix,
    plot_eval_metrics,
    plot_rejoin_effect_metrics,
    plot_test_gallery,
)
from utils.rejoin_data import read_image_rgb
from utils.rejoin_eval import (
    edge_alignment_score,
    find_reference_image,
    reference_ssim,
    stitch_pair,
    summarize_rejoin_effect,
)


@torch.no_grad()
def evaluate_split(models, loader, device, data_dir: Path) -> tuple[dict, list[dict], dict]:
    backbone, classifier, pose_regressor = models
    backbone.eval()
    classifier.eval()
    pose_regressor.eval()

    tp = fp = fn = tn = correct = total = 0
    total_loss = 0.0
    cls_loss_fn = torch.nn.BCELoss()
    pose_loss_fn = torch.nn.SmoothL1Loss()
    records: list[dict] = []

    for batch_idx, (img1, img2, label, pose_gt) in enumerate(loader):
        img1, img2 = img1.to(device), img2.to(device)
        label, pose_gt = label.to(device), pose_gt.to(device)

        feat1, feat2 = backbone(img1), backbone(img2)
        pred_prob = classifier(feat1, feat2)
        pred_pose = pose_regressor(feat1, feat2)

        cls_loss = cls_loss_fn(pred_prob, label)
        mask = label.squeeze(-1) > 0.5
        pose_loss = (
            pose_loss_fn(pred_pose[mask], pose_gt[mask])
            if mask.any()
            else torch.zeros((), device=device)
        )
        total_loss += (cls_loss + pose_loss).item()

        pred_bin = (pred_prob > 0.5).float()
        correct += (pred_bin == label).sum().item()
        total += label.numel()

        for i in range(label.size(0)):
            y = int(label[i].item())
            p = int(pred_bin[i].item())
            conf = float(pred_prob[i].item())
            if y == 1 and p == 1:
                tp += 1
            elif y == 0 and p == 1:
                fp += 1
            elif y == 1 and p == 0:
                fn += 1
            else:
                tn += 1

            global_idx = batch_idx * loader.batch_size + i
            if global_idx >= len(loader.dataset.pairs):
                continue
            p1_rel, p2_rel, _, _, _, _ = loader.dataset.pairs[global_idx]
            path1 = data_dir / p1_rel
            path2 = data_dir / p2_rel
            rgb1 = read_image_rgb(path1)
            rgb2 = read_image_rgb(path2)
            stitch = stitch_pair(rgb1, rgb2)

            edge = edge_alignment_score(rgb1, rgb2) if y == 1 else None
            ref_path = find_reference_image(data_dir, Path(p1_rel))
            ref_s = None
            if y == 1 and p == 1 and ref_path is not None:
                ref_s = reference_ssim(stitch, ref_path)

            records.append(
                {
                    "path1": path1,
                    "path2": path2,
                    "label": y,
                    "pred": p,
                    "confidence": conf,
                    "edge_score": edge,
                    "ref_ssim": ref_s,
                    "stitch": stitch,
                }
            )

    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    cls_metrics = {
        "loss": total_loss / max(len(loader), 1),
        "acc": correct / max(total, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": f1,
    }
    counts = {"tp": tp, "fp": "fn" if False else fn, "tn": tn, "fn": fn}
    # fix counts dict typo
    counts = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    rejoin_metrics = summarize_rejoin_effect(records)
    all_metrics = {**cls_metrics, **rejoin_metrics, "confusion": counts}
    return all_metrics, records, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/sciencedb")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--ckpt", default="checkpoints/rejoin.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--gallery-rows", type=int, default=6)
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt)
    plot_dir = Path(args.plot_dir)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return 1

    (backbone, classifier, pose_regressor), model_cfg = load_rejoin_checkpoint(ckpt_path, device)
    print(f"Loaded model: {model_cfg}")

    dataset = RejoinDataset(data_dir, args.split)
    if len(dataset) == 0:
        print(f"Error: {args.split} set is empty")
        return 1

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics, records, counts = evaluate_split(
        (backbone, classifier, pose_regressor), loader, device, data_dir
    )

    print(f"[{args.split}] classification:")
    print(
        f"  loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    print(f"[{args.split}] rejoin effect:")
    print(f"  positive_pair_acc={metrics['positive_pair_accuracy']:.4f}")
    print(f"  negative_pair_acc={metrics['negative_pair_accuracy']:.4f}")
    print(f"  false_rejoin_rate={metrics['false_rejoin_rate']:.4f}")
    print(f"  mean_edge_alignment={metrics['mean_edge_alignment']:.4f}")
    print(f"  mean_reference_ssim={metrics['mean_reference_ssim']:.4f} "
          f"(n={metrics['reference_pairs_evaluated']})")

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_eval_metrics(metrics, args.split, plot_dir)
    plot_rejoin_effect_metrics(metrics, args.split, plot_dir)
    plot_confusion_matrix(counts["tp"], counts["fp"], counts["tn"], counts["fn"], args.split, plot_dir)
    plot_test_gallery(records, plot_dir, args.split, max_rows=args.gallery_rows)

    out_json = plot_dir / f"eval_full_{args.split}.json"
    export = {k: v for k, v in metrics.items() if k != "confusion"}
    export["confusion"] = counts
    export["samples"] = [
        {
            "path1": str(r["path1"]),
            "path2": str(r["path2"]),
            "label": r["label"],
            "pred": r["pred"],
            "confidence": r["confidence"],
            "edge_score": r["edge_score"],
            "ref_ssim": r["ref_ssim"],
        }
        for r in records
    ]
    out_json.write_text(json.dumps(export, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {plot_dir / f'eval_metrics_{args.split}.png'}")
    print(f"  {plot_dir / f'rejoin_effect_{args.split}.png'}")
    print(f"  {plot_dir / f'confusion_matrix_{args.split}.png'}")
    print(f"  {plot_dir / f'test_gallery_{args.split}.png'}")
    print(f"  {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
