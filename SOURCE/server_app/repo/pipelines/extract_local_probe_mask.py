#!/usr/bin/env python3
"""Extract one non-empty local component region for a deterministic GPU probe."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--radius", type=int, default=96)
    args = p.parse_args()
    a = np.asarray(Image.open(args.input).convert("L"))
    ys, xs = np.where(a > 0)
    if not len(xs):
        raise RuntimeError("input mask is empty")
    # Pick the densest candidate neighbourhood, not merely the first pixel.
    best = None
    for index in np.linspace(0, len(xs) - 1, min(200, len(xs)), dtype=int):
        x, y = int(xs[index]), int(ys[index])
        x1, x2 = max(0, x - args.radius), min(a.shape[1], x + args.radius)
        y1, y2 = max(0, y - args.radius), min(a.shape[0], y + args.radius)
        score = int((a[y1:y2, x1:x2] > 0).sum())
        if best is None or score > best[0]:
            best = (score, x, y)
    _, x, y = best
    out = np.zeros_like(a)
    x1, x2 = max(0, x - args.radius), min(a.shape[1], x + args.radius)
    y1, y2 = max(0, y - args.radius), min(a.shape[0], y + args.radius)
    out[y1:y2, x1:x2] = a[y1:y2, x1:x2]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(path)
    print({"nonzero": int((out > 0).sum()), "center": [x, y], "bbox": [x1, y1, x2, y2]})


if __name__ == "__main__":
    main()
