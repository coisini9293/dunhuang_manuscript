#!/usr/bin/env python3
"""Pre-flight + smoke test: data, rejoin train, rejoin eval, restore eval."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from utils.rejoin_data import demo as data_demo
    from utils.restore_paths import ensure_diffhdr_ready

    ensure_diffhdr_ready()
    print("DiffHDR weights: OK")

    data_demo()
    run([sys.executable, "scripts/prepare_pairs.py"])
    run([
        sys.executable,
        "scripts/train_rejoin.py",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--device",
        "mps",
    ])
    ckpt = ROOT / "checkpoints" / "rejoin.pth"
    if not ckpt.exists():
        print("Failed: checkpoints/rejoin.pth not created")
        return 1
    run([sys.executable, "scripts/eval_rejoin.py", "--split", "val", "--device", "mps"])
    run([sys.executable, "scripts/eval_restore.py", "--device", "mps", "--steps", "10"])
    print("smoke test passed (rejoin + restore)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
