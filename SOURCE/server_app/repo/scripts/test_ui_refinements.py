"""Targeted checks for the pre-AutoDL UI refinements."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.rejoin_eval import stitch_pair, trim_light_margins


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    # A fragment centred in a wide scanner-white canvas must lose that padding.
    fragment = np.full((60, 240, 3), 255, dtype=np.uint8)
    fragment[5:55, 90:150] = (120, 90, 50)
    trimmed = trim_light_margins(fragment, padding=0)
    require(trimmed.shape[:2] == (50, 60), f"unexpected trim: {trimmed.shape}")

    left = np.full((50, 70, 3), 80, dtype=np.uint8)
    panorama = stitch_pair(left, fragment)
    require(panorama.shape[1] < left.shape[1] + fragment.shape[1], "white canvas was not removed")
    require(panorama.shape[1] <= 150, f"unexpected panorama width: {panorama.shape[1]}")

    html = (ROOT / "demo/frontend/index.html").read_text(encoding="utf-8")
    js = (ROOT / "demo/frontend/static/app.js").read_text(encoding="utf-8")
    require('id="maskAdvanced"' in html and '<details class="advanced-tools hidden"' in html,
            "mask editor is not folded into advanced options")
    require('class="result-triptych"' in js, "three-column result view missing")
    for label in ["原图", "缺损检测叠层（Overlay）", "修复结果", "紧凑拼接预览（已去空白边缘）"]:
        require(label in html + js, f"missing label: {label}")
    for obsolete in [">Fragment A<", ">Fragment B<", ">Stitched Preview<", ">Confidence<", " backend</p>"]:
        require(obsolete not in html, f"obsolete mixed-language label remains: {obsolete}")

    print("UI_REFINEMENTS_PASS")


if __name__ == "__main__":
    main()
