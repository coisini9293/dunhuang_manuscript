"""阶段 2 后处理：对 DiffHDR 输出做 mask 外回贴 + 重新校验。

DiffHDR 输出的整张图都经过扩散处理，mask 外区域也会有变化。
用 blend_repaired 在 mask 外回贴原像素，确保 mask 外像素一致性。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.mask_gen import blend_repaired


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "mvp_05" / "diffhdr_abc"


def main():
    print("=" * 60)
    print("阶段 2 后处理：mask 外回贴 + 重新校验")
    print("=" * 60)

    # 加载 degraded + mask
    degraded = np.array(Image.open(OUTPUT_DIR / "degraded.png").convert("RGB"))
    mask = np.array(Image.open(OUTPUT_DIR / "mask.png").convert("L"))
    print(f"degraded: {degraded.shape}, mask: {mask.shape}")

    # BGR for blend_repaired
    degraded_bgr = degraded[:, :, ::-1].copy()

    pixel_integrity = {}
    for source in ["original", "manual", "rag"]:
        restored_path = OUTPUT_DIR / f"restored_{source}.png"
        if not restored_path.exists():
            print(f"[SKIP] {source}: 文件不存在")
            continue

        restored = np.array(Image.open(restored_path).convert("RGB"))
        restored_bgr = restored[:, :, ::-1].copy()

        # blend: mask 外回贴原像素
        blended_bgr = blend_repaired(degraded_bgr, restored_bgr, mask)
        blended_rgb = blended_bgr[:, :, ::-1].copy()

        # 保存 blended 图（覆盖原 restored）
        Image.fromarray(blended_rgb).save(str(restored_path))
        print(f"[BLEND] {source}: mask 外已回贴原像素")

        # 重新校验
        outside = mask == 0
        diff = np.abs(blended_bgr.astype(int) - degraded_bgr.astype(int))
        diff_outside = diff[outside]
        integrity = {
            "max_diff": int(diff_outside.max()),
            "mean_diff": float(diff_outside.mean()),
            "nonzero_pixels": int(np.count_nonzero(diff_outside)),
            "total_outside_pixels": int(outside.sum()),
            "passed": bool(diff_outside.max() == 0),
        }
        pixel_integrity[source] = integrity
        print(f"  mask 外校验: max_diff={integrity['max_diff']}, "
              f"mean_diff={integrity['mean_diff']:.6f}, "
              f"passed={integrity['passed']}")

    # 保存 pixel_integrity
    (OUTPUT_DIR / "pixel_integrity.json").write_text(
        json.dumps(pixel_integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\npixel_integrity.json 已保存")

    # manual 控制能力验证
    print("\n--- manual 控制能力验证 ---")
    if "original" in pixel_integrity and "manual" in pixel_integrity:
        restored_orig = np.array(Image.open(OUTPUT_DIR / "restored_original.png").convert("RGB"))
        restored_manu = np.array(Image.open(OUTPUT_DIR / "restored_manual.png").convert("RGB"))
        inside = mask > 0
        diff_inside = np.abs(restored_manu.astype(int) - restored_orig.astype(int))
        diff_inside_masked = diff_inside[inside]
        control = {
            "max_diff_inside": int(diff_inside_masked.max()),
            "mean_diff_inside": float(diff_inside_masked.mean()),
            "nonzero_inside": int(np.count_nonzero(diff_inside_masked)),
            "has_observable_effect": bool(diff_inside_masked.max() > 5),
        }
        print(f"  mask 内 max_diff: {control['max_diff_inside']}")
        print(f"  mask 内 mean_diff: {control['mean_diff_inside']:.4f}")
        print(f"  有可观测影响: {control['has_observable_effect']}")

        # 更新 run_metadata
        meta_path = OUTPUT_DIR / "run_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["manual_control"] = control
            meta["pixel_integrity_after_blend"] = pixel_integrity
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  run_metadata.json 已更新")

    # 重新生成 comparison_abc.png
    print("\n--- 重新生成 comparison_abc.png ---")
    target_h = 400
    def resize_h(img, th):
        ih, iw = img.shape[:2]
        return cv2.resize(img, (int(iw * th / ih), th))

    panels = []
    for source in ["original", "manual", "rag"]:
        content_p = OUTPUT_DIR / f"content_{source}.png"
        restored_p = OUTPUT_DIR / f"restored_{source}.png"
        if content_p.exists() and restored_p.exists():
            content_img = np.array(Image.open(content_p).convert("RGB"))
            restored_img = np.array(Image.open(restored_p).convert("RGB"))
            content_r = resize_h(content_img, target_h)
            restored_r = resize_h(restored_img, target_h)
            col = np.hstack([content_r, restored_r])
            panels.append(col)

    if panels:
        comparison = np.hstack(panels)
        Image.fromarray(comparison).save(str(OUTPUT_DIR / "comparison_abc.png"))
        print("  comparison_abc.png 已保存")

    print("\n" + "=" * 60)
    print("后处理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
