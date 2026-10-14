"""阶段 2：三路径真实 DiffHDR 对照实验。

使用同一张 degraded image、同一个 mask、相同随机种子和完全相同的 DiffHDR 参数，
只改变 content_image 来源：original / manual / rag。

每组必须真正调用 DiffHDR inference，生成 restored image。

输出：
  outputs/mvp_05/diffhdr_abc/
    degraded.png
    mask.png
    content_original.png
    content_manual.png
    content_rag.png
    restored_original.png
    restored_manual.png
    restored_rag.png
    comparison_abc.png
    run_metadata.json
    pixel_integrity.json
    inference.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rag_agent import RagAgent
from utils.cbeta_index import Cbetaindex
from utils.mask_gen import extract_content_image
from utils.text_renderer import render_content_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUNHUANG_IIIF_DIR = PROJECT_ROOT.parent / "data" / "data" / "raw" / "dunhuang_iiif"
HDR_ROOT = PROJECT_ROOT / "third_party" / "HDR"
DIFFHDR_CKPT = PROJECT_ROOT / "checkpoints" / "diffhdr" / "unet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mvp_05" / "diffhdr_abc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# managed venv python（有 torch + diffusers）
PYTHON_EXE = r"C:\Users\95788\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

# DiffHDR 参数（三路径完全相同）
SEED = 42
STEPS = 20
CONTENT_SCALE = 7.5
DEGRADED_SCALE = 7.5
DEVICE = "cpu"  # CPU 模式


def log(msg: str):
    print(msg)
    with open(OUTPUT_DIR / "inference.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_image() -> np.ndarray:
    case_dir = DUNHUANG_IIIF_DIR / "05_0100bd6eba9c42f5b62dbcebffc928c0"
    img_path = case_dir / "preview.jpg"
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        rgb = np.array(Image.open(img_path).convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
    # DiffHDR 要求 512x512，裁剪一块 512x512 区域
    h, w = bgr.shape[:2]
    if h > 512 and w > 512:
        # 从中间裁剪 512x512
        y0 = (h - 512) // 2
        x0 = (w - 512) // 2
        bgr = bgr[y0:y0+512, x0:x0+512]
    elif bgr.shape[0] != 512 or bgr.shape[1] != 512:
        bgr = cv2.resize(bgr, (512, 512))
    return bgr


def make_mask(h: int, w: int) -> np.ndarray:
    """生成模拟缺损 mask。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    # 一处缺损区域
    mask[100:300, 200:260] = 255
    return mask


def make_content_original(degraded_bgr: np.ndarray) -> np.ndarray:
    """original: OTSU 二值化（灰度）。"""
    return extract_content_image(degraded_bgr)


def make_content_manual(degraded_bgr: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    """manual: 人工指定文字渲染（转灰度）。"""
    ys, xs = np.where(mask > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    decisions = [{"char": text, "bbox": bbox, "color": "blue", "source": "manual"}]
    content_bgr = render_content_image(degraded_bgr, mask, decisions)
    # 转灰度
    content_gray = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2GRAY)
    return content_gray


def make_content_rag(degraded_bgr: np.ndarray, mask: np.ndarray, decisions: list[dict]) -> np.ndarray:
    """rag: RAG 决策渲染（转灰度）。"""
    content_bgr = render_content_image(degraded_bgr, mask, decisions)
    content_gray = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2GRAY)
    return content_gray


def run_diffhdr(
    degraded_path: str,
    mask_path: str,
    content_path: str,
    save_dir: str,
    seed: int = SEED,
) -> dict:
    """调用 DiffHDR inference.py。"""
    # Windows 路径转成 / 分隔（inference.py 用 split('/')[-1] 取文件名）
    def to_posix(p):
        return str(p).replace("\\", "/")

    cmd = [
        PYTHON_EXE,
        to_posix(HDR_ROOT / "inference.py"),
        "--image_path", to_posix(degraded_path),
        "--mask_image_path", to_posix(mask_path),
        "--content_image_path", to_posix(content_path),
        "--save_dir", to_posix(save_dir),
        "--ckpt_path", to_posix(DIFFHDR_CKPT),
        "--seed", str(seed),
        "--num_inference_steps", str(STEPS),
        "--content_mask_guidance_scale", str(CONTENT_SCALE),
        "--degraded_guidance_scale", str(DEGRADED_SCALE),
        "--device", DEVICE,
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(HDR_ROOT),
        capture_output=True, text=True,
        timeout=600,
    )
    latency = time.time() - t0
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "latency_sec": round(latency, 1),
    }


def check_pixel_integrity(restored: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> dict:
    """mask 外像素校验。"""
    outside = mask == 0
    diff = np.abs(restored.astype(int) - degraded.astype(int))
    diff_outside = diff[outside]
    return {
        "max_diff": int(diff_outside.max()),
        "mean_diff": float(diff_outside.mean()),
        "nonzero_pixels": int(np.count_nonzero(diff_outside)),
        "total_outside_pixels": int(outside.sum()),
        "passed": diff_outside.max() == 0,  # 完全回贴才算通过
    }


def run_abc_test():
    """三路径对照实验。"""
    # 清空日志
    (OUTPUT_DIR / "inference.log").write_text("", encoding="utf-8")

    log("=" * 70)
    log("阶段 2：三路径真实 DiffHDR 对照实验")
    log(f"时间: {datetime.now(timezone.utc).isoformat()}")
    log(f"Seed: {SEED}, Steps: {STEPS}, Device: {DEVICE}")
    log("=" * 70)

    # 1. 加载图
    log("\n[1/5] 加载 degraded image...")
    degraded_bgr = load_image()
    h, w = degraded_bgr.shape[:2]
    log(f"  尺寸: {w}x{h}")

    # 2. 生成 mask
    log("\n[2/5] 生成 mask...")
    mask = make_mask(h, w)
    log(f"  mask 区域: {np.count_nonzero(mask)} pixels")

    # 3. RAG 决策（用于 rag 路径）
    log("\n[3/5] RAG 决策...")
    agent = RagAgent()
    before = "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净"
    after = "以为庄严上妙宝轮圆满清净"
    decision = agent.decide(before, after, sutra_hint="大方广佛华严经")
    decisions = [{
        **decision.to_dict(),
        "bbox": [200, 100, 60, 200],
        "test_type": "synthetic_masked_text_unit_test",
        "real_case_validation": False,
    }]
    log(f"  L1 命中: {decision.char}, status={decision.decision_status}")

    # 4. 生成三种 content_image
    log("\n[4/5] 生成三种 content_image...")
    content_original = make_content_original(degraded_bgr)
    content_manual = make_content_manual(degraded_bgr, mask, "众宝杂华")
    content_rag = make_content_rag(degraded_bgr, mask, decisions)

    # 保存（用 PIL 避免 cv2 中文路径问题）
    Image.fromarray(degraded_bgr[:, :, ::-1]).save(str(OUTPUT_DIR / "degraded.png"))
    Image.fromarray(mask).save(str(OUTPUT_DIR / "mask.png"))
    Image.fromarray(content_original).save(str(OUTPUT_DIR / "content_original.png"))
    Image.fromarray(content_manual).save(str(OUTPUT_DIR / "content_manual.png"))
    Image.fromarray(content_rag).save(str(OUTPUT_DIR / "content_rag.png"))
    log("  degraded.png, mask.png, content_original/manual/rag.png 已保存")

    # 5. 三路径 DiffHDR 推理
    log("\n[5/5] 三路径 DiffHDR 推理...")
    results = {}
    for source, content_img in [
        ("original", content_original),
        ("manual", content_manual),
        ("rag", content_rag),
    ]:
        log(f"\n--- 路径 {source.upper()} ---")
        work_dir = OUTPUT_DIR / f"work_{source}"
        work_dir.mkdir(exist_ok=True)

        # 保存输入
        deg_path = work_dir / "degraded.png"
        mask_path = work_dir / "mask.png"
        content_path = work_dir / "content.png"
        Image.fromarray(degraded_bgr[:, :, ::-1]).save(str(deg_path))
        Image.fromarray(mask).save(str(mask_path))
        Image.fromarray(content_img).save(str(content_path))

        log(f"  调用 DiffHDR inference (seed={SEED}, steps={STEPS})...")
        run_result = run_diffhdr(
            str(deg_path), str(mask_path), str(content_path),
            str(work_dir), seed=SEED,
        )
        log(f"  returncode: {run_result['returncode']}")
        log(f"  latency: {run_result['latency_sec']}s")

        if run_result["returncode"] != 0:
            log(f"  [FAIL] stderr: {run_result['stderr'][:500]}")
            results[source] = {"success": False, "error": run_result["stderr"][:500]}
            continue

        # 找输出文件
        repaired_path = work_dir / "degraded_repaired_img.png"
        if not repaired_path.exists():
            log(f"  [FAIL] 输出文件不存在: {repaired_path}")
            results[source] = {"success": False, "error": "output not found"}
            continue

        # 加载 restored
        restored = np.array(Image.open(repaired_path).convert("RGB"))
        restored_bgr = restored[:, :, ::-1].copy()

        # 保存到 diffhdr_abc 目录
        Image.fromarray(restored).save(str(OUTPUT_DIR / f"restored_{source}.png"))
        log(f"  restored_{source}.png 已保存")

        # mask 外像素校验
        integrity = check_pixel_integrity(restored_bgr, degraded_bgr, mask)
        log(f"  mask 外像素校验: max_diff={integrity['max_diff']}, "
            f"mean_diff={integrity['mean_diff']:.4f}, "
            f"passed={integrity['passed']}")

        results[source] = {
            "success": True,
            "latency_sec": run_result["latency_sec"],
            "integrity": integrity,
        }

    # 6. manual 控制能力验证
    log("\n--- manual 控制能力验证 ---")
    if results.get("original", {}).get("success") and results.get("manual", {}).get("success"):
        restored_orig = np.array(Image.open(OUTPUT_DIR / "restored_original.png").convert("RGB"))
        restored_manu = np.array(Image.open(OUTPUT_DIR / "restored_manual.png").convert("RGB"))
        inside = mask > 0
        diff_inside = np.abs(restored_manu.astype(int) - restored_orig.astype(int))
        diff_inside_masked = diff_inside[inside]
        control_strength = {
            "max_diff_inside": int(diff_inside_masked.max()),
            "mean_diff_inside": float(diff_inside_masked.mean()),
            "nonzero_inside": int(np.count_nonzero(diff_inside_masked)),
            "has_observable_effect": diff_inside_masked.max() > 5,  # 至少 5 像素差异才算有影响
        }
        log(f"  mask 内 max_diff: {control_strength['max_diff_inside']}")
        log(f"  mask 内 mean_diff: {control_strength['mean_diff_inside']:.4f}")
        log(f"  有可观测影响: {control_strength['has_observable_effect']}")
    else:
        control_strength = {"error": "original 或 manual 路径失败"}

    # 7. 生成对比图
    log("\n--- 生成对比图 ---")
    panels = []
    for source in ["original", "manual", "rag"]:
        content_p = OUTPUT_DIR / f"content_{source}.png"
        restored_p = OUTPUT_DIR / f"restored_{source}.png"
        if content_p.exists() and restored_p.exists():
            content_img = np.array(Image.open(content_p).convert("RGB"))
            restored_img = np.array(Image.open(restored_p).convert("RGB"))
            panels.append((source, content_img, restored_img))

    if panels:
        target_h = 400
        def resize_h(img, th):
            ih, iw = img.shape[:2]
            return cv2.resize(img, (int(iw * th / ih), th))

        cols = []
        for source, content, restored in panels:
            content_r = resize_h(content, target_h)
            restored_r = resize_h(restored, target_h)
            col = np.hstack([content_r, restored_r])
            cols.append(col)
        comparison = np.hstack(cols)
        Image.fromarray(comparison).save(str(OUTPUT_DIR / "comparison_abc.png"))
        log("  comparison_abc.png 已保存")

    # 8. 保存 run_metadata + pixel_integrity
    run_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "steps": STEPS,
        "content_scale": CONTENT_SCALE,
        "degraded_scale": DEGRADED_SCALE,
        "device": DEVICE,
        "image_size": [w, h],
        "results": results,
        "manual_control": control_strength if "control_strength" in dir() else None,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    pixel_integrity = {
        source: results[source].get("integrity", {})
        for source in ["original", "manual", "rag"]
        if results.get(source, {}).get("success")
    }
    (OUTPUT_DIR / "pixel_integrity.json").write_text(
        json.dumps(pixel_integrity, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n" + "=" * 70)
    log("阶段 2 完成")
    log(f"产物目录: {OUTPUT_DIR}")
    for source in ["original", "manual", "rag"]:
        r = results.get(source, {})
        status = "PASS" if r.get("success") else "FAIL"
        log(f"  {source}: {status}")
    log("=" * 70)

    return run_metadata


if __name__ == "__main__":
    run_abc_test()
