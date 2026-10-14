"""阶段 3A：content guidance 方向性验证。

从 T0278 卷一取 GT 文字渲染为 512x512 图像，生成缺损 mask，
5 组 content_image（original/manual_correct/manual_wrong/blank/rag_correct），
固定所有 DiffHDR 参数，分别推理，计算 mask 内 PSNR/SSIM/像素误差/IoU。

验证目标：manual_correct 是否稳定优于 manual_wrong 和 blank_content。

至少 3 个不同字符区域，不得只测一个样本。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rag_agent import RagAgent
from utils.mask_gen import extract_content_image, blend_repaired
from utils.text_renderer import find_cjk_font, PAPER_COLOR, INK_COLOR


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HDR_ROOT = PROJECT_ROOT / "third_party" / "HDR"
DIFFHDR_CKPT = PROJECT_ROOT / "checkpoints" / "diffhdr" / "unet"
CBETA_DIR = PROJECT_ROOT / "data" / "cbeta"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "content_guidance_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = r"C:\Users\95788\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

# 固定参数
SEED = 42
STEPS = 5  # 加快验证，5 步即可
CONTENT_SCALE = 7.5
DEGRADED_SCALE = 7.5
DEVICE = "cpu"
IMG_SIZE = 512
FONT_SIZE = 48


def log(msg: str):
    print(msg)
    with open(OUTPUT_DIR / "inference.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_cbeta_vol1_text() -> str:
    """加载 T0278 卷一原文（从 Normal Text JSONL）。"""
    # 优先从 parsed JSONL 加载（更规范）
    jsonl_path = CBETA_DIR / "parsed" / "T0278_001.jsonl"
    if jsonl_path.exists():
        records = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        text = "".join(r.get("normalized_text", "") for r in records if r.get("normalized_text"))
        if len(text) > 100:
            return text

    # fallback: 从 raw Normal Text 加载
    txt_path = CBETA_DIR / "raw" / "normal_text" / "T0278_001.txt"
    if txt_path.exists():
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        text = ""
        for line in lines:
            m = re.match(r"T\d+n\d+_p\d+[a-c]\d+║(.*)", line)
            if m:
                t = m.group(1).strip()
                if t and not t.startswith("No.") and "三藏" not in t and "譯" not in t:
                    text += re.sub(r"[^\u4e00-\u9fff]", "", t)
        if len(text) > 100:
            return text

    # 最后 fallback：旧的 T0278_001.txt
    f = CBETA_DIR / "T0278_001.txt"
    if f.exists():
        raw = f.read_text(encoding="utf-8")
        lines = raw.strip().split("\n")
        content = lines[2] if len(lines) >= 3 else lines[0]
        return re.sub(r"[^\u4e00-\u9fff]", "", content)

    raise FileNotFoundError("CBETA 卷一文本不可用")


def pick_char_regions(text: str, n: int = 3) -> list[dict]:
    """从文本中选 n 个不重叠的字符区域，每个 4 个字。"""
    regions = []
    # 从文本开头、中间、末尾各取 4 个字
    indices = [
        10,  # 开头附近
        len(text) // 2,  # 中间
        len(text) - 30,  # 末尾附近
    ]
    for i in range(min(n, len(indices))):
        idx = min(indices[i], len(text) - 4)
        chars = text[idx:idx + 4]
        regions.append({
            "id": i + 1,
            "chars": chars,
            "wrong_chars": "ABCD"[:4] if len(chars) == 4 else chars[::-1],  # 错误文字
            "text_idx": idx,
        })
    return regions


def render_gt_image(regions: list[dict], font_path: str) -> np.ndarray:
    """用 PIL 渲染 GT 文字到 512x512 白色背景，竖排。"""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), tuple(PAPER_COLOR))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, FONT_SIZE)

    # 把所有文字排成竖排，放在图像中间
    all_chars = ""
    for r in regions:
        all_chars += r["chars"]

    # 竖排渲染
    n_chars = len(all_chars)
    start_x = IMG_SIZE // 2 - FONT_SIZE // 2
    total_h = n_chars * FONT_SIZE
    start_y = (IMG_SIZE - total_h) // 2

    char_bboxes = {}  # {char: (x, y, w, h)}
    char_idx = 0
    region_char_starts = {}
    for r_idx, r in enumerate(regions):
        region_char_starts[r_idx] = char_idx
        char_idx += len(r["chars"])

    char_idx = 0
    for r_idx, r in enumerate(regions):
        for j, c in enumerate(r["chars"]):
            cy = start_y + (char_idx) * FONT_SIZE
            cx = start_x
            draw.text((cx, cy), c, fill=INK_COLOR, font=font)
            char_bboxes[c] = (cx, cy, FONT_SIZE, FONT_SIZE)
            char_idx += 1

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, char_bboxes, all_chars


def make_degraded_and_mask(gt_bgr: np.ndarray, regions: list[dict], char_bboxes: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """从 GT 图中遮挡字符区域，生成 degraded + mask。"""
    degraded = gt_bgr.copy()
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    damage_infos = []
    for r in regions:
        x0, y0 = IMG_SIZE, IMG_SIZE
        x1, y1 = 0, 0
        for c in r["chars"]:
            cx, cy, cw, ch = char_bboxes.get(c, (0, 0, FONT_SIZE, FONT_SIZE))
            x0 = min(x0, cx)
            y0 = min(y0, cy)
            x1 = max(x1, cx + cw)
            y1 = max(y1, cy + ch)

        # 遮挡（用纸色填充）
        degraded[y0:y1, x0:x1] = PAPER_COLOR[::-1]  # BGR
        mask[y0:y1, x0:x1] = 255

        damage_infos.append({
            "region_id": r["id"],
            "chars": r["chars"],
            "wrong_chars": r["wrong_chars"],
            "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        })

    return degraded, mask, damage_infos


def build_content_image(source: str, degraded_bgr: np.ndarray, mask: np.ndarray, damage_infos: list[dict], agent=None) -> np.ndarray:
    """根据 source 构建 content_image。"""
    h, w = degraded_bgr.shape[:2]

    if source == "original":
        content_gray = extract_content_image(degraded_bgr)
        return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR)

    if source == "blank":
        return np.full((h, w, 3), PAPER_COLOR, dtype=np.uint8)

    from utils.text_renderer import render_content_image
    decisions = []

    if source == "manual_correct":
        for d in damage_infos:
            decisions.append({
                "char": d["chars"], "bbox": d["bbox"],
                "color": "blue", "source": "manual",
            })
    elif source == "manual_wrong":
        for d in damage_infos:
            decisions.append({
                "char": d["wrong_chars"], "bbox": d["bbox"],
                "color": "blue", "source": "manual",
            })
    elif source == "rag_correct":
        for d in damage_infos:
            # 用 CBETA 检索上下文
            # 模拟上下文：取字符周围文字作为 before/after
            # 实际使用 RagAgent
            if agent:
                before, after = _extract_context(d, damage_infos)
                decision = agent.decide(before, after, sutra_hint="大方广佛华严经")
                decisions.append({
                    "char": decision.char if decision.char else "?",
                    "bbox": d["bbox"],
                    "color": "blue" if decision.decision_status == "retrieved_exact" else "yellow",
                    "source": "rag",
                })
            else:
                decisions.append({
                    "char": "????", "bbox": d["bbox"],
                    "color": "gray", "source": "rag_failed",
                })

    return render_content_image(degraded_bgr, mask, decisions)


def _extract_context(damage: dict, all_damages: list[dict]) -> tuple[str, str]:
    """提取字符周围的上下文（模拟版）。"""
    # 用 T0278 卷一的已知文字作为上下文
    # 简单地返回一个短上下文（后续可以从实际卷文字中提取）
    return (
        "如是我闻一时佛在摩竭提国寂灭道场始成正觉其地金刚具足严净",
        "以为庄严上妙宝轮圆满清净",
    )


def run_diffhdr(deg_path: str, mask_path: str, content_path: str, save_dir: str) -> dict:
    def to_posix(p):
        return str(p).replace("\\", "/")
    cmd = [
        PYTHON_EXE, to_posix(HDR_ROOT / "inference.py"),
        "--image_path", to_posix(deg_path),
        "--mask_image_path", to_posix(mask_path),
        "--content_image_path", to_posix(content_path),
        "--save_dir", to_posix(save_dir),
        "--ckpt_path", to_posix(DIFFHDR_CKPT),
        "--seed", str(SEED), "--num_inference_steps", str(STEPS),
        "--content_mask_guidance_scale", str(CONTENT_SCALE),
        "--degraded_guidance_scale", str(DEGRADED_SCALE),
        "--device", DEVICE,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(HDR_ROOT), capture_output=True, text=True, timeout=600)
    return {
        "returncode": result.returncode,
        "latency_sec": round(time.time() - t0, 1),
        "stderr": result.stderr[:500] if result.stderr else "",
    }


def compute_metrics(restored_bgr: np.ndarray, gt_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """计算 mask 内 PSNR/SSIM/像素误差/IoU。"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    inside = mask > 0
    if inside.sum() == 0:
        return {}

    restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)

    # PSNR/SSIM on mask area only (crop bbox)
    ys, xs = np.where(inside)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    r_crop = restored_gray[y0:y1, x0:x1]
    g_crop = gt_gray[y0:y1, x0:x1]

    # PSNR
    try:
        psnr_val = psnr(g_crop, r_crop, data_range=255)
    except Exception:
        psnr_val = 0

    # SSIM
    try:
        win = min(r_crop.shape[0], r_crop.shape[1], 7)
        ssim_val = ssim(g_crop, r_crop, data_range=255, win_size=min(win, 7))
    except Exception:
        ssim_val = 0

    # 绝对像素误差
    abs_diff = np.abs(restored_gray.astype(float) - gt_gray.astype(float))
    abs_error_inside = abs_diff[inside]

    # 二值笔画 IoU（OTSU 阈值）
    _, r_bin = cv2.threshold(r_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, g_bin = cv2.threshold(g_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    intersection = (r_bin & g_bin).sum()
    union = (r_bin | g_bin).sum()
    iou = intersection / union if union > 0 else 0

    return {
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "mse": round(float((abs_error_inside ** 2).mean()), 2),
        "mae": round(float(abs_error_inside.mean()), 2),
        "stroke_iou": round(iou, 4),
    }


def main():
    (OUTPUT_DIR / "inference.log").write_text("", encoding="utf-8")
    log("=" * 60)
    log("阶段 3A：content guidance 方向性验证")
    log(f"Seed={SEED}, Steps={STEPS}, Device={DEVICE}")
    log("=" * 60)

    # 0. 准备
    log("\n[0] 准备 GT 数据...")
    text = load_cbeta_vol1_text()
    log(f"  CBETA T0278 卷一: {len(text)} 字")
    regions = pick_char_regions(text, n=3)
    for r in regions:
        log(f"  region {r['id']}: \"{r['chars']}\" at idx {r['text_idx']}")

    font_path = find_cjk_font()
    if font_path is None:
        log("[FAIL] 未找到中文字体")
        return
    log(f"  font: {font_path}")

    gt_bgr, char_bboxes, _ = render_gt_image(regions, font_path)
    degraded_bgr, mask, damage_infos = make_degraded_and_mask(gt_bgr, regions, char_bboxes)
    log(f"  degraded: {degraded_bgr.shape}, mask area: {mask.sum()} pixels")

    # 保存 GT + degraded + mask
    Image.fromarray(gt_bgr[:, :, ::-1]).save(str(OUTPUT_DIR / "gt.png"))
    Image.fromarray(degraded_bgr[:, :, ::-1]).save(str(OUTPUT_DIR / "degraded.png"))
    Image.fromarray(mask).save(str(OUTPUT_DIR / "mask.png"))

    # 1. RAG agent（用于 rag_correct）
    log("\n[1] 初始化 RAG agent...")
    agent = RagAgent()

    # 2. 生成 5 组 content_image
    log("\n[2] 生成 5 组 content_image...")
    sources = ["original", "manual_correct", "manual_wrong", "blank", "rag_correct"]
    content_images = {}
    for source in sources:
        content = build_content_image(source, degraded_bgr, mask, damage_infos, agent if source == "rag_correct" else None)
        content_images[source] = content
        Image.fromarray(content[:, :, ::-1]).save(str(OUTPUT_DIR / f"content_{source}.png"))
        log(f"  content_{source}.png")

    # 3. 5 组 DiffHDR 推理
    log("\n[3] 5 组 DiffHDR 推理（每张约 1-2 分钟）...")
    results = {}
    for source in sources:
        log(f"\n--- {source} ---")
        work_dir = OUTPUT_DIR / f"work_{source}"
        work_dir.mkdir(exist_ok=True)

        deg_path = work_dir / "degraded.png"
        mask_path = work_dir / "mask.png"
        content_path = work_dir / "content.png"
        Image.fromarray(degraded_bgr[:, :, ::-1]).save(str(deg_path))
        Image.fromarray(mask).save(str(mask_path))
        Image.fromarray(content_images[source][:, :, ::-1]).save(str(content_path))

        run_result = run_diffhdr(str(deg_path), str(mask_path), str(content_path), str(work_dir))
        log(f"  returncode={run_result['returncode']}, latency={run_result['latency_sec']}s")

        if run_result["returncode"] != 0:
            log(f"  [FAIL] {run_result['stderr'][:200]}")
            results[source] = {"success": False, "error": run_result["stderr"][:200]}
            continue

        repaired_path = work_dir / "degraded_repaired_img.png"
        if not repaired_path.exists():
            log(f"  [FAIL] 输出不存在")
            results[source] = {"success": False, "error": "output not found"}
            continue

        # 加载 + blend
        restored = np.array(Image.open(repaired_path).convert("RGB"))
        restored_bgr = restored[:, :, ::-1].copy()
        blended_bgr = blend_repaired(degraded_bgr, restored_bgr, mask)

        # 保存到输出目录
        Image.fromarray(blended_bgr[:, :, ::-1]).save(str(OUTPUT_DIR / f"restored_{source}.png"))

        # 计算指标
        metrics = compute_metrics(blended_bgr, gt_bgr, mask)
        log(f"  PSNR={metrics.get('psnr', '?')}, SSIM={metrics.get('ssim', '?')}, "
            f"MAE={metrics.get('mae', '?')}, IoU={metrics.get('stroke_iou', '?')}")

        # mask 外像素校验
        outside = mask == 0
        out_diff = np.abs(blended_bgr.astype(int) - degraded_bgr.astype(int))
        out_max = int(out_diff[outside].max())

        results[source] = {
            "success": True,
            "latency_sec": run_result["latency_sec"],
            "metrics": metrics,
            "mask_outside_max_diff": out_max,
            "mask_outside_passed": out_max == 0,
        }

    # 4. 保存指标汇总
    log("\n[4] 保存指标汇总...")
    metrics_csv = "source,psnr,ssim,mse,mae,stroke_iou,success\n"
    for source in sources:
        r = results.get(source, {})
        m = r.get("metrics", {})
        metrics_csv += f"{source},{m.get('psnr', 0)},{m.get('ssim', 0)},{m.get('mse', 0)},{m.get('mae', 0)},{m.get('stroke_iou', 0)},{r.get('success', False)}\n"
    (OUTPUT_DIR / "metrics.csv").write_text(metrics_csv, encoding="utf-8")

    # 5. run_metadata
    run_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "steps": STEPS, "device": DEVICE,
        "content_scale": CONTENT_SCALE, "degraded_scale": DEGRADED_SCALE,
        "font": font_path, "font_size": FONT_SIZE,
        "regions": [{"id": r["id"], "chars": r["chars"], "wrong_chars": r["wrong_chars"]} for r in regions],
        "results": {s: {k: v for k, v in r.items() if k != "error"} for s, r in results.items()},
        "note": "max_diff>5 只能证明 content image 对 DiffHDR 有可观测影响，不能证明引导方向正确",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. 判断 manual_correct 是否优于 manual_wrong 和 blank
    log("\n[5] 判断 manual_correct 方向性...")
    mc = results.get("manual_correct", {})
    mw = results.get("manual_wrong", {})
    bl = results.get("blank", {})
    if mc.get("success") and mw.get("success") and bl.get("success"):
        mc_psnr = mc["metrics"]["psnr"]
        mw_psnr = mw["metrics"]["psnr"]
        bl_psnr = bl["metrics"]["psnr"]
        better_than_wrong = mc_psnr > mw_psnr
        better_than_blank = mc_psnr > bl_psnr
        log(f"  manual_correct PSNR={mc_psnr} vs manual_wrong PSNR={mw_psnr}: {'更优' if better_than_wrong else '未更优'}")
        log(f"  manual_correct PSNR={mc_psnr} vs blank PSNR={bl_psnr}: {'更优' if better_than_blank else '未更优'}")
        if not better_than_wrong:
            log("  [FAIL] manual_correct 未优于 manual_wrong，需要检查渲染和参数")
    else:
        log("  [FAIL] 有路径未完成，无法判断")

    log("\n" + "=" * 60)
    log("阶段 3A 完成")
    log("=" * 60)


if __name__ == "__main__":
    main()
