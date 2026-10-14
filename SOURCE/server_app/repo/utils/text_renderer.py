"""PIL 文字渲染器：把 RAG 决策的字渲染到 mask 位置，生成 content_image。

content_image 用于 DiffHDR 的 content 引导通道：
- 当前问题：content = 退化图副本（等于没用）
- 改进：content = RAG 决策字渲染图（在 mask 位置写上预测文字）

渲染策略（MVP）：
- 创建与原图同尺寸的纸色背景
- 在 mask 区域内，用竖排黑字写上 RAG 决策的字
- 字号自适应 mask 大小
- 非 mask 区域保留纸色（DiffHDR 会用 degraded 图填充）

后期可升级为 AnyText2（书风可控渲染）。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 敦煌写本纸色（泛黄）
PAPER_COLOR = (234, 216, 180)  # 浅黄棕
# 墨色
INK_COLOR = (40, 30, 25)       # 深棕黑


def find_cjk_font() -> Optional[str]:
    """查找系统中可用的中文字体路径。"""
    candidates = [
        # Windows
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/simkai.ttf",     # 楷体（更接近写本风格）
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑粗
        # macOS
        "/System/Library/Fonts/STKaiti.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Songti.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def render_content_image(
    degraded_bgr: np.ndarray,
    mask_gray: np.ndarray,
    decisions: list[dict],
) -> np.ndarray:
    """渲染 content_image：在 mask 位置写上 RAG 决策的字。

    Args:
        degraded_bgr: 原始退化图（BGR，OpenCV 格式）
        mask_gray: mask 灰度图（255=缺损，0=完好）
        decisions: RAG 决策列表，每个 {
            "char": "佛" or None,
            "bbox": (x, y, w, h),  # 缺损区域 bounding box
            "color": "blue" | "yellow" | "gray",
            "source": ...,
            "confidence": ...,
            "citation": ...,
        }

    Returns:
        content_bgr: 渲染后的 content_image（BGR，与原图同尺寸）
    """
    h, w = degraded_bgr.shape[:2]

    # 创建纸色背景
    paper = np.full((h, w, 3), PAPER_COLOR, dtype=np.uint8)

    # 转 PIL 绘制
    paper_rgb = paper[:, :, ::-1].copy()  # BGR -> RGB
    img = Image.fromarray(paper_rgb)
    draw = ImageDraw.Draw(img)

    # 加载字体
    font_path = find_cjk_font()
    if font_path is None:
        print("[text_renderer] 警告：未找到中文字体，文字可能无法渲染")
        return paper

    for d in decisions:
        char = d.get("char")
        color = d.get("color", "gray")
        bbox = d.get("bbox")
        if bbox is None or not char:
            continue  # L4 标灰，不写字

        x, y, bw, bh = bbox
        # 竖排渲染：每个字占一行
        chars = list(char)
        n_chars = len(chars)

        if n_chars == 0:
            continue

        # 字号自适应：根据 bbox 大小和字数
        char_h = min(bh // n_chars, bw)  # 每个字的高度
        char_w = char_h  # 方块字
        font_size = max(12, min(char_h, 80))

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # 颜色按置信度标注
        if color == "blue":
            text_color = INK_COLOR  # 蓝标用墨色（高置信）
        elif color == "yellow":
            text_color = (60, 50, 120)  # 黄标用偏紫色（中置信，区别于墨色）
        else:
            text_color = (150, 150, 150)  # 灰标用灰色

        # 竖排绘制
        for i, c in enumerate(chars):
            cx = x + (bw - char_w) // 2
            cy = y + i * char_h
            # 居中绘制
            draw.text((cx, cy), c, fill=text_color, font=font)

    # 转回 BGR
    content_rgb = np.array(img)
    content_bgr = content_rgb[:, :, ::-1].copy()
    return content_bgr


def render_overlay(
    image_bgr: np.ndarray,
    decisions: list[dict],
) -> np.ndarray:
    """在三色标注图上叠加决策信息（蓝/黄/灰边框 + 字）。

    用于最终输出展示，区别于 content_image（用于 DiffHDR 输入）。
    """
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    # 转 PIL
    overlay_rgb = overlay[:, :, ::-1].copy()
    img = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(img, "RGBA")

    font_path = find_cjk_font()
    font_size = 14
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

    color_map = {
        "blue": (50, 80, 200, 180),     # 蓝标
        "yellow": (200, 180, 30, 180),  # 黄标
        "gray": (128, 128, 128, 180),   # 灰标
    }

    for d in decisions:
        bbox = d.get("bbox")
        if bbox is None:
            continue
        x, y, bw, bh = bbox
        color_name = d.get("color", "gray")
        box_color = color_map.get(color_name, color_map["gray"])

        # 画边框
        draw.rectangle([x, y, x + bw, y + bh], outline=box_color, width=2)

        # 标注信息
        char = d.get("char", "") or "?"
        source = d.get("source", "")
        conf = d.get("confidence", 0)
        label = f"{char} [{source[:4]}] {conf:.0%}"
        draw.text((x, y - 16 if y > 16 else y + bh + 2), label, fill=box_color, font=font)

    overlay_bgr = np.array(img)[:, :, ::-1].copy()
    return overlay_bgr


if __name__ == "__main__":
    # 测试：创建一个假图 + mask + 决策
    print("--- 测试文字渲染 ---")
    degraded = np.full((200, 400, 3), PAPER_COLOR, dtype=np.uint8)
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[50:150, 100:140] = 255  # 一个缺损区域

    decisions = [
        {
            "char": "众宝杂华",
            "bbox": (100, 50, 40, 100),
            "color": "blue",
            "source": "cbeta",
            "confidence": 1.0,
            "citation": "T0278 卷1",
        },
        {
            "char": None,
            "bbox": (200, 50, 40, 100),
            "color": "gray",
            "source": "none",
            "confidence": 0.0,
            "citation": "",
        },
    ]

    content = render_content_image(degraded, mask, decisions)
    print(f"content_image shape: {content.shape}")
    print(f"content_image dtype: {content.dtype}")
    print(f"非零像素: {np.count_nonzero(content - PAPER_COLOR)}")

    overlay = render_overlay(degraded, decisions)
    print(f"overlay shape: {overlay.shape}")

    # 保存测试图
    import cv2
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "test_render"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "content_test.png"), content)
    cv2.imwrite(str(out_dir / "overlay_test.png"), overlay)
    print(f"测试图已保存到 {out_dir}")
