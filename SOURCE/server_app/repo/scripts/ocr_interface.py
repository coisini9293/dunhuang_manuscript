"""阶段 6：OCR 辅助上下文提取接口。

支持三种模式：
- manual: 人工转录
- ocr: 自动 OCR
- ocr_reviewed: OCR + 人工校对

OCR 只提供候选转录，不作为最终事实来源。
MVP 阶段使用 PIL + 简单文字区域检测。
"""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def ocr_extract(image_path: str, mode: str = "ocr") -> dict:
    """OCR 提取文本。mode: ocr | manual | ocr_reviewed。"""
    if mode == "manual":
        return {
            "mode": "manual",
            "text": "",
            "caveat": "需要人工转录。请在 transcription_regions.json 中标注。",
        }

    # MVP OCR 使用简单图像处理检测文字区域
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(image_path).convert("L")
        arr = np.array(img)
        # 简单阈值检测文字区域
        binary = (arr < 128).astype(np.uint8)
        text_area_ratio = binary.sum() / binary.size
        return {
            "mode": mode,
            "text_area_ratio": round(text_area_ratio, 4),
            "caveat": "MVP OCR 仅检测文字区域比例。完整 OCR 需接 PaddleOCR/Tesseract。",
        }
    except Exception as e:
        return {"mode": mode, "error": str(e)}


if __name__ == "__main__":
    # 测试 05 号案例图
    img = r"C:\Users\95788\Desktop\项目\敦煌\data\data\raw\dunhuang_iiif\05_0100bd6eba9c42f5b62dbcebffc928c0\preview.jpg"
    result = ocr_extract(img, mode="ocr")
    print(json.dumps(result, indent=2))
