import cv2
import numpy as np


MIN_MASK_AREA = 50


def align_mask_to_image(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize user mask to match image size and binarize (255 = repair region)."""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def mask_has_damage(mask: np.ndarray, min_area: int = MIN_MASK_AREA) -> bool:
    """Return True if mask contains enough marked repair pixels."""
    if mask is None or mask.size == 0:
        return False
    binary = align_mask_to_image(mask, mask.shape[0], mask.shape[1])
    return int(cv2.countNonZero(binary)) >= min_area


def mask_damage_ratio(mask: np.ndarray) -> float:
    binary = align_mask_to_image(mask, mask.shape[0], mask.shape[1])
    return float(np.mean(binary > 127))


def detect_damage(image, method='adaptive'):
    """检测缺损区域，返回binary mask (0=完好, 255=缺损)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if method == 'adaptive':
        # 自适应阈值：深色孔洞检测
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        # 固定阈值
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 过滤小面积
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50:  # ponytail: 阈值可调
            result[labels == i] = 255
    
    return result


def extract_content_image(image) -> np.ndarray:
    """Extract character stroke map as DiffHDR content guidance (grayscale L)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, content = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return content


def resize_for_diffhdr(image, size: int = 512):
    """DiffHDR requires exactly size×size inputs; letterbox-pad after uniform scale."""
    padded, _ = letterbox_for_diffhdr(image, size=size)
    return padded


def letterbox_for_diffhdr(image: np.ndarray, size: int = 512) -> tuple[np.ndarray, dict]:
    """Scale longest side to `size`, center-pad to a square. Returns (image, meta)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    is_color = len(resized.shape) == 3
    fill = (255, 255, 255) if is_color else 0
    canvas = np.full((size, size, 3), fill, dtype=image.dtype) if is_color else np.full((size, size), fill, dtype=image.dtype)

    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    meta = {"x0": x0, "y0": y0, "w": new_w, "h": new_h, "orig_h": h, "orig_w": w}
    return canvas, meta


def unletterbox(image: np.ndarray, meta: dict) -> np.ndarray:
    """Crop letterbox padding and restore original aspect ratio."""
    crop = image[meta["y0"] : meta["y0"] + meta["h"], meta["x0"] : meta["x0"] + meta["w"]]
    return cv2.resize(crop, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)


def blend_repaired(original, repaired, mask):
    """将修复结果与原图混合：mask外保留原图"""
    mask_3ch = np.stack([mask / 255.0] * 3, axis=-1)
    return (repaired * mask_3ch + original * (1 - mask_3ch)).astype(np.uint8)
