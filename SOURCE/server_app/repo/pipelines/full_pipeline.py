"""Full pipeline: Rejoin -> Stitch -> user mask -> content_image (original/manual/rag) -> DiffHDR restore.

配置开关（.env）：
- RESTORE_CONTENT_SOURCE: original / manual / rag（默认 original）
  - original: 原 extract_content_image（OTSU 二值化），保持原流程
  - manual:  人工指定正确文字
  - rag:     RAG 决策后渲染（需配合 damages 参数）
- LLM_ENABLED: false / true（控制 RAG agent L3 是否调真实 LLM）

异常回退策略：
- rag 模式下 RAG agent 异常 -> 自动回退 original
- manual 模式下未提供 manual_text -> 自动回退 original
- 任何 content_image 生成异常 -> 自动回退 original
- 不允许 RAG 结果无条件进入 DiffHDR
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dotenv import load_dotenv

from pipelines.rejoin import RejoinPipeline
from pipelines.restore import DiffHDRRestorer
from utils.mask_gen import (
    align_mask_to_image,
    blend_repaired,
    extract_content_image,
    letterbox_for_diffhdr,
    mask_damage_ratio,
    mask_has_damage,
    unletterbox,
)
from utils.rejoin_eval import stitch_pair
from utils.restore_paths import DIFFHDR_CKPT, HDR_ROOT, ensure_diffhdr_ready


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

RESTORE_CONTENT_SOURCE = os.environ.get("RESTORE_CONTENT_SOURCE", "original").lower()


class FullPipeline:
    """Mandatory two-stage pipeline: rejoin classification + DiffHDR restoration."""

    def __init__(
        self,
        rejoin_ckpt: str | Path | None = None,
        hdr_root: str | Path | None = None,
        diffhdr_ckpt: str | Path | None = None,
        device: str = "auto",
        content_source: str | None = None,
    ):
        ensure_diffhdr_ready()
        self.rejoin = RejoinPipeline(model_path=rejoin_ckpt, device=device)
        self.restorer = DiffHDRRestorer(
            hdr_root=hdr_root or HDR_ROOT,
            ckpt_path=diffhdr_ckpt or DIFFHDR_CKPT,
        )
        self.device = _resolve_device(device)
        print(f"[full_pipeline] device = {self.device}")
        # content_image 来源（实例级覆盖环境变量）
        self.content_source = (content_source or RESTORE_CONTENT_SOURCE).lower()
        if self.content_source not in ("original", "manual", "rag"):
            print(f"[full_pipeline] 未知 content_source={self.content_source}，回退 original")
            self.content_source = "original"
        self._rag = None  # lazy load
        print(f"[full_pipeline] content_source = {self.content_source}")

    @property
    def rag(self):
        if self._rag is None and self.content_source == "rag":
            try:
                from agents.rag_agent import RagAgent
                self._rag = RagAgent()
            except Exception as e:
                print(f"[full_pipeline] RAG agent 加载失败，回退 original: {e}")
                self._rag = None
                self.content_source = "original"
        return self._rag

    def _build_content_image(
        self,
        degraded_bgr: np.ndarray,
        mask_full: np.ndarray,
        damages: list[dict] | None = None,
        manual_text: str | None = None,
    ) -> tuple[np.ndarray, list[dict], str]:
        """构建 content_image，返回 (content_bgr, decisions, actual_source)。

        三种来源：
        - original: OTSU 二值化（原 extract_content_image）
        - manual:  人工指定文字渲染
        - rag:     RAG 决策渲染

        异常自动回退 original。
        """
        h, w = degraded_bgr.shape[:2]
        decisions: list[dict] = []

        # original 模式（原流程，不碰 RAG）
        if self.content_source == "original":
            content_gray = extract_content_image(degraded_bgr)
            return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"

        # manual 模式
        if self.content_source == "manual":
            if not manual_text:
                print("[full_pipeline] manual 模式未提供 manual_text，回退 original")
                content_gray = extract_content_image(degraded_bgr)
                return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"
            try:
                from utils.text_renderer import render_content_image
                # 把 manual_text 渲染到整个 mask 区域
                mask_bbox = _mask_bbox(mask_full)
                decisions = [{
                    "char": manual_text,
                    "bbox": mask_bbox,
                    "color": "blue",
                    "source": "manual",
                }]
                content = render_content_image(degraded_bgr, mask_full, decisions)
                return content, decisions, "manual"
            except Exception as e:
                print(f"[full_pipeline] manual 渲染失败，回退 original: {e}")
                content_gray = extract_content_image(degraded_bgr)
                return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"

        # rag 模式
        if self.content_source == "rag":
            if self.rag is None:
                print("[full_pipeline] RAG agent 不可用，回退 original")
                content_gray = extract_content_image(degraded_bgr)
                return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"
            if not damages:
                print("[full_pipeline] rag 模式未提供 damages，回退 original")
                content_gray = extract_content_image(degraded_bgr)
                return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"
            try:
                # 1. 基础 content：周围字的二值化 mask
                base_content_gray = extract_content_image(degraded_bgr)
                content_bgr = cv2.cvtColor(base_content_gray, cv2.COLOR_GRAY2BGR)

                # 2. 对每个缺损跑 RAG 决策
                for d in damages:
                    decision = self.rag.decide(
                        context_before=d.get("context_before", ""),
                        context_after=d.get("context_after", ""),
                        sutra_hint=d.get("sutra_hint", ""),
                    )
                    d_dict = decision.to_dict()
                    d_dict["bbox"] = d.get("bbox")
                    decisions.append(d_dict)

                # 3. 在二值化图上叠加 RAG 决策字
                content_bgr = _overlay_rag_text(content_bgr, decisions, w, h)
                return content_bgr, decisions, "rag"
            except Exception as e:
                print(f"[full_pipeline] RAG 渲染失败，回退 original: {e}")
                content_gray = extract_content_image(degraded_bgr)
                return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"

        # fallback
        content_gray = extract_content_image(degraded_bgr)
        return cv2.cvtColor(content_gray, cv2.COLOR_GRAY2BGR), decisions, "original"

    def _restore_bgr(
        self,
        degraded_bgr: np.ndarray,
        work_dir: Path,
        user_mask: np.ndarray,
        damages: list[dict] | None = None,
        manual_text: str | None = None,
    ) -> dict:
        if not mask_has_damage(user_mask):
            raise ValueError("未标注缺损区域：请在前端涂抹需要修复的区域后再执行修复")

        h, w = degraded_bgr.shape[:2]
        mask_full = align_mask_to_image(user_mask, h, w)
        degraded, meta = letterbox_for_diffhdr(degraded_bgr)
        mask = letterbox_for_diffhdr(mask_full)[0]

        # 构建 content_image（三种来源 + 异常回退）
        content_full, decisions, actual_source = self._build_content_image(
            degraded_bgr, mask_full, damages, manual_text
        )
        content = letterbox_for_diffhdr(content_full)[0]

        restore_device = self.device
        if restore_device == "auto":
            restore_device = _resolve_device("auto")
        repaired = self.restorer.restore_image(
            degraded,
            mask,
            content,
            work_dir,
            steps=20,
            content_scale=1.5,
            degraded_scale=1.2,
            device=restore_device,
        )
        if repaired is None:
            raise RuntimeError("DiffHDR returned empty image")
        blended_sq = blend_repaired(degraded, repaired, mask)
        blended = unletterbox(blended_sq, meta)
        return {
            "degraded": degraded_bgr,
            "mask": mask_full,
            "content": content,
            "content_source": actual_source,
            "repaired": repaired,
            "blended": blended,
            "decisions": decisions,
            "mask_area_ratio": mask_damage_ratio(mask_full),
        }

    def rejoin_only(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> dict:
        confidence, pose = self.rejoin.predict_pair(img1_bgr, img2_bgr)
        rgb1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
        panorama_rgb = stitch_pair(rgb1, rgb2)
        panorama_bgr = cv2.cvtColor(panorama_rgb, cv2.COLOR_RGB2BGR)
        verdict = "match" if confidence >= 0.5 else "no_match"
        return {
            "rejoin": {
                "confidence": float(confidence),
                "verdict": verdict,
                "pose": {
                    "dx": float(pose[0]),
                    "dy": float(pose[1]),
                    "rotation": float(pose[2]),
                },
            },
            "panorama": panorama_bgr,
        }

    def restore_only(
        self,
        img1_bgr: np.ndarray,
        img2_bgr: np.ndarray,
        user_mask: np.ndarray,
        damages: list[dict] | None = None,
        manual_text: str | None = None,
    ) -> dict:
        rgb1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
        panorama_bgr = cv2.cvtColor(stitch_pair(rgb1, rgb2), cv2.COLOR_RGB2BGR)
        with tempfile.TemporaryDirectory(prefix="gurou_restore_") as tmp:
            restore_out = self._restore_bgr(panorama_bgr, Path(tmp), user_mask, damages, manual_text)
        return {
            "panorama": panorama_bgr,
            "mask": restore_out["mask"],
            "repaired": restore_out["blended"],
            "content_source": restore_out["content_source"],
            "decisions": restore_out["decisions"],
            "mask_area_ratio": restore_out["mask_area_ratio"],
        }

    def run(
        self,
        img1_bgr: np.ndarray,
        img2_bgr: np.ndarray,
        user_mask: np.ndarray,
        damages: list[dict] | None = None,
        manual_text: str | None = None,
    ) -> dict:
        rejoin_out = self.rejoin_only(img1_bgr, img2_bgr)
        restore_out = self.restore_only(img1_bgr, img2_bgr, user_mask, damages, manual_text)
        return {**rejoin_out, **restore_out}


def _resolve_device(device: str) -> str:
    """自动选择设备：cuda → cpu (Windows 不用 mps)。"""
    if device != "auto":
        return device
    import platform
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """计算 mask 的 bounding box。"""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _overlay_rag_text(content_bgr: np.ndarray, decisions: list[dict], w: int, h: int) -> np.ndarray:
    """在二值化 content_image 上叠加 RAG 决策文字。

    安全规则：只有 decision_status == "manual_confirmed" 或
    allow_auto_render == True 的决策才能渲染到 content_image。
    retrieved_exact / retrieved_ambiguous / llm_suggestion 默认
    不得直接进入 content_image。颜色仅用于 UI，不作为渲染判断。
    """
    from utils.text_renderer import render_content_image
    out = content_bgr.copy()
    mask_dummy = np.zeros((h, w), dtype=np.uint8)

    renderable = []
    for d in decisions:
        char = d.get("char")
        if not char:
            continue
        status = d.get("decision_status", "")
        allow = d.get("allow_auto_render", False)
        if status == "manual_confirmed" or allow is True:
            renderable.append({**d, "color": "blue"})

    if not renderable:
        return out  # 不渲染任何决策（content 保持为 OTSU baseline）

    rendered = render_content_image(out, mask_dummy, renderable)
    for d in renderable:
        bbox = d.get("bbox")
        if bbox is None or not d.get("char"):
            continue
        x, y, bw, bh = bbox
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        x2 = max(0, min(x + bw, w))
        y2 = max(0, min(y + bh, h))
        if x2 > x and y2 > y:
            out[y:y2, x:x2] = rendered[y:y2, x:x2]
    return out
