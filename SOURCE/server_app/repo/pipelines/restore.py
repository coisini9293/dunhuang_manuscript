import subprocess
import sys
from pathlib import Path

import cv2


class DiffHDRRestorer:
    """DiffHDR 修复封装：调用官方 inference.py。"""

    def __init__(self, hdr_root: str | Path, ckpt_path: str | Path):
        self.hdr_root = Path(hdr_root)
        self.ckpt_path = Path(ckpt_path)
        if not (self.hdr_root / "inference.py").exists():
            raise FileNotFoundError(
                f"DiffHDR 仓库不完整: {self.hdr_root}，请先运行 scripts/setup_diffhdr.py"
            )
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"DiffHDR 权重不存在: {self.ckpt_path}，请按 third_party/HDR/README 手动下载"
            )

    def restore(
        self,
        degraded_path: str | Path,
        mask_path: str | Path,
        content_path: str | Path,
        output_dir: str | Path,
        steps: int = 50,
        content_scale: float = 7.5,
        degraded_scale: float = 7.5,
        device: str = "cpu",
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(self.hdr_root / "inference.py"),
            "--image_path",
            str(degraded_path),
            "--mask_image_path",
            str(mask_path),
            "--content_image_path",
            str(content_path),
            "--save_dir",
            str(output_dir),
            "--ckpt_path",
            str(self.ckpt_path),
            "--num_inference_steps",
            str(steps),
            "--content_mask_guidance_scale",
            str(content_scale),
            "--degraded_guidance_scale",
            str(degraded_scale),
            "--device",
            device,
        ]
        subprocess.run(cmd, check=True, cwd=str(self.hdr_root))
        stem = Path(degraded_path).stem
        out = output_dir / f"{stem}_repaired_img.png"
        if not out.exists():
            raise FileNotFoundError(f"DiffHDR 未生成输出: {out}")
        return out

    def restore_image(
        self,
        degraded_bgr,
        mask_gray,
        content_gray,
        work_dir: str | Path,
        **kwargs,
    ):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        deg = work_dir / "degraded.png"
        mask = work_dir / "mask.png"
        content = work_dir / "content.png"
        cv2.imwrite(str(deg), degraded_bgr)
        cv2.imwrite(str(mask), mask_gray)
        cv2.imwrite(str(content), content_gray)
        out = self.restore(deg, mask, content, work_dir, **kwargs)
        return cv2.imread(str(out))
