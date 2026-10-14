"""下载并配置 DiffHDR 修复模块。

官方仓库: https://github.com/yeungchenwa/HDR
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HDR_DIR = PROJECT_ROOT / "third_party" / "HDR"
WEIGHTS_DIR = PROJECT_ROOT / "checkpoints" / "diffhdr"

# 官方 README Model Zoo（2024-12）
GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1ArP21T7vyTpbPb5qC5VV76pMUsQd4tCx?usp=sharing"
BAIDU_URL = "https://pan.baidu.com/s/1XpoGvQHruOQjzJDEymsXzg"
BAIDU_CODE = "x62f"


def clone_hdr() -> None:
    if HDR_DIR.exists():
        print(f"DiffHDR repo exists: {HDR_DIR}")
        return
    HDR_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning DiffHDR repo...")
    subprocess.run(
        ["git", "clone", "https://github.com/yeungchenwa/HDR.git", str(HDR_DIR)],
        check=True,
    )


def install_deps() -> None:
    req = HDR_DIR / "requirements.txt"
    if req.exists():
        print("Installing DiffHDR dependencies...")
        subprocess.run(["pip", "install", "-r", str(req)], check=True)


def print_manual_download_guide() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(WEIGHTS_DIR.rglob("config.json")) + list(WEIGHTS_DIR.glob("**/unet/*"))
    if any(WEIGHTS_DIR.iterdir()):
        print(f"Weights directory already has files: {WEIGHTS_DIR}")
        return

    guide = WEIGHTS_DIR / "DOWNLOAD.md"
    guide.write_text(
        f"""# DiffHDR Manual Weight Download

## Option A: Baidu Netdisk (recommended in China)

1. Open: {BAIDU_URL}
2. Extraction code: `{BAIDU_CODE}`
3. Download the `unet` folder (HuggingFace diffusers format)
4. Place it here:

```
checkpoints/diffhdr/unet/
├── config.json
├── diffusion_pytorch_model.safetensors
└── ...
```

## Option B: Google Drive

1. Open: {GOOGLE_DRIVE_URL}
2. Download the `unet` checkpoint folder
3. Copy to `checkpoints/diffhdr/unet/`

## Verify

```bash
ls checkpoints/diffhdr/unet/config.json
```

## Inference example

```bash
python third_party/HDR/inference.py \\
  --device mps \\
  --image_path data/raw/diffhdr_examples/0_damaged_img.png \\
  --mask_image_path data/raw/diffhdr_examples/0_mask_img.png \\
  --content_image_path data/raw/diffhdr_examples/0_content_img.png \\
  --save_dir outputs/diffhdr \\
  --ckpt_path checkpoints/diffhdr/unet \\
  --num_inference_steps 20
```

Note: DiffHDR officially targets CUDA. On Apple Silicon use `--device mps` or `--device cpu`.
""",
        encoding="utf-8",
    )

    print("=" * 60)
    print("Manual download required for DiffHDR weights")
    print(f"  Baidu:  {BAIDU_URL}  (code: {BAIDU_CODE})")
    print(f"  Google: {GOOGLE_DRIVE_URL}")
    print(f"  Target: {WEIGHTS_DIR / 'unet'}/")
    print(f"  Guide:  {guide}")
    print("=" * 60)


def main() -> None:
    clone_hdr()
    install_deps()
    print_manual_download_guide()
    print("\nDone.")
    print(f"  hdr_root  = {HDR_DIR}")
    print(f"  ckpt_path = {WEIGHTS_DIR / 'unet'}")


if __name__ == "__main__":
    main()
