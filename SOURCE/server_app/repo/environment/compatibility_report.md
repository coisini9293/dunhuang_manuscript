# 环境兼容性报告

## 决策：Situation B — Windows 无 CUDA

- **主项目 (gurou)**: Windows, Python 3.13, CPU only, managed venv
- **AutoHDR Worker**: 需要独立 Linux CUDA 服务器
- **通信方式**: HTTP API (`services/autohdr_worker/app.py`)

## 为什么使用独立环境

1. AutoHDR 依赖 PyTorch CUDA + mmdetection + Qwen2-1.5B/7B
2. 当前主项目使用 diffusers 0.39.0 + PyTorch 2.13.0 CPU
3. 两者版本冲突（mmdetection 需要特定 PyTorch+CUDA 组合）
4. 独立环境避免破坏现有缀合和 DiffHDR 功能

## AutoHDR 环境要求

```bash
conda create -n autohdr python=3.10 -y
conda activate autohdr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 按官方 requirements 安装 mmdetection, transformers 等
```

## 主项目环境（保持不变）

```
Python 3.13, diffusers 0.39.0, PyTorch 2.13.0 CPU
dashscope, zhconv, scikit-image, opencv-python
```

## 通信架构

```
主项目 Demo (FastAPI :8000)
  ├── POST /api/autohdr/analyze → calls AutoHDRClient
  ├── POST /api/autohdr/evidence → CBETA fusion
  ├── POST /api/autohdr/confirm → human confirm
  └── POST /api/autohdr/restore → AutoHDR restore

AutoHDR Worker (FastAPI :8010, 独立 conda 环境)
  ├── GET /health
  ├── POST /analyze → Stage1 damage+OCR
  └── POST /restore → Stage3 appearance
```

## 当前状态

- `autohdr_inference_not_executed`: Worker 代码/接口/schemas 已完成，等待 CUDA 服务器部署
- Legacy DiffHDR 继续作为 baseline 可用
