# AutoDL 部署步骤

## 1. 创建实例

选择 Ubuntu 20.04、CUDA 11.8 兼容镜像和 NVIDIA GPU。上游没有公开最低显存，
不要把未经实测的显存数字写成要求；先用 1.5B 模型做 smoke，再决定是否使用 7B。

## 2. 上传内容

按 `upload_manifest.json` 上传：

1. 固定 commit 的 `third_party/AutoHDR/` 源码；
2. 本目录 `deployment/autodl_autohdr/`；
3. `case_transfer_manifest.csv` 中明确选中的真实图像；
4. 通过官方 Model Zoo 合法取得的模型文件。

不要上传 `.venv`、全量 CBETA/RAG 索引、缓存、历史 outputs/reports、Cookie、
密钥或整个 `data/raw`。

## 3. 校验上游版本

若在服务器重新 clone：

```bash
git clone https://github.com/SCUT-DLVCLab/AutoHDR.git third_party/AutoHDR
git -C third_party/AutoHDR checkout d92b223b3d9274b03ca6f1e4c62b09253f378c98
git -C third_party/AutoHDR rev-parse HEAD
```

最后一行必须逐字符等于固定 commit。

## 4. 创建环境

```bash
conda create -n autohdr python=3.10 -y
conda activate autohdr
cd third_party/AutoHDR
pip install -r requirements.txt
```

然后核验：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

预期主版本为 PyTorch 2.3.0、CUDA 11.8 且 `cuda.is_available()` 为 True。

## 5. 下载官方权重

严格按上游 README 的 Model Zoo 下载并放到 `REQUIRED_MODELS.md` 所列位置。
如果百度网盘要求登录、验证码或人工点击，必须由用户完成最后一步；不得使用
非官方镜像或绕过限制。下载后立即生成 SHA-256 清单并保留来源记录。

## 6. 预检与 smoke

```bash
bash deployment/autodl_autohdr/run_official_smoke.sh \
  data/real_cases/transfer/CASE05-PREVIEW.jpg
```

脚本缺任一模型就会失败退出，不会伪造 PASS。输出仅是
`unreviewed_official_raw/smoke/` 下的待审查工件。

## 7. 真实案例批量 smoke

```bash
ALLOW_MONOLITHIC_OFFICIAL_SMOKE=1 \
bash deployment/autodl_autohdr/run_official_real_cases.sh \
  deployment/autodl_autohdr/case_transfer_manifest.csv
```

该开关只允许执行上游单体管道，不等于允许自动采纳文字或修复结果。逐例核验
输入哈希、日志、输出图和显存峰值；任何失败都保持 FAIL/PENDING。

## 8. 回传与本地接入

只回传：

- 原始官方输出图和日志；
- 输入/输出 SHA-256；
- 运行环境、固定 commit、模型哈希；
- 结构化候选（适配器完成后）。

本地随后执行 CBETA RAG 证据融合和人工确认。只有 `manual_confirmed` 的字符
允许进入正式外观恢复；未确认输出不得显示为“真实修复完成”。
