# AutoDL / Official AutoHDR 就绪包

本目录只提供部署说明、预检脚本、输入清单和运行包装器。它不包含模型权重、
真实原图、Cookie、API Key 或登录会话，也不表示 Official AutoHDR 已经运行。

## 固定上游

- 项目：SCUT-DLVCLab/AutoHDR
- 仓库：https://github.com/SCUT-DLVCLab/AutoHDR
- 固定 commit：`d92b223b3d9274b03ca6f1e4c62b09253f378c98`
- 许可：CC BY-NC-ND 4.0，仅限非商业研究；商业使用须联系上游作者。
- 本地来源记录：`third_party/AutoHDR_MANIFEST.json`

## 当前状态

- 上游源码已在本机 `third_party/AutoHDR/`，但 Git 忽略且不在本就绪包内。
- damage detector、OCR、Qwen2、DiffHDR 等官方权重均未下载。
- Official 代码要求 Ubuntu 20.04、Python 3.10、PyTorch 2.3.0、CUDA 11.8。
- 本地 Windows 只完成 Mock/Replay 契约验证：`OFFICIAL_AUTOHDR_NOT_RUN`。
- CBETA RAG V2 已在本地完成，AutoDL 无需上传 23 GiB 全量索引；官方候选返回
  本地后再走 RAG 证据与人工确认门槛。

## 安全边界

官方 `infer_pipeline.py` 是单体端到端脚本，不能在文字预测与恢复之间自然暂停。
本包只允许把它作为隔离的 smoke/raw-candidate 运行，输出保存在
`unreviewed_official_raw/`，不得自动采纳。正式适配器必须在 `manual_confirmed`
之后才调用外观恢复阶段。

从 `STEP_BY_STEP.md` 开始。
