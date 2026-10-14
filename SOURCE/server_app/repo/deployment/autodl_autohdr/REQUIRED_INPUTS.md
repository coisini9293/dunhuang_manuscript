# 真实案例输入要求

## 服务器目录

将 `case_transfer_manifest.csv` 中的图像逐一复制到：

```text
data/real_cases/transfer/<case_id>.<ext>
```

复制后必须重新计算 SHA-256，并与清单逐字符一致。原图只读保留；旋转、裁剪、
缩放、颜色转换和 ROI 应写入 `work/` 副本。

## 第一轮建议

- `CASE05-PREVIEW`：仅为数字藏经洞官方拼合预览，适合接口 smoke，不可解除
  Case05 阻塞。
- `EDH-DX01962-02052`：数字藏经洞公开 IIIF 的真实拼合图，体积小，适合
  接口 smoke。
- `SCIENCEDB-S4901`：ScienceDB 真实缀合组的高分辨率独立残片。

北敦12288、5753本地文件实际是 `IMAGE TEMPORARILY UNAVAILABLE` 占位图，
不上传、不运行 AutoHDR；只保留论文证据和人工下载队列。

## 禁止上传

- `.venv/`、`cache/`、`logs/`、`outputs/`；
- `artifacts/rag_v2/`、`data/cbeta_v2/`（RAG 已在本地完成）；
- 全量 `data/raw/`；
- 未经许可的馆藏原图、用户账户信息、Cookie、API Key；
- Mock/Replay restored 图作为 Official 输入或 ground truth。

## 人工数据

当前 172 条释文任务均未完成。任何 OCR/AutoHDR/RAG 候选都不能预填
`manual_confirmed`；Official 输出回传后仍需专家复核。
