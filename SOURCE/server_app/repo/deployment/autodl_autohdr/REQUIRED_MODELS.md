# Official AutoHDR 必需模型

当前均为 `not_downloaded`。来源只使用固定上游 README 的 Model Zoo。

| 组件 | 上游下载标识 | 服务器目标位置 | 当前状态 |
|---|---|---|---|
| AutoHDR-Qwen2-1.5B（优先 smoke） | BaiduYun `W2wq` | `third_party/AutoHDR/ckpt/AutoHDR-Qwen2-1.5B/` | MISSING |
| AutoHDR-Qwen2-7B（上游默认代码路径） | BaiduYun `6o84` | `third_party/AutoHDR/ckpt/AutoHDR-Qwen2-7B/` | MISSING |
| Damage Localization Model | BaiduYun `2QC7` | `third_party/AutoHDR/ckpt/damage_detect.pth` | MISSING |
| OCR Model | BaiduYun `1X88` | `third_party/AutoHDR/dist/det_model/` 与 `dist/reg_model/` | MISSING |
| OCR detection weights（随官方 OCR 包核实） | 官方 OCR 包 | `third_party/AutoHDR/ckpt/best.pt` | MISSING |
| DiffHDR | BaiduYun `63a3` | `third_party/AutoHDR/ckpt/unet/` | MISSING |

已存在的小型配置：

- `ckpt/damage_detect.py`
- `ckpt/dic_31524.txt`

注意：本项目已有的
`checkpoints/diffhdr/unet/diffusion_pytorch_model.safetensors`
属于既有 DiffHDR baseline，不能在未验证目录结构和上游兼容性的情况下冒充
Official AutoHDR Model Zoo 的完整 DiffHDR 包。

所有模型下载后都应运行：

```bash
find third_party/AutoHDR/ckpt third_party/AutoHDR/dist -type f -print0 \
  | sort -z | xargs -0 sha256sum > deployment/autodl_autohdr/OFFICIAL_MODEL_SHA256SUMS.txt
```

该清单在真实下载前不得预填。
