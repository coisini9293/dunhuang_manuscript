# AutoHDR 官方代码审计

## 基本信息
- Repo: SCUT-DLVCLab/AutoHDR
- Commit: d92b223b3d9274b03ca6f1e4c62b09253f378c98
- Branch: main
- Paper: arXiv 2507.05108

## 推理入口
1. : CLI 模式
2. : API 模式（含可视化）
3. : Gradio Web 界面

## CLI 参数推测
- --source: NOT VERIFIED (pending real run)
- --output: NOT VERIFIED (pending real run)
- --model_path: 模型权重路径
- --img_size: NOT VERIFIED (pending real run)

## 输入要求
- 单张图像，支持 PNG/JPG
- 无需特定分辨率（内部 resize）

## 模型加载: 需AutoDL实测确认加载顺序和常驻策略
1. Damage Detection: mmdet + 自定义检测器
2. OCR: 自定义识别模型
3. Text Prediction: Qwen2 (1.5B/7B)
4. Restoration: DiffHDR

## 各阶段输出
- Stage1 (Damage+OCR): bbox + char recognition
- Stage2 (Text): Qwen2 Top-k candidates
- Stage3 (Restore): DiffHDR restored image

## 能否单独调用
- 官方为单体管道，不可单独调用各阶段
- 需在 display/restore 阶段做拦截

## 硬编码 CUDA
- device = torch.device(cuda if torch.cuda.is_available() else cpu)
- mmdet 和 Qwen2 需要 CUDA

## AutoDL 环境要求
- Python 3.10+
- PyTorch 2.x + CUDA 11.8+
- mmdet, opencc, zhconv, diffusers, transformers
- 最低显存: 官方未声明, 等待AutoDL实测
