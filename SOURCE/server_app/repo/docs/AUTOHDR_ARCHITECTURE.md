# AutoHDR 架构文档

## 模块分工

```
缀合模块 (models/rejoin/): 同源判断 + 位姿估计 + 全景拼接
AutoHDR (third_party/AutoHDR/): 损坏定位 + OCR + 文字预测 + 外观恢复
CBETA RAG (utils/multi_scale_index.py): 经文证据检索 + 卷次定位 + 版本校验
人工确认 (integrations/decision_policy.py): 所有真实案例进入恢复前的必要步骤
Qwen API (utils/llm_client.py): CBETA与AutoHDR都无法提供可靠结果时的黄色候选
Legacy DiffHDR (pipelines/full_pipeline.py): 保留为baseline和消融对照
PIL/text_renderer (utils/text_renderer.py): 保留用于旧baseline/debug overlay/可视化
```

## 数据流

```
残卷图像
  → 缀合（可选）→ 全景图
  → AutoHDR Stage1: damage detection + OCR
  → AutoHDR Stage2: missing text candidates
  → CBETA RAG: 经文证据检索
  → 决策融合: evidence_agreement/conflict/ambiguous/autohdr_only/abstain
  → 人工确认: 选择候选/自定义文字/选择不修复
  → AutoHDR Stage3: 外观恢复（仅 manual_confirmed 字符）
  → 结果展示: 原图 + 检测框 + OCR + 候选 + 证据 + 修复图
```

## 部署

- 主项目 (gurou): :8000, Windows CPU
- AutoHDR Worker: :8010, Linux CUDA (独立 conda 环境)
- 通信: HTTP API
