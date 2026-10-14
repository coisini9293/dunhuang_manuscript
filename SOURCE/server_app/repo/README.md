# 骨肉重聚 · 李盛铎旧藏敦煌残卷智能缀合与可读化修复

> **骨肉重聚** (Gurou) — AI-assisted rejoin (缀合) and readable restoration (可读化修复) of the
> Li Shengduo (李盛铎) collection of fragmented Dunhuang manuscripts.

---

## 1. 项目概述

本项目面向敦煌残卷的**智能缀合**与**证据约束的可读化修复**两大任务：

- **缀合 (Rejoin)**：判断两片残卷是否同源、估计相对位姿，并拼接全景图。
- **可读化修复 (Readable Restoration)**：在拼接图像上由用户在全景图标注缺损区域，
  借助 DiffHDR 模型与 CBETA 经文 RAG 证据，将模糊/缺损的汉字修复为可读文字。

项目采用**单一统一前端**（官方敦煌视觉主题）+ **FastAPI 后端**，内置两条业务管线，
且严格区分 **Mock（占位/演示）** 与 **Official（真实模型/真实 LLM）** 两种运行模式。

> 注：本项目已完成目录整理，原先的 `归档/` 前缀已消除，所有源码位于仓库根目录
> （见第 3 节目录结构）。相关迁移与清理记录见 `migration_reports/`。

---

## 2. 功能模块

| 模块 | 名称 | 后端路由族 | 启用条件 |
|------|------|-----------|----------|
| **模块一** | 同源识别与智能缀合 (Rejoin + Legacy DiffHDR) | `/api/rejoin`, `/api/restore` | `ENABLE_LEGACY_DIFFHDR=true` |
| **模块二** | 古卷证据约束修复 (AutoHDR + CBETA RAG) | `/api/jobs/...` | 默认开启（`AUTOHDR_BACKEND=mock`） |

- **模块一**：上传残卷 A / B → 获取缀合置信度、位姿、判定与拼接全景图 → 在全景图上手绘标注
  缺损区域 → 执行 DiffHDR 修复，对比前后。
- **模块二**：上传残卷 → 分析 (AutoHDR 检测缺损 + OCR) → CBETA 证据融合
  (AutoHDR / CBETA / Qwen 候选) → 人工逐处确认 (仅 `manual_confirmed` 进入修复) →
  证据约束修复并导出任务包。

---

## 3. 目录结构

```
.
├── demo/                  # 前后端（统一前端 + FastAPI 后端）
│   ├── backend/main.py    # FastAPI 应用（所有真实路由）
│   └── frontend/          # index.html + static/{app.js, styles.css}
├── pipelines/             # Rejoin + DiffHDR 完整管线 (full_pipeline.py)
├── models/                # Rejoin 模型定义 (ResNet18)
├── integrations/          # AutoHDR 后端抽象 + CBETA 证据融合
├── agents/                # RAG 决策 agent (rag_agent.py)
├── core/                  # 任务存储 + 状态机 (job_store, state_machine)
├── utils/                 # CBETA 索引、掩码生成、文本渲染等
├── clients/               # AutoHDR 客户端
├── services/              # autohdr_worker 服务
├── checkpoints/           # rejoin.pth（已跟踪）；diffhdr/unet/*.safetensors（已忽略，体积大）
├── data/                  # cbeta / raw(数据集) / cases / evaluation / llm_cache
├── outputs/               # 评测与验证产出（local_readiness 含测试结果）
├── scripts/               # 脚本（含 test_api.py 连通性测试）
├── tests/                 # pytest 用例 + fixtures/
├── docs/                  # 架构/审计/案例文档 + project-intro.html
├── deploy/                # autodl / mac 部署配置
├── environment/           # 兼容性报告等
├── third_party/           #  vendored AutoHDR / HDR（见 THIRD_PARTY_MANIFEST.json，已忽略）
├── migration_reports/     # 迁移与清理过程报告（Phase 0–5）
├── THIRD_PARTY_MANIFEST.json
├── README.md
├── requirements.txt
├── environment.yml
└── .env                   # 本地环境变量（已忽略，严禁提交密钥）
```

---

## 4. 环境要求

- **Python** 3.11（见 `environment.yml`，推荐用 conda 创建 `gurou-rejoin` 环境）
- **PyTorch** ≥ 2.1（CPU 可运行；CUDA 可选以加速 DiffHDR）
- **FastAPI / Uvicorn**（见 `requirements.txt`）
- OpenCV、NumPy、Pillow、python-dotenv
- 前端无需构建：HTML/CSS/JS 静态托管，GSAP 通过 CDN 加载

---

## 5. 安装依赖

```bash
conda env create -f environment.yml
conda activate gurou-rejoin
pip install -r requirements.txt        # 若仓库提供
```

> DiffHDR 权重位于 `checkpoints/diffhdr/unet/`（`*.safetensors` 体积较大、**已被 .gitignore 忽略**，
> 需自行放置或经 `utils/restore_paths.ensure_diffhdr_ready()` 准备）。Rejoin 权重 `rejoin.pth` 已随仓库跟踪。

---

## 6. 配置（`.env`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 空 | 真实 LLM（通义千问）密钥；**仅从环境变量读取，禁止写入代码或提交** |
| `LLM_ENABLED` | `false` | `true` 时 RAG 的 L3 才调用真实 LLM，否则返回 mock 结果 |
| `AUTOHDR_BACKEND` | `mock` | `mock` / `replay` / `official`；决定模块二后端实现 |
| `ENABLE_LEGACY_DIFFHDR` | `false` | `true` 时挂载模块一 `/api/rejoin`、`/api/restore` |
| `RESTORE_CONTENT_SOURCE` | `original` | `original` / `manual` / `rag`（修复文字来源） |

> **安全**：仓库 `.gitignore` 已屏蔽 `.env`、`*apiKey*.csv`、`*业务空间*.csv`。任何含活密钥的导出文件
> 严禁入库或打包；前端在 Mock 模式下会显示 `MOCK` 角标。

---

## 7. 快速开始（运行后端）

```bash
# 模块一（Legacy DiffHDR）需要显式开启：
ENABLE_LEGACY_DIFFHDR=true python demo/backend/main.py
# 模块二默认开启（AUTOHDR_BACKEND=mock）。
```

启动后访问 <http://localhost:8000/>，通过顶部标签页在「同源识别与智能缀合」与
「古卷证据约束修复」之间切换。

---

## 8. 前端使用

- 顶部导航 **两个标签页** 对应两大模块，共享同一官方敦煌视觉主题（`demo/frontend/static/styles.css`）。
- **模块一**：上传两片残卷 → 「开始缀合分析」→ 查看置信度环与拼接全景 → 在「标注缺损区域」
  画布上涂抹 → 「执行 DiffHDR 修复」→ 拖动滑块对比前后。
- **模块二**：上传残卷 → 依次经历 上传 / 分析 / 证据 / 确认 / 修复 五步；每一步由后端
  `/api/jobs/{run_id}/...` 驱动；确认时仅「人工确认」(`manual_confirmed`) 或「不修复」(`abstain`)
  会被后端接受，空文本或非法来源将被拒绝。
- 当后端处于 Mock 模式时，页面右上角显示 **MOCK** 角标；Mock 修复图带 `"MOCK RESTORATION"` 水印。

---

## 9. API 参考

| 方法 | 路径 | 说明 | 门控 |
|------|------|------|------|
| GET | `/api/system/health` | 健康检查（返回后端类型与 `is_mock`） | 常开 |
| POST | `/api/jobs` | 创建任务（multipart `image`） | 常开 |
| GET | `/api/jobs/{run_id}` | 获取任务状态与字段 | 常开 |
| POST | `/api/jobs/{run_id}/analyze` | 缺损检测 + OCR | 常开 |
| POST | `/api/jobs/{run_id}/evidence` | CBETA 证据融合 | 常开 |
| POST | `/api/jobs/{run_id}/confirm` | 提交人工确认决策 | 常开 |
| POST | `/api/jobs/{run_id}/restore` | 证据约束修复 | 常开 |
| GET | `/api/jobs/{run_id}/report` | 任务 JSON 报告 | 常开 |
| GET | `/api/jobs/{run_id}/download` | 任务 ZIP 包 | 常开 |
| POST | `/api/rejoin` | 缀合分析（返回 fragment/panorama 图） | `ENABLE_LEGACY_DIFFHDR` |
| POST | `/api/restore` | Legacy DiffHDR 修复（返回 before/after/mask 图） | `ENABLE_LEGACY_DIFFHDR` |

> 完整契约见 `migration_reports/03_api_route_map_after.json`。`/api/confirm` 与
> `/api/restore-with-rag` 仅为调试桩，默认（未开 `ENABLE_LEGACY_DEBUG`）返回 404。

---

## 10. 模型与权重

- `checkpoints/rejoin.pth`：缀合二分类 + 位姿回归（ResNet18），随仓库跟踪。
- `checkpoints/diffhdr/unet/`：DiffHDR UNet 权重（`*.safetensors`，约 454 MB，**已忽略**）。
- `third_party/AutoHDR`、`third_party/HDR`：vendored 第三方仓库，来源与许可见
  `THIRD_PARTY_MANIFEST.json`（其嵌套 `.git` 已在清理阶段移除，不应作为子模块提交）。

---

## 11. 数据

- `data/cbeta/`：CBETA T0278 经文解析结果与 BM25 索引，供 RAG 证据检索。
- `data/raw/`：原始数据集（如 `sciencedb` 数据集 ZIP 已移出仓库；解压目录已忽略）。
- `data/cases/`：案例元数据（如第 05 号案例）。
- `data/evaluation/`、`data/llm_cache/`：评测产出与 LLM 调用缓存。

---

## 12. 测试与验证

```bash
pytest tests/                       # 单元/契约测试（fixtures 在 tests/fixtures/）
python scripts/test_api.py          # API 连通性测试（结果写入 outputs/api_test/）
```

- `outputs/local_readiness/`：本地就绪验证结果，含 `unit_test_results.json`、
  `security_audit.json`、`contract_test_results.json`、`local_readiness_report.md`。
- `outputs/` 下其余子目录为评测/审计过程产出。

---

## 13. 安全与合规

- **无密钥入库**：`.gitignore` 屏蔽 `.env` 与 `*apiKey*.csv`；历史上发现的含活密钥导出文件
  已移至回收站，未进入任何交付物。
- **Mock ≠ Official**：前端在 Mock 模式显示 `MOCK` 角标；Mock 修复图带水印；`/api/system/health`
  暴露 `is_mock` 供前端区分。
- **仅人工确认进入修复**：后端对 `/api/jobs/{id}/confirm` 做服务端再校验——只有
  `status=="manual_confirmed"` 且文本非空（AutoHDR 来源还需命中候选）才被接受；`abstain`/空文本被拒绝。
- **第三方许可**：`third_party/` 下仓库的许可归原上游所有，见 `THIRD_PARTY_MANIFEST.json`。

---

## 14. 致谢与许可

- 敦煌研究院、ScienceDB、IDP 提供文献与图像资源；CBETA 提供经文语料。
- DiffHDR、AutoHDR（SCUT-DLVCLab）、HDR（yeungchenwa）等开源工作的模型与代码。
- 缀合分类器基于 ResNet18；RAG 证据融合结合 CBETA 本地检索与同文异本库。
- 本仓库代码以项目自有许可发布；第三方组件许可见其各自上游。

---

*迁移与清理过程记录见 `migration_reports/`；面向 Codex 的交接文档见 `CODEX_HANDOFF.md`。*
