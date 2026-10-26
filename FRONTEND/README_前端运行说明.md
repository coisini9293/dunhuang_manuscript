# 敦煌项目｜前端源码整理版

本目录由 `C:\Users\95788\Desktop\preview-sandbox` 整理复制而来，原始 `preview-sandbox` 未移动、未删除、未修改。

## 运行方式

本前端是静态 HTML/CSS/JavaScript 预览包，不需要 Node.js、npm 或虚拟环境。建议在本目录打开终端后运行：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000/`。

直接双击 HTML 也可以查看部分页面，但本地 HTTP 服务更适合加载图片、视频和页面资源。

## 页面入口

- `index.html`：项目首页
- `history/index.html`：藏经洞历史时间线
- `rejoin.html`：同源识别与智能缀合预览
- `restoration.html`：古卷证据约束修复预览
- `about.html`：项目说明、数据来源与学术边界

## 目录说明

- `static/`：当前页面实际使用的 CSS 和 JavaScript
- `docs/legacy-static/`：预览过程中保留的旧版或未被当前页面直接引用的前端脚本/样式，作为源码留档，不是当前运行必需项
- `assets/`：页面运行所需的图片、视频、回放结果和来源清单
- `history/`：历史总览及历史节点详情页
- `scripts/`：前端素材辅助脚本
- `docs/`：模块迁移边界、预览总结和视频录制说明

## 后端边界

本目录只包含前端。缀合、文献检索、修复、任务状态和服务器 API 需要连接项目服务器；离线打开时只能展示静态界面和已保存的 replay 结果，不能据此宣称完成新的后端计算。

页面中的生成环境图仅用于界面氛围，不属于敦煌文献证据；真实素材、回放结果和来源信息以 `assets/ASSET_SOURCES.md` 及 `assets/asset_manifest.json` 为准。
