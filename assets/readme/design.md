# README visual direction

## Project story

- Audience: 局域网内使用浏览器的内容工作者，以及接入 HTTP / MCP 的 Agent 开发者。
- One-sentence value: 把一台 Windows + NVIDIA + WSL 电脑变成可回听、校对、导出字幕的共享转录服务。
- Primary proof: 2026-10-25 新版工作台截图、63 项线上检查和有限参考文本样本；界面压力样本不作为 ASR 质量证据。
- First successful action: 已部署用户登录工作台、上传素材并导出；新部署者先完成 GPU 与对齐验收、构建网页、创建管理员。
- Visual theme: 声笺工作台的 HCDS 雾蓝、纸白界面，音频波形与可编辑文字之间的转换。

## Art direction

- Palette: paper `#F7F7F4`; ink `#343A46`; atmosphere `#BECBEB`; accent `#0C91FA`; warm `#EBCB75`; muted `#787C83`. 与用户 myContentMachine 的 HCDS Brand DNA 1.3.0 一致。
- Typography: body sans (`MiSans`, `PingFang SC`, `Microsoft YaHei`); system mono for ports and API paths. Hero title 82; section title 44; core diagram text 28–36; supporting labels 22–24 SVG units.
- Shape: 1200-unit canvas; 24-unit outer radius; 2-unit rules; 8-unit spacing grid. Explicit opaque backgrounds support GitHub light and dark surfaces.
- Motif: 与 `frontend/src/icons.tsx` 一致的五线波形；用流程与共享推理关系表达产品，而非装饰图案。
- Composition: technical editorial; one title and workflow, one service map, followed by searchable Markdown. README 图示保持静态；产品本身使用 GSAP 动效。

## Assets and sources

- `workbench-hero.svg`: 项目标题、平台与上传—转录—校对—导出的概念流程，不是运行状态或识别证据。
- `service-map.svg`: 浏览器及 Agent 两种入口、同端口 FastAPI、共享 FIFO、WSL GPU 的简化架构。
- UI screenshot: `docs/validation-artifacts/2026-10-25-workbench-ui/upload-final.png` 与 `long-desktop.png`。测试账号与构造界面样本，图注标明来源及证据边界。
- UI tokens: `frontend/src/styles.css` and `frontend/src/icons.tsx`.
- Product facts: `AGENTS.md`, `docs/product-workbench.md`, `docs/asr-server-prd.md`, deployment scripts and source code.
- Previous `hero.svg` is retained as an existing source asset; the current README uses the new workbench hero.

## Maintenance

SVG files are editable source. Keep commands, limits, authentication and recovery details in Markdown. Inspect visuals at 900px and 360px; mobile readers must have the same information in adjacent text. No remote fonts, scripts, embedded raster layers or essential animation.
