# 2026-10-27 工作台清晰控制台改造与发布

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。

状态：用户选定 A「清晰控制台」后完成改造，并于 2026-10-27 明确授权 Windows/WSL 前端部署。新版已覆盖既有 `18080` 工作台入口；仅更新前端源码、构建与发布记录，没有重启服务、迁移数据库或执行账户/任务写操作。

私人音频、正文、会话、数据库和生产截图没有加入仓库。

## 改造与保留范围

- 保留 React/TypeScript/Vite、HCDS 品牌主色和声笺字标；素材与设置在桌面并排，功能标题替代装饰性标题，任务状态和结果工具层级更清楚。
- 新增 `frontend/src/tokens.css`、`base.css` 和 `frontend/design.md`；控件焦点立即显示，输入边界与核心文字 token 对比度经过检查。
- 补充纯展示加载状态，区分尚未读取、真实空数据和读取失败；部分结果仍遵守完整性和字幕禁用条件。短屏支持工作区滚动，正文与工具区保持独立滚动。
- AST 与改造前源码对照：35 项 API/fetch/jobDetail/确认调用和 181 项事件/禁用/值/引用绑定完全一致。Hallmark kept 检查保留 2/2 links、17/17 fields、96/96 handlers；移除的循环 key 仅属于装饰声纹。
- 后端、API 模块、依赖锁、Vite 配置和官方 ASR 客户端 Skill 未改。项目级第三方 Skill 与安装器锁文件的 Git 忽略规则另以 `3e31051` 提交；Skill 本体没有进入 Git 或生产包。

## 本机验收

`npm run typecheck`、`npm run build` 和 `git diff --check` 通过。Hallmark 静态检查为 0 FAIL、0 REFLEX、0 WARN；原生 `--render` 因缺少 `puppeteer-core` 未运行，没有安装全局依赖，也不将静态结果视为浏览器验收。

使用既有 ego-browser 和隔离本机 mock 服务验证 320×720、390×844、768×1024、1280×800、1920×1080 CSS 视口。登录校验、空任务、文件上传、结果、搜索、时间跳转、编辑新版本、原稿切换、TXT 下载和手机工具切换通过；最初发现的短屏正文高度不足、账户名挤压、搜索框过窄及上传列表内部横向滚动均修复。

1000 段、60 个缺口的构造 UI 样本只挂载 6 个可见片段，远端片段搜索及部分结果字幕禁用通过。该样本和 mock 转录只用于界面验证，不是真实 ASR 数据或长音频质量证据。

## 生产发布

- 应用源码提交：`ac353aa05ac0086a2f02d24a38b028c13a5a1923`。
- 发布包：`workbench-console-20261027-ac353aa.tar.gz`，SHA-256 `1ede59e6b0ec0a14705305512189f9f6ee3fb45168fe55d352c7186a2883ceda`。
- 包含 10 个本轮源码/设计文件及 3 个构建文件，显式包含设计说明；不含依赖目录、Skill、音频、账户或运行数据。
- 运行目录 `/path/to/user/services/asr-server`，为发行目录而非 Git checkout；未在生产执行 `git pull`。19 项前端基线检查全部匹配后才覆盖，避免覆盖未知修改。
- 先落盘新哈希资源，再更新前端源码，最后原子替换 `frontend/dist/index.html`。旧哈希资源保留，入口采用新 mtime/ETag。
- 13 项已部署文件及其实际 HTTP 字节校验通过；LAN 入口、JS、CSS、`/health`、`/readyz` 均为 200。
- 发布前后 PID 均为 `24726`，53 个后端 Python 文件摘要一致。未安装或替换 Node、CUDA、模型或服务依赖。
- 通过 HTTP 与后端检查后原子更新生产根目录的 `deployed-frontend.json`。

线上资源：

| 文件 | SHA-256 |
| --- | --- |
| `index.html` | `86c084f729f97a1efa4cf2527a731edf5dbf0ee6b88a64a1abb8f1243081bc31` |
| `assets/index-DtTsBN0f.js` | `98523b286cb28c3a1de042604d53d1a4556ebd96794a98c4582bf2fc7fddee20` |
| `assets/index-DobOoFVq.css` | `b0ef14876b9637ba0a68680b8eeb3dfce96d0d7dba4b2b59d7499a4ace4d5430` |

## 生产浏览器与验证边界

真实 LAN 浏览器确认加载新 JS/CSS、动作色 `#0865A8`、品牌雾蓝 `#BECBEB`，新建转录页在 1280×800 和 390×844 下没有横向溢出或运行时异常。使用既有会话只读检查个人任务空态，没有提交新音频、创建账户或修改任务。

该生产账户没有任务行，因此结果编辑与导出沿用本机 mock 验收，未称为生产结果内页实测。未新增真实 GPU、长音频准确率/性能或 Safari/Firefox 验收。Ego Lite 提示可升级，本轮未升级。

## 备份与回滚边界

- 旧前端备份：`/path/to/user/services/backups/workbench-console-20261027-ac353aa`，11 个原文件及备份清单完整性已核对。
- 发布目录：`/path/to/user/services/releases/workbench-console-20261027-ac353aa`，保留发布包、清单、激活结果与激活脚本供追溯。
- 旧资源 `index-BOz5fKeL.js`、`index-rMir3oFa.css` 仍在生产静态目录。
- 回滚仅限此次前端：先核对当前发布 ID 和文件摘要，恢复备份源码与发布记录、移除原先不存在的新增源码，最后原子恢复旧入口；两套哈希资源均保留。不恢复或覆盖数据库，不重启推理服务。
- 激活脚本的失败回滚路径经过独立只读审查，临时文件归属保护经过本机验证；本轮没有在生产实际执行回滚。
