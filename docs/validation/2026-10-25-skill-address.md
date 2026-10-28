# 2026-10-25 Skill 下载包自动配置服务地址

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。

本地实现将官方 lan-asr 升级为 1.3.0，增加按部署生成的 `server.json`。仓库模板保留通用内容，不写入个人局域网 IP。以下记录对应本地工程验收，线上发布和局域网下载需要另外核验。

## 行为

- `/skill` 和 ZIP 中的 SKILL.md 从同一模板渲染，显示该部署的地址。ZIP 另含标准库批量客户端和 `server.json`，后者仅有 `base_url`，不含 API Token、Cookie 或其他运行环境值。
- 可用的 `ASR_PUBLIC_BASE_URL` 优先；未设置、loopback 和通配监听地址使用实际下载请求的 URL。这避免将 WSL NAT 内部地址当作 Windows 局域网入口，也不需要为识别 IP 安装依赖或探测其他机器。
- 部署脚本提供 `--public-base-url`，生成对应 systemd 配置；项目 `.env` 可覆盖它。地址配置含凭据或不合法时，文档端点受控返回 `503 skill_address_invalid`。
- 客户端优先级为 `--base-url`、`ASR_BASE_URL`、随包 `server.json`、本机 `config.json`、loopback。随包地址优先于旧机器默认值；损坏的随包配置明确失败。已有批次继续使用已冻结的服务器和请求，不被新配置重定向。
- 带地址的文档和 ZIP 使用 `Cache-Control: no-store`，不同客户端请求不会复用前一个 Host 的生成结果。单独下载的脚本或仓库文件没有随包配置，继续支持显式配置及本机默认值。
- `server.json` 属于生成物，不写入 Git 或服务源码发布包。用户安装 ZIP 时应保留它；IP 变化后重新下载或显式覆盖。

## 验证

完整回归 **422 passed / 4 skipped，23.98 秒**，mypy **119 文件**通过，Skill 验证器和 Bash 语法检查通过。与本次调用链相关的文档、客户端、部署、API 和鉴权检查先行完成 **97 项**。

新行为检查包括：真实解压后的 Python 客户端无参数读取随包地址；未配置/loopback/通配地址下按下载 Host 生成地址；固定部署地址优先于 loopback 下载入口；不同 Host 连续请求不串包；凭据不进入 ZIP；命令行和环境变量覆盖；旧本机默认值不覆盖新安装包；部署地址写入生成的 unit；服务发布包排除生成的 server.json。

默认四项跳过沿用原有真实服务地址及可选 Jinja2 条件。本轮不修改模型推理、任务保留或账号隔离，不提供新的 GPU 质量或吞吐结论。

```bash
uv run --locked pytest -q
uv run --locked mypy asr_server asr_mcp tests scripts
bash -n deploy/wsl-deploy.sh
```
