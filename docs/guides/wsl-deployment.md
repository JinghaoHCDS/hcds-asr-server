# Windows + NVIDIA + WSL 部署

把 Windows 电脑的 NVIDIA GPU 变成局域网 ASR 服务：WSL 运行模型，Agent 客户端通过 HTTP 上传音频。唯一局域网 API 端口是 **18080**；客户端无需安装模型或 CUDA。

已完成真实 GPU 验收的参考环境是 **WSL2 Arch Linux + RTX 5070 Ti 16 GB**。其他 NVIDIA 显卡需要在自己的机器上重新通过 CUDA 和模型 smoke；Ubuntu 下文提供依赖准备参考，尚未完成该发行版的项目 GPU 验收。

部署后从局域网访问 `/skill/hcds-lan-asr.zip`，包内 `server.json` 会自动带上该部署的访问地址。
默认从实际下载 URL 获取，避免把 WSL NAT 内部地址当作 Windows LAN 地址。
需要固定公布地址时，可传 `--public-base-url http://YOUR_WINDOWS_LAN_IP:18080`，
或在项目 `.env` 中设置 `ASR_PUBLIC_BASE_URL`（后者优先）。
显式命令行和 `ASR_BASE_URL` 仍可覆盖客户端的随包地址；修改地址后应重新下载包。

## 1. 准备 Windows 和 WSL

已有 WSL 的用户，在 Windows PowerShell 查看发行版名称和版本：

```powershell
wsl --list --verbose
wsl --version
```

使用 WSL2，Windows 安装支持 WSL CUDA 的 NVIDIA 驱动。WSL 内不要另装 Linux 显卡驱动；这里运行官方 PyTorch CUDA wheels，无需为普通推理安装完整 CUDA Toolkit。[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

常驻服务需要 systemd。确认 `/etc/wsl.conf` 包含以下配置；合并进已有文件，保留其他设置：

```ini
[boot]
systemd=true
```

修改 WSL 配置后，在没有其他重要 WSL 任务运行时，从 PowerShell 执行 `wsl --shutdown`，再进入目标发行版。运行：

```bash
systemctl --user show-environment >/dev/null
nvidia-smi
```

第一条失败时先解决 systemd user manager，第二条失败时先解决 Windows 驱动和 WSL GPU 可见性。[微软 systemd 指南](https://learn.microsoft.com/en-us/windows/wsl/systemd)

## 2. 在 Linux HOME 里克隆

安装到 WSL 原生文件系统，例如 `$HOME/services/hcds-asr-server`。部署脚本默认直接使用当前 checkout，支持不同用户名和目录，不向某个固定用户目录复制。

Arch Linux：

```bash
sudo pacman -Syu --needed git uv rsync ffmpeg libsndfile
mkdir -p "$HOME/services"
git clone https://github.com/JinghaoHCDS/hcds-asr-server.git "$HOME/services/hcds-asr-server"
cd "$HOME/services/hcds-asr-server"
```

Ubuntu 依赖准备参考（尚未 GPU 实测）：

```bash
sudo apt-get update
sudo apt-get install -y git rsync ffmpeg libsndfile1 build-essential
# 如尚未安装 uv，使用官方安装器后重新打开终端
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ubuntu 再执行同样的 `mkdir`、`git clone` 和 `cd`。后续部署命令使用 `--skip-system-packages`，避免让脚本尝试处理发行版系统包。[uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)

## 3. 首次部署 Qwen

准备一小段自己有权使用的语音，例如 `$HOME/audio/smoke.wav`，包含清晰讲话；仓库提供短合成英文样例 `test-fixtures/audio/test_short.wav`，来源、规格与再生成方式见[样例说明](../../test-fixtures/audio/README.md)，许可见[第三方归属](../../THIRD_PARTY_NOTICES.md)。当前合成样例尚未取得独立真实 GPU 验收，不能套用旧录音的历史结果。首次部署会下载运行时依赖和模型，需联网并留出充足磁盘空间。

```bash
bash deploy/wsl-deploy.sh \
  --skip-system-packages \
  --smoke-audio "$HOME/audio/smoke.wav"
```

脚本依次完成：

1. 确认真实部署运行在 WSL，使用 Linux 原生目录和 `18080`。
2. 安装 Python 3.12，执行 `uv sync --frozen`，跑 mock 单元测试与 mypy。
3. 从 PyTorch 官方 **cu128** 源安装 `requirements/wsl-gpu-cu128.txt`；检查 `nvidia-smi`、CUDA torch 和 HF native Transformers 类。
4. 用 `Qwen/Qwen3-ASR-0.6B-hf` 跑独立后端 smoke，要求非空文本。
5. 从 `deploy/asr-server.service` 模板生成当前用户的 unit，启动 API；HTTP smoke 再用同一段音频验证默认 `qwen3-asr-1.7b`。

MOSS 和 Qwen 强制对齐默认关闭；首次部署只验收 Qwen 转录。部署模板不要直接 `cp` 到 systemd 目录，它含有需要渲染的路径占位符。

确认启动：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:18080/health
curl --noproxy '*' -fsS http://127.0.0.1:18080/v1/models
```

常驻服务直接运行当前 `.venv/bin/uvicorn`，避免每次开机通过 `uv run` 同步依赖。默认允许首次加载时从 Hugging Face 下载，解决只预热 0.6B 却调用默认 1.7B 的缓存缺失问题；模型在最后一次转录完成后空闲 180 秒自动卸载。

只想准备环境和验收模型、暂不安装常驻服务：

```bash
bash deploy/wsl-deploy.sh --skip-system-packages \
  --smoke-audio "$HOME/audio/smoke.wav" --no-enable-service
ASR_ADAPTER=qwen .venv/bin/uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

## 4. 启用 MOSS 播客转录

运行时依赖已随部署安装，但 MOSS 模型只有在本机独立 smoke 通过后才可以进入 `/v1/models`：

```bash
.venv/bin/python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio "$HOME/audio/smoke.wav" \
  --language auto
```

确认输出非空文本和可解析 segments，再检查 CUDA 没被依赖替换：

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.version.cuda is not None
assert torch.cuda.is_available()
PY
```

修改 checkout 中的 `.env`，设置 `ASR_ENABLE_MOSS=1`；这个文件被 systemd 加载，且不提交 Git。如文件不存在可从 `.env.example` 复制，保留已有配置。然后：

```bash
systemctl --user restart asr-server.service
curl --noproxy '*' -fsS http://127.0.0.1:18080/v1/models
```

用短音频确认服务接口返回 segments：

```bash
curl --noproxy '*' -fsS \
  -F "file=@$HOME/audio/smoke.wav" \
  -F model=moss-transcribe-diarize-0.9b \
  -F response_format=verbose_json \
  http://127.0.0.1:18080/v1/audio/transcriptions
```

长播客使用 `POST /v1/audio/transcription-jobs`，读取创建响应的 `id` 作为 `GET /v1/jobs/{id}?detail=compact` 的路径参数；完成后用 `detail=full` 取一次最终 `result.text`。

MOSS 只声明 `transformers` 后端和可靠的 `language=auto`。`segments` 包含段级时间与匿名 speaker，不代表逐字时间戳。普通 `purpose=transcription` 且 `speaker_resolution=off` 时，`auto` 在不超过 1801 秒的已验收窗口内使用原生长音频；更长输入退回 1800 秒分块，speaker 身份只保证块内一致。

传入 `purpose=podcast` 会选择播客参数，并默认启用 `speaker_resolution=auto` 的 Anchor Replay；也可显式传 `speaker_resolution=required`，要求无法完整解析时返回错误。Anchor Replay 使用 1200 秒正文窗口和最多 60 秒参考前缀，未显式指定时自动生成预算至少 24000；ID 只对当前 job 有效。只有响应 `speaker_scope=global` 才把标签视为整期一致，`mixed` / unresolved 片段应保留未知状态。

截断或缺尾会受控报错，不能把完成百分比当成转录准确率。71 分钟四人样本产生了完整正文，但片头短句 unresolved，结果为 `partial/mixed`；4–5 小时整期 speaker 一致性仍未完成实测。自动预算下限修改也不能沿用旧样本成功结果作为新预算验收。能力及当前限制以 `/v1/models` 为准；验证证据见 [最新升级记录](../validation/2026-09-19-upgrade.md)、[71 分钟真实播客记录](../validation/2026-07-15-live-chunks-four-speaker-podcast.md)、[MOSS 长音频记录](../validation/2026-07-15-wsl-moss-long-form.md) 和 [Anchor Replay 记录](../validation/2026-07-15-wsl-moss-anchor-replay.md)。

## 5. 局域网连通

推荐使用 Windows 11 22H2 或更高版本的 WSL mirrored networking。在 Windows 的 `%UserProfile%\.wslconfig` 中合并：

```ini
[wsl2]
networkingMode=mirrored
```

重启 WSL 配置后，在管理员 PowerShell 中添加专用网络入站规则，只开放 `18080`：

```powershell
New-NetFirewallRule -DisplayName "ASR Server LAN" `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 18080 `
  -Profile Private -RemoteAddress LocalSubnet
```

如果 WSL Hyper-V 防火墙仍阻拦该端口，可按微软说明添加针对 WSL 的单端口规则：

```powershell
New-NetFirewallHyperVRule -Name "ASRServer18080" -DisplayName "ASR Server WSL" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol TCP -LocalPorts 18080
```

找出 Windows 当前的局域网 IPv4 地址（PowerShell `ipconfig`），从另一台同网设备访问：

```bash
# 替换为 Windows 主机的 LAN IP
ASR_BASE_URL=http://YOUR_WINDOWS_LAN_IP:18080
curl --noproxy '*' -fsS "$ASR_BASE_URL/health"
curl --noproxy '*' -fsS "$ASR_BASE_URL/v1/models"
```

示例 IP 需要替换，建议给 Windows 主机设置路由器 DHCP 地址保留。localhost 成功只证明本机服务正常，第二台设备的请求才能证明 LAN 可达。Windows 默认 NAT 环境没有本项目的 LAN 验收；本指南采用 mirrored，不添加端口代理或公网映射。[微软 WSL 网络文档](https://learn.microsoft.com/en-us/windows/wsl/networking)

## 6. 可选鉴权与离线模型缓存

systemd 读取 checkout 下可选的 `.env`。支持的服务设置包括：

```dotenv
ASR_ENABLE_MOSS=0
ASR_ENABLE_QWEN_ALIGNMENT=0
ASR_IDLE_UNLOAD_SECONDS=180
ASR_MAX_UPLOAD_MB=512
ASR_MAX_QUEUED_JOBS=20
ASR_JOB_RESULT_TTL_SECONDS=3600
```

修改后重启 unit。需要 Bearer Token 时设置 `ASR_API_TOKEN`，并用 `chmod 600 .env` 限制文件权限；`/v1/**` 请求增加 `Authorization: Bearer <token>`，健康检查保持公开。

HTTP smoke 从进程环境读取 `ASR_HTTP_SMOKE_TOKEN`（也接受 `ASR_API_TOKEN`），不自动加载 `.env`。对已开启鉴权的部署重跑检查时，需在当前终端注入与服务端一致的 token。`scripts/asr_client.py` 使用客户端环境变量 `ASR_TOKEN`，MCP bridge 使用 `ASR_MCP_TOKEN`；不要把真实 token 写进提交或命令示例。

离线服务是可选模式。先在线 smoke 预热 **每个要使用的模型**（Qwen 两个尺寸、已启用的 MOSS，以及可选对齐器），再在 `.env` 中增加：

```dotenv
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

也可在部署时传 `--offline-service`；该选项本身不会预下载所有模型。Qwen 默认使用 native `AutoProcessor` + `AutoModelForMultimodalLM`，不依赖默认 `qwen-asr` 包加载路径，不声明 vLLM。

Qwen 强制对齐必须先独立完成 GPU smoke 和 Context/反幻觉验收，然后才能设置 `ASR_ENABLE_QWEN_ALIGNMENT=1`：

```bash
.venv/bin/python scripts/qwen_forced_alignment_smoke.py \
  --audio "$HOME/audio/smoke.wav" --asr-model Qwen/Qwen3-ASR-0.6B-hf --language auto
.venv/bin/python scripts/qwen_asr_backend_smoke.py \
  --backend transformers --audio "$HOME/audio/smoke.wav" \
  --prompt 'Reference vocabulary only; do not invent: ABSENTTERMZXQ' \
  --reject-hallucinated-term ABSENTTERMZXQ
```

## 7. 日常启动、停止和更新

当前 schema 4→5 保留全部任务和 MP3，不运行旧清库工具。具体版本、停写备份与匹配回滚见[升级指南](upgrading.md)。以下 Git 更新只用于 checkout；无 `.git` 的发行目录在新目录解包校验后切换，不执行 `git pull`。

在 WSL checkout 中：

```bash
bash scripts/start_asr_server.sh
bash scripts/shutdown_asr_server.sh
systemctl --user status asr-server.service --no-pager
journalctl --user -u asr-server.service -n 80 --no-pager
```

启动脚本确认 unit 和 health；停止脚本确认 unit inactive 和 health 不再响应。模型卸载、job 取消和服务停止遵循已有生命周期约束。服务重启会丢失内存 job，更新前等正在执行的任务结束。

更新源码后保留已经安装的 GPU 包，并重新做 CUDA 和模型验收：

```bash
git pull --ff-only
bash deploy/wsl-deploy.sh --skip-system-packages --skip-gpu-deps \
  --smoke-audio "$HOME/audio/smoke.wav"
```

`--skip-gpu-deps` 使用 `uv sync --frozen --inexact`，避免清掉已安装的推理包；如果 requirements 改变，应重新执行完整部署。不要并行修改同一个 `.venv`。指定另一处部署目录可用 `--deploy-dir "$HOME/services/asr-production"`；复制排除虚拟环境、缓存、音频运行数据和 `.env`，不会删除目标已有文件。

开发机仅验收 mock 部署流程可用 `--mock`；这不证明真实 CUDA 或模型可用。

## 8. Windows 登录后自动启动

先在 WSL 完成正常部署，并允许用户 manager 在登录会话外运行：

```bash
sudo loginctl enable-linger "$USER"
```

将 `deploy/windows-start-asr.ps1` 复制到 Windows 的固定位置，例如 `C:\tools\asr\windows-start-asr.ps1`。任务计划程序创建**当前 Windows 用户登录时**任务，程序为 `powershell.exe`，参数例如：

```text
-NoProfile -ExecutionPolicy Bypass -File "C:\tools\asr\windows-start-asr.ps1" -Distro "archlinux" -LinuxUser "your-linux-user" -ProjectDir "/path/to/user/services/asr-server"
```

发行版名称来自 `wsl --list --verbose`；用户名和路径来自自己的 WSL。省略 `-Distro` 使用默认发行版，省略 `-LinuxUser` 使用该发行版默认用户，省略 `-ProjectDir` 使用该用户的 `$HOME/services/hcds-asr-server`。

任务应允许长期运行、关闭自动运行时长上限，并设置已有实例运行时不启动新实例；可为失败配置一分钟后重试。Windows 脚本启动已经安装的 systemd unit，保留一个 WSL 前台等待进程，直到 unit 停止。仅启用 systemd 不能保证 WSL 实例一直存活，这是微软明确的 WSL 生命周期限制。手动调用且不需要保持进程时可传 `-NoKeepAlive`。[微软 systemd 生命周期说明](https://learn.microsoft.com/en-us/windows/wsl/systemd)

当前脚本已完成本地静态审查；Windows 任务计划程序及本次产品化后的 WSL 常驻路径需在目标机器重新验收。

## 9. 故障定位

| 现象 | 检查与处理 |
| --- | --- |
| `torch.version.cuda` 是 `None` | 装到了 CPU torch；停止服务，将错误 `.venv` 移走后按完整部署重建，保留模型缓存 |
| `nvidia-smi` 成功，CUDA tensor 仍失败 | 检查 CUDA wheel 与显卡架构；新卡需要支持它的官方 CUDA wheel，不退回 CPU torch |
| 模型加载失败，日志指向离线缓存 | 临时取消 offline 设置，在线完成该模型 smoke，再恢复离线 |
| systemd 显示 active，health 不通 | 查看 journal，确认当前 checkout 的 `.venv/bin/uvicorn` 路径和 `18080` 监听 |
| localhost 成功，LAN 不通 | 核对 mirrored、Windows LAN IP、专用网络及 Hyper-V 单端口规则 |
| MOSS 未出现在模型发现中 | 完成 MOSS smoke，确认 unit 读取的 checkout `.env` 有 `ASR_ENABLE_MOSS=1` 并重启 |
| 长任务查不到 job | job 只保存在内存，重启或 TTL 到期后须重新提交 |

现有 GPU 验收记录描述当时的硬件与固定模型 snapshot，不等于所有 NVIDIA/WSL 环境兼容保证。API 合约见 [PRD](../reference/asr-server-prd.md)，Qwen 运行时验收见 [HF native 记录](../validation/2026-07-02-wsl-hf-native.md)。
