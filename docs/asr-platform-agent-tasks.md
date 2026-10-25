# ASR 改进 Agent 任务包

本文件用于后续派发实现任务。设计与验收标准见 [ASR 服务性能与扩展改进计划](asr-platform-improvement-plan.md)。建议一次派发一个完整工作包，不按每个函数拆 Agent。

基线为 `2fb6a4dca0375ad2cbc3a20b2e3aa5b4b71cf852`。开始任务前重新读取当前源码与 git 状态，继承前包已落地的接口。下列新文件名为建议；允许在不扩大职责的前提下调整。

## 所有任务的共同要求

阅读 `AGENTS.md`、`CLAUDE.md`、`docs/asr-server-prd.md`、`docs/product-workbench.md` 和主设计计划。服务端任务参考 `prompts/server-agent.md`，客户端改动参考 `prompts/request-client-agent.md`；这些提示词不构成远程操作授权，已被工作台契约覆盖的历史限制以当前契约为准。

你不是仓库中唯一工作的 Agent。不要回退别人的改动；发现共享文件变动时先适配，再继续。只修改当前包拥有的文件；跨包变更在交接中说明并取得文件所有权后实施。

先完成本地实现及相称验证，再交付可审查结果。本任务包没有预先授权 Windows/WSL 连接、提交、推送或部署。真实 GPU 验收需要用户在当前任务中指定 Windows 或同意明确范围。

报告中的 4 项仍是待验证线索。验收使用小型 dummy fixture 和有界运行，不把报告文字当作已经观察到的故障。不得为了减少测试而删掉未被其他证据替代的用户行为保护。

每包交付：修改文件、接口/契约变化、关键证据、实际运行的检查、未完成项及下一包需要知道的事项。修改运行行为时同步更新契约，不仅更新这两份计划。

## A 日志与性能基线

目标：给定 job ID，Agent 可以定位最后成功阶段、排队/处理耗时、模型切换和 worker 错误，并能判断后续优化有无收益。

输入：当前代码与主计划第 7、9 节。无前置实现依赖。

文件所有权：

- 新增 `asr_server/observability.py`、`scripts/asr_diagnostics.py`。
- 埋点修改 `main.py`、`jobs.py`、`lifecycle.py`、`transcription.py`、`quality/pipeline.py`、`workers/transport.py`、`workbench/service.py`、两个 adapter 的错误响应。
- 配套 `config.py`、`errors.py`、部署日志配置及诊断文档；上述 Python 路径均在 `asr_server/` 下。

实施：

1. 定义稳定 JSONL schema、ExecutionContext、request/trace/job/RPC 分层 ID；把 context 显式传过 FIFO 和 worker。
2. 增加阶段与错误事件、队列耗时及版本摘要，采用单写者、有界异步日志和限额轮转。保留已有错误码及响应格式。
3. 提供只读 trace/bundle 命令；明确敏感字段白名单、截断与丢失计数。
4. 用小型本地 fixture 记录接口、CPU I/O、测试耗时基线及测量方法；GPU 指标留给获得授权后的实机步骤，不填假值。

完成判据：一个失败任务可关联到 HTTP 请求与具体 RPC/worker；错误栈有可用定位；敏感正文和凭据不进入默认日志；日志饱和不阻塞业务；记录可重复的性能基线命令。

最少验证：关联与失败链一组，敏感字段/有界队列一组，受影响 job/worker 回归。日志文案不逐条快照。完成后交接 context 与事件 helper，后包直接复用。

## B 媒体资源边界与单次归一化

目标：关闭无界窗口及解析/解码资源缺口，移除工作台重复规范音频副本。

依赖：A 的事件与 context 接口稳定。对应审查 1、2、3，及第 4 项的格式策略/验证。

文件所有权：

- `asr_server/audio/splitter.py`、`preprocess.py`、`workspace.py`、必要的 `metadata.py`。
- 新增 `asr_server/admission.py`、必要的音频资产类型。
- 集成修改 `main.py`、`config.py`、`transcription.py`、`quality/pipeline.py`、`workbench/service.py`。
- 对应音频/上传测试和 API 文档；本包期间 C 不同时编辑这些文件。

实施：

1. 用采样整数生成窗口，统一前进、总数量和取消检查；参数错误尽早返回。
2. 在解析前增加正文限制、有限接收槽位和 parser 临时文件预算；按实际同时存在的副本计费；覆盖断连/取消清理。
3. 公共 preprocessor 以实际 PCM 字节限制输出、并行读取有限 stderr，超限或取消后回收 decoder；旧 bytes helper 不能绕过。
4. 实现可信 `NormalizedAudioAsset`，工作台 quality 流程借用一次归一化结果；明确文件所有权及 lease 的移交/释放。
5. 共用输入协议/格式策略，验证部署构建的二级引用行为；记录兼容格式和真实拒绝结果。

完成判据：边界窗口有限退出；超限在落盘越界之前阻断；取消后预算归零；工作台正常首次任务只归一化一次，正常恢复复用原资产；处理策略不能把截断音频标为成功。

最少验证：窗口边界矩阵、两条旧上传入口的解析前限制、约 1 MiB 输出预算的小媒体场景、解码取消、归一化复用/借用清理、代表格式兼容及隔离二级引用检查。沿用现有清理回归，避免按每个 helper 补一套测试。

交接：asset/lease/preprocessor 接口、计费公式、错误码、格式矩阵。性能数字区分消除的 I/O 与实测端到端收益。

## C 统一调度与工作台性能

目标：让所有推理共享可解释的 FIFO，避免请求处理线程被数据库/密码/目录扫描占用，控制轮询流量。

依赖：A、B 完成。主计划第 5 节为实现契约。

文件所有权：

- 新增 `asr_server/scheduler.py`，修改 `jobs.py`、`lifecycle.py`、`main.py`。
- `workbench/service.py`、`store.py`、`router.py`、`security.py`，必要的 schema migration。
- `frontend/src/App.tsx`、`ResultPane.tsx`、`api.ts` 及相关列表组件；对应配置/文档与测试。

实施：

1. 统一同步、旧异步、工作台和编辑对齐的 operation；保留各 owner 的状态、权限与恢复方式，明确 unload/load 的仲裁。
2. 将 SQLite 整个 repository 操作移至有界 executor；加入状态摘要、revision、索引及有版本迁移。容量检查采用账本，目录扫描转后台核对。
3. 缩短全局临界区：每上传锁稳定源文件，摘要计算后重新检查幂等与删除状态；密码计算使用独立有界 executor。
4. 新摘要分页接口和前端按需轮询；现有完整详情/导出语义保留，管理员状态只含白名单字段。health 使用后台 GPU 快照。
5. 基础 FIFO 稳定后增加深度 1 的可关闭预取；在已有任务 GPU 推理期间准备下一项，先取得 B 的资源 lease。

完成判据：混合入口只出现一个 GPU 调用；顺序由提交序号决定；卸载意图及时可见；预取取消不留资源；摘要查询不读取完整结果/扫描目录；迁移前后的任务、编辑版本和配额一致。

最少验证：一组混合调度/取消、原有删除与关停回归、迁移+恢复+容量核对、摘要 owner 分页及前端浏览器流程。性能用固定负载单独测，不能把严格毫秒断言放入普通单元测试。

交接：scheduler 所有权、锁顺序、schema 版本、查询协议与真实基线对比。若预取收益不足，交付可用实现并默认关闭，不能以吞吐承诺代替测量。

## D 多 ASR 接入接口

目标：增加一个基础转写引擎时，改动集中在 adapter 与注册定义；核心调度、HTTP 路由和工作台存储不需要认识新模型名称。

依赖：B 的音频资产与 C 的调度接口。主计划第 6 节为设计边界。

文件所有权：

- `asr_server/registry.py`、`execution.py`、`adapters/base.py`；新增内部 engines 注册模块。
- `adapters/qwen.py`、`adapters/moss.py`、`workers/transport.py` 和公共 IPC codec。
- 装配修改 `main.py`、能力校验修改 `transcription.py`、工作台提交/能力协议和 `frontend/src/UploadPane.tsx`、`api.ts`。
- 涉及固定模型源/revision 的 smoke 工具与模型接入文档。

实施：

1. 集中模型身份、revision、adapter factory、输入规格、pipeline 适用性和 capability 声明；保留旧环境配置兼容映射。
2. 提取公共 IPC envelope 与结果 codec，增加协议版本和可选能力协议；模型 prompt/解析继续由模型模块所有。
3. 工作台可选模型/profile，默认行为不变；幂等与恢复身份覆盖新维度。
4. 在测试注入第三基础引擎，验证无对齐能力时的准确拒绝与普通转写成功；编写新模型接入模板。

完成判据：第三引擎不需要在 main/scheduler/store 添加模型分支；Qwen/MOSS 已有模型行为矩阵不变；不支持能力在耗费资源前拒绝；换模型不会命中旧幂等结果。stub 不出现在生产模型列表。

最少验证：统一 adapter/registry 契约矩阵、第三引擎通过 API 与工作台的集成、模型/profile 幂等冲突；继续保留显式语言、空稿、MOSS long-form 等模型特有回归。

交接：接入模板、一个完整 stub 示例、实际新增模型所需修改清单、运行时依赖冲突的预留点。真实模型接入和 GPU batching 均不在本包中顺带启用。

## E 测试精简与最终验收

目标：实际删除重复/实现镜像断言，形成后续维护成本更低的验证组合，给出整轮改进的可核对结果。

依赖：最终修改在 A–D 集成后进行。可以提前静态整理测试去向表，但不要与其他 Agent 同时编辑其测试文件。

文件所有权：`tests/`、`pyproject.toml` 测试配置、`.github/workflows/ci.yml`、必要的发布检查脚本、验收文档；功能代码的失败回交对应 owner，不自行重写架构。

实施：

1. 执行主计划第 8 节的具体删改清单；区分真正删除、仅合并 fixture、参数化和迁入发布检查。
2. 统一 fixture 和事件同步；保留一个默认完整本地行为套件，GPU/performance/release 走明确标记或命令。mypy 继续检查实际代码和必要测试，不为了少报错移除检查；CI Action 固定审核过的 commit。
3. 对照安全报告逐项记录“改动 / 本地观察 / 剩余验证”，不把加了参数或 mock 通过当作真实 decoder/GPU 验收。
4. 完整本地回归一次，按修改范围完成前端 build/浏览器检查；给出耗时、用例数、性能对照及迁移回滚证据。
5. 准备 WSL 实机验收清单：Qwen 0.6B/1.7B、对齐、已启用 MOSS、模型切换、卸载/重启、真实长素材与质量对照；取得当前任务授权后才执行。

完成判据：每个删除的高价值场景有明确替代用例；无文案快照/固定常量造成的无效维护；新增控制均有行为验证；诊断包可用于交接；实际性能与未实测部分清晰分开。

最终报告按“达成的行为和性能 → 测试精简结果 → 尚未验证的事项”组织。没有固定删测试百分比，也不以把测试移出默认 CI 冒充测试质量改进。

## 派发顺序与整体验收

推荐顺序为 `A → B → C → D → E`。工作包内部的只读文档整理可以并行，核心实现按上游接口落地顺序交接；不需要为了使用多个 Agent 而制造共享文件冲突。

用户可直接发送：“按 `docs/asr-platform-agent-tasks.md` 实现任务包 A，完成本地实现和验证，交付修改及未验收项。”其他包替换字母即可；Windows 使用范围仍需单独明确。

整轮完成应满足：

- 上传/解码/分块有界；归一化不重复；临时与持久资产生命周期正确。
- 一个 job ID 能串起请求、阶段、模型和 worker，日志开销及保存量可控。
- 所有推理共用调度，预取不改变顺序或扩大无界资源占用。
- 新 ASR 仅需 adapter、注册和契约验证；生产能力声明与真实验收一致。
- 低价值单测确实被删改，关键竞态和质量回归仍有覆盖。
- 性能提升有同机对照数据；未经验证的数值仍标为目标。
