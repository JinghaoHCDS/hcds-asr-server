# 接入一个 ASR 引擎

引擎在 `asr_server/engines.py` 显式注册，模型在 `asr_server/registry.py` 定义。注册表示代码可以装配该引擎；只有通过对应真实 smoke 的模型才能加入生产启用列表。测试示例始终通过 `create_app(models=..., engines=...)` 注入，不进入 `default_models()`。

一个基础引擎需要实现 `AsrAdapter` 的 `load / unload / abort / transcribe / transcribe_batch`，复用 `TranscriptionResult`。`AudioInput` 的生产路径是 `AudioPath` 或有界合成描述符；不要把完整长音频或全部 PCM 通过 Pipe 复制。`transcribe_batch` 可逐项调用；当前 Qwen/MOSS 都声明 `batch_mode=serial`，`native_batch_max_items=1` 表示一次真实模型调用只处理一项，不承诺 GPU 原生 batching。原生 batching 需要独立质量、吞吐和显存实验。

需要对齐时另外实现 `TextAlignmentAdapter.align_text`，并准确开启 timestamps/forced_alignment。仅支持基础转写的引擎使用 `pipelines=("legacy",)`、空 timestamps 和 `forced_alignment=False`，不必认识 Qwen 质量复核或 MOSS Anchor Replay。路由会在解码前拒绝不支持的流程和能力。

下面是模型及引擎的最小注册形式。完整可运行的无 GPU 示例位于 `tests/test_engines.py`，同时覆盖普通 HTTP 和个人工作台。

```python
definition = ModelDefinition(
    id="example-asr", provider="Example", default=True,
    repository="organization/frozen-model", revision="PINNED_COMMIT",
    engine_id="example", engine_version="1", pipeline_version="example-v1",
    capabilities=ModelCapabilities(
        transcription=True, streaming=False, timestamps=[], forced_alignment=False,
        languages=["auto", "zh"], chinese_dialects=[], backends=["transformers"],
        context_prompt=False,
    ),
    execution_policy=ModelExecutionPolicy("chunked", 512, 4096),
    audio_spec=AudioSpec(sample_rate=16000, channels=1),
    pipelines=("legacy",), default_pipeline="legacy", resource_type="gpu",
)
engine = EngineRegistration(
    engine_id="example", adapter_factory=make_adapter,
    worker_entry="asr_server.adapters.example:worker_main",
    supported_pipelines=("legacy",),
)
app = create_app(settings, models={definition.id: definition}, engines={"example": engine})
```

`make_adapter(definition, settings)` 接收已经冻结的模型源、revision 和运行设置。Qwen/MOSS adapter 将它们显式传给 worker；兼容模块常量和 smoke 脚本从同一 `MODEL_SOURCES` 派生。`ALIGNER_SOURCE` 同样只有一份。不能只改对外 revision 而继续加载 adapter 内另一份固定模型。

默认模型必须属于启用的模型集合；工作台优先保持可用 Qwen quality 默认，没有质量流程的注册集合可以使用基础转写默认。`ModelDefinition.supports_pipeline()` 决定可用流程，`discovery()` 提供能力选择元数据。`identity()` 包含模型、源、revision、engine/pipeline 版本、输入规格和运行时 profile；工作台幂等及恢复还冻结请求参数、词库和质量流程依赖。模型或规格变化必须产生新身份，不能复用另一模型的结果。

输入规格由 `AudioSpec` 描述。新模型需要较高采样率时，从原始上传生成目标资产；禁止将保留的 16 kHz 资产升采样冒充高采样率原音。资产由其 workspace owner 删除，adapter 只借用视图。

worker 复用 `ProcessRpcTransport` 和 `workers.codec`。内部 envelope 包含 `protocol_version=1` 和 `execution_context`；ping 必须返回协议版本，响应继续使用独立递增的 RPC ID。错误的内部 `diagnostics` 仅给父进程写过滤日志，不能放进对外 `details`。模型 prompt 与输出解析留在引擎模块。当前 `runtime_profile_id=managed-python` 使用同一受管 Python 环境、独立 spawn 进程；其他 profile 会明确拒绝，本轮没有下载器、任意插件加载或外部 HTTP worker。

接入验收至少包括：源与 revision 到达真实加载边界；对应 API/工作台普通转写；不支持的能力在解码前拒绝；模型/profile/参数变化不命中旧幂等结果；取消和卸载保持安全边界；重量依赖不影响无 GPU import。后续真实模型验证按项目的 WSL CUDA 和独立 smoke 门槛执行。新增模型不需要在 HTTP 路由、调度器或持久存储添加名称分支。
