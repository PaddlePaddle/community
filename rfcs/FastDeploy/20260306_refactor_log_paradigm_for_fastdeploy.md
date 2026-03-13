# 为FastDeploy重构log日志打印范式

| 任务名称 | 【Hackathon 9th No.88】为FastDeploy重构log日志打印范式 |
|---|---|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-03-06 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20260306_refactor_log_paradigm_for_fastdeploy.md |

# 一、概述

## 1、相关背景

FastDeploy 作为 PaddlePaddle 的高性能 LLM 推理部署框架，已具备结构化日志模块 (`fastdeploy/logger/`)，
包含 FastDeployLogger 单例、dictConfig 配置、彩色终端输出、文件轮转等基础设施。

然而，当前日志输出在**内容范式**上存在明显问题：279 处 `print()` 调用散落在核心推理模块中，
58.7% 的 `logger` 调用使用 INFO 级别（大量应为 DEBUG），且日志输出缺乏阶段感和请求上下文关联。
运维人员在 INFO 级别下启动一次推理，会看到 100+ 行无结构噪音。

相关背景：
- [PaddlePaddle/Paddle#74773](https://github.com/PaddlePaddle/Paddle/issues/74773) — Hackathon 9th 任务列表
- [PR #6370](https://github.com/PaddlePaddle/FastDeploy/pull/6370) — 已合并的日志格式统一 (qwes5s5)
- [PR #6682](https://github.com/PaddlePaddle/FastDeploy/pull/6682) — Phase A 基础设施改进 (cloudforge1)

## 2、功能目标

为 FastDeploy 大模型推理的**全执行流程**（从启动到输出结果）设计清晰、易读、有条理的日志打印范式：

1. 定义 5 个执行阶段的日志内容规范（打印什么、用什么级别、为什么）
2. 将核心模块 112 个 `print()` 迁移为 `logger` 调用
3. 纠正 INFO 级别过度使用问题，使生产环境日志量从 100+ 行/启动降至约 15 行
4. 建立模块级日志职责表，明确每个源文件的日志行为

## 3、意义

- **运维效率**: INFO 级别下输出约 15 行清晰启动日志 + 每分钟 2 行汇总指标，运维可实时阅读
- **调试能力**: DEBUG 级别保留完整的逐请求追踪，排查问题时切换即可
- **代码质量**: 消除 print/logger 混用，统一日志入口
- **可扩展性**: 范式设计兼容未来 JSON 结构化输出、OpenTelemetry 集成等增强

# 二、飞桨现状

FastDeploy 已有 `fastdeploy/logger/` 模块（6 文件，~1126 行），提供：
- `FastDeployLogger` 单例（线程安全，支持统一接口 + 11 个 legacy logger）
- `dictConfig` 配置（console + 文件 handler，彩色终端，按天轮转）
- 环境变量接口（`FD_LOG_LEVEL`, `FD_LOG_DIR`, `FD_DEBUG`）

**核心问题**:

| 问题 | 影响 |
|------|------|
| 279 个 `print()` 调用（核心模块 112 个） | 不受级别控制，不写入日志文件 |
| INFO 占比 58.7%，大量应为 DEBUG | 生产环境日志冗长 |
| 无执行阶段划分 | 启动/运行/关闭日志混在一起，无法快速定位 |
| `llm_logger` 被 16+ 模块共用 | 日志来源不清晰 |

PR #6370 已解决日志**格式**统一问题；PR #6682 已补充 `FD_LOG_LEVEL`、console handler、`get_logger()` API。
本 RFC 解决**内容范式**问题 — 每个阶段打印什么、什么级别。

详细定量分析见调研文档: [logging_current_state_analysis.md](https://github.com/PaddlePaddle/FastDeploy/pull/6682)

# 三、业内方案调研

| 框架 | 日志方案 | 特点 |
|------|----------|------|
| **vLLM** | Python `logging` + 自定义 `init_logger(__name__)` | 模块级 logger，启动阶段 INFO 输出配置摘要，请求处理阶段 INFO 安静（仅周期性指标），DEBUG 打逐请求日志。支持 `VLLM_LOGGING_LEVEL` 环境变量 |
| **TGI** (HuggingFace) | Rust `tracing` crate + 结构化 span | 按 span 层级过滤，请求自带 trace_id，生产环境极简输出。JSON 格式可选 |
| **SGLang** | Python `logging` + `loguru` 可选 | 类似 vLLM 模式，启动阶段打配置，运行阶段安静。`SGLANG_LOG_LEVEL` 控制 |
| **LMDeploy** | Python `logging` | 较简单，主要靠 print + logger 混用，无明确阶段划分 |

**共性规律**:
1. 启动阶段打配置摘要（1 行），模型加载耗时，缓存分配 — 均为 INFO
2. 请求处理阶段在 INFO 下**完全安静**，仅靠周期性指标汇总
3. 逐请求日志（接收/调度/完成）全部为 DEBUG
4. 关闭阶段打汇总统计 — INFO

# 四、对比分析

| 方案 | 优势 | 劣势 | 适用性 |
|------|------|------|--------|
| **A: 维持现状** | 无修改成本 | 噪音大，print 不可控 | ❌ |
| **B: 全量替换为结构化 JSON** | 机器可解析，对接 ELK | 改动面大，可读性降低，超出⭐⭐任务范围 | ❌ 未来扩展 |
| **C: 内容范式重构（本方案）** | 精准控制每个阶段的输出，最小化改动面，兼容未来 JSON 扩展 | 需逐模块迁移 | ✅ |

选择方案 C：
- 在已有日志基础设施上只改**内容层**（what to log），不改**格式层**（how to format）
- 与 PR #6370（格式）和 PR #6682（基础设施）正交，互不冲突
- 分批迁移（5 批 PR），每批可独立 review

# 五、设计思路与实现方案

## 1、主体设计思路与折衷

### 执行流程划分

将 FastDeploy LLM 推理划分为 **5 个阶段**，每个阶段定义独立的日志范式：

```
阶段1: 初始化与启动  →  配置解析 → 引擎构建 → Worker启动 → 模型加载 → 缓存初始化 → 服务就绪
阶段2: 请求接入      →  API接收 → 参数校验 → 输入预处理 → 入队
阶段3: 调度与执行    →  调度决策 → 缓存分配 → Prefill → Decode → Token输出
阶段4: 周期性运维    →  吞吐量汇总 → 缓存利用率 → 健康检查
阶段5: 关闭          →  信号接收 → 请求排空 → 资源释放 → 统计汇总
```

### 日志级别使用规范

| 级别 | 用途 | 生产环境可见 |
|------|------|-------------|
| **ERROR** | 操作失败，需人工介入 | ✅ 始终 |
| **WARNING** | 异常但可恢复，或即将出问题 | ✅ 始终 |
| **INFO** | 系统级状态变更（启动/就绪/关闭）、周期性汇总 | ✅ 默认 |
| **DEBUG** | 逐请求/逐操作追踪 | ❌ 需 `FD_LOG_LEVEL=DEBUG` |

**核心规则**: 请求处理在 INFO 下**完全安静**。逐请求日志全部为 DEBUG。

### 迁移后效果对比

**迁移前** (INFO, 启动+1个请求): ~100+ 行噪音
**迁移后** (INFO, 启动+1个请求): ~15 行，清晰有阶段感，请求处理零输出

```
INFO  engine.py  FastDeploy 0.5.0 (commit=839bc83, paddle=3.0)
INFO  config.py  Configuration: model=Qwen2.5-7B, tp=4, max_batch=256, dtype=float16
INFO  engine.py  Starting 4 worker process(es)...
INFO  model_base.py  Model loaded: Qwen2.5-7B (12.3s, 14.50 GB)
INFO  engine.py  KV cache: 2048 GPU blocks (8.00 GB)
INFO  engine.py  FastDeploy engine ready (15.0s). Serving on 0.0.0.0:8000
```

## 2、关键技术点/子模块设计与实现方案

### 2.1 print() 迁移策略

| 分类 | 数量 | 策略 |
|------|------|------|
| 核心推理模块 (engine, model_executor, cache_manager, scheduler) | 112 | **全部迁移** → `logger.info/debug` |
| API/服务模块 (entrypoints, output) | ~30 | **全部迁移** → `logger.debug` |
| CLI 用户交互 (entrypoints/cli/) | ~10 | **保留 print()** |
| 基准测试/Demo (benchmarks/, demo/) | ~65 | **保留 print()** |

### 2.2 模块级日志职责表

| 模块 | Logger 名称 | INFO 职责 | DEBUG 职责 |
|------|-------------|-----------|-----------|
| config | `fastdeploy.config` | 配置摘要(1行) | 完整 dump |
| engine | `fastdeploy.engine` | 启动/就绪/关闭 | 组件通信 |
| model_executor | `fastdeploy.model_executor` | 加载完成(名+耗时) | 逐层加载 |
| entrypoints | `fastdeploy.entrypoints` | — (安静) | 逐请求处理 |
| scheduler | `fastdeploy.scheduler` | — (安静) | 调度决策 |
| cache_manager | `fastdeploy.cache_manager` | 缓存初始化完成 | 块分配/驱逐 |
| metrics | `fastdeploy.metrics` | 周期性汇总(60s) | 逐步指标 |

### 2.3 周期性指标汇总 (新增)

每 60 秒输出 2 行 INFO：
```python
logger.info("Throughput: %.1f req/s, %.1f tok/s, running=%d, queued=%d", ...)
logger.info("KV cache usage: %.1f%% (%d/%d blocks)", ...)
```

## 3、主要影响的模块接口变化

### 直接接口变化
- `fastdeploy/logger/__init__.py`: 已公开 `get_logger()` API (PR #6682)
- `fastdeploy/envs.py`: 已新增 `FD_LOG_LEVEL` 环境变量 (PR #6682)

### 对各模块影响

| 模块 | 影响 |
|------|------|
| engine/ | print→logger, INFO→DEBUG 降级 (~20 处) |
| model_executor/ | print→logger (~19 处, model_base 已完成) |
| cache_manager/ | print→logger (~54 处, 含 benchmark 保留) |
| entrypoints/ | print→logger (~30 处, CLI 保留) |
| scheduler/ | 级别调整，无 print 迁移 |
| config.py | print→logger (~9 处) |
| **不受影响** | input/, worker/, trace/, spec_decode/, plugins/ |

# 六、测试和验收的考量

| 验收标准 | 度量方式 |
|----------|----------|
| 核心模块 print() 清零 | `grep -r "print(" engine/ model_executor/ cache_manager/ scheduler/` 返回 0 |
| INFO 级别启动日志 ≤ 20 行 | 启动 Qwen2.5-7B，统计 INFO 行数 |
| INFO 级别请求处理零输出 | 发送 10 个请求，检查 INFO 无逐请求日志 |
| DEBUG 级别完整追踪 | `FD_LOG_LEVEL=DEBUG` 下可看到逐请求全路径 |
| 周期性汇总正常输出 | 运行 2 分钟，检查至少 1 次吞吐量汇总 |
| 现有测试通过 | `pytest tests/` 无回归 |

# 七、影响面

## 对用户的影响
- 启动日志更清晰，噪音大幅减少
- 新增 `FD_LOG_LEVEL` 环境变量提供细粒度控制（已在 PR #6682 中实现）

## 对二次开发用户的影响
- `get_logger(__name__)` 作为推荐的 logger 获取方式（新增 API）
- Legacy logger (`llm_logger` 等) 继续可用，不做 breaking change

## 对框架架构的影响
- 无架构变更，仅修改日志调用方式
- 与 PR #6370（格式统一）完全兼容

## 对性能的影响
- logger 的 `%s` 延迟格式化比 f-string 更优（DEBUG 关闭时零开销）
- 移除高频 print() 减少 stdout I/O
- 无负面性能影响

## 其他风险
- 迁移过程中可能遗漏个别 print()，通过 grep 验收兜底
- 日志级别调整可能影响依赖特定日志输出的外部脚本（风险低）

# 八、排期规划

| 阶段 | 内容 | 状态 | 预计 |
|------|------|------|------|
| Phase A | 基础设施: `FD_LOG_LEVEL`, `get_logger()`, console handler, 示范迁移 | ✅ 已完成 (PR #6682) | — |
| Phase B1 | `config.py` 迁移 (~9 print, ~50 行变更) | 待开始 | 1 天 |
| Phase B2 | `engine/` 迁移 (~20 print, ~80 行变更) | 待开始 | 1 天 |
| Phase B3 | `model_executor/` 迁移 (~19 print, ~60 行变更) | 待开始 | 1 天 |
| Phase B4 | `entrypoints/` 迁移 (~30 print, ~80 行变更) | 待开始 | 1 天 |
| Phase B5 | `scheduler/`, `cache_manager/`, `output/` (~34 print, ~100 行变更) | 待开始 | 2 天 |

每批一个 PR，附迁移前后日志样本对比。

# 名词解释

| 术语 | 含义 |
|------|------|
| 日志范式 | 定义"在什么阶段、打什么内容、用什么级别"的规范 |
| Legacy logger | `utils.py` 中预配置的 11 个 logger（`llm_logger` 等） |
| 周期性汇总 | 每 60 秒输出一次吞吐量和缓存利用率的 INFO 日志 |

# 附件及参考资料

1. [调研文档 (D1)](./logging_current_state_analysis.md) — 日志现状定量分析
2. [PR #6682](https://github.com/PaddlePaddle/FastDeploy/pull/6682) — Phase A 基础设施实现
3. [PR #6370](https://github.com/PaddlePaddle/FastDeploy/pull/6370) — 日志格式统一 (qwes5s5)
4. [vLLM logging](https://github.com/vllm-project/vllm/blob/main/vllm/logger.py) — 业内参考
5. [Python logging Best Practices](https://docs.python.org/3/howto/logging.html)
6. [内部详细设计草稿](./_internal_draft_logging_paradigm_v3.md) — 完整的逐阶段范式表（693 行）
