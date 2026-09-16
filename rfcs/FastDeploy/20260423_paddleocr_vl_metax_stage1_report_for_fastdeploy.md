# FastDeploy PaddleOCR-VL Metax 适配阶段一报告

| 任务名称 | FastDeploy PaddleOCR-VL Metax 适配阶段一报告 |
|------|------|
| 提交作者 | Dryoung |
| 提交时间 | 2026-04-23 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | `paddle==3.4.0.dev20251223` |
| 文件名 | 20260423_paddleocr_vl_metax_stage1_report_for_fastdeploy.md |
| 说明 | 本文用于汇总 `patch15` 收口、服务验证、benchmark、长尾复核与 profiling 准备结果，作为阶段一提交主文档 |

## 1. 背景与目标

本报告用于收敛 `FastDeploy release/2.4` 在 Metax 环境上适配 `PaddleOCR-VL` 的阶段一结果。阶段一的目标不是完成全部性能分析，而是把工作从“离线 smoke 可跑”推进到“在线服务可用、单请求可用、小规模 benchmark 可重复、profiling 已拿到第一层证据”。

本阶段执行边界如下：

- 不改源码
- 不重编译
- 不提交 git
- 优先复用已经 build、install、smoke 成功的 `patch15` 安装结果

阶段一评审关注点包括：

- `patch15` 是否已经形成可复用的最小补丁链路
- 已安装服务是否能稳定跑通 PaddleOCR-VL OCR 请求
- 是否已经形成可引用的小规模 benchmark 结果
- 是否已经具备进入下一阶段 benchmark / profiling 的基础条件

## 2. 环境与版本

### 2.1 路径与组件

- venv：`/data/venvs/paddleocr_vl_metax_fd24`
- FastDeploy 工作目录：`/data/FastDeploy_release_2_4_metax`
- 模型目录：`/mnt/moark-models/PaddleOCR-VL`
- 测试图片：`/data/ocr_demo.jpg`

### 2.2 系统与软件版本

- 操作系统：`Ubuntu 24.04.1 LTS`
- 内核：`5.15.0-58-generic`
- `fastdeploy`：`2.4.0`
- `paddle`：`3.4.0.dev20251223`
- `mx-smi`：已有可用采样能力，前序材料记录版本为 `2.2.9`
- `py-spy`：`0.4.1`
- `perf`：未安装
- `nsys`：未安装

### 2.3 当前服务状态

截至 `2026-04-23 16:27:48 UTC`，当前在线服务 `http://127.0.0.1:8288/health` 返回 `200`。

## 3. patch15 关键改动概述

`patch15` 已完成 build、install、smoke 成功，并已收口为 6 个变更文件，其中 5 个属于主链路必需文件，1 个属于环境兜底文件。

### 3.1 5 个必需文件

1. `build.sh`
   - 改动内容：修正 Metax 自定义 op 的打包落点，优先从 modern package 拷贝到 `fastdeploy/model_executor/ops/gpu/`，并带上 `version.txt`。
   - 解决问题：避免编译成功但 wheel 中缺少运行期需要的 GPU ops。
   - 必要性：直接决定安装包内是否带上可运行 op，属于 build/install 主链路必需项。

2. `custom_ops/setup_ops.py`
   - 改动内容：补入 `unset_data_ipc`、`remote_cache_kv_ipc`、`get_img_boundaries` 等 Metax 自定义 op 及其编译依赖路径。
   - 解决问题：确保 PaddleOCR-VL/Metax 运行时需要的 op 被实际编译和打包。
   - 必要性：没有这部分，运行所需 op 集合不完整，主链路不能成立。

3. `fastdeploy/entrypoints/llm.py`
   - 改动内容：将 `num_requests = len(req_ids)` 提前到 `if use_tqdm:` 之外。
   - 解决问题：修复 `use_tqdm=False` 场景下 `num_requests` 未定义的运行时错误。
   - 必要性：属于请求执行路径上的稳定性修复，直接影响真实请求链路。

4. `fastdeploy/model_executor/layers/backends/metax/attention/flash_attn_backend.py`
   - 改动内容：新增 rotary dim 归一化逻辑，并重写 encoder prefill 阶段 cache 写入逻辑。
   - 解决问题：修复 PaddleOCR/Qwen 类模型在 Metax attention 后端上的 rotary dim 不匹配与多模态 prefill cache 写入问题。
   - 必要性：PaddleOCR-VL 多模态推理会直接经过该链路，属于核心运行路径修复。

5. `fastdeploy/worker/metax_model_runner.py`
   - 改动内容：将 `qwen`/`paddleocr` 的 `rope_head_dim` 适配到 `head_dim`，新增 `extract_vision_features_paddleocr()`，接入 `paddleocr` 视觉特征分支。
   - 解决问题：补齐 PaddleOCR-VL 在 Metax 上的视觉编码与 projector 主运行路径。
   - 必要性：这是 PaddleOCR-VL 真正跑通的核心模型适配点，缺失时主链路不完整。

### 3.2 1 个环境兜底文件

1. `fastdeploy/worker/metax_worker.py`
   - 改动内容：新增 `_query_metax_memory_info()`，优先使用 `pymxsml`，缺失时回退 `mx-smi` 查询显存信息。
   - 解决问题：避免环境缺少 `pymxsml` 时 worker 在显存探测/KV cache 预算阶段直接失败。
   - 必要性：在具备 `pymxsml` 的环境中属于兜底项；若环境缺失 `pymxsml`，则升级为运行必需项。

## 4. 服务跑通证据

### 4.1 patch15 build / install / smoke 证据

- build 成功日志：`/data/FastDeploy_release_2_4_metax/build_metax_patch15.log`
- smoke 成功日志：`/data/FastDeploy_release_2_4_metax/smoke_test_patch15.log`

已确认的关键结果如下：

- build 阶段完成 wheel 构建
- install 阶段日志出现 `Successfully installed fastdeploy-metax-gpu-2.4.0`
- smoke test 成功返回 OCR 文本：`相思那得夢魂来。`

这说明 `patch15` 已经完成“补丁存在、可编译、可安装、可发起真实请求并拿到文本结果”的闭环。

### 4.2 在线服务跑通证据

阶段一在线验证分两步推进：

1. 最小服务验证基线
   - 基线命令来源：`/data/FastDeploy_release_2_4_metax/run_stage_formal/success_config.md`
   - 核心参数：`max_model_len=8192`、`max_num_batched_tokens=8192`、`max_num_seqs=4`
   - 结果：服务启动成功，`/health` 和 `/v1/models` 返回 `200`

2. 形式化 benchmark 稳态配置
   - 当前稳态命令来源：`/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/formal_service_status.json`
   - 核心参数：`max_model_len=2048`、`max_num_batched_tokens=2048`、`max_num_seqs=4`
   - 结果：该配置作为后续矩阵与正式 benchmark 的统一稳态配置，当前仍在线可用

### 4.3 单请求 OCR 证据

固定输入：

- 图片：`file:///data/ocr_demo.jpg`
- prompt：`OCR:`

实际结果：

- 请求接口：`http://127.0.0.1:8288/v1/chat/completions`
- HTTP 状态：`200`
- 返回文本：`相思那得夢魂来。`
- 首次请求耗时：`0.6401504781097174s`
- token 使用：`prompt_tokens=228`，其中 `image_tokens=215`

结论：阶段一已证明 PaddleOCR-VL 在当前安装包和当前服务模式下可以稳定返回 OCR 结果。

## 5. benchmark 小矩阵结果

本轮小矩阵仅扫描两个运行参数：

- `max_num_seqs`：`1`、`2`、`4`
- `max_model_len`：`1024`、`2048`

其余参数保持一致：

- `tensor_parallel_size=1`
- `gpu_memory_utilization=0.7`
- `FD_ENABLE_MAX_PREFILL=1`
- `MACA_VISIBLE_DEVICES=0`
- 每组顺序请求 `10` 次

结果如下：

| 组别 | 成功率 | 平均时延(s) | P50(s) | P95(s) | 吞吐(req/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| `seqs_1_len_1024` | 100% | 1.7370 | 0.3333 | 8.0539 | 0.5757 |
| `seqs_1_len_2048` | 100% | 0.3577 | 0.3420 | 0.4350 | 2.7950 |
| `seqs_2_len_1024` | 100% | 0.3648 | 0.3415 | 0.4791 | 2.7412 |
| `seqs_2_len_2048` | 100% | 0.3590 | 0.3435 | 0.4329 | 2.7855 |
| `seqs_4_len_1024` | 100% | 0.3727 | 0.3483 | 0.4822 | 2.6825 |
| `seqs_4_len_2048` | 100% | 0.3561 | 0.3403 | 0.4319 | 2.8078 |

矩阵观察结论：

- 6 组全部成功，说明当前服务在该小范围参数扫描内具备稳定运行能力。
- 除 `seqs_1_len_1024` 外，其余 5 组的平均时延均集中在 `0.356s` 到 `0.373s` 区间，差异较小。
- 初步最优组为 `seqs_4_len_2048`，同时在平均时延和吞吐上表现最佳。
- `seqs_1_len_1024` 出现一次明显异常长尾，需要独立复核，不能直接作为稳定瓶颈结论。

## 6. 正式 benchmark 结果

正式 benchmark 固定在当前最佳稳态参数：

- `max_num_seqs=4`
- `max_model_len=2048`
- `max_num_batched_tokens=2048`
- `gpu_memory_utilization=0.7`
- `tensor_parallel_size=1`

以下结果统一以原始结果文件 `benchmark_formal.json` / `benchmark_formal.log` 为准。

### 6.1 顺序请求主结果

- 轮数：`50`
- 模式：顺序请求
- 成功数：`50 / 50`
- 成功率：`100%`
- 平均时延：`0.3572s`
- P50：`0.3475s`
- P95：`0.3869s`
- P99：`0.5300s`
- 吞吐：`2.7987 req/s`

结论：

- 正式顺序 benchmark 未出现 8 秒级异常。
- 第 1 次请求明显高于后续稳态请求，但长尾未持续扩散。
- 该结果已经可以作为阶段一的主 benchmark 结论。

### 6.2 小并发补充结果

- 并发度：`4`
- waves：`5`
- 总请求数：`20`
- 成功数：`20 / 20`
- 成功率：`100%`
- 平均时延：`0.7347s`
- P50：`0.7380s`
- P95：`0.7893s`
- P99：`0.7905s`
- 吞吐：`5.3036 req/s`

结论：

- 当前单卡服务在小并发下仍能稳定处理请求。
- 并发补充组的吞吐高于顺序模式，符合服务在小规模并发下的预期表现。
- 阶段一已经具备“顺序主结果 + 小并发补充结果”的正式 benchmark 基础。

## 7. 长尾复核结论

针对矩阵中唯一异常组 `seqs_1_len_1024`，已执行独立复核：

- 参数：`max_num_seqs=1`、`max_model_len=1024`
- 轮数：`20`
- 模式：顺序请求

复核结果：

- 成功数：`20 / 20`
- 成功率：`100%`
- 平均时延：`0.3495s`
- P50：`0.3355s`
- P95：`0.3751s`
- P99：`0.5221s`
- 最慢样本：第 `1` 次请求，耗时 `0.5588s`
- 第 `2` 到第 `20` 次平均时延：`0.3385s`
- `tail_reproduced=false`

复核结论：

- 先前小矩阵中 `seqs_1_len_1024` 的 `8s` 级异常没有稳定复现。
- 当前更合理的判断是：该异常更可能属于冷态扰动、瞬时环境抖动或一次性偶发慢样本，而不是该配置的稳定长尾特征。
- 因此阶段一材料中应保留“矩阵曾出现一次异常长尾”的事实，但不能把它直接上升为稳定瓶颈结论。

## 8. 当前 profiling 证据与限制

### 8.1 已获得的 profiling 证据

基于当前服务，已完成一轮轻量 profiling：

- 连续请求数：`10`
- 请求成功数：`10 / 10`
- profiling 期间平均请求时延：`0.3607s`
- 返回文本集合：仅出现 `相思那得夢魂来。`
- `mx-smi` 样本数：`17`
- 最大 GPU 利用率：`10%`
- 显存使用：约 `23416004 KB`

当前证据说明：

- 服务在 profiling 采样期间保持可用；
- 设备侧采样链路已可工作；
- 已拿到“请求在跑、服务稳定、设备指标有响应”的第一层证据。

### 8.2 当前限制

`py-spy` 已安装完成，但 attach 在线服务时失败，当前明确限制为：

- `kernel.yama.ptrace_scope = 1`
- `/proc/sys` 处于只读挂载
- `py-spy dump --pid <worker_pid>` 返回 `Permission denied (os error 13)`

这意味着：

- 当前已经具备 profiler 工具本体；
- 但深度 profiling 仍受 `ptrace/py-spy` 权限限制；
- 当前问题属于系统/运行时权限问题，不属于代码问题。

### 8.3 解锁方向

如需解锁 `py-spy` attach，需由 root/平台管理员在宿主机侧执行：

```bash
sudo sysctl -w kernel.yama.ptrace_scope=0
```

若不能修改 `ptrace`，替代路径包括：

- 继续使用 `mx-smi` 做服务期连续采样
- 使用 `py-spy record --subprocesses -- python -m fastdeploy.entrypoints.openai.api_server ...` 包裹拉起专用 profiling 副本
- 使用 `python -m cProfile -o service.prof -m fastdeploy.entrypoints.openai.api_server ...` 获取 Python 级热点

## 9. 当前明确瓶颈与未完成项

### 9.1 当前明确瓶颈

1. 深度 profiling 权限受限
   - `py-spy` 已安装，但不能 attach 在线服务；
   - 当前缺少可直接使用的 Python 级火焰图和更细粒度 CPU 热点证据。

2. 多服务并存资源空间有限
   - 在不停止当前在线实例的情况下额外拉起 `seqs_1_len_1024` 复核实例时，worker 初始化阶段出现 `The total number of blocks cannot be less than zero`；
   - 这说明当前单卡资源下，多实例并存会触发显存/KV cache 预算冲突；
   - 因此阶段一的实验主要采用串行复用单实例方式推进。

3. profiling 工具链仍不完整
   - `perf` 缺失；
   - `nsys` 缺失；
   - 尚未引入更深层的 Metax 设备侧专用 profiler。

### 9.2 当前未完成项

- 尚未完成更长时间窗口或更大请求规模的正式 benchmark。
- 尚未形成更高并发档位下的系统性结果矩阵。
- 尚未完成 Python 栈热点、CPU 热点、算子级/框架级 trace 的深度 profiling。
- 尚未对首请求、稳态请求、图像前处理、视觉编码、解码等子路径建立更细颗粒度拆分指标。

## 10. 结论与提交判断

综合 `patch15` 收口材料、在线服务验证、小矩阵结果、正式 benchmark 结果、长尾复核结果和轻量 profiling 证据，可以做出以下判断：

1. `patch15` 已完成从补丁到 build/install/smoke 的闭环验证。
2. 基于已安装包的 PaddleOCR-VL 服务已经跑通，单请求 OCR 成功返回文本。
3. 已形成一套可重复的小矩阵 benchmark 结果，并完成了最佳参数下的正式 benchmark 扩展。
4. `seqs_1_len_1024` 的异常长尾未稳定复现，当前不支持把其认定为稳定瓶颈。
5. 当前已经具备阶段一提交基础。
6. 但深度 profiling 仍受 `ptrace/py-spy` 权限限制。

阶段一提交建议状态：

- 可以提交阶段一正式材料；
- 可以把 `seqs_4_len_2048` 作为后续阶段的默认稳态服务参数；
- 不应把当前 profiling 结果描述为“已完成深度定位”，只能描述为“已完成轻量证据采集，深度采样受系统权限限制”。

## 11. 下一步计划

建议后续按以下顺序推进：

1. 固定 `seqs_4_len_2048` 作为阶段二默认稳态配置。
2. 由 root/平台管理员评估并处理 `ptrace_scope` 放开方案，优先解锁 `py-spy` attach。
3. 若不能放开 `ptrace`，则启动专用 profiling 副本，采用 `py-spy record --subprocesses` 或 `cProfile` 包裹启动。
4. 在当前正式 benchmark 基础上继续扩展：
   - 增加总请求数；
   - 增加并发档位；
   - 区分首请求与稳态请求。
5. 在拿到可用 profiler 结果后，补齐 Python 栈热点、CPU 热点与设备侧证据之间的对应关系，进入更深一层的性能定位。
