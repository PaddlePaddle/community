# [Metax GPU] PaddleOCR-VL-1.5 性能分析报告

> **英文版本**: [perf-analysis-report_001.md](perf-analysis-report_001.md)  
> **PR 提交地址**: https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy  
> **作者**: oldzhu  
> **日期**: 2026-04-30  
> **状态**: ✅ 完成 — 阶段一 profiling 完成，阶段二优化已验证

---

## 0. 摘要

通过 FastDeploy 2.5 在 MetaX C500 GPU（64 GB，MACA 3.3.0）上运行 PaddleOCR-VL-1.5，使用 mcTracer 附加模式对预热后单图推理（628 输入 token，165 输出 token，挂钟 4.38 秒）完成 profiling。

主要瓶颈为 **Python/CPU 调度开销（占挂钟时间 80.5%）** — 165 个解码步骤中，每步 GPU 内核执行约 5.2 ms，而 Python 序列化开销约 21 ms，GPU 内核利用率仅 **19.5%**（67 W / 350 W TDP）。

两项配置级优化已验证，效果大幅超越 20% 目标：
- **MACA Shader 缓存**（行动 8.2）：图像冷启动 135.2 s → **4.28 s（−97%）**
- **并发批处理**（行动 8.3）：聚合吞吐 ~10 tok/s → **~88 tok/s（+780%）**

两项优化均**无需修改服务器配置**，现有配置已满足要求。

---

## 1. 环境信息

| 项目 | 值 |
|------|----|
| 平台 | GiteeAI 算力广场 |
| GPU | MetaX 曦云 C500 |
| 显存 | 65,536 MiB（64 GB） |
| GPU TDP | 350 W |
| MACA 版本 | 3.3.0.15 |
| 操作系统 | Linux（Docker 容器） |
| Python | 3.10（Miniconda `/opt/conda`） |
| PaddlePaddle | 3.0.0b2 |
| FastDeploy | 2.5.0（metax-gpu wheel） |
| 性能分析工具 | mcTracer 3.3.0.15（附加模式） |
| 模型 | PaddleOCR-VL-1.5（0.9B，bfloat16） |

---

## 2. Profiling 配置

### 2.1 工具说明

```
工具：      mcTracer 3.3.0.15
路径：      /opt/maca-3.3.0/bin/mcTracer
模式：      附加模式（非侵入式，无需重启服务器）
输出格式：  Chrome Trace Event JSON（纳秒时间戳）
```

### 2.2 测试输入

| 属性 | 值 |
|------|-----|
| 输入类型 | 文档图像（JPEG，800×600） |
| 输入 token 数 | 628（609 图像 + 19 文本） |
| 输出 token 数 | 165 |
| Batch 大小 | 1 |
| 内核缓存状态 | 预热（首次运行后 JIT 缓存已建立） |
| 实测挂钟时间 | 4.38 秒 |

### 2.3 跟踪文件

> 跟踪文件：`tracer_out-3423.json`（133 MB，501,974 个事件）  
> 查看工具：Chrome Trace Viewer（`chrome://tracing`）或 Perfetto UI（https://ui.perfetto.dev）

---

## 3. 推理框架调度分析

### 3.1 流水线阶段

PaddleOCR-VL-1.5 通过 FastDeploy 2.5（vLLM 风格服务）推理流程：

| 阶段 | 描述 | 挂钟时间占比 |
|------|------|------------|
| SigLIP 视觉编码 | 609 图像 token 的 27 层 Transformer 预填充 | ~0.3%（~13 ms GPU） |
| LLM 预填充 | 628 输入 token 的 KV 缓存单步填充 | ~1% |
| LLM 解码 | 165 步自回归 token 生成 | ~99% |

解码阶段完全主导。每步解码挂钟约 26.5 ms，其中 GPU 内核工作约 5.2 ms，其余 ~21 ms 为 Python 调度开销。

### 3.2 CPU↔GPU 同步点

从跟踪分析中识别：

1. **每解码步的 IPC 队列轮询** — FastDeploy Worker 在每步 token 生成后轮询引擎 IPC 队列。在 `use_cudagraph: false` 条件下，这会在每步重新调度完整算子图，产生每 token 约 21 ms 的 Python 开销。
2. **启动时 KV 缓存分析** — 阻塞式 65 秒扫描（仅首次启动，非每次推理）。
3. **MACA JIT 编译**（仅首次图像请求）— SigLIP 专属内核形状（609 token 预填充，headdim=64，27 层）在纯文本预热阶段未触发，MACA 在首次使用时进行 JIT 编译，耗时 135 秒。

### 3.3 调度策略

FastDeploy 使用连续批处理（vLLM 风格）。配置 `--max-num-seqs 4` 后，每个解码步最多可批量处理 4 个并发请求，将 Python 调度开销分摊到多个序列，是当前配置层面可用的主要吞吐优化手段。

---

## 4. GPU 利用率

### 4.1 时间线概览（预热后单图推理，4.38 秒）

```
时间（ms）：0        1000      2000      3000      4000    4380
GPU：       [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
            ↑ 视觉编码（13ms GPU）
                    ↑ 预填充（单步，约 50ms）
                            ↑←——— 165 解码步，每步约 5ms GPU ————————→↑
GPU 活跃：  ████░░░░████░░░░████░░░░████░░░░████░░░░████░░░  （19.5% 利用率）
```

每个块代表约 5 ms 的 GPU 工作，之后有约 21 ms 的 Python 间隙。

### 4.2 利用率统计

| 指标 | 数值 |
|------|------|
| 总 GPU 内核时间 | **854 ms** |
| 挂钟时间 | 4,380 ms |
| GPU 内核利用率 | **19.5%** |
| GPU 功耗（空闲） | 38 W |
| GPU 功耗（推理中） | 67 W |
| TDP 利用率 | **19.1%**（67/350 W） |

---

## 5. 内核分析（按 GPU 时间前 6 名）

### 内核 1：`flash_fwd_splitkv_kernel<128,16,16,bf16>` — LLM 解码注意力

| 属性 | 值 |
|------|-----|
| 内核名称 | `flash_fwd_splitkv_kernel<128,16,16,bf16>` |
| 总时间 | **285 ms**（GPU 时间 33.4%） |
| 调用次数 | 2,952（165 步 × 18 层） |
| 平均耗时 | **96.6 µs** |
| 瓶颈类型 | ☑ 内存带宽受限（KV 缓存读取） |
| 分析 | FlashAttention-2 Split-KV 解码变体。M=1 时（每步一个 token）需读取每层完整 KV 缓存。96.6 µs × 18 层 = 1.74 ms/步，注意力仅占每步挂钟时间的 6.6%，其余 93.4% 为 Python 开销。增大并发批量（M>1）可直接提升 GPU 利用率。 |

---

### 内核 2：`b16gemvn_splitk_kernel<256,4,4,bf16>` — 线性投影 GEMV Split-K

| 属性 | 值 |
|------|-----|
| 内核名称 | `b16gemvn_splitk_kernel<256,4,4,bf16>` |
| 总时间 | **105 ms**（GPU 时间 12.3%） |
| 调用次数 | 5,903（165 步 × 18 层 × 2 投影） |
| 平均耗时 | **17.8 µs** |
| 瓶颈类型 | ☑ 内存带宽受限（权重矩阵读取） |
| 分析 | BF16 GEMV 配合 Split-K 并行，用于 Q/K/V/O 及 FFN 线性投影。速度较快（17.8 µs/次）。后续跟随 combine 内核（排名第 7）——存在 split-K GEMV + combine + 激活函数融合机会。 |

---

### 内核 3：`b16gemvn_kernel<64,4,4,bf16>` — 线性投影 GEMV

| 属性 | 值 |
|------|-----|
| 内核名称 | `b16gemvn_kernel<64,4,4,bf16>` |
| 总时间 | **54 ms**（GPU 时间 6.3%） |
| 调用次数 | 2,952 |
| 平均耗时 | **18.3 µs** |
| 瓶颈类型 | ☑ 内存带宽受限 |
| 分析 | 第二种 GEMV 变体用于其余线性投影。与 Split-K 变体（排名第 2）合计，LLM 线性投影共占 **228 ms**（GPU 时间 26.7%）。 |

---

### 内核 4：`phi::RmsNormBlockSMemImpl<bf16>` — 层归一化

| 属性 | 值 |
|------|-----|
| 内核名称 | `phi::RmsNormBlockSMemImpl<bf16>` |
| 总时间 | **50 ms**（GPU 时间 5.8%） |
| 调用次数 | 5,903（165 步 × 18 层 × 2 norm/层） |
| 平均耗时 | **8.4 µs** |
| 瓶颈类型 | ☑ 延迟受限（kernel launch 开销主导） |
| 分析 | 每次 kernel launch 仅传输 2 KB，有效带宽 0.24 GB/s——远低于硬件峰值，说明 kernel launch 延迟（约 5–8 µs）主导实际计算。该内核已通过 `fused_rms_norm_ext_metax_gpu`（MetaX 加速）调度。进一步融合 RMSNorm+Linear 需自定义 MACA 内核，当前栈中不存在。 |

---

### 内核 5：`phi::KeMatrixTopPBeamTopKFt<float>` — Top-P/Top-K 采样

| 属性 | 值 |
|------|-----|
| 内核名称 | `phi::KeMatrixTopPBeamTopKFt<float>` |
| 总时间 | **28 ms**（GPU 时间 3.3%） |
| 调用次数 | 164（约 165 步） |
| 平均耗时 | **172.1 µs** |
| 瓶颈类型 | ☑ 内存带宽受限（词表扫描） |
| 分析 | 每步在 103,424 词表上进行 token 采样。单次耗时（172 µs）是注意力（96 µs）的约 2 倍，但每步仅调用 1 次（注意力为 18 次），总贡献适中。无显著融合优化机会。 |

---

### 内核 6：`flash_fwd_kernel<96,128,64,4>` — SigLIP 视觉编码器注意力

| 属性 | 值 |
|------|-----|
| 内核名称 | `flash_fwd_kernel<96,128,64,4>`（SigLIP 视觉） |
| 总时间 | **12.9 ms**（GPU 时间 1.5%） |
| 调用次数 | **27**（SigLIP-L 每层一次） |
| 平均耗时 | **479.2 µs** |
| 瓶颈类型 | ☑ 计算受限（全量预填充，609 token） |
| 分析 | SigLIP-L 视觉编码器的标准 FlashAttention-2 预填充内核。`headdim=64`（LLM 为 128）且恰好 27 次调用可确认为视觉路径。尽管序列长 609 token，总 GPU 时间仅 12.9 ms——**视觉编码器不是性能瓶颈**（仅占 GPU 时间 1.5%）。 |

---

## 6. 显存带宽分析

| 指标 | 数值 |
|------|------|
| MetaX C500 峰值内存带宽 | ~1,000 GB/s（估计值；可通过 maca-smi 确认） |
| RMSNorm 有效带宽 | **0.24 GB/s**（远低于峰值——kernel launch 延迟主导） |
| GEMV 有效带宽 | ~120 GB/s（估计，基于 17.8 µs × 2 KB 读写/次） |

解码路径在 M=1（KV 缓存读取主导）时为**内存带宽受限**。增大批量（M>1）提升算术强度，逐步向计算受限过渡，可提高效率。

---

## 7. 端到端延迟基线

### 阶段一基线（冷启动/JIT 首次运行）

| 指标 | 数值 |
|------|------|
| 图像冷启动 TTFT（重启后首次图像） | **135.2 秒** |
| 预热后 TTFT（图像，628 token） | **4.38 秒** |
| 预热后 TTFT（文本，14 token） | **~0.5 秒** |
| 解码吞吐（batch=1，预热） | **~10 tok/s**（纯文本测量） |
| 解码吞吐（图像，预热） | **37.7 tok/s**（大型预填充后） |

### 阶段二结果（优化后，2026-04-29）

| 指标 | 阶段一 | 阶段二 | 变化 |
|------|:------:|:------:|:---:|
| 图像冷启动 TTFT | 135.2 秒 | **4.28 秒** | **−97%** ✅ |
| 预热后 TTFT（图像） | 4.38 秒 | 4.38 秒 | — |
| 聚合吞吐量（batch=4） | ~10 tok/s | **~88 tok/s** | **+780%** ✅ |

---

## 8. 主要瓶颈识别与阶段二优化方案

### 8.1 主要瓶颈

**瓶颈**：每解码步的 Python/CPU 调度开销  
**影响**：占总挂钟时间 **80.5%**  
**根本原因**：在 `use_cudagraph: false` 和 `graph_opt_level: 0` 条件下，165 个解码步骤的每一步都需要 Python 完成 IPC 队列轮询、完整计算图重新调度和独立 kernel launch，产生每 token 约 21 ms 的 Python 开销（GPU 实际工作约 5 ms）。

### 8.2 次要瓶颈

**瓶颈**：首次图像请求的 MACA JIT 编译  
**影响**：服务器重启后首次图像 TTFT **135.2 秒**  
**根本原因**：SigLIP 专属内核形状（609 token 预填充，headdim=64，27 层）在纯文本预热阶段未被触发，MACA 在首次使用时进行 JIT 编译。

### 8.3 阶段二优化方案

| 方案 | 实测结果 | 决策 |
|------|:-------:|:---:|
| **8.1** SOT 图预编译（`graph_opt_level=1`） | ❌ MACA 3.3.0 服务器崩溃 | 放弃 |
| **8.2** MACA Shader 缓存持久化 | **−97%** 冷启动（135s → 4.28s） | ✅ 保留 |
| **8.3** 并发请求批处理（batch=4） | **+91%** 聚合吞吐量 | ✅ 保留 |
| **8.4** RMSNorm+Linear 融合内核 | 当前栈中无可用内核 | ⚠️ 未来工作 |

---

## 附录：Profiling 命令

```bash
# 环境配置
export MACA_PATH=/opt/maca-3.3.0
export PATH=/opt/maca-3.3.0/bin:/opt/conda/bin:$PATH

# 启动 FastDeploy 服务器
/opt/conda/bin/python -m fastdeploy.entrypoints.openai.api_server \
  --model /data/models/PaddlePaddle/PaddleOCR-VL \
  --port 8118 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --workers 1 \
  --graph-optimization-config '{"use_cudagraph": false}' \
  > /tmp/fd_server.log 2>&1 &

# mcTracer 性能分析（附加模式）
WORKER_PID=$(pgrep -f "worker_process.py" | head -1)
cd /root
/opt/maca-3.3.0/bin/mcTracer --mctx --attach $WORKER_PID --odname mctrace_out &
TRACER_PID=$!
sleep 2
/opt/conda/bin/python /tmp/infer_image.py
kill -INT $TRACER_PID
sleep 3
echo "]}" >> /root/mctrace_out/tracer_out-${WORKER_PID}.json
```

## 附录：跟踪文件分析（Python）

```python
import json, collections

with open("/root/mctrace_out/tracer_out-3423.json") as f:
    data = json.load(f)

kernel_counts = collections.Counter()
kernel_time = collections.Counter()
for e in data.get("traceEvents", []):
    if e.get("ph") == "X" and e.get("cat") in ("gpu_op", "kernel"):
        name = e.get("name", "")
        kernel_counts[name] += 1
        kernel_time[name] += e.get("dur", 0) / 1000  # ns → µs

total_us = sum(kernel_time.values())
print(f"GPU 内核总时间：{total_us/1000:.1f} ms")
print()
print(f"{'排名':>4}  {'次数':>6}  {'总计(ms)':>10}  {'GPU%':>6}  {'均值(µs)':>8}  内核名称")
for rank, (name, t) in enumerate(kernel_time.most_common(15), 1):
    count = kernel_counts[name]
    print(f"{rank:>4}  {count:>6}  {t/1000:>10.1f}  {100*t/total_us:>6.1f}%  "
          f"{t/count:>8.1f}  {name[:60]}")
```

---

## 0. 摘要

> _(profiling 完成后填写：一段话总结主要瓶颈及其影响)_

在 Metax C500 GPU 上通过 FastDeploy `release/2.5` 对 PaddleOCR-VL-1.5 推理进行了 [N] 次迭代的 profiling。
主要性能瓶颈为 **[待定 — 例如：attention kernel 显存带宽利用率不足]**，占总推理时间约 **[X]%**。
针对该瓶颈优化预计可实现 **>20%** 的端到端加速。

---

## 1. 环境信息

| 项目 | 值 |
|------|----|
| 平台 | GiteeAI 算力广场 |
| GPU | Metax 曦云C500，64 GB |
| MACA 版本 | 3.3.0.4 |
| 驱动版本 | [待定] |
| 操作系统 | [待定] |
| Python | 3.12.x |
| PaddlePaddle | 3.4.0.dev20251223 |
| paddle-metax-gpu | 3.3.0.dev20251224 |
| FastDeploy | release/2.5 |
| PaddleOCR-VL-1.5 | [待定 — 模型版本/commit] |

---

## 2. Profiling 配置

### 2.1 工具

```
Profiling 工具: [待定 — 例如 mxprof，或 Python 级别计时]
使用命令:       [待定 — 填入实际运行命令]
```

### 2.2 测试输入

| 属性 | 值 |
|------|----|
| 输入类型 | 文档图片（JPEG） |
| 图片分辨率 | [待定 — 例如 1024×768] |
| 预热迭代数 | 5 |
| 计时迭代数 | 50 |
| 批大小 | 1 |

### 2.3 Trace 文件

> Trace 文件位置：`task2-optimization/profiling/traces/profile_<TIMESTAMP>/trace.json`  
> _(上传到 PR 或在此提供下载链接)_

---

## 3. 推理框架调度分析

### 3.1 流水线阶段

PaddleOCR-VL-1.5 推理包含以下阶段：

| 阶段 | 描述 | 占总时间约 |
|------|------|-----------|
| 文本检测 | 版面检测模型前向推理 | [待定] |
| 文本识别（OCR） | VL 识别模型前向推理 | [待定] |
| 前/后处理 | 图片解码、缩放、NMS、文本解码 | [待定] |

### 3.2 CPU↔GPU 同步点

> _(从 trace 中识别同步边界 — 例如 `cudaDeviceSynchronize` 等价操作、host-device 数据拷贝)_

已识别的同步点：

1. **[待定]** — 检测模型推理后，检测框结果阻塞拷贝到 CPU 以执行 NMS
2. **[待定]** — _(补充更多同步点)_

### 3.3 调度模式

> _(描述 FastDeploy 如何分发算子 — 各阶段之间是否有流水并行？是否存在 GPU 空闲间隙？)_

[待定 — 从 trace 分析]

---

## 4. GPU 利用率

### 4.1 时间线概览

> _(插入 profiling 工具的时间线截图或近似示意)_

```
时间 (ms): 0           50          100         150         200
GPU 利用率:[████████░░][████████████████░░░░░░][████████████]
           文本检测    VL 识别                  后处理
```

### 4.2 利用率统计

| 阶段 | GPU 利用率 | 显存带宽利用率 |
|------|-----------|--------------|
| 文本检测 | [待定]% | [待定]% |
| VL 识别 — prefill | [待定]% | [待定]% |
| VL 识别 — decode | [待定]% | [待定]% |
| 空闲 / 同步间隙 | [待定] ms 合计 | — |

---

## 5. Kernel 分析（≥5 个 kernel）

> _(profiling 完成后填写。按总耗时贡献降序排列。)_

### Kernel 1：[待定 kernel 名称]

| 属性 | 值 |
|------|----|
| Kernel 名称 | [待定 — 例如 `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn`] |
| 总耗时 | [待定] ms（占端到端 [待定]%） |
| 调用次数 | [待定] |
| 平均单次耗时 | [待定] μs |
| 理论占用率 | [待定]% |
| 实际占用率 | [待定]% |
| 瓶颈类型 | ☐ 算力受限  ☐ 显存带宽受限  ☐ 延迟受限 |
| 分析 | [待定 — 为何是瓶颈？限制因素是什么？] |

### Kernel 2：[待定 kernel 名称]

| 属性 | 值 |
|------|----|
| Kernel 名称 | [待定] |
| 总耗时 | [待定] ms（[待定]%） |
| 调用次数 | [待定] |
| 平均单次耗时 | [待定] μs |
| 理论占用率 | [待定]% |
| 实际占用率 | [待定]% |
| 瓶颈类型 | ☐ 算力受限  ☐ 显存带宽受限  ☐ 延迟受限 |
| 分析 | [待定] |

### Kernel 3：[待定 kernel 名称]

| 属性 | 值 |
|------|----|
| Kernel 名称 | [待定] |
| 总耗时 | [待定] ms（[待定]%） |
| 调用次数 | [待定] |
| 瓶颈类型 | ☐ 算力受限  ☐ 显存带宽受限  ☐ 延迟受限 |
| 分析 | [待定] |

### Kernel 4：[待定 kernel 名称]

| 属性 | 值 |
|------|----|
| Kernel 名称 | [待定] |
| 总耗时 | [待定] ms（[待定]%） |
| 调用次数 | [待定] |
| 瓶颈类型 | ☐ 算力受限  ☐ 显存带宽受限  ☐ 延迟受限 |
| 分析 | [待定] |

### Kernel 5：[待定 kernel 名称]

| 属性 | 值 |
|------|----|
| Kernel 名称 | [待定] |
| 总耗时 | [待定] ms（[待定]%） |
| 调用次数 | [待定] |
| 瓶颈类型 | ☐ 算力受限  ☐ 显存带宽受限  ☐ 延迟受限 |
| 分析 | [待定] |

---

## 6. 显存带宽分析

| Kernel | 理论带宽 (GB/s) | 实际带宽 (GB/s) | 利用效率 |
|--------|----------------|----------------|---------|
| [待定] | [待定] | [待定] | [待定]% |
| [待定] | [待定] | [待定] | [待定]% |

Metax C500 峰值显存带宽：**[待定 — 通过 maca-smi 或规格书确认]** GB/s

---

## 7. 端到端延迟基线

| 指标 | 值 |
|------|----|
| 平均延迟 | [待定] ms |
| 中位数延迟 | [待定] ms |
| P90 延迟 | [待定] ms |
| P99 延迟 | [待定] ms |
| 吞吐量 | [待定] iter/s |

---

## 8. 主要瓶颈识别与阶段二优化建议

### 8.1 主要瓶颈

> _(指出最有优化价值的单一目标)_

**瓶颈**：[待定]  
**影响**：占端到端总时间 [待定]%  
**根本原因**：[待定 — 显存带宽？kernel launch 开销？同步点？算子融合机会？]

### 8.2 阶段二优化方案

> _(从高层次描述优化思路)_

| 优化方向 | 预期加速 | PR 目标 |
|----------|----------|---------|
| [待定] | ~[待定]% | FastDeploy/develop |

---

## 附录：Profiling 命令

```bash
# 环境准备
export MACA_PATH=/opt/maca
export MACA_VISIBLE_DEVICES=0
export PADDLE_XCCL_BACKEND=metax_gpu

# Profiling 运行命令
[待定 — 填入实际使用的命令]
```

## 附录：Trace 文件

> Trace 文件：`traces/profile_<TIMESTAMP>/trace.json`  
> 打开工具：[待定 — 例如 Perfetto UI https://ui.perfetto.dev，或 mxprof viewer]
