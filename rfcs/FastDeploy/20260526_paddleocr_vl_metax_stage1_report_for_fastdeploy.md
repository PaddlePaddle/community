# PaddleOCR-VL-1.5 在 MetaX C500 上的性能瓶颈分析

> Stage 1 RFC for the Hackathon 10 沐曦赛题
> *优化 PaddleOCR-VL-1.5 + Metax GPU*. To be submitted to
> [`PaddlePaddle/community/rfcs/FastDeploy`](https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy).
> GitHub ID: **linkeLi0421**.
> Repo with scripts and traces: <https://github.com/linkeLi0421/metax-paddleocr-vl>.

---

> **范围**：本文档是赛题 **Stage 1** 提交内容（性能瓶颈分析评估报告），
> 不包含 Stage 2 的优化承诺。Stage 2 由厂商 review 阶段一报告后 comment 指定
> 优化方向，作者再据此提交 PR 到 `PaddlePaddle/FastDeploy@develop`。
>
> 我们基于本次 profile 给出的 Stage 2 优化建议见
> [`stage2_optimization_plan.md`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/rfc/stage2_optimization_plan.md)（独立文档，
> 非本 RFC 的承诺部分）。

## 0. 摘要 / Summary

在 MetaX C500 64 GiB GPU 上，对 PaddleOCR-VL-1.5 v1.5 pipeline 做了一次端到端
推理 profiling。**入口是 `fastdeploy.LLM(...)` 进程内 API**，走的是 FastDeploy
的 metax 后端（`metax/attention/flash_attn_backend.py` → `flash_attn_kvcache`
custom op）。被测样本是 1100×690 的中文奖牌榜表格图片，`max_tokens=256`、
`quantization=wint8`、`use_cudagraph=False`。

### 关键观测

- **稳态端到端延迟 5.4 s/image**（warmup 后稳态，~21 ms/token）。
- **GPU device-busy 比例 ~22 %**——剩余 78 % wall 时间花在 host 端的 Python
  dispatch、kernel launch、CUDA runtime 调用上。
- 模型 = Siglip 视觉塔（27 层，每图 ~600 ms）+ Ernie-4.5 解码器（18 层 GQA
  `num_heads=16 / num_kv_heads=2`，每图自回归 ~256 tokens）。瓶颈在解码器。
- **Top 2 GPU kernel 占 device 时间 75.8 %**，且都是**已经深度融合**的算子：

| # | kernel | %device | 含义 |
|---|---|--:|---|
| 1 | `flash_fwd_splitkv_kernel` (FlashAttention split-K) | 47.9 % | 注意力主体（Q·K^T + softmax + ·V 全融合） |
| 2 | `mctlass::Kernel<...MacaGemmUniversal>` (wint8 GEMM) | 27.9 % | weight-only int8 的 QKV/O/MLP 投影 |
| 3 | `phi::RmsNormBlockSMemImpl` | 5.4 % | LayerNorm（部分已有 fused 版本） |
| 4 | `update_value_by_repeat_times` | 3.1 % | sampler 状态更新 |
| 5 | `b16gemvn_row_double_buffer_kernel` (prefill GEMV) | 2.6 % | prefill 阶段的 GEMV 变体 |

### 关键结论

- **算子端融合已经做得相当彻底**：FlashAttention 吸收了 softmax / GQA broadcast /
  attention scale；mctlass GEMM 接近 mcblas 带宽上限；KV cache 用 paged
  layout；MLP 激活已经融合。
- **真正可观的剩余空间在 host 端**：78 % wall 时间花在 Python dispatch / kernel
  launch / CUDA runtime 调用上。

Stage 2 该选哪些算子优化由厂商 review 后指定；我们基于本数据给出的优先级
建议见 [`stage2_optimization_plan.md`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/rfc/stage2_optimization_plan.md)。

---

## 1. 环境 / Environment

| 项 | 值 |
|---|---|
| Hardware | MetaX C500 64 GiB（曦云 C5xx，pcie 0000:38:00.0） |
| Driver | MX-SMI 2.2.9, Kernel Mode Driver 3.6.11, MACA 3.3.0.15 |
| OS / kernel | Ubuntu 24.04.1 LTS, Linux 5.15 |
| CPU / RAM | Intel Xeon-class, 128 logical cores / 128 GiB |
| Python | 3.10.10 (`/opt/conda`) |
| paddlepaddle | 3.4.0.dev20260127, commit `d979b30f5c35` |
| paddle-metax-gpu | 3.3.0.dev20260128+maca3.3.0.15 |
| FastDeploy | `release/2.5 @ a60a3e630`（编译自源码） |
| PaddleOCR / PaddleX | 3.5.0 / 3.5.2 |
| 模型 | PaddleOCR-VL-1.5（自动从 PaddleX 模型源下载），权重 `model.safetensors` 1.79 GiB |
---

## 2. 工作负载 / Workload

| 项 | 值 |
|---|---|
| 入口 | `fastdeploy.LLM(model=PaddleOCR-VL-1.5, ...)` Python API（in-process） |
| 输入图像 | `medal_table.png`，1100×690 RGB，965 KB |
| 内容 | 15 行 × 6 列的中文体育奖牌榜表格 |
| Prompt | `"OCR:"` + 图片 |
| 生成长度 | `max_tokens=256`、`temperature=0` |
| 数据类型 | 模型权重 wint8（int8 weight + bf16 input/output）；KV cache bf16 |
| 量化 | `quantization="wint8"` |
| CUDA Graph | `use_cudagraph=False`（为了拿干净的 per-kernel 数据；§4 详述） |
| 并行 | `tensor_parallel_size=1`，`max_num_seqs=4`，`max_model_len=2048` |

选这张图的原因：表格类输入会同时触发布局检测器（小型 CNN）和 PaddleOCR-VL 的
VL 解码（长 token 序列），既覆盖 vision encoder 又能让 decoder 跑到稳态，是
评估端到端瓶颈最有代表性的 workload。

### 2.1 稳态延迟

| 模式 | 每张图稳态延迟（warmup 后） |
|---|---:|
| 关闭 profiler（4 iters, 1 warmup）| **5.45 ± 0.31 s** |
| 打开 profiler（worker-side, 200 步 capture）| ~5.7 s |

profiler 只在 worker 子进程内某个窗口启用，主 wall time 几乎不受影响。

### 2.2 Profiler 实现

`paddle.profiler` 在 LLM 主进程里**抓不到 GPU 活动**——因为 FastDeploy 的
model forward 跑在 worker 子进程里。所以我们做了一个 worker-side patch：在
`fastdeploy/worker/worker_process.py` 的 `event_loop_normal` 里插入一段，
读 `FD_PROFILER_DIR` / `FD_PROFILER_WARMUP` / `FD_PROFILER_RECORD` 三个环境
变量，套 `paddle.profiler.Profiler` 包住 `execute_model` 调用：

- patch 脚本：[`scripts/patch_worker.py`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/scripts/patch_worker.py)
- profile driver：[`scripts/profile_fd.py`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/scripts/profile_fd.py)
- 本次抓了 **200 个连续 decode step** 的 trace，~320 MB

---

## 3. 推理框架调度 / Framework scheduling

整个 trace 共 **1,104,354 个 event**。按 category 汇总（200 步 capture，wall ~5.0 s）：

| category | calls | sum (ms) | %sum | 备注 |
|---|--:|--:|--:|---|
| Forward | 40,795 | 13,058 | 57.2 % | 模型嵌套 forward（与下层 overlap） |
| ProfileStep | 200 | 4,994 | 21.9 % | 200 步 capture 的 wall（每步 ~25 ms） |
| UserDefined | 38,422 | 1,164 | 5.1 % | pybind wrappers |
| **Kernel** | **43,585** | **1,077** | **4.7 %** | **device 执行** |
| Operator | 43,409 | 1,056 | 4.6 % | dygraph op dispatch |
| DygraphKernelLaunch | 45,598 | 731 | 3.2 % | kernel launch 胶水 |
| CudaRuntime | 396,772 | 662 | 2.9 % | `cudaLaunchKernel` 等 |
| Memcpy | 4,801 | 42 | 0.2 % | |

### 3.1 关键观察

1. **GPU device-busy 比例 ≈ 22 %**（device kernel 时间 1.077 s ÷ wall ~5.0 s）。
   剩下 78 % wall 落在 host 端：Python 字节码、operator 元数据准备、
   `cudaLaunchKernel` API、stream 同步。
2. **每次 decode step（生成 1 token）触发 ~2 K cudaRuntime 调用 + ~218 Kernel
   events**。其中 `cudaLaunchKernel_v7000` 占比最大，与 `DygraphKernelLaunch`
   数量近似匹配——平均每个 dygraph op 还是触发一次独立的 GPU launch。
3. **没有 CUDA Graph capture**。`use_cudagraph=False`（默认）且没有静态图。
   FastDeploy 实际上有 CUDA Graph 基础设施（`fastdeploy/model_executor/
   graph_optimization/`），只是默认未开启，且当前只 bucket
   `batch_size=1/2/4 × num_tokens=1`，不覆盖变长 KV。这是 §6 Plan G 的入口。
4. **没有显著的 H2D/D2H bottleneck**：Memcpy 仅 0.2 %。

### 3.2 解码循环形状（从 Forward 计数反推）

| Forward 节点 | calls | 每 step | 解读 |
|---|--:|--:|---|
| `PaddleOCRVLForConditionalGeneration` | 199 | 1 | 每 step 一次调用 |
| `Ernie4_5_DecoderLayer` | 3,582 | 18 | 18 layers × 199 steps |
| `Ernie4_5_Attention` | 3,582 | 18 | 每层 1 次 |
| `Attention` | 3,582 | 18 | FastDeploy `Attention` 抽象层 |
| `Ernie4_5_MLP` | 3,582 | 18 | 每层 1 次 |
| `RMSNorm` | 7,363 | ~37 | 略多于 36 = 18 × 2，可能含 sample/embedding 路径 |
| `QKVParallelLinear` | 3,582 | 18 | QKV 投影 |
| `MergedColumnParallelLinear` | 3,582 | 18 | MLP 第一层（gate+up 合并？） |
| `RowParallelLinear` | 7,164 | 36 | output proj + MLP down |

由此推出模型结构（与正式 spec 一致）：

```
[image] → SiglipVisionTransformer (27 layers, 一次, ~600 ms / image)
            ↓ visual tokens
[prompt + visual tokens] → Ernie-4.5 decoder
                            18 transformer layers
                            num_heads=16, num_kv_heads=2 (GQA, 8× share)
                            自回归生成 ~256 tokens
                            ~4.6 s / image (decode 主导)
```

### 3.3 每 token wall 时间分解

| Phase | wall (ms / token) | 来源 |
|---|--:|---|
| Decode step total | ~25 | ProfileStep 平均 (4994 ms / 200) |
| —— 其中 device 计算 | ~5.4 | Kernel sum / 200 |
| —— 其中 host overhead (dispatch / launch / sync) | ~20 | 差值 |

**host overhead 仍然是 device 计算的 ~4×**，跟原始 paddleformers offline 路径
的"~2×"相比反而更悬殊——因为 FastDeploy 把每个 step 的 device 计算压紧之后，
host 路径没有变快多少，相对占比就上升了。

→ 这直接指向 **CUDA Graph 是最大杠杆**。

---

## 4. GPU 利用率 / GPU utilization

`cat == "Kernel"` 时间合计 1.077 s / capture wall ~5.0 s = **~21.5 % device-busy**。

按 phase 分布（FastDeploy 不像 PaddleOCR-VL CLI 那样把 vision tower 和
decoder 分到独立 Forward node，但可以从 `DispatchCacheKVWithRopeVecKernel`
触发次数推断 200 步 capture 几乎全部在 decode 阶段）：

| Phase | wall (ms) | Kernel (ms) | busy % |
|---|--:|--:|--:|
| Decode（本次 capture 覆盖） | ~5,000 | 1,077 | ~21.5 % |

mx-smi 推理期间 `GPU-Util` 在 0 % 与瞬时高峰之间快速跳变（采样间隔 1 s，分
辨率不足以观察单 token 内的占空比）；峰值显存 ~23 GiB / 64 GiB（含 wint8
权重 + KV cache + 中间激活）。device 远没打满。

### 4.1 22 % 这个数为什么这么稳

我们也独立测过 paddleformers offline 路径（不走 FastDeploy 后端，详见
[`exploration_paddleformers_offline.md`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/rfc/exploration_paddleformers_offline.md)），
GPU device-busy 同样是 **~22 %**——尽管那条路径的算子完全没融合、CUDA runtime
调用是这条的 8 倍。

**这说明算子端融合并不能直接提升 device-busy**：FastDeploy 后端节省下来的 host
时间立即让下一 token 提前开始，host 空隙等比例缩小，device-busy 几乎不变。

要真正打破 22 % 上限，必须 **CUDA Graph** 把 host loop 整体消掉。

---

## 5. 热点 kernel 分析 / Hot-kernel deep dive

完整 top-30 在 [`traces/fd_backend/top_kernels.txt`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/traces/fd_backend/top_kernels.txt)。
本节按 device 时间占比详细分析前 5 个，再加 5 个补充。

### 5.1 `flash_fwd_splitkv_kernel` —— 47.9 %

| 项 | 值 |
|---|---|
| 全称 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, ...>>` |
| 调用次数 | 3,582 = 18 layers × 199 decode steps |
| 总时间 | 515.5 ms / 200 步 |
| 平均时长 | 143.9 µs / call |
| 输入形状 | q=`[1, 16, 1, 128]`、k_cache/v_cache=`[blocks, block_size=64, 2, 128]`（GQA broadcast 在 kernel 内处理） |
| Verdict | **memory-bound**（FlashAttention split-K 已经是 SOTA，主要受 HBM 带宽限制） |

**为什么是 splitkv 变体**：解码阶段每步 q=1 个新 query token、k_cache 长度
~700-1100、num_heads_q=16、num_kv_heads=2。Split-K 把 K 维度切片让多个 thread
block 并行处理同一 query 行——这是 FA decoding 的标准做法。

**还能不能优化**：
- 143 µs / call 已经接近 mcdnn 的 mxbench 单点能拿到的水平。继续吃 device
  时间空间小。
- 但**这个 kernel 仍占 wall 时间的 ~9.6 %**（47.9 % × 22 % = 10.5 % wall）；
  如果能让 18 层 attention 在 stream 上 overlap（目前是串行），能省下一部分
  launch latency。CUDA Graph 帮不上这个忙（FA 本身已经是单 kernel launch）。

### 5.2 `mctlass::Kernel<...MacaGemmUniversal>` —— 27.9 %

| 项 | 值 |
|---|---|
| 全称 | mctlass 模板 GEMM kernel（fp32 accumulator + bf16 input + int8 weight） |
| 调用次数 | 14,328 = ~4 GEMM/layer × 18 layers × 199 steps |
| 总时间 | 300.5 ms / 200 步 |
| 平均时长 | 21.0 µs / call |
| 对应 paddle 算子 | `weight_only_linear` |
| 对应 LLM 层 | QKV projection、output projection、MLP up/down/gate |
| Verdict | **memory-bound**（wint8 weight 读取，从 HBM 读 int8 + dequant 到 bf16） |

**优化方向**：
1. **`mctlass` 是闭源的**——能改的不是 kernel 本身，而是**调度它的代码**：
   - 把 attention 的 Q/K/V/O 4 个 `weight_only_linear` 合并成 `merged_linear`
     （或叫 `QKVParallelLinear`，但当前观察到的实现似乎没用最大批化）
   - 调 `MacaGemmUniversal` 的 split-K / tile size 参数（mctlass 是模板库，
     不同输入形状需要不同 tile，可能存在一组没调到最优的 shape）
2. **从 wint8 进一步降到 wint4**：weight bandwidth 减半，预期 ~1.4× 提速
   （前提是数值精度可接受，需要小批校准）
3. **观察单 call 时长**：21 µs × 14,328 = 301 ms，距离 mcblas 上界差距小；
   最大杠杆是减少 call 次数（合并 Linear）而不是单 kernel 优化。

### 5.3 `phi::RmsNormBlockSMemImpl` —— 5.4 %

| 项 | 值 |
|---|---|
| 调用次数 | 7,164 = 2 × 18 × 199 |
| 总时间 | 57.9 ms / 200 步 |
| 平均时长 | 8.1 µs / call |

模型每层有 2 个 RMSNorm（attention 前 + MLP 前）。FastDeploy 已经有融合 op
`fused_rms_norm_quant`（把 RMSNorm + quantize 合并成一个 op），但**只在
attention 前的 norm 走融合路径**，MLP 前的 norm 仍调原始 `cuApplyRMSNorm`。

→ **§6 Plan R**：审计 `paddleocr_vl.py` / `ernie4_5_moe.py` 里两处 RMSNorm
的 dispatch，让 MLP-前那一处也走 fused 版本。低风险，落地快。

### 5.4 `update_value_by_repeat_times<float>` —— 3.1 %

| 项 | 值 |
|---|---|
| 调用次数 | 199 = 每 step 1 次 |
| 总时间 | 33.6 ms / 200 步 |
| 平均时长 | 168.8 µs / call |

调用在 sampler / scheduler 路径（非 attention/MLP 主链）。**168 µs / call 是
个相对大的开销**，每 step 都跑一次。值得 follow-up 看是什么场景下用到的，
可能存在一个 trivial 的 short-circuit 优化。

### 5.5 `b16gemvn_row_double_buffer_kernel<256,4,8,bf16>` —— 2.6 %

| 项 | 值 |
|---|---|
| 调用次数 | 199 = 每 step 1 次 |
| 总时间 | 28.1 ms / 200 步 |
| 平均时长 | 141.4 µs / call |

每 step 一次的 GEMV，141 µs 偏大。**很可能是 LM head 投影**（hidden → vocab，
`[1, hidden]` × `[hidden, vocab≈100K]`）。LM head 没走 wint8 quant 的话会留
在 bf16 直接 GEMV，从 HBM 读 ~256 MB weight，141 µs 对应 ~1.8 TB/s 有效带宽
（接近 C500 上限）。优化空间小。

### 5.6 补充 5 个

| # | 名称 | %device | 一句话定位 |
|---|---|--:|---|
| 6 | `DispatchCacheKVWithRopeVecKernel<bf16,...>` | 2.5 % | RoPE + 写 KV cache 的融合 op（已经是 fused 实现，每层每 step 1 次） |
| 7 | `TopPSamplingFromProbKernel` | 2.4 % | sampler，每 step 1 次 |
| 8 | `phi::fusion::ActFFNGlu<bf16,...>` | 2.1 % | SiLU + Mul 融合（MLP 激活）—— 已经是 fused |
| 9 | `mcdnn::KernelSoftmaxForwardInstanceLdgB128` | 1.4 % | 残留的 softmax（只在 sampler 处用，主体已融入 FlashAttention） |
| 10 | `phi::funcs::VectorizedElementwiseKernel<int,...>` | 0.6 % | 小型 elementwise |

**Top 10 累计 95.4 %**。Top 2 (FlashAttention + mctlass GEMM) 一共 75.8 %，
是 device 端两个最大杠杆。

---

## 6. 复现 / Reproduction

仓库：<https://github.com/linkeLi0421/metax-paddleocr-vl>

### 环境

```bash
# 在 GiteeAI compute 的 MetaX C500 容器中
source scripts/fd_env.sh                  # MACA + cu-bridge 环境变量
export PATH=/opt/conda/bin:$PATH          # conda base = Python 3.10
# 已安装：
#   paddlepaddle==3.4.0.dev20260127
#   paddle-metax-gpu==3.3.0.dev20260128+maca3.3.0.15
#   paddleocr==3.5.0 (含 doc-parser extras)
#   fastdeploy-metax-gpu==2.5.0 (源码编译，见 docs/00_warmup_checkin.md)
```

### 复现端到端推理（基线时延）

```bash
python scripts/profile_fd.py \
    --image /data/work/poc_test/doc.png \
    --iters 4 --warmup 1 --no-profiler \
    --max-tokens 256 \
    --out-dir /data/work/poc_test/traces_fd_smoke
```

### 复现 profiling trace（§3–§5 数据）

```bash
# 1) patch worker_process.py once so the worker subprocess can profile
python scripts/patch_worker.py

# 2) capture trace via in-process LLM (worker writes to FD_PROFILER_DIR)
FD_PROFILER_DIR=/data/work/poc_test/traces_fd_worker \
FD_PROFILER_WARMUP=10 \
FD_PROFILER_RECORD=200 \
python scripts/profile_fd.py \
    --image /data/work/poc_test/doc.png \
    --iters 3 --warmup 1 --no-profiler \
    --max-tokens 256

# 3) aggregate + top kernels
python scripts/aggregate_trace.py \
    /data/work/poc_test/traces_fd_worker/host_*.paddle_trace.json \
    --top 25 --out traces/fd_backend/summary.json
python scripts/top_kernels.py \
    /data/work/poc_test/traces_fd_worker/host_*.paddle_trace.sanitized.json \
    --top 50 > traces/fd_backend/top_kernels.txt
```

### 已提交的产物

| 文件 | 大小 | 内容 |
|---|--:|---|
| [`traces/fd_backend/wall_times.json`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/traces/fd_backend/wall_times.json) | 373 B | 各 iter 的 wall time |
| [`traces/fd_backend/summary.json`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/traces/fd_backend/summary.json) | 70 KB | 全部 event 按 category × name 聚合 |
| [`traces/fd_backend/top_kernels.txt`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/traces/fd_backend/top_kernels.txt) | 3.3 KB | 设备端 top-30 kernel |
| [`scripts/profile_fd.py`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/scripts/profile_fd.py) | — | profile driver（`fastdeploy.LLM` in-process） |
| [`scripts/patch_worker.py`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/scripts/patch_worker.py) | — | 给 worker_process.py 注入 paddle.profiler |
| [`scripts/aggregate_trace.py`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/scripts/aggregate_trace.py) | — | 流式解析 chrome_tracing JSON，聚合 |
| [`scripts/top_kernels.py`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/scripts/top_kernels.py) | — | top GPU kernel 排序 |
| [`docs/JOURNAL.md`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/docs/JOURNAL.md) | — | 实施过程的时序记录 |
| [`stage2_optimization_plan.md`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/rfc/stage2_optimization_plan.md) | — | Stage 2 优化建议（独立文档，非本 RFC 承诺） |
| [`exploration_paddleformers_offline.md`](https://github.com/linkeLi0421/metax-paddleocr-vl/blob/main/rfc/exploration_paddleformers_offline.md) | — | 早期 paddleformers offline 路径的探索性 profile（非本 RFC 承诺） |

原始 320 MB chrome_tracing JSON 保留在容器 `/data/work/poc_test/traces_fd_worker/`
下，若评审需要可按需上传。

---
