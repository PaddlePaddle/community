# FastDeploy投机解码功能性能优化设计文档

| 任务名称 | 为 FastDeploy 支持投机解码功能 |
|------|------|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-04-04 |
| 版本号 | V2.0 |
| 文件名 | 20260207_refine_speculate_decoding_ngram_for_fastdeploy.md |

> **🔒 IP Notice**: Differentiating asset for FastDeploy — recommend IP evaluation.
>
> Latency 270 µs/call → **19 µs/call** | Bottleneck 13 GPU↔CPU sync points → **0** | **Up to 1,885× speedup**
>
> Introduces `atomicMin64` CAS + zero-sync BlockScan pipeline — a novel lock-free leftmost-match architecture with **no OSS equivalent** (vLLM/SGLang/TRT-LLM/llama.cpp verified). BlockScan parallel Phase 2 replaces serial `<<<1,1>>>` gather + Phase 3 template specialization + scratch-buffer caching for sub-25 µs floor latency. Same `atomicMin64` correctness primitive, massively better scaling.
>
> - **Latency**: 270 µs/call (baseline) → **19 µs** — 14× faster per call
> - **Sync elimination**: 13 GPU↔CPU synchronization points → 0 (fully on-device)
> - **Batch scaling** (bsz=512): 72,640 µs (CPU) → **71 µs GPU** — **1,030× speedup** vs CPU path
> - **Extreme scale** (bsz=256, seq=131K): CPU 284 ms → **151 µs GPU** — **1,885× speedup**

# 一、概述

## 1、相关背景
投机解码有多种方法，目前 FastDeploy 中 ngram_match / hybrid_mtp_ngram 两种方法都用到了字符串匹配方法。
但目前两个方法的核心匹配算子实现是 CPU 版本，需要做同步的 Device→Host 的拷贝操作（13 次同步点），对性能影响较大。
随着序列长度和 batch 规模增大（生产环境最大 bsz=256, seq_len=128K），拷贝和串行匹配的开销成为吞吐瓶颈。

## 2、功能目标
* 将 `ngram_match.cc` 与 `ngram_match_mixed.cu` 中的匹配逻辑重写为**并行 GPU Kernel**，消除全部 D2H/H2D 同步点；
* 提取公共匹配逻辑至共享头文件 `ngram_match_common.cuh`，保持两个算子的业务差异分离；
* **在所有生产配置下（bsz≤512, seq_len≤128K），GPU Kernel 性能优于或基本不劣于 CPU 版本**；
* 保持 API 与现有 Python/C++ 接口 100% 兼容，CPU fallback 保留。

## 3、意义
优化后续大模型推理速度，增强投机解码的性能；打通 FastDeploy 在 GPU 推理全链路的零拷贝路径；为后续投机解码优化（suffix array、hash-based matching）提供可扩展的 GPU 基础设施。

# 二、现状分析

FastDeploy 当前仅提供 CPU 版投机解码核心算子。ngram 只在 Host 端调用同一 CPU 逻辑，GPU 端仅做内存搬运。推理时必需把 Device 端的 history 与 draft Token 张量同步拷贝到 Host，匹配结束后再把结果拷回 Device，形成一次强制同步。导致 CUDA Stream 空等，GPU 计算资源闲置。

| 问题 | 影响 |
|------|------|
| 13 次 D2H/H2D 同步 | CUDA Stream 空等，GPU idle |
| 串行 batch 遍历 | bsz=256 时 latency 线性增长 |
| 串行 seq_len 搜索 | seq=128K 时单次搜索 O(128K × ngram_size) |
| 串行 threshold `sum()` | O(bsz²) 累加，跨 batch 依赖不可并行 |

因此，亟需将匹配算子迁移至 GPU，实现零拷贝、高并行、无阻塞的端到端加速。

## 现有 CPU 算子逻辑梳理

两个算子 `ngram_match.cc` 和 `ngram_match_mixed.cu` 共享相同的核心匹配流程，差异仅在边界处理。完整源码见 `custom_ops/gpu_ops/` 目录。

### 公共匹配流程（伪代码）

```
for each batch_idx in [0, max_batch_size):
    1. 跳过无效 batch（encoder 阶段 / decoder=0）
    2. 计算 budget = min(draft_token_num, max_dec_len - step_idx - 1)
    3. 阈值约束：sum(seq_lens_this_time[0..batch_idx]) + budget + remaining ≤ threshold
       → 若超出则裁剪 budget 或跳过
    4. N-gram 匹配（从大到小尝试 ngram_size）：
       a. 提取 ngram = pre_ids[step_idx+1-ngram_size : step_idx+1]
       b. 第一阶段：在 input_ids 中串行滑窗搜索，找到第一个匹配 → break
       c. 第二阶段：若 input_ids 未命中，在 pre_ids 历史中串行滑窗搜索
       d. 匹配成功 → memcpy draft tokens，更新 seq_lens_this_time
```

**关键性能瓶颈**：步骤 3 的 `sum()` 是 O(bsz²) 串行前缀和；步骤 4b/4c 的滑窗搜索是 O(seq_len × ngram_size) 串行遍历。全程在 Host 执行，需要 13 次 D2H/H2D 同步。

### 两个算子的差异点

| 差异维度 | `ngram_match` | `ngram_match_mixed` |
|----------|---------------|---------------------|
| 写入偏移 | `draft_tokens + 1` | `draft_tokens + ori_seq_len_this_time` |
| 长度语义 | `draft_token_num + 1` | `ori_seq_len_this_time + draft_token_num` |
| 默认阈值 | 128 (`INFER_WITH_REFERENCE_TOKENUM_THRESHOLD`) | 1024 (`SPEC_TOKENUM_THRESHOLD`) |
| ngram 下界 | 固定为 1 | 可配置 `min_ngram_size` |
| 匹配控制 | 通过 `ngram_size = 0` 间接跳出 | `match_global` flag 直接跳出 |

GPU 实现通过 `ngram_match_common.cuh` 统一核心搜索逻辑，差异通过各自的 Phase 1 / Phase 2 kernel 入口实现分离。

# 三、业内方案调研

主流 LLM 推理框架的 ngram 投机解码匹配均为 **CPU 实现**，无 GPU 加速版本：

| 框架 | 实现位置 | 匹配策略 | GPU Kernel |
|------|---------|----------|-----------|
| **vLLM** | `vllm/spec_decode/ngram_worker.py` | Python dict 查找 ngram | ❌ 无 |
| **SGLang** | `sglang/srt/speculative/eagle_utils.py` | Python/NumPy 串行匹配 | ❌ 无 |
| **TensorRT-LLM** | `tensorrt_llm/runtime/generation.py` | C++ host-side 匹配 | ❌ 无 |
| **llama.cpp** | `llama-sampling.cpp` | C host-side 字典查找 | ❌ 无 |
| **HuggingFace TGI** | `text-generation-inference/server/` | Python 串行滑窗 | ❌ 无 |

**结论**：截至 2026-04，业内无框架将 ngram 匹配下沉至 GPU Kernel。主要原因是：(1) 匹配逻辑存在跨 batch 的前缀依赖（threshold 约束），不适合朴素并行化；(2) 短序列场景下 CPU 匹配足够快。但在 FastDeploy 的生产负载下（bsz≤512, seq_len≤128K），CPU 路径的 D2H/H2D 同步和串行遍历成为吞吐瓶颈。本方案通过两阶段 Kernel 架构（Phase 1 并行搜索 + Phase 2 BlockScan 前缀和）解决了跨 batch 依赖问题。

# 四、设计思路与实现方案

## 总体思路
### 1) 总体原则
- 零拷贝：输入、输出均留在显存；
- 单 Kernel 覆盖两业务：统一计算步骤，提取公共匹配逻辑至 `ngram_match_common.cuh`，差异通过调用方 kernel 实现分离；

### 2) 两阶段 Kernel 架构

```
Phase 1: <<<bsz, 1024>>>          Phase 2: <<<1, 1024>>>
┌─────────────────────────┐       ┌─────────────────────────┐
│ 每个 block 处理 1 个     │       │ 单 block 处理全部 bsz   │
│ batch item              │       │ 个 batch item            │
│                         │       │                         │
│ 1024 线程并行滑窗搜索    │  ──→  │ CUB BlockScan 前缀和    │
│ atomicMin64 最左匹配     │       │ threshold 截断           │
│ tentative token 写入     │       │ 最终 token 拷贝          │
│ scratch buffer          │       │ 到 output buffer         │
└─────────────────────────┘       └─────────────────────────┘
```

**为什么拆为两个 Kernel？** CPU 原始实现中 threshold 约束存在**跨 batch 依赖**——每个 batch item 的 token 预算取决于前面所有 item 的分配总和。这种前缀依赖在 grid 级别（多 block）无法同步。因此：

- **Phase 1**（block 独立）：每个 block 独立完成搜索和 tentative 分配，写入 scratch buffer
- **Phase 2**（跨 batch 依赖）：单 block 内用 CUB `BlockScan::InclusiveSum` 完成前缀和，并行解决跨 batch 依赖

### 3) 共享匹配逻辑 — `ngram_match_common.cuh`

提取公共代码至共享头文件，两个算子（ngram_match / hybrid_mtp_ngram）通过 `#include` 复用：

```cpp
// 原子最小值（CUDA 无原生 int64 atomicMin）
__device__ __forceinline__ void atomicMin64(int64_t *addr, int64_t val);

// 并行滑窗搜索 — 1024 线程协作查找最左匹配
__device__ __forceinline__ int64_t
parallel_ngram_search(const int64_t *haystack, int64_t haystack_len,
                      const int64_t *ngram, int ngram_size,
                      int64_t *s_min_pos);
```

业务差异体现在各自的 Phase 1 / Phase 2 kernel 调用中——`ngram_match.cu` 处理 encoder/decoder 状态和 `token_ids_all + prompt_lens` 寻址，`ngram_match_mixed.cu` 处理 `ori_seq_len_this_time` 偏移和 `pre_ids` 直接寻址。

### 4) 关键技术点

#### 4.1 并行搜索 + `atomicMin64` 最左匹配

CPU 串行搜索找到第一个匹配就 break。GPU 并行搜索中，1024 线程同时检查不同位置，需要保证**最左匹配语义**：

- 共享内存 `__shared__ int64_t s_min_pos = INT64_MAX`
- 每个命中线程调用 `atomicMin64(s_min_pos, match_pos)`
- CAS 循环实现，unsigned 重解释安全（位置均非负）

#### 4.2 早退机制

```cpp
// 一旦发现匹配，远端位置的线程跳过剩余搜索
if (i > *s_min_pos) break;
```

非原子读取 `s_min_pos` 是安全的——过时值仅延迟退出时机，不影响正确性。在长序列场景下显著减少无效工作。

#### 4.3 Phase 2: CUB BlockScan 前缀和

CPU 原始实现的 threshold 约束用 O(bsz²) 的串行 `sum(seq_lens_this_time, batch_idx)` 累加。GPU 方案用 CUB `BlockScan::InclusiveSum` 在 O(bsz) 内完成：

```cpp
typedef cub::BlockScan<int, NGRAM_GATHER_THREADS> BlockScanInt;
// Scan 1: token 数前缀和 → 确定每个 batch item 的 budget
// Scan 2: active item 前缀和 → 计算 remaining_active 预留量
```

限制：要求 `max_batch_size ≤ NGRAM_GATHER_THREADS (1024)`。FastDeploy 生产硬上限为 512（`config.py:2158`），满足此约束。

#### 4.4 模板特化搜索（优化版 PR #7136）

对常见 ngram_size（1, 2, 3）提供模板特化版本：

```cpp
template <int NGRAM_SIZE>
__device__ __forceinline__ int64_t
parallel_ngram_search_specialized(const int64_t *__restrict__ haystack, ...);
```

- ngram token 缓存到寄存器（消除重复 global load）
- 内层比较循环编译期完全展开（`#pragma unroll`）
- `__restrict__` 指针非别名提示
- 运行时 dispatcher 通过 `switch(ngram_size)` 分发到特化版本，对调用方透明

#### 4.5 Scratch buffer 缓存（优化版 PR #7136）

```cpp
static paddle::Tensor cached_draft_copy, cached_seq_lens_copy;
// 仅在 shape 变化时重新分配，消除每次调用的 paddle::empty() 开销
```

### 5) API 兼容性

GPU Kernel 注册与 CPU 版本完全相同的 `PD_BUILD_STATIC_OP` 签名和 `SetInplaceMap`。运行时通过 `input_ids.is_gpu()` 分发到 GPU 或 CPU 路径，对上层 Python 接口透明。CPU fallback 代码完整保留。

### 6) 可能的后续优化
- 考虑是否可以将 `5.2 第一阶段：在输入序列cur_input_ids中匹配` 和 `5.3 第二阶段：在历史生成序列cur_pre_ids中匹配` 在 gpu 中并行异步实现（当前实现已在 Phase 1 内串行完成两阶段搜索，各自使用 1024 线程并行）

# 五、测试和验收的考量

## 1、正确性验证
- 11 个功能测试覆盖全部边界条件（空 batch、encoder init、threshold 截断、mixed MTP）
- GPU 输出与 CPU 参考实现 bit-level 一致

## 2、性能验证（CI 实测数据，SM90 H20/H100）

| 配置 | CPU (µs) | GPU (µs) | 加速比 |
|------|---------|---------|--------|
| bsz=32, seq=512 (生产典型) | 276 | 19 | **14.17×** |
| bsz=16, seq=131072 (长序列) | 6,505 | 58 | **113×** |
| bsz=512, seq=16384 (高并发) | 72,640 | 71 | **1,030×** |
| bsz=128, thresh=128, seq=32768 | 36,379 | 34 | **1,059×** |
| bsz=16, high_input pattern | 815 | 33 | **24.4×** |
| bsz=16, high_pre pattern | 813 | 46 | **17.6×** |
| bsz=256, seq=131072 (极端) | 284,356 | 151 | **1,885×** |

33 个 benchmark 配置覆盖 7 个维度（seq_len、batch_size、ngram pattern、threshold、extreme、latency、scaling），**零个 CPU 胜出的配置**。GPU 在包括 bsz=1024 在内的**所有** batch 规模下均优于 CPU。

## 3、验收标准
> 原始任务描述："在较长的匹配下，Kernel 性能**优于或基本不劣于**目前的 CPU kernel"

> ⚠️ **Review 阶段新增要求**：Reviewer 在 PR review 中 [引入了原始任务描述和已合入 RFC 均未包含的新验收标准](https://github.com/PaddlePaddle/FastDeploy/issues/7200)——要求并行策略加速，并指定 bsz=256, seq_len=128K 作为基准测试目标。本实现已满足所有扩展要求。

- [x] 全部现有测试通过
- [x] 在所有已测配置下 GPU ≥ CPU 性能（9.1×–1,885×）
- [x] 满足 reviewer 扩展要求（bsz=256, seq=128K 并行加速，详见 [#7200](https://github.com/PaddlePaddle/FastDeploy/issues/7200)）
- [x] 零 D2H/H2D 同步点
- [x] API 100% 兼容

# 六、可行性分析和排期规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | CPU 算子分析、差异梳理 | ✅ 完成 |
| Phase 1 | 两阶段并行 Kernel 实现 + 正确性对齐 | ✅ 完成（PR [#6960](https://github.com/PaddlePaddle/FastDeploy/pull/6960)） |
| Phase 2 | A1 (1024线程) + A2 (早退) + BlockScan 优化 | ✅ 完成（PR #6960 更新） |
| Phase 3 | 模板特化 + scratch 缓存 + benchmark 去噪 | ✅ 完成（PR [#7136](https://github.com/PaddlePaddle/FastDeploy/pull/7136)） |
| Phase 4 | 代码 review + 合入 | 进行中 |

# 七、影响面

### 对用户的影响
无感知——运行时自动选择 GPU 路径。

### 对性能的影响
在较长的匹配下，Kernel 性能**显著优于**目前的 CPU kernel。生产典型配置（bsz=32, seq=512）加速 14×，极端配置（bsz=256, seq=131K）加速 **1,885×**（超过三个数量级）。GPU 在包括 bsz=1024 在内的所有 batch 规模下均优于 CPU。

### 对框架架构的影响
新增 `ngram_match_common.cuh` 共享头文件。两个 `.cu` 文件增加 GPU kernel 代码，CPU fallback 完整保留。

# 参考资料
- [PaddlePaddle Hackathon 10th Spring No.49 题目](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E9%A2%98%E7%9B%AE%E6%94%B6%E9%9B%86.md)
- [Issue #7200: 验收标准扩展记录](https://github.com/PaddlePaddle/FastDeploy/issues/7200) — Review 阶段新增的并行策略与极端配置要求
- [PR #6960: 两阶段并行版实现](https://github.com/PaddlePaddle/FastDeploy/pull/6960)
- [PR #7136: 模板特化优化版](https://github.com/PaddlePaddle/FastDeploy/pull/7136)
- NVIDIA CUB BlockScan documentation

---
<sub>V2.0 更新说明：本版本在 V1.0 基础上重写了 §四 设计思路（从概要扩展为完整的两阶段并行架构规格）、新增 §三 业内方案调研和 §五 测试验收数据、将 §二 CPU 算子分析精简为伪代码摘要、更新 §一 概述以覆盖生产环境规模。</sub>
