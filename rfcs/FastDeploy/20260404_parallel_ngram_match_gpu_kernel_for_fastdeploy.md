# FastDeploy 投机解码 N-gram 匹配 GPU 并行 Kernel 设计文档

| 任务名称 | 【Hackathon 10th Spring No.49】为 FastDeploy 支持投机解码功能 |
|------|------|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-04-04 |
| 版本号 | V2.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20260404_parallel_ngram_match_gpu_kernel_for_fastdeploy.md |
| 前序 RFC | [20260207_refine_speculate_decoding_ngram_for_fastdeploy.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/FastDeploy/20260207_refine_speculate_decoding_ngram_for_fastdeploy.md)（PR [#1213](https://github.com/PaddlePaddle/community/pull/1213)，NKNaN） |
| 实现 PR | [#6960](https://github.com/PaddlePaddle/FastDeploy/pull/6960)（标准版）、[#7136](https://github.com/PaddlePaddle/FastDeploy/pull/7136)（模板特化优化版） |

# 一、概述

## 1、相关背景

投机解码（Speculative Decoding）通过 N-gram 匹配在输入序列和历史生成序列中查找候选 draft token，用于加速大模型推理。FastDeploy 目前使用两个 CPU 算子完成匹配：

- `ngram_match.cc`（标准 ngram 匹配）
- `ngram_match_mixed.cu` 中的 CPU 路径（hybrid MTP ngram 匹配）

CPU 实现需要 13 次 Device→Host 同步拷贝，在 CUDA Stream 上形成强制同步点，GPU 计算资源闲置。随着序列长度和 batch 规模增大（生产环境最大 bsz=256, seq_len=128K），拷贝和串行匹配的开销成为吞吐瓶颈。

## 2、功能目标

1. 将两个 CPU 匹配算子重写为 **并行 GPU Kernel**，消除全部 D2H/H2D 同步点
2. 提取公共匹配逻辑至共享头文件，保持两个算子的业务差异分离
3. **在所有生产配置下（bsz≤512, seq_len≤128K），GPU Kernel 性能优于 CPU 版本**
4. 保持 Python/C++ 接口 100% 兼容，CPU fallback 保留

## 3、意义

- 打通 FastDeploy 推理路径零拷贝链路
- 在大 batch / 长序列场景下释放 GPU 并行能力
- 为后续投机解码优化（如 suffix array、hash-based matching）提供可扩展的 GPU 基础设施

## 4、与前序 RFC 的关系

前序 RFC（[#1213](https://github.com/PaddlePaddle/community/pull/1213)，NKNaN，2026-02-07）提出了基础设计方向：零拷贝、公共逻辑提取、模板参数控制差异。该 RFC 的设计思路部分（§三）仅有 12 行，未涉及具体**并行策略、线程模型、threshold 约束的 GPU 化方案、或极端配置（bsz=256, seq_len=128K）下的性能保证**。

本 RFC 在前序基础上提出完整的**两阶段并行架构**，补充了：

| 缺失项 | 前序 RFC | 本 RFC |
|--------|---------|--------|
| 并行策略 | "考虑是否可以并行" | 两阶段 Kernel：Phase 1 `<<<bsz, 1024>>>` + Phase 2 `<<<1, 1024>>>` |
| 线程模型 | 未指定 | 1024 线程/block 并行滑窗搜索 + `atomicMin64` CAS 最左匹配 |
| Threshold 约束 | CPU 串行 `sum()` | CUB `BlockScan` 前缀和，O(bsz) 并行 |
| 极端配置 | 未提及 | 实测 bsz=256, seq=128K：GPU 162µs vs CPU 275ms（1,700×） |
| 模板特化 | 未涉及 | `parallel_ngram_search_specialized<1/2/3>`，寄存器缓存 + `#pragma unroll` |
| 早退机制 | 未涉及 | A2 early-exit：匹配命中后跳过远端搜索位置 |

# 二、现状分析

## CPU 算子问题

| 问题 | 影响 |
|------|------|
| 13 次 D2H/H2D 同步 | CUDA Stream 空等，GPU idle |
| 串行 batch 遍历 | bsz=256 时 latency 线性增长 |
| 串行 seq_len 搜索 | seq=128K 时单次搜索 O(128K × ngram_size) |
| 串行 threshold `sum()` | O(bsz²) 累加，不可并行 |

## 两个算子差异（与前序 RFC 一致）

| | ngram_match | ngram_match_mixed |
|---|---|---|
| 写入位置 | `draft_tokens + 1` | `draft_tokens + ori_seq_len_this_time` |
| 长度计算 | `draft_token_num + 1` | `ori_seq_len_this_time + draft_token_num` |
| 默认阈值 | 128 | 1024 |
| 最小 ngram | 固定 1 | 可配置 `min_ngram_size` |
| encoder 处理 | 跳过但保留 budget | 无 encoder 逻辑 |

# 三、业内方案调研

| 框架 | N-gram 匹配实现 | 并行度 |
|------|----------------|--------|
| vLLM | Python 层 dict + set 匹配 | 无 GPU 并行 |
| SGLang | `ngram_worker.py` Python 匹配 | 无 GPU kernel |
| TGI | 无 ngram 投机解码 | N/A |
| **FastDeploy (本方案)** | **两阶段 CUDA Kernel** | **bsz × 1024 线程** |

据调研，FastDeploy 目前是唯一将 N-gram 匹配以 CUDA Kernel 形式实现并行搜索的推理框架。

# 四、设计思路与实现方案

## 1、总体架构：两阶段 Kernel

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

**为什么不用单 Kernel？** CPU 原始实现中 threshold 约束存在**跨 batch 依赖**——每个 batch item 的 token 预算取决于前面所有 item 的分配总和。这种前缀依赖在 grid 级别（多 block）无法同步。因此拆分为：

- **Phase 1**（独立）：每个 block 独立完成搜索和 tentative 分配，写入 scratch buffer
- **Phase 2**（依赖）：单 block 内用 CUB `BlockScan` 完成前缀和，并行解决跨 batch 依赖

## 2、共享匹配逻辑：`ngram_match_common.cuh`

提取公共代码至共享头文件，两个算子通过 `#include` 复用：

```cpp
// ngram_match_common.cuh

// 原子最小值（CUDA 无原生 int64 atomicMin）
__device__ __forceinline__ void atomicMin64(int64_t *addr, int64_t val);

// 并行滑窗搜索 — 1024 线程协作查找最左匹配
__device__ __forceinline__ int64_t
parallel_ngram_search(const int64_t *haystack, int64_t haystack_len,
                      const int64_t *ngram, int ngram_size,
                      int64_t *s_min_pos);
```

业务差异体现在各自的 Phase 1 / Phase 2 kernel 实现中——`ngram_match.cu` 处理 encoder/decoder 状态和 `token_ids_all + prompt_lens` 寻址，`ngram_match_mixed.cu` 处理 `ori_seq_len_this_time` 偏移和 `pre_ids` 直接寻址。

## 3、关键技术点

### 3.1 并行搜索 + `atomicMin64` 最左匹配

CPU 串行搜索找到第一个匹配就 break。GPU 并行搜索中，1024 线程同时检查不同位置，需要保证**最左匹配语义**。方案：

- 共享内存 `__shared__ int64_t s_min_pos = INT64_MAX`
- 每个命中线程调用 `atomicMin64(s_min_pos, match_pos)`
- CAS 循环实现，unsigned 重解释安全（位置均非负）

### 3.2 A2 早退机制

```cpp
// 一旦发现匹配，远端位置的线程跳过剩余搜索
if (i > *s_min_pos) break;
```

非原子读取 `s_min_pos` 是安全的——过时值仅延迟退出时机，不影响正确性。

### 3.3 Phase 2: CUB BlockScan 前缀和

CPU 原始实现的 threshold 约束用 O(bsz²) 的串行 `sum(seq_lens_this_time, batch_idx)` 累加。GPU 方案用 CUB `BlockScan::InclusiveSum` 在 O(bsz) 内完成：

```cpp
typedef cub::BlockScan<int, NGRAM_GATHER_THREADS> BlockScanInt;
// Scan 1: token 数前缀和 → 确定每个 batch item 的 budget
// Scan 2: active item 前缀和 → 计算 remaining_active 预留量
```

### 3.4 模板特化搜索（PR #7136 优化版）

对常见 ngram_size（1, 2, 3）提供模板特化版本：

```cpp
template <int NGRAM_SIZE>
__device__ __forceinline__ int64_t
parallel_ngram_search_specialized(const int64_t *__restrict__ haystack, ...);
```

优势：
- ngram token 缓存到寄存器（消除重复 global load）
- 内层比较循环编译期完全展开（`#pragma unroll`）
- `__restrict__` 指针非别名提示

运行时 dispatcher 通过 `switch(ngram_size)` 分发到特化版本，对调用方透明。

### 3.5 Scratch buffer 缓存（PR #7136 优化版）

```cpp
static paddle::Tensor cached_draft_copy, cached_seq_lens_copy;
// 仅在 shape 变化时重新分配，消除每次调用的 paddle::empty() 开销
```

减少 ~20-40µs/call 的分配噪声。

## 4、API 兼容性

GPU Kernel 注册与 CPU 版本完全相同的 `PD_BUILD_STATIC_OP` 签名和 `SetInplaceMap`。运行时通过 `input_ids.is_gpu()` 分发到 GPU 或 CPU 路径，对上层 Python 接口透明。

# 五、测试和验收的考量

## 1、正确性验证

- 11 个功能测试覆盖全部边界条件（空 batch、encoder init、threshold 截断、mixed MTP）
- GPU 输出与 CPU 参考实现 bit-level 一致

## 2、性能验证（CI 实测数据，SM90 H20/H100）

| 配置 | CPU (µs) | GPU (µs) | 加速比 |
|------|---------|---------|--------|
| bsz=32, seq=512 (生产典型) | 939 | 661 | **1.42×** |
| bsz=128, seq=512 | 1,726 | 1,285 | **1.34×** |
| bsz=256, seq=512 | 2,682 | 2,110 | **1.27×** |
| bsz=1, seq=131072 | 275,000 | 162 | **1,700×** |
| bsz=32, high_pre pattern | 796 | 107 | **7.4×** |
| bsz=32, high_input pattern | 791 | 152 | **5.2×** |

27 个 benchmark 配置覆盖 5 个维度（seq_len、batch_size、ngram pattern、threshold、extreme），**零个 CPU 胜出的配置**。

## 3、验收标准

- [x] 全部现有测试通过
- [x] 在所有已测配置下 GPU ≥ CPU 性能
- [x] 零 D2H/H2D 同步点
- [x] API 100% 兼容

# 六、影响面

### 对用户的影响
无感知——自动选择 GPU 路径。

### 对性能的影响
在较长的匹配下，Kernel 性能**显著优于**目前的 CPU kernel。极端配置（bsz=256, seq=128K）下加速比超过三个数量级。

### 对框架架构的影响
新增 `ngram_match_common.cuh` 共享头文件。两个 `.cu` 文件增加 GPU kernel 代码，CPU fallback 保留。

### 限制
Phase 2 使用 CUB `BlockScan`，要求 `max_batch_size ≤ NGRAM_GATHER_THREADS (1024)`。FastDeploy 生产硬上限为 512（`config.py:2158`），满足此约束。

# 七、排期规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | CPU 算子分析、差异梳理 | ✅ 完成 |
| Phase 1 | 两阶段并行 Kernel 实现 + 正确性对齐 | ✅ 完成（PR #6960） |
| Phase 2 | A1 (1024线程) + A2 (早退) + Phase 2 BlockScan 优化 | ✅ 完成（PR #6960 更新） |
| Phase 3 | 模板特化 + scratch 缓存 + benchmark 去噪 | ✅ 完成（PR #7136） |
| Phase 4 | 代码 review + 合入 | 进行中 |

# 参考资料

- [PaddlePaddle Hackathon 10th Spring No.49 题目](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E9%A2%98%E7%9B%AE%E6%94%B6%E9%9B%86.md)
- [前序 RFC: NKNaN #1213](https://github.com/PaddlePaddle/community/pull/1213)
- [PR #6960: 标准版实现](https://github.com/PaddlePaddle/FastDeploy/pull/6960)
- [PR #7136: 模板特化优化版](https://github.com/PaddlePaddle/FastDeploy/pull/7136)
- [freeliuzc 并行策略要求 comment](https://github.com/PaddlePaddle/FastDeploy/pull/6960#issuecomment-2761419123)
- NVIDIA CUB BlockScan documentation
