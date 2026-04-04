# FastDeploy投机解码功能性能优化设计文档

| 任务名称 | 【Hackathon 10th Spring No.49】为 FastDeploy 支持投机解码功能 |
|------|------|
| 提交作者 | NKNaN（V1.0）、cloudforge1（V2.0） |
| 提交时间 | 2026-02-07（V1.0）、2026-04-04（V2.0） |
| 版本号 | V2.0 |
| 文件名 | 20260207_refine_speculate_decoding_ngram_for_fastdeploy.md |
| 实现 PR | [#6960](https://github.com/PaddlePaddle/FastDeploy/pull/6960)（两阶段并行版）、[#7136](https://github.com/PaddlePaddle/FastDeploy/pull/7136)（模板特化优化版） |

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

## 现有cpu算子逻辑梳理 - ngram_match.cc
```cpp
void find_candidate_pred_tokens(const int64_t *input_ids, //[batch_size, seq_len]
        const int64_t *input_ids_len, //[batch_size]
        const int64_t *pre_ids, //[batch_size, max_dec_len]
        const int64_t *step_idx, //[batch_size]
        const int *draft_token_num, //[batch_size]
        int64_t *draft_tokens, //[batch_size, max_draft_tokens]
        int32_t *seq_lens_this_time, //[batch_size]
        int32_t *seq_lens_encoder, //[batch_size]
        int32_t *seq_lens_decoder, //[batch_size]
        int64_t *max_dec_len, //[batch_size]
        int64_t input_ids_stride,
        int64_t pre_ids_stride,
        int64_t draft_tokens_stride,
        int64_t max_batch_size,
        int max_ngram_size = 3,
        int max_draft_tokens = 10) {
    // 1. 初始化和阈值设置
    int threshold = 128;
    char *env_var = getenv("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD");
    if (env_var) {
        threshold = std::stoi(env_var);
    }

    // 2. 统计待处理的 batch 数量
    int unprocessed_batch_size = 0;
    for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
        if (seq_lens_encoder[batch_idx] > 0 || seq_lens_decoder[batch_idx] > 0) {
            unprocessed_batch_size++;
        }
    }

    // 3. 遍历每个 batch 进行处理
    for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
        // 3.1 计算当前 batch 的最大草稿 token 数
        max_draft_tokens = std::min(static_cast<int64_t>(
            draft_token_num[batch_idx]), max_dec_len[batch_idx] - step_idx[batch_idx] - 1);
        // 3.2 跳过 encoder 阶段或无效的 batch
            if (seq_lens_encoder[batch_idx] > 0) {
            continue;
        } else if (seq_lens_decoder[batch_idx] == 0) {
            seq_lens_this_time[batch_idx] = 0;
            continue;
        }
        // printf("bid: %d. enc: %d. dec. %d\n", batch_idx, seq_lens_encoder[batch_idx], seq_lens_decoder[batch_idx]);
        
        // 3.3 准备当前 batch 的数据指针
        const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
        int64_t *cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
        const int64_t *cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
        const int64_t cur_step_idx = step_idx[batch_idx];
        const int64_t cur_input_ids_len = input_ids_len[batch_idx];
        seq_lens_this_time[batch_idx] = 1;
        unprocessed_batch_size--;

        // 4. 动态调整 max_draft_tokens（基于阈值约束
        auto sum_token_num = sum(seq_lens_this_time, batch_idx);
        int left_min_token_num = unprocessed_batch_size;

        if (sum_token_num + max_draft_tokens + left_min_token_num > threshold) {
            int tmp_max_draft_tokens = threshold - sum_token_num - left_min_token_num;
            max_draft_tokens = tmp_max_draft_tokens < max_draft_tokens ? tmp_max_draft_tokens : max_draft_tokens;
        }

        if (sum_token_num + left_min_token_num >= threshold - 1) {
            continue;
        }
        
        // 5. 核心 N-gram 匹配算法
        // 5.1 外层循环：从大到小尝试不同的 n-gram 大小
        for (int ngram_size = max_ngram_size; ngram_size > 0; --ngram_size) {
            // Extract the last n tokens as our search ngram
            if (cur_step_idx < ngram_size) {
                continue;
            }
            const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);
            
            // 5.2 第一阶段：在输入序列中匹配
            // Iterate through sliding windows of size ngram_size
            bool match_input = false;
            for (int64_t i = 0; i <= cur_input_ids_len - ngram_size; ++i) {
                // Check if the current window matches the ngram
                bool match = true;
                for (int j = 0; j < ngram_size; j++) {
                    if (ngram[j] != cur_input_ids[i + j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    int64_t start_idx = i + ngram_size;
                    int64_t end_idx = std::min(start_idx + max_draft_tokens, cur_input_ids_len);
                    if (start_idx >= end_idx)
                        continue;

                    int64_t cur_draft_token_num = end_idx - start_idx;

                    seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
                    memcpy(cur_draft_tokens + 1, cur_input_ids + start_idx, sizeof(int64_t) * cur_draft_token_num);
                    // To break the current batch_idx for-loop
                    ngram_size = 0;
                    match_input = true;
                    break;
                    // }
                }
            }
            // 5.3 第二阶段：在历史生成序列中匹配
            if (!match_input) {
                for (int64_t i = 0; i <= cur_step_idx - ngram_size; ++i) {
                    // Check if the current window matches the ngram
                    bool match = true;

                    for (int j = 0; j < ngram_size; j++) {
                        if (ngram[j] != cur_pre_ids[i + j]) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
                        int64_t start_idx = i + ngram_size;
                        int64_t end_idx = std::min(start_idx + max_draft_tokens, cur_step_idx);
                        int64_t cur_draft_token_num = end_idx - start_idx;
                        if (start_idx >= end_idx)
                            continue;

                        seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
                        memcpy(cur_draft_tokens + 1, cur_pre_ids + start_idx, sizeof(int64_t) * cur_draft_token_num);
                        ngram_size = 0;
                        break;
                    }
                }
            }
        }
    }
}

```


## 现有cpu算子逻辑梳理 - ngram_match_mixed.cu
```cpp
void find_candidate_pred_tokens_mixed(const int64_t *input_ids, // [bsz, seq_len]
        const int64_t *input_ids_len, // [bsz]
        const int64_t *pre_ids, // [bsz, max_dec_len]
        const int64_t *step_idx, // [bsz]
        const int *draft_token_num, // [bsz]
        int64_t *draft_tokens, // [bsz, max_draft_tokens]
        int32_t *seq_lens_this_time, // [bsz]
        int32_t *seq_lens_decoder, // [bsz]
        int64_t *max_dec_len, // [bsz]
        int64_t input_ids_stride,
        int64_t pre_ids_stride,
        int64_t draft_tokens_stride,
        int64_t max_batch_size,
        int max_ngram_size = 3,
        int min_ngram_size = 1,
        const int max_draft_tokens = 10) {
    // 1. 初始化阈值
    int threshold = 1024;
    // dynamic in future
    char *env_var = getenv("SPEC_TOKENUM_THRESHOLD");
    if (env_var) {
        threshold = std::stoi(env_var);
    }
    // 2. 统计待处理 batch
    int unprocessed_batch_size = 0;
    for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
        if (seq_lens_decoder[batch_idx] > 0) {
            unprocessed_batch_size++;
        }
    }
    // 3. 核心循环：处理每个 batch
    for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
        // 3.1 获取当前已有的 draft tokens 数量
        const int ori_seq_len_this_time = seq_lens_this_time[batch_idx];
        int max_draft_tokens_query = std::min(static_cast<int64_t>(
            max_draft_tokens - ori_seq_len_this_time + 1), max_dec_len[batch_idx] - step_idx[batch_idx] - 1);

        // 3.2 跳过无效或已满的 batch
        if (ori_seq_len_this_time == 0 || max_draft_tokens_query <= 0) {
            continue;
        }

        // 3.3 准备当前 batch 的数据指针
        const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
        int64_t *cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
        const int64_t *cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
        const int64_t cur_step_idx = step_idx[batch_idx];
        const int64_t cur_input_ids_len = input_ids_len[batch_idx];
        unprocessed_batch_size--;

        // 4. 动态调整 max_draft_tokens_query（阈值控制）
        auto sum_token_num = sum_mixed(seq_lens_this_time, batch_idx);
        int left_min_token_num = unprocessed_batch_size;

        if (sum_token_num + max_draft_tokens_query + left_min_token_num > threshold) {
            int tmp_max_draft_tokens = threshold - sum_token_num - left_min_token_num;
            max_draft_tokens_query = std::min(max_draft_tokens_query, tmp_max_draft_tokens);
        }

        if (sum_token_num + left_min_token_num >= threshold - 1) {
            continue;
        }

        // 5. 核心 N-gram 匹配算法
        // 5.1 外层循环：从大到小尝试不同的 n-gram 大小
        bool match_global = false;
        // apply ngram_match in input_ids
        for (int ngram_size = max_ngram_size; ngram_size >= min_ngram_size && !match_global; --ngram_size) {
            // Extract the last n tokens as our search ngram
            if (cur_step_idx < ngram_size) {
                continue;
            }
            const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

            // 5.2 第一阶段：在输入序列中匹配
            // Iterate through sliding windows of size ngram_size
            // bool match_input = false;
            for (int64_t i = 0; i <= cur_input_ids_len - ngram_size && !match_global; ++i) {
                // Check if the current window matches the ngram
                bool match_local = true;
                for (int j = 0; j < ngram_size; j++) {
                    if (ngram[j] != cur_input_ids[i + j]) {
                        match_local = false;
                        break;
                    }
                }
                if (match_local) {
                    int64_t start_idx = i + ngram_size;
                    int64_t end_idx = std::min(start_idx + max_draft_tokens_query, cur_input_ids_len);
                    if (start_idx >= end_idx)
                        continue;

                    int64_t cur_draft_token_num = end_idx - start_idx;

                    seq_lens_this_time[batch_idx] = ori_seq_len_this_time + cur_draft_token_num;
                    memcpy(cur_draft_tokens + ori_seq_len_this_time, cur_input_ids + start_idx, sizeof(int64_t) * cur_draft_token_num);
                    // To break the current batch_idx for-loop
                    match_global = true;
                    break;
                }
            }
            // 5.3 第二阶段：在历史生成序列中匹配
            // apply ngram_match in generated tokens
            if (!match_global) {
                for (int64_t i = 0; i <= cur_step_idx - ngram_size && !match_global; ++i) {
                    // Check if the current window matches the ngram
                    bool match_local = true;

                    for (int j = 0; j < ngram_size; j++) {
                        if (ngram[j] != cur_pre_ids[i + j]) {
                            match_local = false;
                            break;
                        }
                    }
                    if (match_local) {
                        int64_t start_idx = i + ngram_size;
                        int64_t end_idx = std::min(start_idx + max_draft_tokens_query, cur_step_idx);

                        int64_t cur_draft_token_num = end_idx - start_idx;

                        if (start_idx >= end_idx)
                            continue;
                        // printf("match in Output with Ngram_size %d. %lld:[%lld,%lld]\n",ngram_size, cur_draft_token_num, start_idx, end_idx);

                        seq_lens_this_time[batch_idx] = ori_seq_len_this_time + cur_draft_token_num;
                        memcpy(cur_draft_tokens + ori_seq_len_this_time, cur_pre_ids + start_idx, sizeof(int64_t) * cur_draft_token_num);
                        match_global = true;
                        break;
                    }
                }
            }
        }
    }
}
```

## 两个算子计算步骤差异分析

| 	| find_candidate_pred_tokens	| find_candidate_pred_tokens_mixed |
|---|---|---|
|写入位置	| cur_draft_tokens + 1	| cur_draft_tokens + ori_seq_len_this_time|
|长度计算	| cur_draft_token_num + 1	| ori_seq_len_this_time + cur_draft_token_num|
|默认阈值	| 128	| 1024|
|最小 ngram_size | 固定为 1	| 可配置 min_ngram_size|
|匹配控制	| 匹配到ngram后没有直接跳出最外层循环	| 匹配到ngram后会直接跳出最外层循环 |


# 三、设计思路与实现方案

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

# 四、测试和验收的考量

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
> 原始任务描述："在较长的匹配下，Kernel 性能**优于或基本不劣于**目前的 CPU kernel"

- [x] 全部现有测试通过
- [x] 在所有已测配置下 GPU ≥ CPU 性能（1.27×–1,700×）
- [x] 零 D2H/H2D 同步点
- [x] API 100% 兼容

# 五、可行性分析和排期规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | CPU 算子分析、差异梳理（V1.0 §二） | ✅ 完成 |
| Phase 1 | 两阶段并行 Kernel 实现 + 正确性对齐 | ✅ 完成（PR [#6960](https://github.com/PaddlePaddle/FastDeploy/pull/6960)） |
| Phase 2 | A1 (1024线程) + A2 (早退) + BlockScan 优化 | ✅ 完成（PR #6960 更新） |
| Phase 3 | 模板特化 + scratch 缓存 + benchmark 去噪 | ✅ 完成（PR [#7136](https://github.com/PaddlePaddle/FastDeploy/pull/7136)） |
| Phase 4 | 代码 review + 合入 | 进行中 |

# 六、影响面

### 对用户的影响
无感知——运行时自动选择 GPU 路径。

### 对性能的影响
在较长的匹配下，Kernel 性能**显著优于**目前的 CPU kernel。极端配置（bsz=256, seq=128K）下加速比超过三个数量级。

### 对框架架构的影响
新增 `ngram_match_common.cuh` 共享头文件。两个 `.cu` 文件增加 GPU kernel 代码，CPU fallback 完整保留。

# 参考资料
- [PaddlePaddle Hackathon 10th Spring No.49 题目](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E9%A2%98%E7%9B%AE%E6%94%B6%E9%9B%86.md)
- [PR #6960: 两阶段并行版实现](https://github.com/PaddlePaddle/FastDeploy/pull/6960)
- [PR #7136: 模板特化优化版](https://github.com/PaddlePaddle/FastDeploy/pull/7136)
- NVIDIA CUB BlockScan documentation
