# FastDeploy投机解码功能性能优化设计文档

| 任务名称 | 为 FastDeploy 支持投机解码功能 |
|------|------|
| 提交作者 | NKNaN |
| 提交时间 | 2026-02-07 |
| 版本号 | V1.0 |
| 文件名 | 20260207_refine_speculate_decoding_ngram_for_fastdeploy.md |

# 一、概述

## 1、相关背景
投机解码有多种方法，目前 FastDeploy 中 ngram_match / hybrid_mtp_ngram 两种方法都用到了字符串匹配方法。
但目前两个方法的核心匹配算子实现是 CPU 版本，需要做同步的 Device->CPU 的拷贝操作，对性能影响较大

## 2、功能目标
* 将 FastDeploy/custom_ops/gpu_ops/speculate_decoding/ngram_match.cc 与 FastDeploy/custom_ops/gpu_ops/speculate_decoding/draft_model/ngram_match_mixed.cu 中的匹配逻辑重写为 统一、通用、可扩展的 GPU Kernel；
* Kernel 性能在长序列场景下不劣于 CPU 版本
* 保持 API 与现有 Python/C++ 接口 100% 兼容。

## 3、意义
优化后续大模型推理速度，增强投机解码的性能；打通 FastDeploy 在 GPU 推理全链路的零拷贝路径；

# 二、现状分析
- FastDeploy 当前仅提供 CPU 版投机解码核心算子。但是ngram只在 Host 端调用同一 CPU 逻辑，GPU 端仅做内存搬运。推理时必需把 Device 端的 history 与 draft Token 张量同步拷贝到 Host，匹配结束后再把结果拷回 Device，形成一次强制同步。导致 CUDA Stream 空等，GPU 计算资源闲置。随着序列长度增加，拷贝耗时线性增长，成为整体吞吐的硬性瓶颈。因此，亟需将匹配算子迁移至 GPU，实现零拷贝、高并行、无阻塞的端到端加速。

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
- 单 Kernel 覆盖两业务：统一计算步骤，提取公共匹配逻辑，差异通过模板参数/分支常量控制；

### 2) Kernel设计和实现
- 提取ngram公共逻辑部分，在分别设计对应Kernel及其调用方式
- 完成基础功能并做算法功能与精度对齐

### 3) 可能的优化方案
- 考虑是否可以将`5. 核心 N-gram 匹配算法`中的`5.2 第一阶段：在输入序列cur_input_ids中匹配`和`5.3 第二阶段：在历史生成序列cur_pre_ids中匹配`在gpu中并行异步实现

# 四、可行性分析和排期规划
* 了解当前整体FastDeploy的pipeline和投机解码的功能原理，已基本完成。
* 分析当前ngram的两个实现文件，并设计kennel函数进行实现，预计两周。
* 功能对齐和性能优化&测试，预计两周。

# 五、影响面
在保证正确性的前提下，在较长的匹配下，Kernel 性能优于或基本不劣于目前的 CPU kernel。
