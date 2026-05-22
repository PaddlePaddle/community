# FastDeploy Ngram 及 Hybrid MTP-Ngram 投机解码端到端验证设计文档

| 任务名称 | ngram 及 hybrid_mtp_ngram 端到端验证 |
|------|------|
| 提交作者 | NKNaN |
| 提交时间 | 2026-05-15 |
| 版本号 | V1.0 |
| 文件名 | 20260515_ngram_e2e_validation_for_fastdeploy.md |

# 一、概述

## 1、相关背景

投机解码（Speculative Decoding）通过预先生成草稿 token 并批量验证，减少串行 forward 次数，从而加速 LLM 推理。FastDeploy 当前支持两种基于字符串匹配的投机解码方法：

- **Ngram**：利用 n-gram 在 prompt 和已生成序列中匹配，提出草稿 token。
- **Hybrid MTP-Ngram**：在 MTP（Multi-Token Prediction）草稿模型推测的基础上，叠加 ngram 补充草稿 token。

**前置任务（NO.49）已完成的工作**：将两种方法的核心匹配算子从 CPU 版本（`ngram_match.cc` / `ngram_match_mixed.cu` 的 CPU 路径）重写为统一的 GPU Kernel（`ngram_match.cu`），消除了推理关键路径上的强制 Device→Host→Device 同步拷贝，在较长序列场景下 kernel 性能不劣于 CPU 版本。同时，新 GPU Kernel 采用了与推理引擎 `share_inputs` 缓冲区直接对齐的接口（`token_ids_all` + `prompt_lens`），适配了 `NgramProposer` 的调用逻辑。

**本任务（NO.54）**：在 NO.49 GPU Kernel 工作的基础上，进行完整的端到端验证，证明两种投机解码方法在完整推理服务链路中的正确性与兼容性。

## 2、功能目标

**第一个 PR（本文档对应）：Ngram 方法端到端验证**

- 新增 operator 层单元测试（`tests/operators/test_ngram_match.py`），覆盖新 GPU Kernel 接口的基础匹配、无匹配等场景；
- 新增 serving 层 E2E 测试（`tests/e2e/test_ernie_03b_ngram.py`），验证 ngram 投机解码在 overlap schedule、cudaGraph、logprob 同时开启时的功能正确性，以及 speculate_metrics 的有效性。

**第二个 PR：Hybrid MTP-Ngram 方法端到端验证**

- 对 `ngram_match_mixed.cu` 中对应的 GPU Kernel 进行相同验证；
- 新增 operator 单元测试与 serving E2E 测试。

## 3、意义

- 以可复现的测试用例自证 GPU Kernel 替换后功能正确性未回退；
- 建立 E2E 测试覆盖，使接受率（accept ratio）、逐头接受数（accepted_tokens_per_head）等关键指标可量化；
- 验证零 H↔D 拷贝路径在 overlap schedule 和 cudaGraph 下的端到端兼容性。

# 二、现状分析

## 2.1 GPU Kernel 接口现状（`ngram_match.cu`）

`ngram_match.cu` 中的 GPU kernel 原始接口将 prompt（input_ids）和已生成序列（pre_ids）作为两个独立的张量传入：

```cpp
// 原始接口（简化）
void NgramMatchKernel(
    const int64_t *input_ids,      // [bsz, prompt_len]  prompt tokens（haystack第一部分）
    const int64_t *input_ids_len,  // [bsz]              实际 prompt 长度
    const int64_t *pre_ids,        // [bsz, max_dec_len] 已生成 token（haystack第二部分）
    const int64_t *step_idx,       // [bsz]              已生成 token 数量
    ...
);
```

而 `NgramProposer._run_impl` 中，`token_ids_all` 已将 prompt 和生成序列拼接在同一块显存中：

```
token_ids_all[b] = [prompt_token_0, ..., prompt_token_{P-1},
                    gen_token_0,    ..., gen_token_{N-1}, 0, ...]
                   |<--- prompt_lens[b] --->|<--- step_idx[b] --->|
```

由于接口不匹配，proposer 原来需要在 CPU 侧进行切片和指针偏移，实质上产生了 D→H 数据访问，破坏了零拷贝路径。

## 2.2 step_idx 语义现状

原始 CPU 算子中 `step_idx` 被用作"最后一个有效生成 token 的位置索引"（last-valid-index，0-based），即已生成 N 个 token 时 `step_idx = N - 1`。

这导致 ngram 提取公式为：
```cpp
const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);
```

此语义与 `NgramProposer` 实际传入的值（`len(request.output_token_ids)`，即生成 token 的数量）不一致，需要在接口层进行 +1/-1 的修正，增加了理解成本和出错风险。

## 2.3 测试覆盖现状

| 测试层级 | 现状 |
|---|---|
| Operator 单元测试 | `test_ngram_match.py` 存在，但使用旧的 13 参数 API，与新接口不兼容 |
| Serving E2E 测试 | 无针对 ngram 投机解码的 E2E 测试 |
| 兼容性测试 | overlap + cudaGraph + logprob 同时开启时无覆盖 |

## 2.4 Hybrid MTP-Ngram 方法现状（第二个 PR 范围）

**调用结构**

- `MTPProposer.hybrid_mode` 在 `mtp_strategy == "with_ngram"` 且 `max_draft_token_num > num_model_steps` 时被启用：

  ```python
  # fastdeploy/spec_decode/mtp.py:82
  self.hybrid_mode = self.mtp_strategy == "with_ngram" and self.max_draft_token_num > self.num_model_steps
  ```

- `MTPProposer._run_impl` 在完成多步 MTP 推测后，调用 `_extend_draft_token_with_ngram_match()` 用 ngram 匹配的 token 补齐剩余草稿位置（`mtp.py:680-681`）；
- CUDA 平台下，`MTPProposerCUDA._extend_draft_token_with_ngram_match()` 调用 `hybrid_mtp_ngram` 算子（`mtp_cuda.py:395-410`）。

**算子接口现状（NO.49 完成后）**

`hybrid_mtp_ngram` 算子（`custom_ops/gpu_ops/speculate_decoding/draft_model/ngram_match_mixed.cu`）当前仍采用旧式分离接口：

```cpp
PD_BUILD_STATIC_OP(hybrid_mtp_ngram)
    .Inputs({"input_ids", "input_ids_len", "pre_ids", "step_idx",
             "draft_token_num", "draft_tokens",
             "seq_lens_this_time", "seq_lens_decoder", "max_dec_len"})
    ...
```

与 `ngram_match.cu` 的统一 `token_ids_all` 接口尚未对齐。

**Python 调用侧的潜在 H↔D 拷贝**

`MTPProposerCUDA._extend_draft_token_with_ngram_match` 中存在显式的 Host→Device 拷贝（`mtp_cuda.py:398-399`）：

```python
hybrid_mtp_ngram(
    self.model_inputs["input_ids_cpu"].cuda(),     # 同步 H→D
    self.model_inputs["input_ids_len"].cuda(),      # 同步 H→D
    ...
)
```

**测试覆盖现状**

| 测试层级 | 现状 |
|---|---|
| Operator 单元测试 | `hybrid_mtp_ngram` 算子无独立单测 |
| Serving E2E 测试 | 现有 `test_ernie_21b_mtp_multistep.py` 仅覆盖 MTP（不含 with_ngram 策略），未覆盖 hybrid 模式 |
| 兼容性测试 | hybrid 模式 + overlap + cudaGraph + logprob 同时开启无覆盖 |

# 三、设计思路与实现方案

## 3.1 GPU Kernel 接口重设计

将 `token_ids_all`（prompt + generated token 的统一缓冲区）和 `prompt_lens` 作为输入，替换原来分离的 `input_ids`、`input_ids_len`、`pre_ids`，接口精简为 11 个参数：

```cpp
void NgramMatchKernel(
    const int64_t *token_ids_all,   // [bsz, prompt_len + max_dec_len] 统一 token 缓冲区
    const int64_t *prompt_lens,     // [bsz]   prompt 的实际长度，用于切分 haystack
    const int64_t *step_idx,        // [bsz]   已生成 token 数量（count 语义，非 index）
    const int      *draft_token_num, // [bsz]   最大草稿 token 数
    int64_t        *draft_tokens,   // [bsz, max_draft_tokens]  输出草稿 token
    int32_t        *seq_lens_this_time,  // [bsz]
    int32_t        *seq_lens_encoder,    // [bsz]
    int32_t        *seq_lens_decoder,    // [bsz]
    int64_t        *max_dec_len,         // [bsz]
    int             max_ngram_size,
    int             max_draft_tokens
);
```

**数据布局约定**

```
token_ids_all[b]:
  [0 .. prompt_lens[b]-1]               <- haystack 第一部分（prompt）
  [prompt_lens[b] .. prompt_lens[b]+step_idx[b]-1]  <- haystack 第二部分（已生成）
```

**step_idx 语义统一**

`step_idx[b]` 表示第 b 个请求已生成 token 的数量（count），与 `NgramProposer` 中 `len(request.output_token_ids)` 直接对应，无需偏移。

ngram 提取公式调整为：
```cpp
const int64_t *pre_ids = token_ids_all + batch_idx * stride + prompt_lens[batch_idx];
const int64_t *ngram   = pre_ids + (cur_step_idx - ngram_size);
```

即从生成序列末尾取最后 `ngram_size` 个 token 作为查询模式。

## 3.2 NgramProposer 适配

`fastdeploy/speculate/proposer/ngram_proposer.py` 中的 `_run_impl` 做如下修改：

1. 将 `token_ids_all` 整体传入 kernel，不再手动切分 prompt / pre_ids；
2. 使用 `share_inputs["prompt_lens"]`（已有字段）代替原来的 `input_ids_len`；
3. `step_idx` 直接从 `share_inputs["step_idx"]` 读取，无需 ±1 修正。

该修改对其他 model runner（MetaX、XPU、DCU 等）透明，因为它们均通过同一个 `NgramProposer._run_impl` 调用 kernel，无需分别修改。

## 3.3 单元测试设计（`tests/operators/test_ngram_match.py`）

更新为新的 11 参数 API，覆盖以下场景：

| 测试用例 | 描述 |
|---|---|
| `test_basic_match` | prompt 中存在匹配，草稿 token 正确提取 |
| `test_no_match` | prompt 和生成序列中均无匹配，seq_len 退化为 1 |
| `test_multi_batch`（如有）| 多 batch 同时处理，各 batch 独立正确 |

**关键参数示例（`test_basic_match`）**

```python
# token_ids_all = [prompt(6 tokens) + generated(4 tokens) + padding]
token_ids_all = [[10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 0, 0]]
prompt_lens   = [[6]]       # prompt: [10,20,30,40,50,60]
step_idx      = [4]         # generated: [10,20,30,40]，查询 ngram=[30,40]

# 期望在 prompt 中匹配到 [30,40] 后的 [50,60]
# seq_lens_this_time = 3（1 个当前 token + 2 个草稿 token）
```

## 3.4 E2E 测试设计（`tests/e2e/test_ernie_03b_ngram.py`）

使用 ERNIE-4.5-0.3B-Paddle 模型，以 subprocess 启动推理服务，通过 HTTP 接口进行测试，覆盖以下场景：

**服务启动配置**

```python
speculative_config = {
    "method": "ngram",
    "num_speculative_tokens": 5,
    "max_ngram_size": 3,
    "min_ngram_size": 1,
}
# 同时开启：--enable-overlap-schedule  --enable-logprob
# graph-optimization-config: use_cudagraph=true
```

**测试用例列表**

| 测试用例 | 验证内容 |
|---|---|
| `test_ngram_stream` | 流式生成返回非空结果，token 数量在 [min_tokens, max_tokens] 范围内 |
| `test_ngram_non_stream` | 非流式生成返回非空结果，token 数量合法 |
| `test_ngram_speculate_metrics` | speculate_metrics 存在且不为 None；包含 accepted_tokens、rejected_tokens、accept_ratio、avg_accept_len、accepted_tokens_per_head、accept_ratio_per_head 字段（使用含重复片段的 prompt 以提高 ngram 命中率） |
| `test_ngram_speculate_metrics_with_logprobs` | logprobs 与 speculate_metrics 同时返回且格式正确（token / logprob / top_logprobs 字段），验证两者兼容 |

**speculate_metrics 字段含义**

| 字段 | 含义 |
|---|---|
| `accepted_tokens` | 验证通过的草稿 token 总数 |
| `rejected_tokens` | 被拒绝的 token 总数（含强制 target token） |
| `accept_ratio` | 整体接受率 = accepted / (accepted + rejected) |
| `avg_accept_len` | 平均每轮接受草稿 token 数 |
| `accepted_tokens_per_head[i]` | 第 i 个 draft 位置累计被接受的次数（单调不增） |
| `accept_ratio_per_head[i]` | 条件接受率：给定前 i 个 draft token 均被接受时，第 i+1 个被接受的概率 |

## 3.5 零拷贝路径验证

通过以下方式确认无新增 H↔D 拷贝：

1. `NgramProposer._run_impl` 中不调用 `.numpy()`、`.cpu()`、`paddle.to_tensor()` 等触发同步的接口；
2. 所有输入张量（`token_ids_all`、`prompt_lens`、`step_idx` 等）已存在于 `share_inputs` 的显存缓冲区中，直接传指针给 kernel；
3. 输出 `draft_tokens` 和 `seq_lens_this_time` 写回显存，供后续 target model 验证阶段直接消费。

## 3.6 第二个 PR：Hybrid MTP-Ngram 端到端验证方案

### 接口对齐与零拷贝改造（待 PR 中实现，初步可行性已确认）

经代码确认：`MTPProposer.model_inputs`（`ProposerInputBatch`）已包含从 target model clone 而来的 `token_ids_all`（`input_batch.py:778-779`）与 `prompt_lens`（`input_batch.py:826`），且 `init_share_inputs` 中 `pre_ids` 一处带有 `# TODO: delete pre_ids in mtp` 注释，表明此方向已是计划内的演进路径。基于此，第二个 PR 的接口对齐改造在数据可用性上无障碍，具体实施步骤：

1. **算子层（`ngram_match_mixed.cu`）**：参考 `ngram_match.cu` 已完成的接口形态，将 `input_ids` + `input_ids_len` + `pre_ids` 替换为 `token_ids_all` + `prompt_lens`，复用 `ngram_match_common.cuh` 中的公共匹配逻辑；`step_idx` 语义统一为 count。需保留 mixed 版本相对于基础版本的业务差异（写入位置 `cur_draft_tokens + ori_seq_len_this_time`、长度累加、`min_ngram_size` 可配等）。
2. **调用层（`mtp_cuda.py:_extend_draft_token_with_ngram_match`）**：将当前的 `self.model_inputs["input_ids_cpu"].cuda()` 与 `self.model_inputs["input_ids_len"].cuda()` 替换为 `self.model_inputs["token_ids_all"]` 与 `self.model_inputs["prompt_lens"]`，消除两次同步 H→D 拷贝。
3. **风险评估**：算子内部的核心匹配算法不变，主要风险来自接口替换时的索引偏移与边界处理；需通过 operator 单元测试在算子层先行对齐。XPU 路径（`mtp_xpu.py`）当前未实现 `_extend_draft_token_with_ngram_match`（基类为 no-op），不在本 PR 修改范围。

### Operator 单元测试设计（`tests/operators/test_hybrid_mtp_ngram.py`）

| 测试用例 | 描述 |
|---|---|
| `test_mtp_with_ngram_extend` | 已有 MTP 推测的 `ori_seq_len_this_time > 0`，验证 ngram 在其后追加，最终长度等于 `ori_seq_len_this_time + cur_draft_token_num` |
| `test_mtp_only_no_ngram_match` | 无 ngram 命中时保持 MTP 原有草稿不变 |
| `test_min_ngram_size_boundary` | 验证 `min_ngram_size` 起作用，小于该值的匹配被忽略 |

### Serving E2E 测试设计（`tests/e2e/test_ernie_21b_mtp_ngram.py`）

使用 ERNIE-4.5-21B-A3B-Paddle 模型（与现有 `test_ernie_21b_mtp_multistep.py` 一致），通过设置 `speculative_config.mtp_strategy = "with_ngram"` 启用 hybrid 模式：

```python
speculative_config = {
    "method": "mtp",
    "model": mtp_model_path,
    "num_speculative_tokens": 5,
    "num_model_steps": 3,
    "mtp_strategy": "with_ngram",   # 触发 hybrid_mode
    "max_ngram_size": 3,
    "min_ngram_size": 1,
}
# 同时开启：--enable-overlap-schedule  --enable-logprob
# graph-optimization-config: use_cudagraph=true, draft_model_use_cudagraph=true
```

**测试用例列表**

| 测试用例 | 验证内容 |
|---|---|
| `test_mtp_ngram_stream` | 流式生成在 hybrid 模式下返回正确结果 |
| `test_mtp_ngram_non_stream` | 非流式生成功能正常 |
| `test_mtp_ngram_speculate_metrics` | speculate_metrics 字段完整；`accepted_tokens_per_head` 长度等于 `num_speculative_tokens`，覆盖 MTP 头 + ngram 补充头两段 |
| `test_mtp_ngram_with_logprobs` | logprobs 与 hybrid 模式 + speculate_metrics 三者共存 |

### 兼容性验证关注点

- **CUDA Graph 兼容**：`draft_model_use_cudagraph=true` 时，hybrid 模式下 `_extend_draft_token_with_ngram_match` 在 graph capture/replay 路径中能正确执行；
- **Overlap Schedule**：MTP 推测与 ngram 扩展的整体时序与 overlap schedule 不冲突；
- **比例验证**：在 prompt 含重复片段的场景下，hybrid 模式整体接受率应不低于纯 MTP 模式（ngram 补充位置带来额外接受）。

# 四、可行性分析和排期规划

| 阶段 | 内容 | 状态 |
|---|---|---|
| 接口分析与设计 | 梳理现有 CPU/GPU 算子接口与 proposer 调用方式，确定新接口方案 | 已完成 |
| GPU Kernel 修复 | 修改 `ngram_match.cu`，对齐 `token_ids_all` 数据布局和 `step_idx` count 语义 | 已完成 |
| Proposer 适配 | 修改 `NgramProposer._run_impl` 调用新接口 | 已完成 |
| 单元测试更新 | 更新 `test_ngram_match.py` 为新 API | 已完成 |
| E2E 测试新增 | 新增 `test_ernie_03b_ngram.py`，覆盖流式/非流式/metrics/logprobs | 已完成 |
| 第二个 PR：算子接口对齐 | `hybrid_mtp_ngram` 改用 `token_ids_all` + `prompt_lens` 接口，消除 `_extend_draft_token_with_ngram_match` 中的 `.cuda()` 同步拷贝 | 待进行 |
| 第二个 PR：单测新增 | 新增 `test_hybrid_mtp_ngram.py`，覆盖 MTP 补齐、无匹配、`min_ngram_size` 边界 | 待进行 |
| 第二个 PR：E2E 测试新增 | 新增 `test_ernie_21b_mtp_ngram.py`，覆盖 hybrid + overlap + cudaGraph + logprob | 待进行 |

# 五、影响面

- **功能正确性**：GPU kernel 输出与原 CPU 实现逻辑等价，通过单元测试对比验证；
- **接口兼容性**：NgramProposer 为 ngram 方法的唯一调用入口，其他 model runner（GPU、MetaX、XPU、DCU）均通过同一 proposer 间接调用 kernel，无需分别修改；第二个 PR 中 `hybrid_mtp_ngram` 仅有 CUDA 路径实现（XPU 路径为 no-op），修改范围可控；
- **兼容性**：第一个 PR E2E 测试覆盖 ngram + overlap schedule + cudaGraph + logprob；第二个 PR 进一步覆盖 hybrid 模式下 `draft_model_use_cudagraph` 的兼容性；
- **性能**：消除推理关键路径上的 H↔D 同步拷贝（第一个 PR 已消除 ngram 路径，第二个 PR 进一步消除 `_extend_draft_token_with_ngram_match` 现存的两次 `.cuda()` 调用），与 overlap schedule 无干扰；
- **回归风险**：第一个 PR 修改范围仅限 `ngram_match.cu` 与 `ngram.py`，不影响 MTP、Medusa 等其他方法；第二个 PR 修改 `ngram_match_mixed.cu` 与 `mtp_cuda.py`，需通过 `test_ernie_21b_mtp_multistep.py` 现有 baseline 回归验证纯 MTP（`mtp_strategy != "with_ngram"`）模式不退化。
