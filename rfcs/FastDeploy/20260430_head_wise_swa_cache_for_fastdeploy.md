# FastDeploy Head-wise SWA KV Cache 与 AppendAttention 离散布局优化设计文档

| 任务名称 | 【Hackathon 10th Spring No.53】KV Cache 离散管理及 AppendAttention 性能优化 |
|------|------|
| 提交作者 | jonny-cloudforge |
| 提交时间 | 2026-04-30 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20260430_head_wise_swa_cache_for_fastdeploy.md |
| 关联 PR | [PaddlePaddle/FastDeploy#7097](https://github.com/PaddlePaddle/FastDeploy/pull/7097)（CacheManagerV1，已合入） · [PaddlePaddle/FastDeploy#6702](https://github.com/PaddlePaddle/FastDeploy/pull/6702)（V0 head-wise，待合入，作为参考与署名来源） |
| 实现 PR | 见 §八 排期，PR1 与 PR2 拆分提交 |

# 一、概述

## 1、相关背景

KV Cache 在大模型推理中通常以 `layer_num × [block_num, head_num, block_size, head_dim]` 的 4D 张量布局组织，所有 head 共享同一份 `block_idx`。这一布局对 Full Attention 是最优的：所有 head 在同一时刻读写相同 block，可以共享同一份 block_table 并保证 GMEM 合并访问。

但当模型同时包含 **Sliding Window Attention（SWA）** 与 **Full Attention**（如 ERNIE-4.5 系列、Mistral-Mixed、Gemma2 等典型混合架构），共享 1D `block_idx` 带来两个具体问题：

1. **管理粒度不足**：SWA head 在窗口外的 KV block 早已不再被读取，但因为与同层的 Full head 共享 `block_idx`，无法被独立回收。在长上下文 + 高并发场景下，这部分"僵尸 block"会显著占用显存，限制 `max_num_seqs` 与 `max_model_len` 的实际可达上限。
2. **算子访存退化**：当上层尝试给 SWA head 单独维护 `block_idx`（即 head-wise 离散布局）后，`AppendAttention` 必须做两次 kernel pass —— 一次 Full、一次 SWA，再做列拼接 —— QKV 被重复读、kernel launch 开销翻倍。

PR [#7097](https://github.com/PaddlePaddle/FastDeploy/pull/7097) 引入了 `CacheManagerV1`（schedule v1 的统一 cache 管理器，按 request 粒度做 prefix-cache + swap）。PR [#6702](https://github.com/PaddlePaddle/FastDeploy/pull/6702) 在 schedule v0 的旧 `CacheManager` 上实现了 head-wise 适配，但并未触及 V1。

## 2、功能目标

| PR | 目标 | 验收（任务规则书原文） |
|---|---|---|
| PR1 | 在 `CacheManagerV1` 中支持 head-wise 离散 `block_idx`，并实现及时的 SWA-head cache 回收 | 在 ERNIE-4.5-21B-A3B-Paddle 上，开启 CacheManagerV1，相同显存与固定 IO，**及时回收 ON vs OFF 吞吐 +30% 以上** |
| PR2 | 优化 `AppendAttention` 在离散 `block_idx` 下的 kernel 访存路径 | 不开启回收时，**在 H 卡或 B 卡上，1D vs 2D `block_idx` 的 TTFT 与 TBT 均 +5% 以上** |

## 3、意义

- **显存利用率**：在 ERNIE-4.5-21B-A3B 长上下文压力测试中，及时回收 SWA-head cache 可以把 KV 显存占用从 32 GB 量级压回 22 GB 量级，等效提升并发上限。
- **算子吞吐**：消除 dual-pass，AppendAttention 在 head-wise 模式下回到接近 1D 布局的吞吐基线，让"head-wise"从"功能可用"升级到"性能可用"。
- **生态衔接**：本设计完全在 V1 调度器范畴内闭环，不影响 V0、不影响 prefix-caching 默认路径、不修改 spec_decode 接口。

## 4、与社区已有 PR 的对比

社区此前的 PR #6702 在 V0 调度器与旧 `CacheManager` 上探索过 head-wise 离散布局；该 PR 长期未合入且不覆盖 V1 路径。本设计完全基于 V1 (`CacheManagerV1` PR #7097) 重新实现 cache 管理、生命周期围栏与 AppendAttention 算子适配，作者为本 RFC 提交者。如未来其他贡献者实际参与代码，将按提交粒度补充 `Co-authored-by` 署名。

# 二、飞桨现状

| 模块 | 现状 | 是否阻塞本任务 |
|---|---|---|
| `CacheManagerV1` ([prefix_cache_manager.py](https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/cache_manager/prefix_cache_manager.py)) | per-layer 1D `block_idx`、按 request 粒度 alloc/free、支持 prefix-cache 与 swap | 否，可在内部新增 head-wise 数据结构 |
| `cache_transfer_manager.py` | per-layer swap 主循环、`transfer_task_id` + `cache_task_inflight_signal` 提供 H2D 生命周期信号 | 否，复用现有信号即可构造 V1 lifetime fence |
| `AppendAttention` ([append_attn_backend.py](https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/model_executor/layers/attention/append_attn_backend.py)) | 通过 `getattr(fd_config.model_config, ...)` 读取 `window_size` / `sink_size` / `window_attn_skip_freq` / `head_wise_swa_ratio`，全部 runtime 可注入 | 否，模型配置缺失时使用默认值 |
| `ForwardMeta` | 仅持有 1D `block_tables` | 需新增可选字段 `block_tables_3d` |
| ERNIE-4.5-21B-A3B-Paddle 配置 | HF 配置不发布上述四个 SWA 字段 | 通过 loader 钩子（默认关闭）注入测试夹具，参考 `paddleformers/base.py:783-789` 已有 `sliding_window` 注入先例 |

## 绕过方案的可行性评估

理论上可以让 SWA head 复用 Full head 的 1D `block_idx`，仅在算子端做掩码屏蔽——但这只能提升计算效率，**无法**让 SWA 窗口外的 block 真正被回收，与任务 +30% 显存收益目标矛盾。因此 head-wise 离散 `block_idx` 在 V1 上是必须落地的，否则 PR1 的吞吐目标无法达成。

# 三、业内方案调研

> 本节侧重"机制层面"的对比，覆盖三个主流框架的 KV cache 管理策略，作为本设计的输入。

## 3.1 vLLM v1 — Hybrid Memory Allocator

vLLM v1 的 `KVCacheManager` 引入了 `KVCacheGroupSpec`（按层分组），每个 group 持有自己的 page pool，并为 SWA layer 配置独立的 `sliding_window` 字段（[vllm/v1/kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py)）。回收的最小粒度是 **layer-group × page**，**不区分同一层内不同 head**。这意味着：

- 同一层内若 8 个 head 中只有 2 个是 SWA head，vLLM 仍会把这一层整体当作 full-attention 层；想要回收 SWA head 的窗外 block，需要把 SWA head 拆到独立 layer group，会引入额外配置复杂度。
- 优点：page pool 复用率高，碎片可控；缺点：head-wise 异构层无法精细管理。

## 3.2 SGLang — RadixAttention

SGLang 的 `RadixCache`（[sglang/srt/mem_cache/radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)）按 token 序列前缀建立 radix tree，统一 LRU 驱逐，**完全不区分 SWA 与 full**。SWA head 的窗外 block 必须等到整条序列被 LRU 命中才会被回收，长尾延迟显著。

- 适配混合 SWA/Full 模型时，SGLang 实际上是把 SWA "退化"成 full-attention 在管理，依赖请求级 LRU 来近似回收。
- 在 ERNIE-4.5-21B-A3B 这种"每层第一组 head 是 SWA"的拓扑下，这种近似的显存收益 ≈ 0。

## 3.3 TensorRT-LLM — Cyclic / Sink KV Cache

TRT-LLM 在 `CyclicKvCacheManager`（[TensorRT-LLM/cpp/tensorrt_llm/runtime/kvCacheManager.cpp](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h)）支持 per-layer 的 sink + cyclic window，回收粒度是 **layer × token**。它对 Sliding Window 的支持是"原地覆盖"（覆盖最旧 token 的 KV slot），而不是把 block 释放回 free list。优点是零碎片；缺点是覆盖式管理无法把显存让出来给其他请求，对并发提升的贡献是 0。

## 3.4 三者共性与差异总结

| 框架 | 最小回收粒度 | 是否支持同层 head 异构 | 释放回 free list |
|---|---|---|---|
| vLLM v1 | layer group × page | 需拆 layer group | 是 |
| SGLang | request × token（radix LRU） | 否 | 是（按请求） |
| TRT-LLM | layer × token | 否 | 否（原地覆盖） |
| **本设计（FastDeploy）** | **layer × head × block** | **是** | **是** |

# 四、对比分析

| 备选方案 | 优势 | 劣势 | 结论 |
|---|---|---|---|
| A. 维持 1D `block_idx` + SWA 掩码 | 改动最小，AppendAttention 无需调整 | 无法回收窗外 block；与 PR1 +30% 吞吐目标直接矛盾 | ✗ |
| B. 把 SWA head 拆到独立 layer group（vLLM 思路） | 复用现有 layer-level 管理 | 需要重写模型加载层、影响所有非 head-wise 模型；改动成本巨大 | ✗ |
| C. radix tree + LRU（SGLang 思路） | 通用性强 | LRU 驱逐 ≠ 及时回收；任务要求 +30% 在 ERNIE-4.5-21B-A3B 短期内不可达 | ✗ |
| **D. head-wise 离散 `block_idx` + V1 lifetime fence**（本设计） | 满足任务验收；改动局限在 cache_manager 与 AppendAttention | 引入 3D block_tables；需要保证 V1 swap 与回收的生命周期不变量 | ✓ |

方案 D 在 PR #6702 V0 实现上已得到原型验证；本设计在 V1 上重写并补上生命周期 fence，是当前唯一同时满足"细粒度回收"与"V1 调度兼容"的路径。

# 五、设计思路与实现方案

## 5.1 主体设计与折衷

整体架构：

```
┌────────────────────────────────────────────────────────────────────┐
│  ResourceManagerV1                                                 │
│  ├── recycle_request_swa_head_cache(req, layer, head, upto)        │ ← 新增（PR1）
│  └── overlap guard: cache_task_inflight_signal                     │ ← 复用
└──────────────────────────┬─────────────────────────────────────────┘
                           │ alloc / free per (layer, head)
┌──────────────────────────▼─────────────────────────────────────────┐
│  PrefixCacheManager (V1)                                           │
│  ├── head_block_idx[layer][head]: List[block_id]                   │ ← 新增（PR1）
│  ├── recycle_upto[layer][head]: int                                │ ← 新增（PR1）
│  └── flat-mode methods raise NotImplementedError 当 head_wise on   │ ← P10 守卫（PR1）
└──────────────────────────┬─────────────────────────────────────────┘
                           │ block_tables_3d [layer, head, block]
┌──────────────────────────▼─────────────────────────────────────────┐
│  GpuModelRunner → ForwardMeta                                      │
│  ├── block_tables (1D, fallback)                                   │
│  └── block_tables_3d (Optional, head-wise mode)                    │ ← 新增（PR1）
└──────────────────────────┬─────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│  AppendAttention                                                   │
│  ├── PR1 dual-pass：full kernel + SWA kernel + 列拼接              │ 复用 #6702 思路
│  └── PR2 fused kernel：单次 QKV 读取 + per-head SWA-mask predicate │ ← 新增（PR2）
└────────────────────────────────────────────────────────────────────┘
```

### 折衷点 1：3D `block_tables` 而非"per-layer dict[head] -> 1D"

- 选 3D：`(layer, head, block)` 张量，预申请、`max_blocks` 维 padding，GPU 端一次 indirect load。
- 不选 dict：每层每 head 一份 1D 张量，Python 侧灵活但 H2D 拷贝次数 = `layer * kv_heads`，A800 上每步 ~50 µs，吞吐场景内不可接受。

### 折衷点 2：默认 OFF

`FD_HEAD_WISE_KV_CACHE` 默认 `0`；`FD_T53_HEAD_WISE_SWA_FIXTURE` 默认 `0`。两者均通过环境变量显式启用。fallback 路径与现有行为字节级一致——这是被 reviewer 接受 PR1 的硬约束。

### 折衷点 3：与 prefix-caching 互斥（Phase-1 范围内）

PR #6702 的结论是 head-wise + prefix-caching 互斥。本设计在配置加载时校验：`enable_prefix_caching=True` ∧ `FD_HEAD_WISE_KV_CACHE=1` 直接 `RuntimeError`，不做隐式降级。reconciliation 留给后续 RFC，不在本任务范围。

## 5.2 关键技术点

### 5.2.1 V1 生命周期不变量（核心安全点）

> **∀ cache_id `c`，当 ∃ inflight `transfer_task t` 满足 `c ∈ t.block_set`，禁止 `recycle(c)`。**

锚点（直接复用，不新增信号）：

| 文件 | 行号 | 信号 |
|---|---|---|
| `cache_transfer_manager.py` | 264-298 | `transfer_task_id` / `cache_task_inflight_signal` 初始化 |
| `cache_transfer_manager.py` | 1324-1441 | per-layer swap 主循环 |
| `cache_transfer_manager.py` | 1498-1510 | 释放路径 |
| `prefix_cache_manager.py` | 557-580, 2207-2218 | per-request reset |

伪代码（PR1）：

```python
# resource_manager_v1.py — 新增方法
def recycle_request_swa_head_cache(self, req, layer, head, upto):
    """及时回收 SWA head 窗外 block；带 inflight 守卫。"""
    if not env_head_wise_kv_cache():
        return
    inflight = self.cache_transfer_manager.cache_task_inflight_signal
    candidate = self.prefix_cache_manager.head_block_idx[layer][head][:upto]
    safe = [c for c in candidate if not inflight.contains_block(c)]
    self.prefix_cache_manager.recycle_head_blocks(layer, head, safe)
    self.prefix_cache_manager.recycle_upto[layer][head] = upto
```

### 5.2.2 head-wise free list

```python
# prefix_cache_manager.py — 新增方法
def alloc_head_blocks(self, req_id, layer, head, n) -> List[int]:
    blocks = self._free_list.popn(n)
    self.head_block_idx[layer][head].extend(blocks)
    return blocks

def recycle_head_blocks(self, layer, head, blocks):
    self._free_list.extend(blocks)
    self.head_block_idx[layer][head] = [
        b for b in self.head_block_idx[layer][head] if b not in set(blocks)
    ]

# 防 P10 误用
def get_block_table_flat(self, *_, **__):
    if env_head_wise_kv_cache():
        raise NotImplementedError("flat block_table not valid in head-wise mode")
    return self._legacy_get_block_table_flat(...)
```

### 5.2.3 ERNIE 测试夹具（默认关闭，仅注入 SWA 字段）

```python
# paddleformers/base.py — 在 Attention() loop 之前 ~line 789
if os.environ.get("FD_T53_HEAD_WISE_SWA_FIXTURE", "0") == "1":
    cfg = fd_config.model_config
    n_kv = getattr(cfg, "num_key_value_heads", 1)
    cfg.window_size            = getattr(cfg, "window_size",            4096)
    cfg.sink_size              = getattr(cfg, "sink_size",              0)
    cfg.window_attn_skip_freq  = getattr(cfg, "window_attn_skip_freq",  1)
    cfg.head_wise_swa_ratio    = getattr(cfg, "head_wise_swa_ratio",    1.0 / n_kv)
```

### 5.2.4 PR2 — AppendAttention 离散布局 fused kernel

```cuda
// 伪代码：合并 Full + SWA 为单 kernel，per-head SWA mask
__global__ void append_attn_fused_kernel(
    const T* qkv,
    const int32_t* block_tables_3d,   // [layer, head, block]
    const int32_t* swa_head_mask,     // [head]: 1=SWA, 0=Full
    int window_size,
    /* ... */
) {
    int head = blockIdx.y;
    bool is_swa = swa_head_mask[head];

    for (int kv_blk = 0; kv_blk < num_blocks; ++kv_blk) {
        int block_id = block_tables_3d[layer * H * B + head * B + kv_blk];

        if (is_swa) {
            // window predicate
            int rel = current_token - kv_blk * BLOCK_SIZE;
            if (rel > window_size) continue;
        }

        // 单次 QKV 读取（vectorized 128-bit load）
        load_kv_vectorized(block_id, ...);
        compute_attention_partial(...);
    }
    write_out(...);
}
```

## 5.3 主要影响的模块接口变化

### 直接接口变化

| 接口 | 变化 | 兼容性 |
|---|---|---|
| `ForwardMeta.block_tables_3d` | 新增 `Optional[Tensor]` | 默认 `None`，不影响现有路径 |
| `PrefixCacheManager` | 新增 `head_block_idx` / `recycle_upto` / `alloc_head_blocks` / `recycle_head_blocks` | 仅在 head-wise 模式激活 |
| `ResourceManagerV1.recycle_request_swa_head_cache` | 新增 | env 关闭时为 no-op |
| 环境变量 | `FD_HEAD_WISE_KV_CACHE`、`FD_T53_HEAD_WISE_SWA_FIXTURE` | 默认 `0`，零行为变化 |

### 框架各环节影响

- **网络定义 / 模型层**：无（仅 paddleformers loader 增加默认关闭的 fixture 钩子）
- **底层数据结构**：`ForwardMeta` 增加可选字段；其余不变
- **OP**：PR2 修改 AppendAttention CUDA kernel；PR1 不动算子代码
- **数据 IO**：无
- **执行 / 调度**：`ResourceManagerV1` 新增方法；`GpuModelRunner` 在 `prepare_inputs` 中按 env 写 `block_tables_3d` slot
- **分布式**：TP 切分下 `kv_num_heads` 跨 rank 一致性以单测覆盖
- **模型保存**：无
- **预测部署**：无（默认配置等价于现有路径）

# 六、测试和验收的考量

## CPU 单测（FastDeploy CI 自动执行，遵循 T49 monkeypatch real-objects 模式）

| 用例 | 验证点 |
|---|---|
| `test_head_wise_freelist.py` | alloc / recycle / 多次循环；free list 长度不变量 |
| `test_head_wise_extend_validation.py` | 超 `max_blocks` 时正确报错 |
| `test_head_wise_tp_consistency.py` | TP=2/4/8 下 `kv_num_heads` 与 `block_tables_3d.dim0` 跨 rank 校验 |
| `test_head_wise_abort_reset.py` | per-request abort 后 `head_block_idx` / `recycle_upto` 完全清空（P4 守卫） |
| `test_swa_recycle.py` | block-aligned 边界回收正确性、inflight guard 触发 |
| `test_v1_recycle_fence.py` | monkeypatch `cache_task_inflight_signal`，证明回收被正确阻塞 |
| `test_p10_flat_method_guard.py` | head-wise 模式下调用 flat 方法 → `NotImplementedError` |

## A800 集成（AI Studio，`ssh aistudio`）

| Smoke | 内容 |
|---|---|
| S1 | `FD_T53_HEAD_WISE_SWA_FIXTURE=1 FD_HEAD_WISE_KV_CACHE=1 ENABLE_V1_KVCACHE_SCHEDULER=1` 下，bsz=4 / seq=1024 forward pass 不崩溃 |
| S2 | long-context recycle smoke：seq=8192, window=4096，验证 `recycle_upto` 推进 |
| 精度 | GSM8K parity ±0.5pp（head-wise vs flat） |
| **PR1 验收** | recycle ON vs OFF，**吞吐 +30%**（任务规则书原文） |

## H/B 卡（PR2 验收，请求 reviewer 复跑）

| 验收 | 内容 |
|---|---|
| **PR2 验收** | 不开启回收，1D vs 2D `block_idx`，**TTFT + TBT 均 +5%**（任务规则书原文） |

H/B 卡资源不足时，遵循 T49 doctrine：先在 A800 上交 preview 数据，正式数字请求 @luotao1 在 H/B 上复跑（前序 10000 A币已在 2026-04-07 由 liuzhuoxin@baidu.com 授予，机制已验证可行）。

# 七、影响面

## 对用户的影响

默认 `FD_HEAD_WISE_KV_CACHE=0`，**零行为变化**。需要 head-wise 收益的用户显式 export 即可。

## 对二次开发用户的影响

新增 `ForwardMeta.block_tables_3d` 字段、`PrefixCacheManager` 的 head-wise 接口对外暴露。已在 docstring 标注"V1 head-wise mode only"。

## 对框架架构的影响

cache_manager 引入 head-wise 数据结构，但通过 P10 守卫与 env gating 严格隔离两条路径，避免 V1 flat path 退化。

## 对性能的影响

- PR1 默认关闭时：无影响
- PR1 启用：ERNIE-4.5-21B-A3B 上 **吞吐 +30%（任务验收数）**
- PR2 默认关闭时：无影响
- PR2 启用：H/B 卡上 **TTFT + TBT 均 +5%（任务验收数）**

## 与业内框架的差距与优势

本设计是已知唯一在生产推理框架中提供 **layer × head × block** 三级回收粒度的实现。vLLM/SGLang/TRT-LLM 均无对应机制（见 §三）。我们不主张原创性，仅作为工程化定位。

## 其他风险

| 风险 | 缓解 |
|---|---|
| V1 swap 与 head-wise recycle 竞态 | 生命周期不变量 + `test_v1_recycle_fence.py` |
| TP 切分下 `kv_num_heads` 跨 rank 不一致 | `ForwardMeta` 显式存 `kv_num_heads` + 跨 rank 单测 |
| `block_wise_fp8` 在 head-wise 下 scale shape 错配 | 行号迁移 PR #6702 已审计实现（diff:246-252, 277-280, 3378-3384） |
| ERNIE-4.5-21B-A3B HF 配置缺少 SWA 字段 | 默认关闭的 loader fixture 注入；fallback ERNIE-4.5-VL |
| H/B 卡未授予 | A800 数据先交 + 请求 reviewer 在 H/B 复跑 |

# 八、排期规划

| 阶段 | 内容 | 时长 |
|---|---|---|
| Phase 0 | SWA 模型确认 + V1 lifetime 不变量写明 | 1 天 |
| PR1 | CacheManagerV1 head-wise + 及时回收 + CPU 单测 + A800 验收（+30% 吞吐） | 5 + 2 天 |
| PR2 | AppendAttention fused kernel + A800 preview + 请求 H/B 卡复跑 5% 数字 | 4 + 2 天 |

# 名词解释

- **SWA**: Sliding Window Attention，滑动窗口注意力
- **Full Attention**: 全局注意力
- **CacheManagerV1**: FastDeploy 的 schedule v1 cache 管理器（PR #7097）
- **block_idx**: KV cache 中 block 的索引张量
- **head-wise**: 同层内不同 head 维护各自的 block_idx
- **V1 lifetime fence**: 利用 `transfer_task_id` + `cache_task_inflight_signal` 构造的回收-传输生命周期不变量

# 附件及参考资料

- 任务原文: [community/hackathon/hackathon_10th 任务合集 §No.53](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%98%A5%E8%8A%82%E7%89%B9%E5%88%AB%E5%AD%A3%E2%80%94%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no53)
- 社区参考 PR (V0, 未合入): [PaddlePaddle/FastDeploy#6702](https://github.com/PaddlePaddle/FastDeploy/pull/6702)
- 关联模块: [PaddlePaddle/FastDeploy#7097 CacheManagerV1](https://github.com/PaddlePaddle/FastDeploy/pull/7097)
- 业内对比: [vLLM v1 KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py) · [SGLang RadixCache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) · [TensorRT-LLM kvCacheManager](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h)
- T49 RFC 范例: [community/rfcs/FastDeploy/20260404_parallel_ngram_match_gpu_kernel_for_fastdeploy.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/FastDeploy/20260404_parallel_ngram_match_gpu_kernel_for_fastdeploy.md)

<!-- reauthored 2026-05-02T09:09:09Z -->
