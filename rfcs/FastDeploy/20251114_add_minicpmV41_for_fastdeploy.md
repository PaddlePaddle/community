# **项目：在 FastDeploy 中原生支持 MiniCPM4.1-8B**

## **一、背景与目标**
**背景**: 

MiniCPM4.1-8B 是 decoder-only 文本生成模型，使用 GQA、LongRoPE、muP 缩放和混合推理模式，并以 InfLLM-V2 支持长上下文稀疏注意力。FastDeploy 在基线中没有注册 `MiniCPMForCausalLM`，也没有对应的模型组网、权重加载、LongRoPE、InfLLM-V2 后端或 MiniCPM4.1 端到端测试。

**目标**: 实现一个高性能、功能完整的 MiniCPM4.1-8B 推理后端，支持其 InfLLM-V2 稀疏注意力、混合推理模式等核心特性，并集成 FastDeploy 现有的低bit量化能力。

**核心技术路径**:

1. **复用**: 最大化复用 FastDeploy 现有的 attention backend 和量化组件。
2. **开发**: 实现 InfLLM-V2 动态稀疏注意力的高性能 CUDA 算子。
3. **集成**: 在 Python 层构建完整的 MiniCPM4.1-8B 模型结构，并适配 FastDeploy 现有的 WINT4/WINT8 量化框架。(现有的WINT2Config 是全局 WINT2 MoE 语义，不支持 Dense 模型。)

## **核心策略**: 并行推进算子开发与模型集成，以4周时间实现核心功能，通过充分复用现有基础设施快速交付。

---


## **二、总体设计**

```text
Hugging Face MiniCPM4.1 checkpoint
        |
        +-- ModelRegistry: MiniCPMForCausalLM
        |       |
        |       +-- MiniCPM41ForCausalLM
        |               +-- GQA + LongRoPE + paged KV cache
        |               +-- gated MLP + muP scaling
        |
        +-- WINT4/WINT8 converter
                +-- quant_weight + weight_scale
                +-- existing weight-only Linear runtime

Attention backend
        +-- FLASH_ATTN: dense reference/fallback
        +-- INFLLMV2_ATTN
                +-- Stage 1: dynamic block selection
                +-- Stage 2: selected blocks + recent dense window
```
## **三、实现方案**

### **Phase 1: [快速集成] FastDeploy低bit整数量化支持**

**目标**: 利用FastDeploy现有的WINT4/WINT8量化基础设施，为MiniCPM4.1-8B快速启用低bit整数量化推理能力。

**现状评估**:
- ✅ **FastDeploy已完整支持**: WINT4/WINT8量化
- ✅ **CUDA算子**: 高性能weight_only_linear实现
- ✅ **配置系统**: WINT4Config/WINT8Config
- ✅ **MiniCPM已有基础**: 已实现QuantizedMiniCPM41MLP和QuantizedMiniCPM41Attention

**任务 1.1: 扩展MiniCPM4.1-8B量化配置支持**

- **目标**: 完善MiniCPM4.1-8B的量化配置解析和模型注册
- **修改文件**: `fastdeploy/model_executor/layers/quantization/__init__.py`

```python
def parse_minicpm41_quant_config(args, model_config):
    """解析MiniCPM4.1-8B量化配置"""
    if args.quantization in ["wint4", "wint8"]:
        # 使用FastDeploy内置INT量化配置
        from .weight_only import WINT4Config, WINT8Config

        config_map = {
            "wint4": WINT4Config,
            "wint8": WINT8Config
        }

        quant_config = config_map[args.quantization].from_config({
            "is_quantized": True,
            "group_size": getattr(args, "group_size", 128)
        })

        return quant_config
```

**任务 1.2: 完善MiniCPM4.1-8B量化层支持**

- **目标**: 完善现有量化实现，确保WINT4/WINT8全面支持
- **修改文件**: `fastdeploy/model_executor/models/minicpm41_quant.py`

```python
# 扩展支持的量化类型 (第71行)
supported_quant_types = ["w4afp8", "w4a8", "wint8", "wint4", "fp8"]

# 扩展量化层自动选择 (第130行和第196行)
if self.quant_config and self.quant_config.quant_type in ["w4afp8", "w4a8", "wint8", "wint4"]:
    # 使用量化线性层
    linear_class = QuantizedRowParallelLinear  # 复用FastDeploy现有实现
```

**任务 1.3: 模型注册和配置更新**

- **目标**: 将MiniCPM4.1-8B添加到FastDeploy模型注册表，明确支持的量化类型
- **修改文件**: `fastdeploy/model_executor/models/__init__.py`

```python
# 注册MiniCPM4.1-8B支持
ModelRegistry.register_model(
    model_name="MiniCPM4.1-8B",
    model_class=MiniCPM41ForCausalLM,
    supported_quantizations=["wint4", "wint8", "w4afp8", "w4a8", "fp8"]
)
```

**任务 1.4: 测试用例和验证**

- **目标**: 验证WINT4/WINT8量化在MiniCPM4.1-8B上的正确性
- **创建文件**: `tests/models/test_minicpm41_int_quant.py`

```python
def test_minicpm41_wint_quantization():
    """测试MiniCPM4.1-8B INT量化功能"""
    for quant_type in ["wint4", "wint8"]:
        config = FDConfig(
            model="openbmb/MiniCPM4.1-8B",
            quantization=quant_type
        )
        model = MiniCPM41ForCausalLM(config)
        assert hasattr(model, 'quant_config')
        assert model.quant_config.quant_type == quant_type

        # 验证量化层正确初始化
        for layer in model.layers:
            assert hasattr(layer, 'mlp')
            assert hasattr(layer, 'self_attn')
```

**预期成果**:
- ✅ **WINT8**: 50%压缩率，<1%精度损失，1.5x推理加速
- ✅ **WINT4**: 75%压缩率，<2-3%精度损失，2x推理加速

**工作量**: 仅需1-2天的配置和测试工作，基于FastDeploy成熟的INT量化基础设施。

---

### **Phase 2: [核心开发] 实现 InfLLM-V2 注意力后端架构 (1-1.5 周)**

按照FastDeploy现有的attention backend管理方式实现InfLLM-V2，而不是作为独立的算子。

**任务 2.1: 实现 InfLLM-V2 注意力后端类**

- **目标**: 继承FastDeploy的AttentionBackend基类，实现InfLLM-V2特有的稀疏注意力
- **创建文件**: `fastdeploy/model_executor/layers/attention/infllmv2_attention_backend.py`
- **技术架构**:
  ```python
  from fastdeploy.model_executor.layers.attention.base_attention_backend import AttentionBackend
  from fastdeploy.model_executor.layers.attention.attention_metadata import AttentionMetadata
  from dataclasses import dataclass

  @dataclass
  class InfLLMV2AttentionMetadata(AttentionMetadata):
      """InfLLM-V2特有的注意力元数据"""
      kernel_select_mask: Optional[paddle.Tensor] = None
      topk_indices: Optional[paddle.Tensor] = None
      kernel_size: int = 32
      kernel_stride: int = 16
      topk: int = 64
      dense_len: int = 8192
      block_size: int = 64

  class InfLLMV2AttentionBackend(AttentionBackend):
      """InfLLM-V2稀疏注意力后端"""

      def __init__(self, fd_config, kv_num_heads, num_heads, head_dim, **kwargs):
          super().__init__()
          self.fd_config = fd_config
          self.kv_num_heads = kv_num_heads
          self.num_heads = num_heads
          self.head_dim = head_dim
          # InfLLM-V2特有参数
          self.kernel_size = getattr(fd_config, 'kernel_size', 32)
          self.kernel_stride = getattr(fd_config, 'kernel_stride', 16)
          self.topk = getattr(fd_config, 'topk', 64)
          self.dense_len = getattr(fd_config, 'dense_len', 8192)

      def init_attention_metadata(self, forward_meta: ForwardMeta):
          """初始化InfLLM-V2特有的元数据"""
          forward_meta.attn_metadata = InfLLMV2AttentionMetadata(
              kernel_size=self.kernel_size,
              kernel_stride=self.kernel_stride,
              topk=self.topk,
              dense_len=self.dense_len
          )

      def forward_decode(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta):
          """Decode模式前向传播 - 适用于逐token生成"""
          return self._infllmv2_forward(q, k, v, forward_meta, mode="decode")

      def _infllmv2_forward(self, q, k, v, forward_meta, mode):
          """核心InfLLM-V2前向传播逻辑"""
          # 调用底层CUDA算子实现
          from fastdeploy import custom_ops
          return custom_ops.infllmv2_attention_forward(
              q, k, v, forward_meta.attn_metadata,
              kernel_size=self.kernel_size,
              kernel_stride=self.kernel_stride,
              topk=self.topk,
              mode=mode
          )
  ```

**任务 2.2: 实现 InfLLM-V2 CUDA 核心算子**

- **目标**: 实现底层CUDA算子，被后端类调用
- **创建文件**: `custom_ops/gpu_ops/infllmv2_attention/infllmv2_impl.cuh`
- **技术要点**:
  ```cpp
  template <paddle::DataType D>
  __global__ void InfLLMV2AttentionKernel(
      const paddle::Tensor& query,                    // [bs, seq_len, num_heads, head_dim]
      const paddle::Tensor& key_cache,                // 分块KV缓存
      const paddle::Tensor& value_cache,
      const paddle::Tensor& kernel_select_mask,       // 动态核选择掩码
      const paddle::Tensor& topk_indices,             // TopK相关块索引
      const int kernel_size,
      const int kernel_stride,
      const int topk,
      const int dense_len,
      const std::string& mode,                        // "decode"
      paddle::Tensor& output                         // 输出注意力结果
  ) {
      // 1. 动态核选择算法
      // 2. 块级稀疏注意力计算
      // 3. 根据mode选择不同的计算策略
      // 目标：在长上下文下优化计算效率
  }
  ```

**任务 2.3: 注册 InfLLM-V2 后端到 FastDeploy 架构**

1. **添加后端枚举**:
   - 编辑 `fastdeploy/model_executor/layers/attention/attention_selecter.py`
   - 在 `_Backend` 枚举中添加 `INFLLMV2_ATTN = enum.auto()`

2. **实现平台映射**:
   - 编辑各平台的 `get_attention_backend_cls()` 方法
   - 添加 InfLLM-V2 后端的映射关系

3. **暴露CUDA算子**:
   ```cpp
   // 在 custom_ops/gpu_ops/infllmv2_attention/infllmv2.cu 中注册
   PD_BUILD_STATIC_OP(infllmv2_attention_forward)
       .Inputs({
           paddle::Tensor("query"),
           paddle::Tensor("key_cache"),
           paddle::Tensor("value_cache"),
           paddle::Tensor("metadata")
       })
       .Outputs({ "Out" })
       .Attrs({
           paddle::CustomOpAttr("kernel_size", paddle::DataType::INT32),
           paddle::CustomOpAttr("kernel_stride", paddle::DataType::INT32),
           paddle::CustomOpAttr("topk", paddle::DataType::INT32),
           paddle::CustomOpAttr("mode", paddle::DataType::STRING)
       })
       .SetKernelFn(PD_KERNEL(InfLLMV2AttentionForwardKernel));
   ```

4. **更新配置系统**:
   - 在 `fastdeploy/config.py` 中添加 `FD_ATTENTION_BACKEND="INFLLMV2_ATTN"` 支持
   - 在 `fastdeploy/envs.py` 中注册环境变量

**里程碑**: 完成并编译通过后，拥有了一个完整的InfLLM-V2注意力后端，可通过 `FD_ATTENTION_BACKEND=INFLLMV2_ATTN` 环境变量启用。

---

### **Phase 3: 混合推理模式支持 (0.5 周)**

**任务 3.1: 混合推理模式支持**

- **目标**: 支持MiniCPM4.1-8B的混合推理模式（thinking tokens）
- **创建文件**: `fastdeploy/model_executor/models/minicpm41/hybrid_reasoning.py`
- **技术要点**:
  ```python
  class HybridReasoningMode:
      def __init__(self, fd_config):
          self.reasoning_tokens = getattr(fd_config, 'reasoning_tokens', "thinking")
          self.max_thinking_length = getattr(fd_config, 'max_thinking_length', 512)

      def process_thinking_tokens(self, input_ids, logits):
          # 处理thinking tokens的特殊逻辑
          # 支持推理过程的多轮思考
  ```
- MiniCPM4.1 的分词器可能会将 `<think>`、`</think>` 以及强制结束文本分别编码为多个 token，因此不能使用**硬编码 token ID**。
- 实现分为通用层和模型层：
  - 通用层提供一个可由不同模型实现的钩子函数，用于通过当前模型的 tokenizer，将思考起始标记、结束标记和强制结束文本转换为 token ID 序列，并校验三类序列是否合法。
  - 模型层：提供 MiniCPM4.1 tokenizer 的具体标记序列

---

### **Phase 4: 构建 Python 接口与模型集成 (1.5 周)**

**任务 4.1: 集成 InfLLM-V2 后端到模型架构 (0.5 周)**

- **文件**: `fastdeploy/model_executor/layers/attention/__init__.py` (添加后端导出)
- **重点**: 确保 InfLLM-V2 后端能被模型正确加载和使用
- **测试**: 验证 `FD_ATTENTION_BACKEND=INFLLMV2_ATTN` 环境变量生效

**任务 4.2: 实现核心 MiniCPM4.1-8B 模型 (1 周)**

- **文件**: `fastdeploy/model_executor/models/minicpm41/minicpm41.py`
- **核心逻辑**:
  - 复用现有 ModelForCausalLM 架构
  - 自动检测并应用 InfLLM-V2 后端 (通过 `FD_ATTENTION_BACKEND=INFLLMV2_ATTN`)
  - 集成 `FastDeploy` WINT4/WINT8 量化支持
  - 支持混合推理模式

```python
# fastdeploy/model_executor/models/minicpm41/minicpm41.py 核心框架
class MiniCPM41ForCausalLM(ModelForCausalLM):
    def __init__(self, fd_config):
        super().__init__(fd_config)
        # 自动检测InfLLM-V2配置
        if hasattr(fd_config, 'use_infllmv2') and fd_config.use_infllmv2:
            # 确保使用InfLLM-V2后端
            os.environ['FD_ATTENTION_BACKEND'] = 'INFLLMV2_ATTN'

        # 初始化模型特有组件
        self._init_hybrid_reasoning(fd_config)
```

**任务 4.3: 快速量化集成 (0.5 周)**

```python
# 复用FastDeploy现有量化框架，全面支持WINT量化
def quantize_model(self, quant_config):
    if quant_config.quant_type in ["wint4", "wint8"]:
        # 直接使用FastDeploy内置的WINT量化实现
        self.quant_config = quant_config
    else:
        # 复用现有 W4A8, W4AFP8, FP8 实现
        super().quantize_model(quant_config)
```

---

### **Phase 5: 性能优化与测试 (1.5 周)**

**任务 5.1: 核心功能快速验证 (0.5 周)**

- 基础模型加载和推理测试
- InfLLM-V2 稀疏注意力功能验证
- FastDeploy WINT4/WINT8 量化功能验证

**任务 5.2: 性能优化 (0.5 周)**

- 专注核心瓶颈优化
- KV缓存优化
- 算子融合（SiLU+GLU等）

**任务 5.3: 集成测试与文档 (0.5 周)**

- 端到端测试
- 基础文档编写
- 部署指南

**任务 5.4: 核心测试用例**

```python
# tests/models/test_minicpm41.py - 精简版核心测试
def test_minicpm41_basic_functionality():
    """基础功能测试"""
    config = FDConfig(model="openbmb/MiniCPM4.1-8B")
    model = MiniCPM41ForCausalLM(config)
    # 基础推理验证

def test_infllmv2_core_features():
    """InfLLM-V2核心特性验证"""
    # 长上下文基础验证
    # 稀疏注意力功能确认

def test_wint_quantization():
    """FastDeploy WINT量化基础测试"""
    for quant_type in ["wint4", "wint8"]:
        config = FDConfig(model="openbmb/MiniCPM4.1-8B", quantization=quant_type)
        model = MiniCPM41ForCausalLM(config)
        # 验证量化功能正确性
```

**任务 5.5: 快速性能基准**

- 基础延迟和吞吐量测试
- 关键性能指标验证
- 与官方实现对比

**任务 5.6: 核心文档**

- **文件**: `docs/get_started/minicpm41.md`
- 重点：快速部署指南和核心API文档

```markdown
# MiniCPM4.1-8B 部署指南

## 环境配置
```bash
# 启用 InfLLM-V2 稀疏注意力
export FD_ATTENTION_BACKEND=INFLLMV2_ATTN

# 启用 WINT 量化 (可选: wint4, wint8)
export FD_QUANT_TYPE=WINT4
```

## 基础使用
```python
from fastdeploy import FDConfig, MiniCPM41ForCausalLM

# WINT4 量化示例
config = FDConfig(
    model="openbmb/MiniCPM4.1-8B",
    quantization="wint4",
    use_infllmv2=True
)

model = MiniCPM41ForCausalLM(config)

# WINT8 量化示例
config_wint8 = FDConfig(
    model="openbmb/MiniCPM4.1-8B",
    quantization="wint8"
)

model_wint8 = MiniCPM41ForCausalLM(config_wint8)

```

新增 `scripts/convert_minicpm41_wint.py`，将原始 BF16 safetensors 转为 FastDeploy 可加载的预量化 checkpoint：

- 使用 `paddle.nn.quant.weight_quantize` 的 `weight_only_int4`/`weight_only_int8`，输出 `.quant_weight` 和 `.weight_scale`。
- 保留 tokenizer、配置和非目标 tensors，并重建 safetensors index。
- 在 `config.json` 写入与 `--quantization wint4|wint8` 一致的配置。
- 校验输入模型 architecture、目标 tensor 集合、shape/alignment、输出目录和转换前后字节数。
- 原始模型与 tokenizer 的 pad token 配置不一致时，在输出 checkpoint 中显式对齐并记录；不得静默生成不可启动的模型。

运行时复用 FastDeploy 的 `WINT4Config`、`WINT8Config` 和 weight-only Linear。MiniCPM4.1 的 fused QKV 在 WINT4 下存在输出维 2:1 packing，需在模型目录实现只针对该参数的 packed shard loader；其他参数继续使用通用 loader。TP1 和 TP2 都必须验证 Q/K/V shard offset、KV head replication 和真实权重加载。

启动方式为：

```bash
python scripts/convert_minicpm41_wint.py \
  --input-dir /path/to/MiniCPM4.1-8B \
  --output-dir /path/to/MiniCPM4.1-8B-WINT4 \
  --quantization wint4 \
  --device gpu:0

FD_ATTENTION_BACKEND=FLASH_ATTN \
python -m fastdeploy.entrypoints.openai.api_server \
  --model /path/to/MiniCPM4.1-8B-WINT4 \
  --quantization wint4 \
  --tensor-parallel-size 1
```

## **四、交付内容清单**

### **4.1 代码交付**

1. **模型实现文件**
   ```
   fastdeploy/model_executor/models/minicpm41/
   ├── __init__.py
   ├── hybrid_reasoning.py
   ├── minicpm41.py
   └── config_minicpm41.py
   ```

2. **Attention后端**
   ```
   fastdeploy/model_executor/layers/attention/
   ├── infllmv2_attention_backend.py     # 核心后端实现
   └── __init__.py                      # 后端导出
   ```

3. **自定义算子**
   ```
   custom_ops/gpu_ops/
   └── infllmv2_attention/              # InfLLM-V2 CUDA算子
       ├── infllmv2_impl.cuh
       └── infllmv2.cu
   ```

4. **量化配置文件**
   ```
   fastdeploy/model_executor/models/minicpm41_quant.py    # 量化模型实现
   tests/models/test_minicpm41_int_quant.py              # WINT量化测试
   ```

### **4.2 配置与注册**

- 在 `fastdeploy/model_executor/layers/attention/attention_selecter.py` 中注册 `INFLLMV2_ATTN` 枚举
- 更新各平台 `get_attention_backend_cls()` 方法支持 InfLLM-V2 后端
- 模型注册到FastDeploy模型库，支持 ["wint4", "wint8", "w4afp8", "w4a8", "fp8"]
- 更新`supported_models.md`
- 添加默认配置模板

### **4.3 测试套件**

```
tests/models/test_minicpm41.py
tests/models/test_minicpm41_int_quant.py              # WINT量化专项测试
tests/integration/test_minicpm41_e2e.py
benchmarks/minicpm41_performance.py
```

### **4.4 文档**

- 部署指南：`docs/get_started/minicpm41.md`
- WINT量化使用指南：`docs/quantization/minicpm41_quant.md`
- API文档
- 最佳实践指南
- 性能基准报告

### **5.1 量化效果**
- **WINT4压缩**: 75%参数压缩，精度损失<2-3%，推理速度提升2x
- **WINT8压缩**: 50%参数压缩，精度损失<1%，推理速度提升1.5x
- **W4A8量化**: 75%内存节省，推理速度提升2x
- **FP8量化**: 87.5%内存节省，推理速度提升2x

### **5.2 混合推理**
- **推理速度**: 3x解码加速
- **复杂任务**: 提升复杂推理任务的准确率

### **6.1 必要的公共文件修改**

- `fastdeploy/model_executor/layers/rotary_embedding.py`：在已有 architecture 分派中增加 LongRoPE。
- `fastdeploy/platforms/base.py`、`fastdeploy/platforms/cuda.py` 和 attention package：注册 InfLLM-V2。
- `custom_ops/setup_ops.py`、`custom_ops/gpu_ops/cpp_extensions.cc`：构建并暴露 InfLLM-V2 op。
- `fastdeploy/config.py`、engine/worker 的通用接口和 `thinking_budget.py`：增加模型无关的多 token marker 与 logits processor hook。
- quantization 配置入口：仅在通用解析无法表达 WINT4/WINT8 checkpoint 契约时增加最小映射，并限制到 `MiniCPMForCausalLM`。
- `docs/supported_models.md`、`docs/zh/supported_models.md`：只列出经过验收的 BF16/WINT4/WINT8。

## **六、风险评估与缓解策略**

### **6.1 技术风险**

**风险1**: 多硬件平台适配问题
**缓解**: 优先支持NVIDIA GPU，再扩展到其他平台

**风险2**: 公共代码影响面
**缓解**: 通用 engine、worker、logits processor 的修改需要回归现有推理模型，不能引入 MiniCPM 专属判断或改变单 token 语义。

### **6.2 时间风险**

**缓解策略**:
- 并行开发多个算子
- 充分利用现有FastDeploy基础设施
- 建立MVP（最小可行产品）版本

## **七、4周精简实施计划**

### **Week 1-2: 基础搭建与WINT量化集成**
- **Days 1-2**: InfLLM-V2 稀疏注意力核心算子开发（优先级最高）
- **Days 3-4**: FastDeploy WINT4/WINT8量化集成配置
- **Days 5-8**: 项目设置与配置（目录结构、FDConfig更新）
- **Days 9-14**: InfLLM-V2 CUDA算子完善和基础测试

### **Week 3: 模型集成与开发**
- **Days 1-2**: 混合推理模式支持实现
- **Days 3-5**: MiniCPM4.1-8B 核心模型实现
- **Days 6-7**: WINT量化功能集成和测试

### **Week 4: 测试验证与优化**
- **Days 1-1**: 跑通 serving 和短/长上下文基线。
- **Days 2-3**: 性能瓶颈分析和优化
- **Days 4-5**: 端到端集成测试和问题修复
- **Days 6-7**: 文档编写和部署指南

**总计**: 4周完成核心功能交付，专注于基础推理和量化支持

### **加速策略**
1. **WINT量化复用**: 充分利用FastDeploy现有的WINT4/WINT8量化基础设施，仅需1-2天配置工作
2. **复用优先**: 最大化复用FastDeploy现有组件，减少开发工作量
3. **后端架构**: 按照现有attention backend模式实现，确保架构一致性
4. **MVP策略**: 先实现核心功能，高级特性后续迭代
5. **聚焦重点**: 优先实现InfLLM-V2稀疏注意力和WINT量化等核心特性

## **架构优势**

**采用Attention Backend架构的核心优势:**

1. **无缝集成**: 完全符合FastDeploy现有的attention管理机制
2. **配置统一**: 通过 `FD_ATTENTION_BACKEND=INFLLMV2_ATTN` 环境变量统一管理
3. **平台兼容**: 自动适配各硬件平台(CUDA/XPU/NPU等)
4. **测试友好**: 集成到现有的后端测试框架
5. **扩展性强**: 后续可轻松添加新的注意力变体

此实施方案基于FastDeploy现有架构设计，充分利用FastDeploy成熟的WINT量化基础设施，确保MiniCPM4.1-8B模型能够快速集成到现有框架中。通过1-2天即可完成WINT4/WINT8量化支持，为用户提供低bit压缩、高性能的大模型推理解决方案。
