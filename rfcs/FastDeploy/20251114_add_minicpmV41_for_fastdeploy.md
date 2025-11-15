# **项目：在 FastDeploy 中原生支持 MiniCPM4.1-8B**

**目标**: 实现一个高性能、功能完整的 MiniCPM4.1-8B 推理后端，支持其 InfLLM-V2 稀疏注意力、LongRoPE 扩展位置编码、BitCPM4 三元量化及混合推理模式等核心特性。

**核心技术路径**:

1. **复用**: 最大化复用 FastDeploy 现有的 append_attention 和 MLA attention 组件。
2. **开发**: 实现 InfLLM-V2 动态稀疏注意力和 LongRoPE 扩展位置编码的高性能 CUDA 算子。
3. **集成**: 在 Python 层构建完整的 MiniCPM4.1-8B 模型结构，并适配 FastDeploy 现有的量化框架。

## **核心策略**: 并行推进算子开发与模型集成，以6周时间实现完整功能，通过充分复用现有基础设施和并行开发提高效率。

### **Phase 0: 项目设置与配置 (1 天)**

在写任何代码之前，先搭建好项目的骨架。

**任务 1: 创建目录结构**
在 FastDeploy 代码库中，创建以下新文件和目录的占位符：

```bash
# 1. 创建 MiniCPM4.1-8B 模型目录
mkdir -p fastdeploy/model_executor/models/minicpm41

# 2. 创建 InfLLM-V2 算子的 C++/CUDA 目录
mkdir -p custom_ops/gpu_ops/infllmv2_attention

# 3. 创建 LongRoPE 算子的目录
mkdir -p custom_ops/gpu_ops/longrope_embedding

# 4. 创建 BitCPM4 三元量化算子目录
mkdir -p custom_ops/gpu_ops/ternary_quantize

# 5. 创建模型文件占位符
touch fastdeploy/model_executor/models/minicpm41/__init__.py
touch fastdeploy/model_executor/models/minicpm41/minicpm41.py
touch fastdeploy/model_executor/models/minicpm41/attention_minicpm41.py
touch fastdeploy/model_executor/models/minicpm41/config_minicpm41.py

# 6. 创建多模态处理器（如需要）
touch fastdeploy/input/minicpm41_processor.py
```

**任务 2: 更新 `FDConfig`**
编辑 `fastdeploy/config.py`，在 `ModelConfig` 类中添加 MiniCPM4.1-8B 特有的配置项。

```python
# fastdeploy/config.py -> class ModelConfig

class ModelConfig:
    def __init__(self, model: str, **args):
        # ...
        self.partial_rotary_factor: float = 1.0

        # === MiniCPM4.1-8B 特有配置 ===
        # InfLLM-V2 稀疏注意力配置
        self.use_infllmv2: bool = False
        self.kernel_size: int = 32
        self.kernel_stride: int = 16
        self.block_size: int = 64
        self.topk: int = 64
        self.dense_len: int = 8192
        self.nope: bool = True

        # LongRoPE 配置
        self.use_longrope: bool = False
        self.max_position_embeddings: int = 8192
        self.scaling_factor: float = 2.0
        self.short_factor: list = [1.0, 1.0, 1.0]
        self.long_factor: list = [1.0, 1.0, 1.0]

        # BitCPM4 三元量化配置
        self.use_ternary_quant: bool = False
        self.quant_threshold: float = 0.1
        self.compression_ratio: float = 0.1

        # 混合推理模式配置
        self.use_hybrid_reasoning: bool = False
        self.reasoning_tokens: str = "thinking"
        self.max_thinking_length: int = 512
        # ===============================

        for key, value in args.items():
            # ...
```

---

### **Phase 1: [核心开发] 实现 InfLLM-V2 稀疏注意力 CUDA 算子 (1-1.5 周)**

这是整个项目中技术含量最高的部分，实现 MiniCPM4.1-8B 的核心创新技术。

**任务 1.1: 实现 InfLLM-V2 动态稀疏注意力核心算子**

- **目标**: 实现 MiniCPM 特有的可训练稀疏注意力机制
- **创建文件**: `custom_ops/gpu_ops/infllmv2_attention/infllmv2_impl.cuh`
- **技术要点**:
  ```cpp
  template <paddle::DataType D>
  __global__ void InfLLMV2AttentionKernel(
      const paddle::Tensor& query,                    // [bs, seq_len, num_heads, head_dim]
      const paddle::Tensor& key_cache,                // 分块KV缓存
      const paddle::Tensor& value_cache,
      const paddle::Tensor& block_tables,             // 块表映射
      const paddle::Tensor& kernel_select_mask,       // 动态核选择掩码
      const paddle::Tensor& topk_indices,             // TopK相关块索引
      const int kernel_size,
      const int kernel_stride,
      const int topk,
      paddle::Tensor& output                         // 输出注意力结果
  ) {
      // 1. 动态核选择算法
      // 2. 块级稀疏注意力计算
      // 3. 内存访问优化
      // 目标：在128K上下文下仅需5%的token-to-token计算
  }
  ```

**任务 1.2: 编写 C++ Host 启动器**

- **目标**: 编写调度逻辑，处理 prefill 和 decode 阶段的不同策略
- **创建文件**: `custom_ops/gpu_ops/infllmv2_attention/infllmv2.cu`
- **代码框架**:

  ```cpp
  #include "infllmv2_impl.cuh"
  #include "paddle/extension.h"

  void InfLLMV2AttentionForwardKernel(
      const paddle::Tensor& query, const paddle::Tensor& key_cache,
      const paddle::Tensor& value_cache, const paddle::Tensor& block_tables,
      const paddle::Tensor& kernel_select_mask, const paddle::Tensor& topk_indices,
      const int kernel_size, const int kernel_stride, const int topk,
      paddle::Tensor& output, const bool is_prefill
  ) {
      // 1. 根据 is_prefill 选择不同的计算策略
      if (is_prefill) {
          // Prefill阶段：高效处理长序列输入
          // 启动 InfLLMV2PrefillKernel
      } else {
          // Decode阶段：逐token高效生成
          // 启动 InfLLMV2DecodeKernel
      }

      // 2. 动态核选择和稀疏注意力计算
      // 3. 结果聚合和输出
  }
  ```

**任务 1.3: 暴露自定义算子**

1. 在 `infllmv2.cu` 文件末尾添加算子注册代码：
   ```cpp
   PD_BUILD_STATIC_OP(infllmv2_attention_forward)
       .Inputs({
           paddle::Tensor("query"),
           paddle::Tensor("key_cache"),
           paddle::Tensor("value_cache"),
           paddle::Tensor("block_tables"),
           paddle::Tensor("kernel_select_mask"),
           paddle::Tensor("topk_indices")
       })
       .Outputs({ "Out" })
       .Attrs({
           paddle::CustomOpAttr("kernel_size", paddle::DataType::INT32),
           paddle::CustomOpAttr("kernel_stride", paddle::DataType::INT32),
           paddle::CustomOpAttr("topk", paddle::DataType::INT32),
           paddle::CustomOpAttr("is_prefill", paddle::DataType::BOOL)
       })
       .SetKernelFn(PD_KERNEL(InfLLMV2AttentionForwardKernel));
   ```
2. 编辑 `custom_ops/gpu_ops/cpp_extensions.cc`，添加 Python 绑定
3. 更新 `custom_ops/setup_ops.py`，将新文件加入编译列表

**里程碑**: 完成并编译通过后，拥有了一个高性能的 InfLLM-V2 稀疏注意力算子。

---

### **Phase 2: 并行开发 LongRoPE 和 BitCPM4 算子 (1 周)**

**任务 2.1: 实现 LongRoPE 核心算子**

- **目标**: 支持 64K+ 上下文的旋转位置编码
- **创建文件**: `custom_ops/gpu_ops/longrope_embedding/longrope_impl.cuh`
- **技术要点**:
  ```cpp
  __global__ void LongRoPEKernel(
      const paddle::Tensor& positions,          // 位置索引
      const paddle::Tensor& short_sin_cos,      // 短序列sin/cos
      const paddle::Tensor& long_sin_cos,       // 长序列sin/cos
      const float scaling_factor,                # 扩展因子
      const int max_position_embeddings,         # 最大位置数
      paddle::Tensor& rotary_emb                 # 输出旋转编码
  ) {
      // 实现LongRoPE的动态因子调整
      // 支持长度缩放的插值算法
      // 兼容标准RoPE格式
  }
  ```

**任务 2.2: 集成到现有 RoPE 框架**

- 扩展 `custom_ops/gpu_ops/fused_rotary_position_encoding.cu`
- 添加 LongRoPE 模式支持
- 确保与现有注意力算子的兼容性

**任务 2.3: 并行实现 BitCPM4 三元量化算子**

- **目标**: 实现 1.58 位三元量化，达到 90% 模型压缩
- **创建文件**: `custom_ops/gpu_ops/ternary_quantize/ternary_quantize.cu`
- **技术要点**:
  ```cpp
  template <typename T>
  __global__ void TernaryQuantizeKernel(
      const paddle::Tensor& input_weights,        // 原始权重
      const float threshold,                       # 量化阈值
      paddle::Tensor& quant_weights,              # 量化权重
      paddle::Tensor& scales,                     # 缩放因子
      paddle::Tensor& signs                       # 符号位
  ) {
      // 实现三元量化算法
      // 值域：{-1, 0, +1} * scale
      // 高效的位打包和解包
      // 目标：<2% 精度损失，90% 压缩率
  }
  ```

**任务 2.4: 快速集成到 FastDeploy 量化框架**

- 复用现有量化后端架构
- 重点支持 BitCPM4 格式（核心差异化特性）
- W8A16、W4A8、FP8 格式在基础版本中复用现有实现

---

### **Phase 3: 构建 Python 接口与模型集成 (1.5 周)**

**任务 3.1: 并行创建 Python 包装器和后端 (0.5 周)**

- **文件**: `fastdeploy/model_executor/layers/attention/ops/infllmv2_attention.py`
- **文件**: `fastdeploy/model_executor/layers/attention/infllmv2_backend.py`
- **重点**: 复用现有架构，快速集成核心算子

**任务 3.2: 实现核心 MiniCPM4.1-8B 模型 (1 周)**

- **文件**: `fastdeploy/model_executor/models/minicpm41/minicpm41.py`
- **核心逻辑**:
  - 复用现有 ModelForCasualLM 架构
  - 重点实现 InfLLM-V2 注意力集成
  - 简化版本先支持基础功能，高级功能后续迭代

**任务 3.3: 快速量化集成 (0.5 周)**

```python
# 复用现有量化框架，重点支持 BitCPM4
def quantize_model(self, quant_config):
    if quant_config.quant_type == "BITCPM4":
        self._apply_ternary_quantization(threshold=self.quant_threshold)
    else:
        # 复用现有 W8A16, W4A8, FP8 实现
        super().quantize_model(quant_config)
```

---

### **Phase 4: 性能优化与测试 (1.5 周)**

**任务 4.1: 核心功能快速验证 (0.5 周)**

- 基础模型加载和推理测试
- InfLLM-V2 稀疏注意力功能验证
- BitCPM4 量化基础测试

**任务 4.2: 性能优化 (0.5 周)**

- 专注核心瓶颈优化
- KV缓存优化
- 算子融合（SiLU+GLU等）

**任务 4.3: 集成测试与文档 (0.5 周)**

- 端到端测试
- 基础文档编写
- 部署指南

**任务 4.4: 核心测试用例**

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

def test_bitcpm4_quantization():
    """BitCPM4量化基础测试"""
    # 压缩率和精度基础验证
```

**任务 4.5: 快速性能基准**

- 基础延迟和吞吐量测试
- 关键性能指标验证
- 与官方实现对比

**任务 4.6: 核心文档**

- **文件**: `docs/get_started/minicpm41.md`
- 重点：快速部署指南和核心API文档

## **四、交付内容清单**

### **4.1 代码交付**

1. **模型实现文件**
   ```
   fastdeploy/model_executor/models/minicpm41/
   ├── __init__.py
   ├── minicpm41.py
   ├── attention_minicpm41.py
   └── config_minicpm41.py
   ```

2. **自定义算子**
   ```
   custom_ops/gpu_ops/
   ├── infllmv2_attention/
   │   ├── infllmv2_impl.cuh
   │   └── infllmv2.cu
   ├── longrope_embedding/
   │   ├── longrope_impl.cuh
   │   └── longrope.cu
   └── ternary_quantize/
       ├── ternary_quantize_impl.cuh
       └── ternary_quantize.cu
   ```

3. **多模态处理器**（如需要）
   ```
   fastdeploy/input/minicpm41_processor.py
   ```

### **4.2 配置与注册**

- 模型注册到FastDeploy模型库
- 更新`supported_models.md`
- 添加默认配置模板

### **4.3 测试套件**

```
tests/models/test_minicpm41.py
tests/integration/test_minicpm41_e2e.py
benchmarks/minicpm41_performance.py
```

### **4.4 文档**

- 部署指南：`docs/get_started/minicpm41.md`
- API文档
- 最佳实践指南
- 性能基准报告

## **五、预期性能指标**

### **5.1 长上下文处理**
- **128K上下文**: 相比标准注意力实现7x解码加速
- **内存效率**: 显著减少显存占用
- **准确率保持**: 长上下文理解准确率不低于基准模型

### **5.2 量化效果**
- **BitCPM4压缩**: 90%参数压缩，精度损失<2%
- **W8A16量化**: 4x内存节省，推理速度提升2x
- **FP8量化**: 8x内存节省，推理速度提升3x

### **5.3 混合推理**
- **推理速度**: 3x解码加速
- **复杂任务**: 提升复杂推理任务的准确率

## **六、风险评估与缓解策略**

### **6.1 技术风险**

**风险1**: InfLLM-V2算法复杂度超出预期
**缓解**: 分阶段实现，先支持基础稀疏注意力，再添加动态核选择

**风险2**: 量化精度损失过大
**缓解**: 支持多种量化策略，提供精度-性能权衡选项

**风险3**: 多硬件平台适配问题
**缓解**: 优先支持NVIDIA GPU，再扩展到其他平台

### **6.2 时间风险**

**缓解策略**:
- 并行开发多个算子
- 充分利用现有FastDeploy基础设施
- 建立MVP（最小可行产品）版本

## **七、6周加速实施计划**

### **Week 1: 基础搭建与核心算子**
- **Days 1-2**: 项目设置与配置（目录结构、FDConfig更新）
- **Days 3-5**: InfLLM-V2 稀疏注意力核心算子开发（优先级最高）
- **Days 6-7**: 基础算子编译验证和调试

### **Week 2: 并行算子开发**
- **Days 1-3**: LongRoPE 扩展位置编码算子
- **Days 4-6**: BitCPM4 三元量化算子
- **Day 7**: 算子集成测试和问题修复

### **Week 3: Python接口与模型集成**
- **Days 1-2**: Python包装器和注意力后端
- **Days 3-5**: MiniCPM4.1-8B 核心模型实现
- **Days 6-7**: 快速量化集成和基础测试

### **Week 4: 性能优化与调试**
- **Days 1-3**: 性能瓶颈分析和优化
- **Days 4-5**: 内存优化和算子融合
- **Days 6-7**: 集成调试和问题解决

### **Week 5: 测试验证**
- **Days 1-2**: 核心功能测试
- **Days 3-4**: 性能基准测试
- **Days 5-7**: 端到端集成测试

### **Week 6: 文档与交付**
- **Days 1-2**: 文档编写和部署指南
- **Days 3-4**: 最终集成测试和问题修复
- **Days 5-7**: 代码交付和验收准备

**总计**: 6周完成核心功能交付，后续可根据需要进行功能扩展

### **加速策略**
1. **并行开发**: 多个算子同时开发，减少串行等待时间
2. **复用优先**: 最大化复用FastDeploy现有组件，减少开发工作量
3. **MVP策略**: 先实现核心功能，高级特性后续迭代
4. **聚焦重点**: 优先实现InfLLM-V2和BitCPM4等差异化特性

此实施方案基于FastDeploy现有架构设计，确保MiniCPM4.1-8B模型能够无缝集成到现有框架中，同时充分利用FastDeploy的高性能推理能力，为用户提供长上下文、高压缩、高性能的大模型推理解决方案。