# 为FastDeploy集成 SageAttn v2/2++

| 方案名称                         |  为FastDeploy集成 SageAttn v2/2++  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | fangfangssj                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2025-09-16                              | 
| 版本号                                                      | V1.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==3.2.0                     | 
| 文件名                                                      | 20250916_FastDeploy_add_sageattention.md<br> | 

# 一、概述
## 1、相关背景
随着深度学习模型规模的不断扩大，特别是在大语言模型（LLM）领域，模型的计算和存储需求呈指数级增长。这对实际部署带来了严峻挑战，包括显存瓶颈、计算延迟和能耗问题。为了应对这些挑战，模型量化技术成为关键解决方案之一，通过降低模型权重和激活值的数值精度来减少资源消耗。
## 2、功能目标
为FastDeploy集成 SageAttn v2++，测试SageAttn在EB，Qwen上的表现。
## 3、意义
SageAttn（Sparse Attention with Group-wise Quantization）是一种结合稀疏注意力和分组量化技术的创新方法，专门针对Transformer架构中的注意力机制进行优化。该方法在保持模型精度的同时，显著降低计算复杂度和内存使用量，特别适合处理长序列输入场景。

# 二、FastDeploy现状
目前FastDeploy中未集成SageAttn v2++

# 三、业内方案调研
vllm与sglang中尚未集成SageAttn系列

# 四、对比分析
参考之前FastDeploy的后端接入SageAttn v2++

# 五、设计思路与实现方案
```text
fastdeploy/
├── model_executor/
│   ├── layers/
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py          # 基础Attention类
│   │   │   ├── sage_attn_backend.py  # 新增：SageAttention后端实现
│   │   │   └── ...                   # 其他attention后端
│   │   └── ...
│   ├── ops/
│   │   ├── triton_ops/
│   │   │   ├── __init__.py
│   │   │   ├── sage_attention.py     # 新增：SageAttention系列Triton算子
│   │   │   └── ...                   # 其他Triton算子
│   │   └── ...
└── platforms/
    ├── cuda.py                       # 需要修改：添加SageAttentionBackend注册
    └── ...
```

在fastdeploy/model_executor/layers/attention/下，新增sage_attn_backend.py文件，继承AttentionBackend，AttentionMetadata两个基类，实现SageAttentionMetadata，SageAttentionBackend两个类。

添加对应的自定义triton算子，将SageAttention中sageattention/triton下的算子转化后放在fastdeploy/model_executor/ops/triton_ops/目录下，同时增加对应的测试

在fastdeploy/model_executor/layers/attention/__init__.py中添加对应的SageAttention注册

在fastdeploy/platforms/cuda.py中，添加对应SageAttentionBackend。

# 六、测试和验收的考量
tests/layers/下增加SageAttention的测试

tests/operators/下增加新增triton算子测试

SageAttention在EB，Qwen上测试性能

# 七、影响面
为FastDeploy集成 SageAttn v2++，不影响其他部分

# 八、排期规划
* 2025-9-18 ~ 2025-10-7：完成集成代码开发
* 2025-9-28 ~ 2025-10-7：完成代码测试
* 2025-10-1 ~ 2025-10-7： 完成部署示例及文档

# 九、参考资料

[SageAttention](https://github.com/thu-ml/SageAttention)

[FD接入MobaAttn的PR](https://github.com/PaddlePaddle/FastDeploy/pull/3209)
