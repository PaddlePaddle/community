> ### NOTE：不在RFC需求列表里的任务可直接开发，不需要提交RFC

## 框架任务

| 序号 |   难度    | 任务标题                                     | 是否需要提交 RFC |
| :--: | :-------: | -------------------------------------------- | :--------------: |
|  109   |  ⭐️⭐️   | 基于 Setuptools 80+ 版本自定义算子机制适配          |        是        |

## FastDeploy任务

| 序号 |   难度    | 任务标题                                     | 是否需要提交 RFC |
| :--: | :-------: | -------------------------------------------- | :--------------: |
|  86   |  ⭐️⭐️   | FastDeploy编译加速                  |        是        |
|  87   |  ⭐️⭐️   | 为FastDeploy增加Profiler模块                   |        是        |
|  88   |  ⭐️⭐️   | 为FastDeploy重构log日志打印范式 |        是        |
|  89   | ⭐️⭐️ | 为FastDeploy集成 SageAttn v2/2++         |        是        |
|  90   |  ⭐️⭐️   | 为FastDeploy集成  SpargeAttn                  |        是        |
|  91   |  ⭐️⭐️   | FastDeploy中的MoE GroupGEMM支持INT8*INT8实现   |        是        |
|  92   |  ⭐️⭐️   | 为 FastDeploy 新增 K2模型 |        是        |
|  93   | ⭐️⭐️ | 为 FastDeploy 新增 MiniMax-M1模型         |        是        |
|  94   |  ⭐️⭐️⭐️   | 为 FastDeploy 新增 SD、Flux扩散模型 |        是        |
|  95   | ⭐️⭐️ | 为 FastDeploy 新增 MTP 的 Multi-layer功能         |        是        |
|  96   |  ⭐️⭐️⭐️   | 为FastDeploy新增MLA的FP8版本实现 |        是        |

## 编译机床任务

| 序号 |   难度    | 任务标题                                | 是否需要提交 RFC |
| :--: | :-------: | ---------------------------------------------------------------------------------------------- | :--------------: |
|  97   |  ⭐  | 适配 tvm 编译器          |        否        |
|  98   |  ⭐  | 适配 xla 编译器          |        否        |
|  99   |  ⭐  | 适配 TensorRT 编译器     |        否        |
|  100   |  ⭐  | 适配 BladeDISC 编译器   |        否        |
|  101   |  ⭐  | 多图抽取问题修复         |        否        |
|  102   |  ⭐  | vmap抽取问题修复        |        否        |
|  110   |  ⭐  | AI4C计算图分解验证器        |        否        |
| 111 | 0.25⭐ | （GraphNet样本修复）batch_norm算子添加weight_meta约束 | 否
| 112 | 0.5⭐ | （GraphNet样本修复）非法Torch样本修复 | 否
| 113 | 0.025⭐ | torch._C._fft.fft_irfft API转换 | 否
| 114 | 0.025⭐ | torch._C._fft.fft_rfft API转换 | 否
| 115 | 0.025⭐ | torch._C._fft.fft_fftn API转换 | 否
| 116 | 0.025⭐ | torch._C._linalg.linalg_vector_norm API转换 | 否
| 117 | 0.025⭐ | torch._C._linalg.linalg_norm API转换 | 否
| 118 | 0.025⭐ | torch._C._nn.softplus API转换 | 否
| 119 | 0.025⭐ | torch._C._nn.one_hot API转换 | 否
| 120 | 0.025⭐ | torch._C._special.special_logit API转换 | 否
| 121 | 0.05⭐ | torch._C._set_grad_enabled API转换 | 否
| 122 | 0.075⭐ | torch._C._log_api_usage_once API转换 | 否
| 123 | 0.1⭐ | torch._C._nn.pad API转换 | 否
| 124 | 0.1⭐ | torch._C._nn.avg_pool2d API转换 | 否
| 125 | 0.15⭐ | torch._C._nn.gelu API转换 | 否
| 126 | 0.2⭐ | torch._C._nn.scaled_dot_product_attention API转换 | 否
| 127 | 0.25⭐ | torch._C._nn.linear API转换 | 否
| 128 | ⭐ | PyTorch to Paddle 计算图转换 | 否
| 129 | ⭐ | ai4c计算图粗分解器设计与实现 | 否
| 130 | ⭐ | GraphNet Analysis功能及ESt绘图优化 | 否
| 131 | ⭐ | GraphNet自动样本抽取Agent | 否

## 科学计算任务

| 序号 |   难度    | 任务标题                                | 是否需要提交 RFC |
| :--: | :-------: | ---------------------------------------------------------------------------------------------- | :--------------: |
|  103   | ⭐️⭐️ | 基于Paddle实现第三方库e3nn               |        是        |
|  104   | ⭐️⭐️⭐️ | 基于Paddle实现第三方库torchmetrics               |        是        |
|  105   | ⭐️⭐️ | 基于Paddle实现CoNFiLD流场生成模型.       |        是        |
|  106   | ⭐️⭐️ | 基于Paddle实现符号深度学习模型，用于流体力学方程发现   |        是        |
|  107   | ⭐️⭐️ | 基于PaddleScience复现Aurora模型推理，使用小样本数据能够实现微调及训练   |    是    |
|  108   | ⭐️⭐️ | 基于PaddleScience复现neuralgcm模型推理，使用小样本数据能够实现训练 |   是    |

