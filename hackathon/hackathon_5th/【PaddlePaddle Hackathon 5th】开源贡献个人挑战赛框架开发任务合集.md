此文档展示 **PaddlePaddle Hackathon 第五期活动——开源贡献个人挑战赛框架开发任务** 详细介绍，更多详见  [PaddlePaddle Hackathon 说明](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_cn)。

## 【开源贡献个人挑战赛-框架开发任务（除API开发）】任务详情

### No.42：为Paddle代码转换工具新增API转换规则

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **PaddlePaddle Code Convert** Toolkits。此次需要你完成 [**API转换名单**](https://shimo.im/sheets/RKAWVnVNopC1NKk8/LmUrM) **中第1组（编号为1 ~ 20）** 的API转换规则开发。

**提交内容：**

- **API映射关系文档**：具体文档模板及要求请参考 [《API映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 。PR提交到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 下，需要为每个 API 新增对应的 md 文件并放入`docs/guides/model_convert/convert_from_pytorch/api_difference` 对应目录下，文件名为PyTorch API名。如果已存在该API的映射关系，则无需新增 md 文件，只需要**检查并校正之前的文档正确性**。
- **API转换规则**：请Fork代码仓库 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 来提交代码，具体开发步骤，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)步骤
- **单测代码**：提交到  [PaConvert](https://github.com/PaddlePaddle/PaConvert) 中，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤5，注意单测规范与要求。
- **注**：如果发现由于对应Paddle API的 **功能缺失、功能Bug、功能diff** 等问题，导致无法实现转换，请在API映射关系文档中说明，同时需开发屏蔽版本的测试case（参考已有代码）。

**技术要求：**

- 熟练掌握Python
- 熟悉Pytorch、Paddle两者API的使用，善于捕捉并分析细节差异

### No.43：为Paddle代码转换工具新增API转换规则

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **PaddlePaddle Code Convert** Toolkits。此次需要你完成 [**API转换名单**](https://shimo.im/sheets/RKAWVnVNopC1NKk8/LmUrM) **中第2组（编号为21 ~ 41）** 的API转换规则开发。

**提交内容：**

- **API映射关系文档**：具体文档模板及要求请参考 [《API映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 。PR提交到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 下，需要为每个 API 新增对应的 md 文件并放入`docs/guides/model_convert/convert_from_pytorch/api_difference` 对应目录下，文件名为PyTorch API名。如果已存在该API的映射关系，则无需新增 md 文件，只需要**检查并校正之前的文档正确性**。
- **API转换规则**：请Fork代码仓库 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 来提交代码，具体开发步骤，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)步骤
- **单测代码**：提交到  [PaConvert](https://github.com/PaddlePaddle/PaConvert) 中，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤5，注意单测规范与要求。
- **注**：如果发现由于对应Paddle API的 **功能缺失、功能Bug、功能diff** 等问题，导致无法实现转换，请在API映射关系文档中说明，同时需开发屏蔽版本的测试case（参考已有代码）。

**技术要求：**

- 熟练掌握Python
- 熟悉Pytorch、Paddle两者API的使用，善于捕捉并分析细节差异

### No.44：为Paddle代码转换工具新增API转换规则

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **PaddlePaddle Code Convert** Toolkits。此次需要你完成 [**API转换名单**](https://shimo.im/sheets/RKAWVnVNopC1NKk8/LmUrM) **中第3组（编号为42 ~ 61）** 的API转换规则开发。

**提交内容：**

- **API映射关系文档**：具体文档模板及要求请参考 [《API映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 。PR提交到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 下，需要为每个 API 新增对应的 md 文件并放入`docs/guides/model_convert/convert_from_pytorch/api_difference` 对应目录下，文件名为PyTorch API名。如果已存在该API的映射关系，则无需新增 md 文件，只需要**检查并校正之前的文档正确性**。
- **API转换规则**：请Fork代码仓库 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 来提交代码，具体开发步骤，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)步骤
- **单测代码**：提交到  [PaConvert](https://github.com/PaddlePaddle/PaConvert) 中，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤5，注意单测规范与要求。
- **注**：如果发现由于对应Paddle API的 **功能缺失、功能Bug、功能diff** 等问题，导致无法实现转换，请在API映射关系文档中说明，同时需开发屏蔽版本的测试case（参考已有代码）。

**技术要求：**

- 熟练掌握Python
- 熟悉Pytorch、Paddle两者API的使用，善于捕捉并分析细节差异

### No.45：为Paddle代码转换工具新增API转换规则

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **PaddlePaddle Code Convert** Toolkits。此次需要你完成 [**API转换名单**](https://shimo.im/sheets/RKAWVnVNopC1NKk8/LmUrM) **中第4组（编号为62 ~ 83）** 的API转换规则开发。

**提交内容：**

- **API映射关系文档**：具体文档模板及要求请参考 [《API映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 。PR提交到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 下，需要为每个 API 新增对应的 md 文件并放入`docs/guides/model_convert/convert_from_pytorch/api_difference` 对应目录下，文件名为PyTorch API名。如果已存在该API的映射关系，则无需新增 md 文件，只需要**检查并校正之前的文档正确性**。
- **API转换规则**：请Fork代码仓库 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 来提交代码，具体开发步骤，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)步骤
- **单测代码**：提交到  [PaConvert](https://github.com/PaddlePaddle/PaConvert) 中，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤5，注意单测规范与要求。
- **注**：如果发现由于对应Paddle API的 **功能缺失、功能Bug、功能diff** 等问题，导致无法实现转换，请在API映射关系文档中说明，同时需开发屏蔽版本的测试case（参考已有代码）。

**技术要求：**

- 熟练掌握Python
- 熟悉Pytorch、Paddle两者API的使用，善于捕捉并分析细节差异

### No.46：为Paddle代码转换工具新增API转换规则

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **PaddlePaddle Code Convert** Toolkits。此次需要你完成 [**API转换名单**](https://shimo.im/sheets/RKAWVnVNopC1NKk8/LmUrM) **中第5组（编号为84 ~ 102）** 的API转换规则开发。

**提交内容：**

- **API映射关系文档**：具体文档模板及要求请参考 [《API映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 。PR提交到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 下，需要为每个 API 新增对应的 md 文件并放入`docs/guides/model_convert/convert_from_pytorch/api_difference` 对应目录下，文件名为PyTorch API名。如果已存在该API的映射关系，则无需新增 md 文件，只需要**检查并校正之前的文档正确性**。
- **API转换规则**：请Fork代码仓库 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 来提交代码，具体开发步骤，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)步骤
- **单测代码**：提交到  [PaConvert](https://github.com/PaddlePaddle/PaConvert) 中，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤5，注意单测规范与要求。
- **注**：如果发现由于对应Paddle API的 **功能缺失、功能Bug、功能diff** 等问题，导致无法实现转换，请在API映射关系文档中说明，同时需开发屏蔽版本的测试case（参考已有代码）。

**技术要求：**

- 熟练掌握Python
- 熟悉Pytorch、Paddle两者API的使用，善于捕捉并分析细节差异

### No.47：为Paddle代码转换工具新增API转换规则

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **PaddlePaddle Code Convert** Toolkits。此次需要你完成 [**API转换名单**](https://shimo.im/sheets/RKAWVnVNopC1NKk8/LmUrM) **中第6组（编号为103 ~ 124）** 的API转换规则开发。

**提交内容：**

- **API映射关系文档**：具体文档模板及要求请参考 [《API映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 。PR提交到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 下，需要为每个 API 新增对应的 md 文件并放入`docs/guides/model_convert/convert_from_pytorch/api_difference` 对应目录下，文件名为PyTorch API名。如果已存在该API的映射关系，则无需新增 md 文件，只需要**检查并校正之前的文档正确性**。
- **API转换规则**：请Fork代码仓库 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 来提交代码，具体开发步骤，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)步骤
- **单测代码**：提交到  [PaConvert](https://github.com/PaddlePaddle/PaConvert) 中，参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤5，注意单测规范与要求。
- **注**：如果发现由于对应Paddle API的 **功能缺失、功能Bug、功能diff** 等问题，导致无法实现转换，请在API映射关系文档中说明，同时需开发屏蔽版本的测试case（参考已有代码）。

**技术要求：**

- 熟练掌握Python
- 熟悉Pytorch、Paddle两者API的使用，善于捕捉并分析细节差异

### No.48：ContiguousKernel、StridedCopyKernel算子CPU、GPU性能优化

**详细描述：**

- 在Paddle支持了 stride 后，大多模型会大量执行ContiguousKernel、StridedCopyKernel，因此两个 kernel 的性能尤为关键。
- 目前两个 kernel 都是通过 numel index 计算数据偏移地址，需要一个 for 循环做计算，计算偏移地址效率低，导致 kernel 性能差。如果改成跟进 stride 计算下一个 element 的偏移地址，则性能会大幅提升。[PR56866](https://github.com/PaddlePaddle/Paddle/pull/56866) 提供了初步示例，仍需调试，可以参考。更好的办法是先优化CPU kernel，再优化GPU，因为CPU易调试。设计文档示例：[op_optimization_example.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/OPs-Perf/op_optimization_example.md)。优化方法参考：[算子性能优化方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/op_optimization/op_optimization_method_introduction_cn.html)。

**提交内容：**

- 代码提交至 [contiguous_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/cpu/contiguous_kernel.cc)，[contiguous_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/contiguous_kernel.cu)，[strided_copy_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/cpu/strided_copy_kernel.cc)，[strided_copy_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/strided_copy_kernel.cu)
- 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见[示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。
- 测试case需要覆盖全部的计算分支，同时至少覆盖fp32，fp16两种数据类型。

**技术要求：**

- 了解stride原理
- 了解CUDA编程、kernel性能优化

### No.49：python端补齐OpResult的patch方法

**详细描述：**

- 在新IR升级项目中，python端的Variable会被替换为OpResult。原来通过Variable.xx()方式调用的方法在新IR下也需要通过OpResult.xx()的方式进行调用，所以需要补齐这些方法
- 详细描述：
  - 补齐类似__add__等加、减、乘、除等操作符方法
  - 补齐类似OpResult.reshape()这这类调用paddle api的方法
  - 具体的实现方法可以参考: python/paddle/base/layers/math_op_patch.py

**提交内容：**

- 参考python/paddle/base/layers/math_op_patch.py实现OpResult的patch逻辑
- 参考test/legacy_test/test_math_op_patch.py添加单测进行验证

**技术要求：**

- 熟练掌握Python
- 对新老IR体系有一定了解

### No.50：为 Paddle 新增 slice 的 spmd 切分推导规则

**详细描述：**

实现 slice 的切分推导规则，包括正向推导和逆向推导：

- 正向推导，根据输入的切分状态推导输出的切分状态
- 逆向推导，根据输出的切分状态推导输入的切分状态
- 可参考 [**切分推导规则参考文档**](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/auto_parallel/spmd_rules.md)，paddle 中现有的切分推导规则 (Paddle/paddle/phi/infermeta/spmd_rules，可参考 split)，tensorflow (tensorflow/tensorflow/dtensor/mlir/expansions/slice_spmd_expander.cc) 和 pytorch 中的实现逻辑 (pytorch/torch/distributed/_tensor/ops/tensor_ops.py)

**提交内容：**

- 推导规则实现代码（C++），放在Paddle/paddle/phi/infermeta/spmd_rules 目录
- 单测代码（Python），放在 Paddle/test/auto_parallel/spmd_rules 目录

**技术要求：**

- 熟练掌握 C++，Python
- 了解分布式训练

### No.51：为 Paddle 新增 flatten 的 spmd 切分推导规则

**详细描述：**

实现 flatten 的切分推导规则，包括正向推导和逆向推导：

- 正向推导，根据输入的切分状态推导输出的切分状态
- 逆向推导，根据输出的切分状态推导输入的切分状态
- 可参考 [**切分推导规则参考文档**](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/auto_parallel/spmd_rules.md)，paddle 中现有的切分推导规则 (Paddle/paddle/phi/infermeta/spmd_rules)，tensorflow (tensorflow/tensorflow/dtensor/mlir/expansions) 和 pytorch 中相关的实现逻辑 (pytorch/torch/distributed/_tensor/ops)

**提交内容：**

- 推导规则实现代码（C++），放在Paddle/paddle/phi/infermeta/spmd_rules 目录
- 单测代码（Python），放在 Paddle/test/auto_parallel/spmd_rules 目录

**技术要求：**

- 熟练掌握 C++，Python
- 了解分布式训练

### No.52：为 Paddle 新增 squeeze 和 unsqueeze 的 spmd 切分推导规则

**详细描述：**

实现  squeeze 和 unsqueeze 的切分推导规则，包括正向推导和逆向推导：

- 正向推导，根据输入的切分状态推导输出的切分状态
- 逆向推导，根据输出的切分状态推导输入的切分状态
- 可参考 [**切分推导规则参考文档**](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/auto_parallel/spmd_rules.md)，paddle 中现有的切分推导规则 (Paddle/paddle/phi/infermeta/spmd_rules)，tensorflow (tensorflow/tensorflow/dtensor/mlir/expansions/squeeze_spmd_expander.cc) 和 pytorch 中的实现逻辑 (pytorch/torch/distributed/_tensor/ops/tensor_ops.py)

**提交内容：**

- 推导规则实现代码（C++），放在Paddle/paddle/phi/infermeta/spmd_rules 目录
- 单测代码（Python），放在 Paddle/test/auto_parallel/spmd_rules 目录

**技术要求：**

- 熟练掌握 C++，Python
- 了解分布式训练

### No.101：将paddle内部的fused_multi_transformer/fused_multi_transformer_int8算子及其kernel实现从fluid下迁移到phi下

**详细描述：**

将paddle内部的fused_multi_transformer/fused_multi_transformer_int8算子及其kernel从fluid下迁移到phi下，包括如下工作：

- 将fluid下的手写op定义删除，配置fused_ops.yaml自定生成op定义
- 将对应的kernel迁移到phi下
- 迁移前后保证单测test_fused_multi_transformer_op.py/test_fused_multi_transformer_int8_op.py运行成功
- 开启FLAGS_enable_new_ir_in_executor=1，单测也可以运行成功

**提交内容：**

- kernel迁移到 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/pull/56846/files#diff-2b738e4c56d8686ac926760061fbf0e7d2131dd26ed086b60dcb4821fa332203)/fusion目录下
- 算子定义在fused_ops.yaml下配置

**技术要求：**

- 熟练掌握 C++，Python

**参考PR：**

https://github.com/PaddlePaddle/Paddle/pull/56846

### No.102：将paddle内部的fused_embedding_eltwise_layernorm、fusion_transpose_flatten_concat和fused_fc_elementwise_layernorm算子及其kernel实现从fluid下迁移到phi下

**详细描述：**

将paddle内部的fused_embedding_eltwise_layernorm/fusion_transpose_flatten_concat/fused_fc_elementwise_layernorm算子及其kernel从fluid下迁移到phi下，包括如下工作：

- 将fluid下的手写op定义删除，配置fused_ops.yaml自定生成op定义
- 将对应的kernel迁移到phi下
- 迁移前后保证单测test_emb_eltwise_layernorm_fuse_pass.py/test_ir_embedding_eltwise_layernorm_fuse_pass.py/test_transpose_flatten_concat_fuse_pass.py/test_fused_fc_elementwise_layernorm_op.py运行成功
- 开启FLAGS_enable_new_ir_in_executor=1，单测也可以运行成功

**提交内容：**

- kernel迁移到 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/pull/56846/files#diff-2b738e4c56d8686ac926760061fbf0e7d2131dd26ed086b60dcb4821fa332203)/fusion目录下
- 算子定义在fused_ops.yaml下配置

**技术要求：**

- 熟练掌握 C++，Python

**参考PR：**

https://github.com/PaddlePaddle/Paddle/pull/56846

### No.103：将paddle内部的skip_layernorm、fc和fused_bias_dropout_residual_layer_norm算子及其kernel实现从fluid下迁移到phi下

**详细描述：**

将paddle内部的skip_layernorm/fc/fused_bias_dropout_residual_layer_norm算子及其kernel从fluid下迁移到phi下，包括如下工作：

- 将fluid下的手写op定义删除，配置fused_ops.yaml自定生成op定义
- 将对应的kernel迁移到phi下
- 迁移前后保证单测test_trt_skip_layernorm_fuse_pass.py/test_fc_op.py/test_fused_bias_dropout_residual_layer_norm_op.py运行成功
- 开启FLAGS_enable_new_ir_in_executor=1，单测也可以运行成功

**提交内容：**

- kernel迁移到 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/pull/56846/files#diff-2b738e4c56d8686ac926760061fbf0e7d2131dd26ed086b60dcb4821fa332203)/fusion目录下
- 算子定义在fused_ops.yaml下配置

**技术要求：**

- 熟练掌握 C++，Python

**参考PR：**

https://github.com/PaddlePaddle/Paddle/pull/56846

### No.104：将paddle内部的self_dp_attention和fusion_repeated_fc_relu/fusion_squared_mat_sub算子及其kernel实现从fluid下迁移到phi下

**详细描述：**

将paddle内部的self_dp_attention/fusion_repeated_fc_relu/fusion_squared_mat_sub算子及其kernel从fluid下迁移到phi下，包括如下工作：

- 将fluid下的手写op定义删除，配置fused_ops.yaml自定生成op定义
- 将对应的kernel迁移到phi下
- 迁移前后保证单测test_fused_vit_attention.py/test_fusion_repeated_fc_relu_op.py/test_fusion_squared_mat_sub_op.py运行成功
- 开启FLAGS_enable_new_ir_in_executor=1，单测也可以运行成功

**提交内容：**

- kernel迁移到 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/pull/56846/files#diff-2b738e4c56d8686ac926760061fbf0e7d2131dd26ed086b60dcb4821fa332203)/fusion目录下
- 算子定义在fused_ops.yaml下配置

**技术要求：**

- 熟练掌握 C++，Python

**参考PR：**

https://github.com/PaddlePaddle/Paddle/pull/56846

### No.105：将paddle内部的fusion_gru、fusion_seqconv_eltadd_relu和fusion_seqexpand_concat_fc算子及其kernel实现从fluid下迁移到phi下

**详细描述：**

将paddle内部的fusion_gru/fusion_seqconv_eltadd_relu/fusion_seqexpand_concat_fc算子及其kernel从fluid下迁移到phi下，包括如下工作：

- 将fluid下的手写op定义删除，配置fused_ops.yaml自定生成op定义
- 将对应的kernel迁移到phi下
- 迁移前后保证单测test_fusion_gru_op.py/test_fusion_seqconv_eltadd_relu_op.py/test_fusion_seqexpand_concat_fc_op.py运行成功
- 开启FLAGS_enable_new_ir_in_executor=1，单测也可以运行成功

**提交内容：**

- kernel迁移到 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/pull/56846/files#diff-2b738e4c56d8686ac926760061fbf0e7d2131dd26ed086b60dcb4821fa332203)/fusion目录下
- 算子定义在fused_ops.yaml下配置

**技术要求：**

- 熟练掌握 C++，Python

**参考PR：**

https://github.com/PaddlePaddle/Paddle/pull/56846


### No.112：将paddle内部的read_file、fused_gemm_epilogue算子及其kernel实现从fluid下迁移到phi下；添加identity_loss的yaml配置

**详细描述：**

将paddle内部的read_file、fused_gemm_epilogue算子及其kernel实现从fluid下迁移到phi下，包括如下工作：

- 将fluid下的手写op定义删除，配置yaml文件生成op定义。read_file配置在ops.yaml文件内，fused_gemm_epilogue配置在fused_ops.yaml内。
- 将对应的kernel迁移到phi下
- 迁移前后保证单测 test_read_file.py 和 test_fused_gemm_epilogue_op.py 运行成功
- 开启FLAGS_enable_new_ir_in_executor=1，单测也可以运行成功

添加identity_loss的yaml配置，包括如下工作：

- 将fluid下的手写op定义删除，配置ops.yaml自定生成op定义

**提交内容：**

- read_file迁移到paddle/phi/kernels目录下，fused_gemm_epilogue迁移到paddle/phi/kernels/fusion目录下
- 算子定义在ops.yaml和fused_ops.yaml下配置

**技术要求：**

- 熟练掌握 C++，Python

**参考PR：**

https://github.com/PaddlePaddle/Paddle/pull/56846


### No.113：为paddle.nn.functional.embedding增加参数max_norm/norm_type/scale_grad_by_freq

**详细描述：**

torch.nn.functional.embedding支持max_norm/norm_type/scale_grad_by_freq参数，而paddle不支持，需要调研这三个参数的功能，并且为paddle.nn.functional.embedding增加这三个参数。
需要注意同时修改paddle.nn.Embedding。


**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch对应API参考：torch.nn.functional.embedding/torch.nn.Embedding


### No.114：为paddle.linalg.norm进行功能对齐与功能增强

**详细描述：**

与torch.norm/torch.linalg.norm相比，paddle.linalg.norm需要进行以下方面的修改：

1）【功能对齐】求解p范数时，torch是对矩阵求p范数，而paddle是将矩阵直接展平为向量求p范数，两者计算结果不一致，需要对paddle的实现逻辑进行调整
2）【功能增强】torch支持负实数的范数，paddle不支持
3）【功能增强】torch支持p='nuc'，paddle不支持


**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch对应API参考：torch.norm/torch.linalg.norm


### No.115：为paddle.nn.LSTM/RNNBase/paddle.quantile/nanquantile功能增强

**详细描述：**

为以下多个API进行功能增强：

1）【功能增强】torch.nn.LSTM支持参数proj_size，表示将隐藏状态h的维度映射到proj_size对应的大小，而paddle.nn.LSTM不支持
2）【功能增强】torch.nn.RNNBase的mode参数，可取值为 `'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'`，而paddle.nn.RNNBase只支持 `'LSTM', 'GRU'`，不支持其他两种
3）【功能增强】torch.quantile/torch.nanquantile的输入q支持1D Tensor表示1个list，而paddle.quantile/nanquantile不支持输入1D Tensor表示1个list
4）【功能增强】torch.quantile/torch.nanquantile支持interpolation参数，而paddle不支持


**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

与上述列出的4个问题点一一对应：

1）https://github.com/PaddlePaddle/Paddle/pull/56460
2）https://github.com/PaddlePaddle/Paddle/pull/56460
3）https://github.com/PaddlePaddle/Paddle/pull/56461
4）https://github.com/PaddlePaddle/Paddle/pull/56461

Pytorch对应API参考：torch.nn.LSTM/torch.nn.RNNBase/torch.quantile/torch.nanquantile


### No.116：为paddle.histogram/paddle.nn.functional.threshold进行功能对齐与功能增强

**详细描述：**

为以下多个API进行功能增强和功能对齐：

1）【功能增强】torch.histogram支持参数weight、density，而paddle不支持，需要调研这两个参数的功能，并且为paddle.histogram增加这两个参数
2）【功能对齐】torch.histogram返回两个Tensor：hist、bin，而paddle仅返回一个hist，需要增加一个histogram_bin_edges，支持返回bin
2）【功能增强】torch.nn.functional.threshold支持value参数，而paddle不支持，需要调研这个参数的功能，并且为paddle.nn.functional.threshold增加这个参数


**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

与上述列出的3个问题点一一对应：

1）https://github.com/PaddlePaddle/Paddle/pull/56771
2）https://github.com/PaddlePaddle/Paddle/pull/56771
3）https://github.com/PaddlePaddle/Paddle/pull/56853

Pytorch对应API参考： torch.histogram/torch.nn.functional.threshold


### No.117：为paddle.nn.functional.upsample/paddle.nn.initializer.XavierNormal/XavierUniform/KaimingNormal/KaimingUniform进行功能增强

**详细描述：**

为以下多个API进行功能增强：

1）【功能增强】paddle.nn.functional.upsample中目前data_format默认值始终固定为NCHW，但由于这个API支持3D/4D/5D，建议data_format默认值能根据输入维度自动切换NCW/NCHW/NCDHW
2）【功能增强】torch.nn.init.xavier_normal_/xavier_uniform_均支持参数gain，paddle.nn.initializer.XavierNormal/XavierUniform缺少参数gain，需增加该参数
3）【功能增强】torch.nn.init.kaiming_normal_/kaiming_uniform_缺少参数mode，当mode="fan_out"时，paddle.nn.initializer.KaimingNormal/KaimingUniform缺少对应可替代的功能，需增加mode参数或fan_out参数，从而补齐该功能


**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

https://github.com/PaddlePaddle/Paddle/pull/56471

Pytorch对应API参考： torch.nn.funciton.upsample/torch.nn.init.xavier_normal_/torch.nn.init.xavier_uniform_/torch.nn.init.kaiming_normal_/torch.nn.init.kaiming_uniform_


### No.118：为paddle.io.RandomSampler/random_split/Layer.clear_gradients进行功能增强

**详细描述：**

为以下多个API进行功能增强：

1）【功能增强】paddle.io.RandomSampler当参数 replacement = False时，不允许指定 num_samples，而torch.utils.data.RandomSampler则无此限制，需要增强该功能
2）【功能增强】torch.utils.data.random_split的lengths参数支持比例方式划分，而paddle.io.random_split不支持，需要增强该功能
3）【功能增强】paddle.nn.Layer.clear_gradients需要暴露底层的set_to_zero参数，从而和torch.nn.Module.zero_grad的set_to_none参数功能对应

**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

Pytorch对应API参考： torch.utils.data.RandomSampler/torch.utils.data.random_split/torch.nn.Module.zero_grad


### No.119：为paddle.round/paddle.nn.functional.max_pool1d/max_pool2d/max_pool3d进行功能增强

**详细描述：**

为以下多个API进行功能增强：

1）【功能增强】torch.round支持decimals 参数，表示舍入的小数点位数，paddle不支持，需要增加该参数
2）【功能增强】torch.nn.functional.max_pool1d/max_pool2d/max_pool3d支持dilation参数空洞池化，paddle不支持，需要增加该参数


**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中


**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

Pytorch对应API参考： torch.round/torch.nn.functional.max_pool1d/max_pool2d/max_pool3d


### No.120：为paddle.nn.functional.max_unpool1d/max_unpool2d/max_unpool3d/paddle.nn.functional.kl_div进行功能增强或Bug修复

**详细描述：**

为以下多个API进行功能增强或Bug修复：

1）【功能增强】torch.nn.functional.max_unpool1d/max_unpool2d/max_unpool3d支持int64输入，而paddle不支持，需要增加该功能
2）【Bug修复】paddle.nn.functional.max_unpool1d/max_unpool2d/max_unpool3d的output_size参数的判断有bug，输入正确的output_size会报错
3）【功能增强】torch.nn.functional.kl_div支持参数log_target，而paddle不支持，需要增加该参数

**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

Pytorch对应API参考： torch.nn.functional.max_unpool1d/max_unpool2d/max_unpool3d/torch.nn.functional.kl_div


### No.121：为paddle.nn.functional.max_pool1d/max_pool2d/max_pool3d/paddle.signal.stft进行Bug修复

**详细描述：**

为以下多个API进行Bug修复：

1）【Bug修复】paddle.nn.functional.max_pool1d/max_pool2d/max_pool3d当return_mask=True时，ceil_mode不生效。[问题case链接](https://github.com/PaddlePaddle/PaConvert/blob/master/tests/test_nn_functional_max_pool1d.py#L93-L106)。

2）【Bug修复】paddle.signal.stft计算结果与torch.stft有较大差距，需要分析该问题并给出正确的解决方案

**提交内容：**

- 算子Kernel 和 API的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的API或OP测试代码，如未有现成代码，则需自行增加测试case，对改动功能点需要着重测试并对齐Pytorch计算结果
- API中文文档，如果有API参数的增删或功能的修改，需修改API文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python


**参考内容：**

Pytorch对应API参考： torch.nn.functional.max_pool1d/max_pool2d/max_pool3d/torch.stft
