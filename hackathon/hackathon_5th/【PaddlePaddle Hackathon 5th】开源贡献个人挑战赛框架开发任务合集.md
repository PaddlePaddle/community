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
- 可参考 paddle 中现有的切分推导规则 (Paddle/paddle/phi/infermeta/spmd_rules) 和 tensorflow (tensorflow/tensorflow/dtensor/mlir/expansions/slice_spmd_expander.cc) 和 pytorch 中的实现逻辑 (pytorch/torch/distributed/_tensor/ops/tensor_ops.py)

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
- 可参考 paddle 中现有的切分推导规则 (Paddle/paddle/phi/infermeta/spmd_rules) 和 tensorflow (tensorflow/tensorflow/dtensor/mlir/expansions) 和 pytorch 中x相关的实现逻辑 (pytorch/torch/distributed/_tensor/ops)

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
- 可参考 paddle 中现有的切分推导规则 (Paddle/paddle/phi/infermeta/spmd_rules) 和 tensorflow (tensorflow/tensorflow/dtensor/mlir/expansions/squeeze_spmd_expander.cc) 和 pytorch 中的实现逻辑 (pytorch/torch/distributed/_tensor/ops/tensor_ops.py)

**提交内容：**

- 推导规则实现代码（C++），放在Paddle/paddle/phi/infermeta/spmd_rules 目录
- 单测代码（Python），放在 Paddle/test/auto_parallel/spmd_rules 目录

**技术要求：**

- 熟练掌握 C++，Python
- 了解分布式训练
