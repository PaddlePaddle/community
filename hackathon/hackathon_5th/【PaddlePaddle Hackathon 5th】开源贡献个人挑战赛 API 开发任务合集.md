此文档展示 **PaddlePaddle Hackathon 第五期活动——开源贡献个人挑战赛 API 开发任务** 详细介绍，更多详见  [PaddlePaddle Hackathon 说明](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_cn)。

## 【开源贡献个人挑战赛-热身赛】任务详情：

为飞桨框架新增一系列 API，提交流程请参考 [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项：

- 合入标准
  - 按 [API 设计规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) 完成 API设计文档；需要在设计文档中说明支持哪些数据类型（默认都要支持fp16/bf16/complex64/complex128），对不支持的要给出理由
  - 按 [API 验收标准](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) 完成 API功能实现、单测、API文档；
- 参考内容
  - [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)
  - [新增 API 设计模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)
  - [飞桨API Python 端开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html)
  - [C++ 算子开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
  - [飞桨API文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
  - [API单测开发及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)
  - 复数数据类型相关资料：
    - [On the Computation of Complex-valued Gradients with Application to Statistically Optimum Beamforming](https://arxiv.org/abs/1701.00392)
    - [复数梯度推导计算](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/complex_autograd) 
    - [paddlepaddle支持复数任务](https://github.com/PaddlePaddle/Paddle/issues/56145)

### No.1：为 Paddle 新增 copysign API

**详细描述：**

根据两个输入逐元素地计算结果张量，其结果由第一个输入的绝对值大小及第二个输入的符号组成。此任务的目标是在 Paddle 框架中，新增 copysign API ，调用路径为：

- paddle.copysign 作为独立的函数调用，非 inplace
- paddle.copysign_，作为独立的函数，inplace 地修改输入；
- Tensor.copysign做为 Tensor 的方法使用，非 inplace;
- Tensor.copysign_做为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/tensor/math.py)文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 copysign & copysign_ API，以支持 paddle.Tensor.copysign & paddle.Tensor.copysign_ 的调用方式；
- C ++/CUDA 实现代码，头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inplace.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace api 单测
- 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。

**技术要求：**

- 熟悉 copysign 函数的计算原理和适用场景
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.2：为 Paddle 新增 index_fill API

**详细描述：**

对于 nd tensor, 沿着某个轴 axis 取 (n-1)d 的切片，索引位置是 index, 并且将 value 中值填充到这些切片上。其中 value 是一个 scalar 或者 0d tensor, 该运算需要支持微分。调用路径：

- paddle.index_fill 作为独立的函数调用，非 inplace
- paddle.index_fill_，作为独立的函数，inplace 地修改输入；
- Tensor.index_fill， 作为 Tensor 的方法使用，非 inplace;
- Tensor.index_fill_，作为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py) 文件。
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inplace.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace api 单测
- 中文API文档，在 docs repo 的  [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 了解 index_fill的计算逻辑和适用场景
- 熟练掌握 Python

### No.3：为 Paddle 新增 masked_fill API

**详细描述：**

对于一个Tensor，根据mask信息，将value 中的值填充到该Tensor中mask对应为True的位置。调用路径：

- paddle.masked_fill 作为独立的函数调用，非 inplace
- paddle.masked_fill_，作为独立的函数，inplace 地修改输入；
- Tensor.masked_fill，作为 Tensor 的方法使用，非 inplace;
- Tensor.masked_fill_，作为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py) 文件。
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inplace.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace api 单测
- 中文API文档，在 docs repo 的  [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 了解 masked_fill的计算逻辑和适用场景
- 熟练掌握 Python

### No.4：为 Paddle 新增 masked_scatter API

**详细描述：**

根据mask信息，将value中的值逐个拷贝到原Tensor的对应位置上。此任务的目标是在 Paddle 框架中，新增 masked_scatter API ，调用路径为：

- paddle.masked_scatter 作为独立的函数调用，非 inplace
- paddle.masked_scatter_，作为独立的函数，inplace 地修改输入；
- Tensor.masked_scatter作为 Tensor 的方法使用，非 inplace;
- Tensor.masked_scatter_作为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 masked_scatter & masked_scatter_ API，以支持 paddle.Tensor.masked_scatter & paddle.Tensor.masked_scatter_ 的调用方式；
- C ++/CUDA 实现代码，头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inplace.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace api 单测
- 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。

**技术要求：**

- 熟悉 masked_scatter 函数的计算原理和适用场景；
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.5：为 Paddle 增强 scatter API

**详细描述：**

当前paddle.scatter API提供了根据index信息更新原Tensor的功能，但缺少指定轴和归约方式等功能。本任务希望在此基础上，进一步增强该API的功能，实现可以根据给定的归约方式，将update中的值按顺序根据index信息累计到原Tensor的对应位置上，即对应index_reduce操作。注意索引规则与scatter_reduce / put_along_axis的区别。

**提交内容：**

- API 的增强设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py) 文件；
- C ++/CUDA 实现代码，头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录
- 补充新增功能单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inplace.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中，在原单测的基础上补充新增功能的测试
- 更新中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件。

**技术要求：**

- 理解不同的索引方式的规则和差异，包括scatter、scatter_nd 、put_along_axis等；
- 熟悉 index_reduce 函数的计算原理和适用场景；
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.6：为 Paddle 增强put_along_axis API

**详细描述：**

当前paddle.put_along_axis API提供了根据index信息和归约方式，沿指定轴将value中的值按顺序根据index信息累计到原Tensor的对应位置上。本任务希望在此基础上，进一步增强该API的功能，覆盖更全面的归约方式与功能。即对应scatter_reduce操作。注意索引规则与index_reduce / scatter的区别。

**提交内容：**

- API 的增强设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py) 文件；
- C ++/CUDA 实现代码，头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录
- 补充新增功能单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inplace.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中，在原单测的基础上补充新增功能的测试
- 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件

**技术要求：**

- 理解不同的索引方式的规则和差异，包括scatter、scatter_nd 、put_along_axis等；
- 熟悉 scatter_reduce 函数的计算原理和适用场景；
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.7：为 Paddle 新增 apply API

**详细描述：**

对输入Tensor中的每个元素，应用输入的Python函数（`callable`）得到结果：

- Tensor.apply(`callable`) 做为 Tensor 的方法使用，返回新的Tensor，存放计算结果
- Tensor.apply_(`callable`) 做为 Tensor 的方法使用， inplace 修改输入Tensor

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- 使用C++ 在 pybind中实现apply/apply_的逻辑代码 并 bind相关方法到Tensor module上，以实现通过Tensor.apply/apply_的调用
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录, 同时在[paddle/test/legacy_test/test_inpalce.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace api 单测inplace api 单测
- 中文API文档，在 docs repo 的  [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.8：为 Paddle 新增 hypot API

**详细描述：**

实现直角三角形斜边长度求解函数的计算：

- paddle.hypot 作为独立的函数调用，非 inplace
- paddle.hypot_ 作为独立的函数，inplace 地修改输入；
- Tensor.hypot(input1, input2) 做为 Tensor 的方法使用，非 inplace;
- Tensor.hypot_(input1, input2) 做为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 ﻿PaddlePaddle/community﻿ 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 ﻿python/paddle/tensor/math.py﻿文件；并在 ﻿python/paddle/tensor/__init__.py﻿ 中，添加 hypot & hypot_  API，以支持 paddle.Tensor.hypot & paddle.Tensor.hypot_ 的调用方式；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录增加非inplace的hypot单测, 同时在[paddle/test/legacy_test/test_inpalce.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace的hypot_单测
- 中文API文档，在 docs repo 的  ﻿docs/api/paddle/﻿ 目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.9：为 Paddle 新增 multigammaln API

**详细描述：**

实现多元对数伽马函数的计算：

- paddle.multigammaln 作为独立的函数调用，非 inplace
- paddle.multigammaln_ 作为独立的函数，inplace 地修改输入；
- Tensor.multigammaln(input, other) 做为 Tensor 的方法使用，非 inplace;
- Tensor.multigammaln_(input, other) 做为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/tensor/math.py)文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 mvlgamma & mvlgamma_  API，以支持 paddle.Tensor.mvlgamma & paddle.Tensor.mvlgamma_ 的调用方式；
- 单测代码，在 Paddle repo 的[test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录增，同时在[paddle/test/legacy_test/test_inpalce.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应的inplace api 单测。
- 中文API文档，在 docs repo 的  [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验

### No.10：为 Paddle 新增 bernoulli_ / log_normal_ / log_normal API

**详细描述：**

内容一：指定概率p，实现inplace的伯努利分布，新增`paddle.bernoulli_`；

内容二：指定均值和方差，实现对数正态分布，新增 `paddle.lognormal/lognormal_` API；其中`lognormal`可通过paddle.gaussian和paddle.exp组合实现，`lognormal_`可通过paddle.normal_和paddle.exp_组合实现，`bernoulli_`可通过paddle.uniform_等来组合实现。调用路径为：

1. paddle.bernoulli_(x, p=0.5) 可以inplace的修改输入x，填充伯努利分布的值
2. paddle.Tensor.bernoulli_(p=0.5) 作为paddle.bernoulli_的Tensor类方法使用
3. paddle.log_normal_(x, mean=1.0, std=2.0) 可以inplace的修改输入x，填充对数正态分布的值
4. paddle.Tensor.log_normal_(mean=1.0, std=2.0) 作为paddle.log_normal_的Tensor类方法使用
5. paddle.log_normal(mean=1.0, std=2.0, shape=None, dtype=None) 作为非 inplace的API，可以创建一个对数正态分布的Tensor

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/random.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/tensor/random.py)文件增加 bernoulli_ / lognormal_ / log_normal，以支持`paddle.bernoulli_/lognormal_/lognormal`的调用；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加  bernoulli_/ log_normal_，以支持`paddle.Tensor.bernoulli_``/log_normal_`的调用；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录增加非inplace的单测, 同时在[paddle/test/legacy_test/test_inpalce.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应inplace的单测
- 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。

**技术要求：**

- 熟悉 bernoulli / log_normal 函数的计算原理和适用场景；
- 熟练掌握 C++，Python

### No.11：为 Paddle 新增 igamma 和 igammac API

**详细描述：**

实现不完整伽马函数的计算：

- paddle.igamma 和 paddle.igammac 作为独立的函数调用，非 inplace
- paddle.igamma_ 和 paddle.igammac_，作为独立的函数，inplace 地修改输入；
- Tensor.igamma_(input, other) 和 Tensor.igammac_(input, other) 做为 Tensor 的方法使用，非 inplace;
- Tensor.igamma_(input, other) 和 Tensor.igammac_(input, other) 做为 Tensor 的方法使用， inplace 修改输入；

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py) 文件。
- C ++/CUDA 实现代码，头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录
- 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录
- 中文API文档，在 docs repo 的  [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 C++，Python
- 有一定的 CUDA 开发经验
