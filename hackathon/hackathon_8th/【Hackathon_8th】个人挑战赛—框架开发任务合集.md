此文档展示 **PaddlePaddle Hackathon 第八期活动——开源贡献个人挑战赛框架开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-API 开发】任务详情

注：为飞桨框架新增一系列 API，提交流程请参考 [新增 API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项：

- 合入标准
  - 按 [API 设计规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) 完成 API 设计文档；需要在设计文档中说明支持哪些数据类型（默认都要支持 fp16/bf16/complex64/complex128），对不支持的要给出理由
  - 按 [API 验收标准](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) 完成 API 功能实现、单测、API 文档；
  - 按 [API 映射关系-格式规范](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 完成 **_API 映射关系文档_** 的编写，文件统一提交到 [convert_from_pytorch/api_difference](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/) 中相应目录，需详细描述与竞品（Pytorch）的差异之处，对差异之处需要在 **_API 设计文档_** 中阐述理由；
- 参考内容
  - [新增 API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)
  - [新增 API 设计模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)
  - [飞桨 API Python 端开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html)
  - [C++ 算子开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
  - [飞桨 API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
  - [API 映射关系-格式规范](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md)
  - [API 单测开发及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)
  - 复数数据类型相关资料：
    - [On the Computation of Complex-valued Gradients with Application to Statistically Optimum Beamforming](https://arxiv.org/abs/1701.00392)
    - [复数梯度推导计算](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/complex_autograd)
    - [paddlepaddle 支持复数任务](https://github.com/PaddlePaddle/Paddle/issues/61975)

### NO.1 为 Paddle 新增 lu_solve API

**详细描述：**

使用 LU 分解 来求解线性方程组 AX=B，A 为 1 个或多个矩阵，A.shape=[m, n] or [batch, m, n]，B.shape=[m, k] or [batch, m, k]，A 和 B 已知，通过 LU 分解方阵 A 来加速求解 X。需要满足 LU, pivots, info =paddle.linalg.lu(A); X = paddle.linalg.lu_solve(B, LU, pivots) 与 使用 X=paddle.linalg.solve(A, B) 直接求解线性方程组的结果一样。此任务的目标是在 Paddle 框架中，新增 lu_solve API，调用路径为：paddle.linalg.lu_solve 和 Tensor.lu_solve

**提交内容：**

- API 的设计文档，并提 PR 至 ﻿PaddlePaddle/community﻿ 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)文件。并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 lu_solve API，以支持 Tensor.lu_solve 的调用方式；
- C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系 文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下

**技术要求：**

- 熟悉矩阵 LU 分解原理，了解当前 paddle.linalg.lu 的逻辑；
- 熟悉 lapack/cublas 库；
- 熟练掌握 Python。

### NO.2 为 Paddle 新增 baddbmm API

**详细描述：**

为 Paddle 新增 baddbmm API，以实现 β∗input+α(A@B)的功能。

此任务的目标是在 Paddle 框架中，新增 baddbmm API，调用路径为：paddle.baddbmm 和 Tensor.baddbmm

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 目录下新建文件
- 并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 API，以支持 Tensor.baddbmm 的调用方式
- C++ 实现代码，在 Paddle repo 放置。头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录
- 单测代码，在 Paddle repo 的[ python/paddle/test/legacy_test](https://github.com/PaddlePaddle/Paddle/tree/develop/test/legacy_test) 目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系 文件，请务必遵守[《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉 Paddle 的参数初始化方式
- 熟练掌握 C++, CUDA, Python

---

## 【开源贡献个人挑战赛-分布式方向】任务详情

### NO.3 分布式训练自动并行与通信相关的老 IR 逻辑清理

**详细描述：**

- 分布式训练在老通信算子退场时，还清理相关的单测、框架代码。
- 这些单测是带有 deprecated 后缀的，在老通信库算子退场时遇到的报错单测。

**验收说明：**

- Paddle 自动并行通信库老 IR 相关单测删除
- 与上述单测相关的框架代码逻辑清理

**技术要求**

- 熟悉 Paddle 分布式通信库代码逻辑
- 熟练掌握 Python C++ 语言

### NO.4 Paddle 旧通信库和 FleetExecutor 退场

**详细描述：**

- Paddle 框架中目前存在新旧通信库两套代码，当前已默认用新通信库，旧通信库相关的模块和代码逻辑需要清理掉。新通信库 FLAG: FLAGS_dynamic_static_unified_comm=1
  - 移除 FLAGS_dynamic_static_unified_comm != 1 的代码逻辑
  - 移除与上述相关联的模块
- Paddle 框架中目前存在新旧执行器两套代码，FleetExecutor 已废弃，需要清理掉相关代码。
  - 移除 FleetExecutor、以及派生出的相关执行器
  - 移除与上述执行器相关联的模块
  - 移除与 FleetExecutor 相关的 Python 端类、单测

**验收说明：**

- Paddle 框架无旧通信库相关类、函数和代码判断逻辑
- Paddle 框架无旧执行器关类、函数和单测
- 上下游关联模块同步删除，CMakeLists.txt 删除对应编译依赖

**技术要求**

- 熟悉 Paddle 分布式通信库代码逻辑
- 了解执行器模块代码
- 熟练掌握 Python C++ 语言
- 熟悉 Pybind 的使用
