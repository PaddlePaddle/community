此文档展示 **PaddlePaddle Hackathon 第六期活动——开源贡献个人挑战赛框架开发任务** 详细介绍，更多详见 [PaddlePaddle Hackathon 说明](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/10_contribution/hackathon_cn.md)。

## 【开源贡献个人挑战赛-API 开发】任务详情

为飞桨框架新增一系列 API，提交流程请参考 [新增 API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项：

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

### NO.1 为 Paddle 新增 AdaptiveLogSoftmaxWithLoss API

**详细描述：**

AdaptiveLogSoftmaxWithLoss 来源于 [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309)，其中的 Adaptive Softmax 方法是对一种高效实现 softmax 函数近似计算的方法。

此任务的目标是在 Paddle 框架中，新增 AdaptiveLogSoftmaxWithLoss API，调用路径为：paddle.nn.AdaptiveLogSoftmaxWithLoss 和 paddle.nn.functional.adaptive_log_softmax_with_loss。可以在之前开发者 [未开发完的 PR](https://github.com/PaddlePaddle/Paddle/pull/59623) 基础上进行开发。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/nn/layer/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/loss.py) 文件 和 [python/paddle/nn/functional/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/loss.py) 文件
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的 docs/api/paddle/ 目录

**技术要求：**

- 熟悉 Adaptive Softmax 算法原理和适用场景
- 熟练掌握 Python

### NO.2 为 Paddle 新增 cholesky_inverse API

**详细描述：**

- 使用 Cholesky 因子 U 计算对称正定矩阵的逆矩阵：返回矩阵`inv`。使用 LAPACK 例程`dpotri`和`spotri`(以及相应的 MAGMA 例程)计算逆。

  - 下三角矩阵
    
    $$inv = (uu^T)^{-1}$$
    
  - 上三角矩阵
    
    $$inv = (u^Tu)^{-1}$$

- 调用形式
  - paddle.cholesky_inverse , 作为独立的函数调用
  - Tenso.cholesky_inverse , 作为 Tensor 的方法使用

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)paddle/linalg.py 文件。
- 单测代码，Paddle repo 的 [test/legacy_test](https://github.com/PaddlePaddle/Paddle/tree/develop/test/legacy_test)目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 熟练掌握 Cholesky 数学语言以及线性代数中矩阵相关运算
- 熟练掌握 Python

### NO.3 为 Paddle 新增 ZeroPad1D / ZeroPad3D / block_diag  API

**详细描述(ZeroPad1D/ ZeroPad3D）：**

- 用零填充输入张量边界，1D填充最后一个维度，3D填充最后三个维度即可。
- 调用形式
  - paddle.nn.ZeroPad1d
  - paddle.nn.ZeroPad3d

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)paddle/nn/layer目录新增padding.py文件
- 单测代码，Paddle repo 的 [test/legacy_test](https://github.com/PaddlePaddle/Paddle/tree/develop/test/legacy_test)目录
- 中文API文档，在 docs repo 的 [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**详细描述(block_diag)：**

- 从提供的张量列表中创建一个块对角矩阵,返回一个二维张量，所有输入张量按顺序排列，使得它们的左上角和右下角对角相邻。所有其他元素都被设置为0
- 调用形式
  - paddle.block_diag , 作为独立的函数调用

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)paddle/tensor/manipulation.py文件。
- 单测代码，Paddle repo 的 [test/legacy_test](https://github.com/PaddlePaddle/Paddle/tree/develop/test/legacy_test)目录
- 中文API文档，在 docs repo 的 [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 熟练掌握python
- 熟练掌握矩阵操作
- 熟练掌握c++，了解 Paddle 算子开发流程

### NO.4 为 Paddle 新增 ormqr API

**详细描述：**

计算一个普通矩阵与 Householder 矩阵的乘积。计算维度为(m, n)的矩阵 C（由 other 给出）和一个矩阵 Q 的乘积， 其中 Q 由 Householder 反射系数 (x, tau) 表示。

调用路径为：paddle.linalg.ormqr 作为独立函数调用，Tensor.ormqr 作为 Tensor 方法调用。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)文件；并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 ormqr API，以支持 Tensor.ormqr 的调用方式；在[python/paddle/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/linalg.py)中，添加 ormqr API，以支持 paddle.linalg.ormqr 的调用方式
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 熟悉 Householder 矩阵含义，了解当前 paddle.linalg.householder_product 的逻辑；
- 熟悉 lapack/cublas 库；
- 了解 Paddle 算子开发流程；
- 熟练掌握 Python；

### NO.5 为 Paddle 新增 Chi2 / LKJCholesky API

**详细描述：**

内容一：实现卡方分布。调用路径为:paddle.distribution.chi2，作为独立的函数调用。

内容二：实现相关矩阵的下三角 Choleskey 因子的 LJK 分布。调用路径为:paddle.distribution.LKJCholesky，作为独立的函数调用。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python](https://github.com/PaddlePaddle/Paddle/tree/release/2.6/python)/[paddle](https://github.com/PaddlePaddle/Paddle/tree/release/2.6/python/paddle)/[distribution](https://github.com/PaddlePaddle/Paddle/tree/release/2.6/python/paddle/distribution)目录增加 chi2.py 和 lkj_cholesky.py
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python
- 熟悉卡方分布和 LJK 分布。

### NO.6 为 Paddle 新增 MultivariateNormal / StudentT API

**详细描述：**

内容一：实现多变量正态分布（又称高斯分布），调用路径为:paddle.distribution.MultivariateNormal，作为独立的函数调用。

内容二：实现学生 t 分布，调用路径为:paddle.distribution.StudentT，作为独立的函数调用。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python](https://github.com/PaddlePaddle/Paddle/tree/release/2.6/python)/[paddle](https://github.com/PaddlePaddle/Paddle/tree/release/2.6/python/paddle)/[distribution](https://github.com/PaddlePaddle/Paddle/tree/release/2.6/python/paddle/distribution)目录增加 multivariate_normal.py 和 student_t.py
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python
- 熟悉多变量正态分布和 t 分布。

### NO.7 为 Paddle 新增 sinc / sinc\_ API

**详细描述：**

计算输入的归一化 sinc 函数。需要实现`paddle.sinc、Tensor.sinc`，及对应的 inplace 函数（`paddle.sinc_、Tensor.sinc_`）。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/optimizer/lr.py 中实现 PolynomialDecay 类（注意命名有风格变化）；在 python/paddle/tensor/math.py 中实现 sinc、sinc\_ API，并在 python/paddle/tensor/init.py 中，添加 sinc、sinc\_ API，
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python

### NO.8 为 Paddle 新增 FeatureAlphaDropout API

**详细描述：**

实现 FeatureAlpha 方式的 Dropout。详情可以参考论文：Self-Normalizing Neural Networks，新增`paddle.nn.FeatureAlphaDropout`。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/nn/layer/common.py 中实现 FeatureAlphaDropout 类，并在 python/paddle/nn/layer/**init**.py、python/paddle/nn/**init**.py 添加对应调用。
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python

### NO.9 为 Paddle 新增 cartesian_prod API

**详细描述：**

paddle.cartesian_prod 作为独立的函数调用，对给定的张量序列进行笛卡尔积。该行为类似于 python 的 `itertools.product` 。相当于把所有输入的张量转成列表，对这些列表做`itertools.product`，最后把得到的列表转成张量。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的 docs/api/paddle/ 目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python

### NO.10 为 Paddle 新增 isposinf / isneginf / isreal / isin API

**详细描述：**

- paddle.isposinf 作为独立的函数调用，Tensor.isposinf(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为正无穷大。
- paddle.isneginf 作为独立的函数调用，Tensor.isneginf(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为负无穷大。
- paddle.isreal 作为独立的函数调用，Tensor.isreal(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为实值。
- paddle.isin 作为独立的函数调用，Tensor.isin(x) 做为 Tensor 的方法使用。测试 `elements` 的每个元素是否在 `test_elements` 中。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/tensor/math.py 文件；并在 python/paddle/tensor/**init**.py 中，添加 isposinf / isneginf / isreal / isin API，以支持 paddle.Tensor. isposinf / isneginf / isreal / isin 的调用方式；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的 docs/api/paddle/ 目录

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python

### NO.11 为 Paddle 新增 bernoulli\_ / log_normal\_ / log_normal API

**详细描述：**

内容一：指定概率 p，实现 inplace 的伯努利分布，新增`paddle.bernoulli_`；

内容二：指定均值和方差，实现对数正态分布，新增`paddle.lognormal/lognormal_`API；其中`lognormal`可通过 paddle.gaussian 和 paddle.exp 组合实现，`lognormal_`可通过 paddle.normal*和 paddle.exp*组合实现，`bernoulli_`可通过 paddle.uniform\_等来组合实现。调用路径为：

1. paddle.bernoulli\_(x, p=0.5) 可以 inplace 的修改输入 x，填充伯努利分布的值
2. paddle.Tensor.bernoulli*(p=0.5) 作为 paddle.bernoulli*的 Tensor 类方法使用
3. paddle.log*normal*(x, mean=1.0, std=2.0) 可以 inplace 的修改输入 x，填充对数正态分布的值
4. paddle.Tensor.log*normal*(mean=1.0, std=2.0) 作为 paddle.log*normal*的 Tensor 类方法使用
5. paddle.log_normal(mean=1.0, std=2.0, shape=None, dtype=None) 作为非 inplace 的 API，可以创建一个对数正态分布的 Tensor

可以在之前开发者 [未开发完的 PR](https://github.com/PaddlePaddle/Paddle/pull/58432) 基础上进行开发。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的[rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/random.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/tensor/random.py)文件增加 bernoulli* / log_normal* / log*normal，以支持`paddle.bernoulli*/log*normal*/log*normal`的调用；并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 bernoulli*/ log*normal*，以支持`paddle.Tensor.bernoulli_/log_normal_`的调用；
- 单测代码，在 Paddle repo 的[test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录增加非 inplace 的单测, 同时在[paddle/test/legacy_test/test_inpalce.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_inplace.py)中新增对应 inplace 的单测
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录和[docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst)文件，同时需要在[docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst)文件中添加 API 介绍。

**技术要求：**

- 熟悉 bernoulli / log_normal 函数的计算原理和适用场景；
- 熟练掌握 C++，Python

### NO.12 为 Paddle 新增 lu_solve API

**详细描述：**

使用 LU 分解 来求解线性方程组 AX=B，A 为 1 个或多个矩阵，A.shape=[m, n] or [batch, m, n]，B.shape=[m, k] or [batch, m, k]，A 和 B 已知，通过 LU 分解方阵 A 来加速求解 X。需要满足 LU, pivots, info =paddle.linalg.lu(A); X = paddle.linalg.lu_solve(B, LU, pivots) 与 使用 X=paddle.linalg.solve(A, B) 直接求解线性方程组的结果一样。此任务的目标是在 Paddle 框架中，新增 lu_solve API，调用路径为：paddle.linalg.lu_solve 和 Tensor.lu_solve

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)文件。并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 lu_solve API，以支持 Tensor.lu_solve 的调用方式；
- C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 熟悉矩阵 LU 分解原理，了解当前 paddle.linalg.lu 的逻辑；
- 熟悉 lapack/cublas 库；
- 熟练掌握 Python。

### NO.13 为 Paddle 新增 RAdam / NAdam API

**详细描述：**

- 参考 [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) 为 Paddle 新增 RAdam 优化器。RAdam 通过对 Adam 的动量项进行了修正提升训练初期稳定性。参数命名风格注意和飞桨现有优化器体系保持一致。调用路径为 paddle.optimizer.RAdam.
- 参考 [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) 为 Paddle 新增 NAdam 优化器。NAdam 将 Nesterov 动量与 Adam 结合，利用了 Adam 的适应性学习率的优势，又结合了 Nesterov 动量的优点。参数命名风格注意和飞桨现有优化器体系保持一致。调用路径为 paddle.optimizer.NAdam.

**提交内容：**

- API 的设计文档，并提 PR 至 PaddlePaddle/community 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/optimizer](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/optimizer) 目录下新增 radam.py/nadam.py 文件。并在[python/paddle/optimizer/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/optimizer/__init__.py)中，添加 RAdam / NAdam API，以支持 paddle.optimizer.RAdam 和 paddle.optimizer.NAdam 的调用方式；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 熟悉 Adam 系列优化器原理，了解飞桨当前优化器体系实现逻辑；
- 熟练掌握 Python。

### NO.14 为 Paddle 新增 tensorinv / tensorsolve API

**详细描述：**

- 使用 tensorsolve 来求解线性方程组 paddle.tensordot(A, X, axes=X.ndim) = B，A 和 B 已知、可以为多维 Tensor 且满足 prod(A.shape[:B.ndim]) == prod(A.shape[B.ndim:])，X 为计算结果
- 使用 tensorinv 求多维 Tensor A 关于 paddle.tensordot 的逆，求的时候需要指定 A 的分割轴 axis，A 需要满足 prod(A.shape[:axis]) == prod(A.shape[axis:])，结果需要满足 paddle.tensordot(paddle.linalg.tensorinv(A), B) == paddle.linalg.tensorsolve(A, B)，
- 此任务的目标是在 Paddle 框架中，新增 tensorinv / tensorsolve API，调用路径为：paddle.linalg.tensorinv/Tensor.tensorinv 和 paddle.linalg.tensorsolve/Tensor.tensorsolve

**提交内容：**

- API 的设计文档，并提 PR 至 PaddlePaddle/community 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)文件。并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 tensorinv / tensorsolve API，以支持 Tensor.tensorinv 和 Tensor.tensorsolve 的调用方式；
- C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 熟悉 tensordot 原理，了解当前 paddle.tensordot 的逻辑；
- 熟悉 lapack/cublas 库；
- 熟练掌握 Python。

### NO.15 为 Paddle 新增 ldl_factor / ldl_solve API

**详细描述：**

- 使用 ldl_factor 计算厄米特矩阵或对称矩阵的 LDL 分解，设 A 为 1 个或多个厄米特矩阵或对称矩阵，即 A.shape=[n, n] or [batch, n, n]，得到的 LD 的 shape 和 A 一样，包含对应的 1 个或多个下三角矩阵；得到特征值对角线矩阵为压缩表示形式 shape=[n] or [batch, n]
- 使用 ldl_solve 来求解线性方程组 AX=B，A 为 1 个或多个厄米特矩阵或对称矩阵 A.shape=[n, n] or [batch, n, n]，B.shape=[n, k] or [batch, n, k]，A 和 B 已知，通过 LDL 分解 A 以能够加速求解 X，需要满足 LD, pivots, info=paddle.linalg.ldl_factor(A); X = paddle.linalg.ldl_solve(B, LU, pivots) 与 使用 X=paddle.linalg.solve(A, B) 直接求解线性方程组的结果一样
- 此任务的目标是在 Paddle 框架中，新增 ldl_factor / ldl_solve API，调用路径为：paddle.linalg.ldl_factor/Tensor.ldl_factor 和 paddle.linalg.ldl_solve/Tensor.ldl_solve

**提交内容：**

- API 的设计文档，并提 PR 至 PaddlePaddle/community 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)文件。并在[python/paddle/tensor/**init**.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 ldl_factor / ldl_solve API，以支持 Tensor.ldl_factor 和 Tensor.ldl_solve 的调用方式；
- C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录；
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录

**技术要求：**

- 熟悉 ldl_factor 分解原理；
- 熟悉 lapack/cublas 库；
- 熟练掌握 Python。

**注意事项：**

- 调研时关注 torch.linalg.ldl_factor 和 torch.linalg.ldl_factor_ex

### NO.16 为 Paddle 新增 LPPool1D / LPPool2D API

**详细描述：（第四期的题目，如现在没有 fluid 目录了）**

- 用于求解一维的幂平均池化 (power-average pooling)
  - 每个窗口的计算过程：
    $$f(X) = \sqrt[p]{\sum_{x \in X} x^p}$$
    - 当 p=∞ 时等同于最大池化
    - 当 p=1 时等同于累加池化
- 调用形式
  - paddle.nn.LPPool1D
  - paddle.nn.LPPool2D
  - paddle.nn.functional.lp_pool1d
  - paddle.nn.functional.lp_pool2d

可以在之前开发者 [未开发完的 PR](https://github.com/PaddlePaddle/Paddle/pull/58433) 基础上进行开发。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，
  - lp_pool1d 和 lp_pool2d 在 Paddle repo 的 python/paddle/nn/functional/pooling.py 文件。
  - LPPool1D 和 LPPool2D 在 Paddle repo 的 python/paddle/nn/layer/pooling.py 文件。
- C ++/CUDA 实现代码，头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录
- 单测代码，Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/) 目录

**技术要求：**

- 熟练掌握 lp_pool1d 和 lp_pool2d 语义及计算过程
- 掌握 Paddle 算子开发流程
- 熟练掌握 Python
- 熟悉 c++
- 了解 cuda 编程

### NO.17 为 Paddle 新增 sparse.mask_as API

**详细描述：**

针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 mask 掩码逻辑，将稠密 Tensor 根据 mask（稀疏 Tensor）的形式来获得一个稀疏 Tensor，一共需要新增 2 个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。可参考框架中代码 [mask_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/cpu/mask_kernel.cc)、[mask_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/gpu/mask_kernel.cu)，分析是否能复用该代码逻辑再进行开发。该 API 对标`torch.Tensor.sparse_mask`的功能，

需要实现 paddle 调用方式`paddle.sparse.mask_as`。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/binary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/binary.py) 文件实现`paddle.sparse.mask_as`API；
- C++ kernel 实现代码，在 Paddle repo 的 [paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
- 单测代码，在 Paddle repo 新建 [test/legacy_test/test_sparse_mask_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test) 文件；
- yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。

**技术要求：**

- 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
- 熟悉稀疏 Tensor 的 concat 计算逻辑；
- 熟练掌握 Python、C++、CUDA 代码编写。

### NO.18 为 Paddle 新增 sparse.concat API

**详细描述：**

针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 concat 的计算逻辑，一共需要新增 2 个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。可参考之前开发者 [设计文档](https://github.com/PaddlePaddle/community/pull/504) ，自行决定是否需要重新设计。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/binary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/binary.py) 文件；
- C++ kernel 实现代码，在 Paddle repo 的 [paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
- 单测代码，在 Paddle repo 新建 [test/legacy_test/test_sparse_concat_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test) 文件；
- yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。

**技术要求：**

- 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor / SparseCsrTensor 数据结构；
- 熟悉 稀疏 Tensor 的 concat 计算逻辑；
- 熟练掌握 Python、C++、CUDA 代码编写。

### NO.19 为 Paddle 新增 sparse.stack API

**详细描述：**

针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 stack 的计算逻辑，一共需要新增 2 个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/binary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/binary.py) 文件；
- C++ kernel 实现代码，在 Paddle repo 的 [paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
- 单测代码，在 Paddle repo 新建 [test/legacy_test/test_sparse_stack_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test) 文件；
- yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。

**技术要求：**

- 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor / SparseCsrTensor 数据结构；
- 熟悉 稀疏 Tensor 的 stack 计算逻辑；
- 熟练掌握 Python、C++、CUDA 代码编写。

### NO.20 为 Paddle 新增 sparse.nn.Conv2DTranspose / Conv3DTranspose API

**详细描述：**

针对 Paddle 的稀疏组网类 API，当前已支持`sparse.nn.Conv2D/ Conv3D`等网络，但还需要新增 `sparse.nn.Conv2DTranspose` / `Conv3DTranspose`，需先开发 `sparse.nn.functional.*`函数式 API，再开发`sparse.nn.*`组网类 API，只需实现 COO 逻辑。

【一些参考思路】对标竞品为 spconv 库，可参考其 [代码实现](https://github.com/traveller59/spconv/blob/master/spconv/pytorch/conv.py#L943)，并与之对比结果一致。在 Paddle 框架中已有 [sparse.nn.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/layer/conv.py)、 [sparse.nn.functional.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/functional/conv.py) 、[conv_kernel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/gpu/conv_kernel.cu) 等稀疏卷积 API 及 kernel，建议对其进行**基类增强设计**，使卷积 API 尽可能共用相同基类，并尽可能复用代码逻辑。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [sparse.nn.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/layer/conv.py)、 [sparse.nn.functional.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/functional/conv.py) 文件；
- C++ kernel 实现代码，在 Paddle repo 的 [conv_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/cpu/conv_kernel.cc)、[conv_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/gpu/conv_kernel.cu) 文件中新增卷积 kernel 的实现；
- 单测代码，在 Paddle repo 的 [test/legacy_test/test_sparse_conv_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_sparse_conv_op.py) 文件新增相应测试 case；
- yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。

**技术要求：**

- 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
- 熟悉 稀疏 Tensor 的 卷积 计算逻辑；
- 熟练掌握 Python、C++、CUDA 代码编写。

### NO.21 为 Paddle 新增 sparse.nn.InverseConv2D / InverseConv3D API

**详细描述：**

针对 Paddle 的稀疏组网类 API，当前已支持`sparse.nn.Conv2D/ Conv3D`等网络，但还需要新增 `sparse.nn.InverseConv2D` / `InverseConv3D`，需先开发 `sparse.nn.functional.*`函数式 API，再开发`sparse.nn.*`组网类 API，只需实现 COO 逻辑。

【一些参考思路】对标竞品为 spconv 库，可参考其 [代码实现](https://github.com/traveller59/spconv/blob/master/spconv/pytorch/conv.py#L1077)，并与之对比结果一致。在 Paddle 框架中已有 [sparse.nn.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/layer/conv.py)、 [sparse.nn.functional.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/functional/conv.py) 、[conv_kernel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/gpu/conv_kernel.cu) 等稀疏卷积 API 及 kernel，建议对其进行**基类增强设计**，使卷积 API 尽可能共用相同基类，并尽可能复用代码逻辑。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [sparse.nn.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/layer/conv.py)、 [sparse.nn.functional.\*](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/functional/conv.py) 文件；
- C++ kernel 实现代码，在 Paddle repo 的 [conv_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/cpu/conv_kernel.cc)、[conv_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/gpu/conv_kernel.cu) 文件中新增卷积 kernel 的实现；
- 单测代码，在 Paddle repo 的 [test/legacy_test/test_sparse_conv_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_sparse_conv_op.py) 文件新增相应测试 case；
- yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。

**技术要求：**

- 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
- 熟悉 稀疏 Tensor 的 卷积 计算逻辑；
- 熟练掌握 Python、C++、CUDA 代码编写。

### NO.22 为 Paddle 增强 sparse.add / subtract / multiply / divide API

**详细描述：**

针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，当前稀疏 Tensor 之间的加减乘除运算（Sparse [ +-*/ ] Sparse）在 CPU 上已经实现，需要对齐增强，支持 GPU 上的计算，一共需要新增 8 个 GPU kernel 的前向与反向：

- add_coo_coo
- add_csr_csr
- subtract_csr_csr
- subtract_csr_csr
- multiply_coo_coo
- multiply_csr_csr
- divide_coo_coo
- divide_csr_csr

【一些参考思路】在 paddle 中当前已对 add_coo_coo kernel 进行了实现，但仅支持相同 indices 的加法运算，不规范且不通用，需要将其删除并重新支持通用的运算，即任意 indices 的稀疏 Tensor 之间的运算。其中加法/乘法运算是 x 与 y 的 indices 的并集，可考虑拼接 indices 然后再 coalesce 合并排序的算法；乘法运算是 x 与 y 的 indices 的交集。

**提交内容：**

- API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
- Python 实现代码 & 英文 API 文档，在 Paddle repo 新建 [python/paddle/sparse/binary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/binary.py) 文件；
- CUDA kernel 实现代码，在 Paddle repo 的[paddle/phi/kernels/sparse/gpu/elementwise_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/sparse/gpu/elementwise_kernel.cu) 目录中新增对应 kernel 的实现代码；
- 单测代码，在 Paddle repo 新建 [test/legacy_test/test_sparse_elementwise_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_sparse_elementwise_op.py) 文件新增 CUDA kernel 对应的测试；
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。

**技术要求：**

- 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
- 熟悉 稀疏 Tensor 的 加减乘除（Sparse [ +-*/ ] Sparse） 计算逻辑；
- 熟练掌握 Python、C++、CUDA 代码编写，重点掌握 CUDA 代码的编写

### NO.23 为 paddle.nn.functional.embedding/paddle.nn.Embedding 增加参数 max_norm/norm_type/scale_grad_by_freq

**详细描述：**

torch.nn.functional.embedding 支持 max_norm/norm_type/scale_grad_by_freq 参数，而 paddle 不支持，需要调研这三个参数的功能，并且为 paddle.nn.functional.embedding 增加这三个参数。 需要注意同时修改 paddle.nn.Embedding。

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch 对应 API 参考：torch.nn.functional.embedding/torch.nn.Embedding

### NO.24 为 paddle.nn.LSTM/RNNBase /paddle.quantile/nanquantile 功能增强

**详细描述：**

为以下多个 API 进行功能增强：

1. 【功能增强】torch.nn.LSTM 支持参数 proj_size，表示将隐藏状态 h 的维度映射到 proj_size 对应的大小，而 paddle.nn.LSTM 不支持
2. 【功能增强】torch.nn.RNNBase 的 mode 参数，可取值为 `'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'`，而 paddle.nn.RNNBase 只支持 `'LSTM', 'GRU'`，不支持其他两种
3. 【功能增强】torch.quantile/torch.nanquantile 的输入 q 支持 1D Tensor 表示 1 个 list，而 paddle.quantile/nanquantile 不支持输入 1D Tensor 表示 1 个 list
4. 【功能增强】torch.quantile/torch.nanquantile 支持 interpolation 参数，而 paddle 不支持

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

与上述列出的 4 个问题点一一对应：

1. [PaddlePaddle/Paddle#56460](https://github.com/PaddlePaddle/Paddle/pull/56460)
2. [PaddlePaddle/Paddle#56460](https://github.com/PaddlePaddle/Paddle/pull/56460)
3. [PaddlePaddle/Paddle#56461](https://github.com/PaddlePaddle/Paddle/pull/56461)
4. [PaddlePaddle/Paddle#56461](https://github.com/PaddlePaddle/Paddle/pull/56461)

Pytorch 对应 API 参考：torch.nn.LSTM/torch.nn.RNNBase/torch.quantile/torch.nanquantile

### NO.25 为 paddle.histogram/paddle.nn.functional.threshold 进行功能对齐与功能增强

**详细描述：**

为以下多个 API 进行功能增强和功能对齐：

1. 【功能增强】torch.histogram 支持参数 weight、density，而 paddle 不支持，需要调研这两个参数的功能，并且为 paddle.histogram 增加这两个参数
2. 【功能对齐】torch.histogram 返回两个 Tensor：hist、bin，而 paddle 仅返回一个 hist，需要增加一个 histogram_bin_edges，支持返回 bin
3. 【功能增强】torch.nn.functional.threshold 支持 value 参数，而 paddle 不支持，需要调研这个参数的功能，并且为 paddle.nn.functional.threshold 增加这个参数

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

与上述列出的 3 个问题点一一对应：

1. [PaddlePaddle/Paddle#56771](https://github.com/PaddlePaddle/Paddle/pull/56771)
2. [PaddlePaddle/Paddle#56771](https://github.com/PaddlePaddle/Paddle/pull/56771)
3. [PaddlePaddle/Paddle#56853](https://github.com/PaddlePaddle/Paddle/pull/56853)

Pytorch 对应 API 参考： torch.histogram/torch.nn.functional.threshold

### NO.26 为 paddle.view/paddle.nn.initializer.XavierNormal / XavierUniform / KaimingNormal / KaimingUniform 进行功能增强

**详细描述：**

为以下多个 API 进行功能增强：

1. 【Bug 修复】两个 Bug：a）paddle 的 view 的形状推导存在问题，对存在“-1”的情形会报错，b）view 的 shape_or_dtype 参数 不支持 paddle 原生数据类型，如 padlde.int32 作为输入。

```python
import paddle
x = paddle.rand([1,2, 4, 6], dtype="float32")
x.contiguous().view([8, -1])
```

1. 【功能增强】torch.nn.init.xavier_normal\_ / xavier_uniform\_ 均支持参数 gain，paddle.nn.initializer.XavierNormal / XavierUniform 缺少参数 gain，需增加该参数
2. 【功能增强】torch.nn.init.kaiming_normal\_ / kaiming_uniform\_缺少参数 mode，当 mode="fan_out"时，paddle.nn.initializer.KaimingNormal / KaimingUniform 缺少对应可替代的功能，需增加 mode 参数或 fan_out 参数，从而补齐该功能

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

[PaddlePaddle/Paddle#56471](https://github.com/PaddlePaddle/Paddle/pull/56471)

Pytorch 对应 API 参考： torch.nn.init.xavier_normal\_ / torch.nn.init.xavier_uniform\_ / torch.nn.init.kaiming_normal\_ / torch.nn.init.kaiming_uniform\_

### NO.27 为 paddle.io.RandomSampler/random_split / Layer.clear_gradients 进行功能增强

**详细描述：**

为以下多个 API 进行功能增强：

1. 【功能增强】paddle.io.RandomSampler 当参数 replacement = False 时，不允许指定 num_samples，而 torch.utils.data.RandomSampler 则无此限制，需要增强该功能
2. 【功能增强】torch.utils.data.random_split 的 lengths 参数支持比例方式划分，而 paddle.io.random_split 不支持，需要增强该功能
3. 【功能增强】paddle.nn.Layer.clear_gradients 需要暴露底层的 set_to_zero 参数，从而和 torch.nn.Module.zero_grad 的 set_to_none 参数功能对应

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch 对应 API 参考： torch.utils.data.RandomSampler/torch.utils.data.random_split/torch.nn.Module.zero_grad

### NO.28 为 paddle.round / paddle.nn.functional.max_pool1d / max_pool2d/max_pool3d 进行功能增强

**详细描述：**

为以下多个 API 进行功能增强：

1. 【功能增强】torch.round 支持 decimals 参数，表示舍入的小数点位数，paddle 不支持，需要增加该参数
2. 【功能增强】torch.nn.functional.max_pool1d /max_pool2d / max_pool3d 支持 dilation 参数空洞池化，paddle 不支持，需要增加该参数

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch 对应 API 参考： torch.round / torch.nn.functional.max_pool1d / max_pool2d / max_pool3d

### NO.29 为 paddle.nn.functional.max_unpool1d / max_unpool2d / max_unpool3d/paddle.nn.functional.kl_div 进行功能增强

**详细描述：**

为以下多个 API 进行功能增强或 Bug 修复：

1. 【功能增强】torch.nn.functional.max_unpool1d / max_unpool2d / max_unpool3d 支持 int64 输入，而 paddle 不支持，需要增加该功能
2. 【Bug 修复】paddle.nn.functional.max_unpool1d / max_unpool2d / max_unpool3d 的 output_size 参数的判断有 bug，输入正确的 output_size 会报错
3. 【功能增强】torch.nn.functional.kl_div 支持参数 log_target，而 paddle 不支持，需要增加该参数

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch 对应 API 参考： torch.nn.functional.max_unpool1d / max_unpool2d / max_unpool3d / torch.nn.functional.kl_div

### NO.30 为 paddle.nn.functional.max_pool1d / max_pool2d / max_pool3d / paddle.signal.stft 进行功能增强

**详细描述：**

为以下多个 API 进行功能增强或 Bug 修复：

1. 【Bug 修复】paddle.nn.functional.max_pool1d / max_pool2d / max_pool3d 当 return_mask=True 时，ceil_mode 不生效。[问题 case 链接](https://github.com/PaddlePaddle/PaConvert/blob/master/tests/test_nn_functional_max_pool1d.py#L93-L106)。
2. 【Bug 修复】paddle.signal.stft 计算结果与 torch.stft 有较大差距，需要分析该问题并给出正确的解决方案

**提交内容：**

- 算子 Kernel 和 API 的修改文档，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- 单测代码，提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中，需自行寻找相应的 API 或 OP 测试代码，如未有现成代码，则需自行增加测试 case，对改动功能点需要着重测试并对齐 Pytorch 计算结果
- API 中文文档，如果有 API 参数的增删或功能的修改，需修改 API 文档并提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

Pytorch 对应 API 参考： torch.nn.functional.max_pool1d / max_pool2d / max_pool3d / torch.stft

### NO.31 paddle Normal 分布支持复数

**详细描述：**

Normal（Gaussian）作为最常见的分布之一，paddle 目前不支持生成复数域的正太分布。本任务需要去支持 paddle.normal, paddle.distribution.Normal, paddle.nn.initializer.Normal 等 API 的复数正太分布生成。

**提交内容：**

- 复数域生成正态分布的的设计文档，并提 PR 至 PaddlePaddle/community 的 rfcs/APIs 目录
- 实现代码 & 英文文档说明支持复数，
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档说明支持复数，在 docs repo 的 docs/api/paddle/ 目录

**技术要求：**

- 熟悉 Normal 分布的算法原理和适用场景
- 熟练掌握 Python，c++

### NO.32 paddle Adam 优化器支持复数

**详细描述：**

Adam 优化器出自 [Adam 论文](https://arxiv.org/abs/1412.6980) 的第二节，能够利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。本任务需要去支持 Adam 对于复梯度的优化，并需要提供详尽的业界调研及理论支撑。

**提交内容：**

- Adam 在复数域优化的设计文档，并提 PR 至 PaddlePaddle/community 的 rfcs/APIs 目录
- 实现代码
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中/英文文 API 文档说明支持复数，在 docs repo 的 docs/api/paddle/ 目录

**技术要求：**

- 熟悉 Adam 算法原理和适用场景
- 了解复梯度的计算和优化器优化
- 熟练掌握 Python， c++

## 【开源贡献个人挑战赛-分布式】任务详情

### NO.33 支持动态图流水并行设定多个损失函数，并返回多个 loss

**详细描述：**

- 流水线并行（Pipeline Parallel）是指将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗，从而实现超大规模模型训练的方法。当前 Paddle 在动态图中通过派生 `paddle.distributed.fleet.meta_parallel.PipelineLayer`子类的方式完成对流水线并行模型的定义，并通过指定 `loss_fn`参数以表明流水线并行的模型使用的损失函数，在流水线的最后一个 stage 会根据模型定义时指定的 `loss_fn`计算出 loss。当前 Paddle 动态图流水并行只支持设定一个损失函数，`train_batch`返回该损失函数对应的 loss。
- 为了提升分布式训练的易用性，增强 Paddle 流水并行模型定义时的表达能力，以支持更复杂的场景，你的任务是：支持动态图流水线并行组网时指定多个损失函数，并返回多个 loss，用户可以指定针对哪一个 loss 做 backward。默认下，只对返回的第 0 个 loss 做 backward。最后一个 stage 计算得到多个 loss 值后，需要广播到其他的 GPU 上。

**提交内容：**

- 实现代码 & PR 描述：实现上述功能，并 PR 描述中说明实现方法
- 单测代码：在 `test/collective/fleet/`目录下添加单测，测试在 non-interleaved 和 interleaved 模式下指定多个损失函数并返回多个 loss 功能的正确性

**技术要求：**

- 熟悉分布式流水并行训练的基本原理
- 熟悉 Paddle 分布式流水并行代码
- 熟练掌握 Python

**参考资料：**

- [Paddle 流水并行原理介绍及使用方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/pipeline_parallel_cn.html)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)

### NO.34 支持动态图流水并行时返回 micro batch 的 loss

**详细描述：**

- 流水线并行（Pipeline Parallel）是指将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗，从而实现超大规模模型训练的方法。目前 Paddle 动态图流水线支持 1F1B 模式中的 non-interleaved 和 interleaved 模式，相关源码在 [Pipeline Parallel Scheduler 源码](https://github.com/PaddlePaddle/Paddle/blob/e213188465e9d3b89ed29fedf98dbe7b846c9576/python/paddle/distributed/fleet/meta_parallel/pipeline_parallel.py)。当前 Paddle 动态图中的流水并行会将一个 batch 的数据切分成若干个 micro-batch，在流水线的最后一个 stage 会对 batch 内所有的 micro-batch 计算得到的 loss 做求和平均得到 batch 的 loss，并将 batch 的 loss 广播到所有 GPU 上。当前`train_batch`也只会返回 batch 的 loss 而不是 batch 内所有 micro-batch 的 loss。
- 为了提升分布式训练的易用性，你的任务是：为动态图流水并行设置一个开关，当开关打开时，`train_batch`返回一个 batch 内所有 micro-batch 的 loss；但是做 backward 时只会对 batch 的 loss 做 backward (与原来的行为对齐)

**提交内容：**

- 实现代码 & PR 描述：实现上述功能，并在 PR 描述中说明实现方法
- 单测代码：在 `test/collective/fleet/`目录下添加单测，测试在 non-interleaved 和 interleaved 模式下流水并行时返回 micro batch 的 loss 功能的正确性

**技术要求：**

- 熟悉分布式流水并行训练的基本原理
- 熟悉 Paddle 分布式流水并行代码
- 熟练掌握 Python

**参考资料：**

- [Paddle 流水并行原理介绍及使用方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/pipeline_parallel_cn.html)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)

### NO.35 前向重计算函数在 `use_reentrant == True` 时支持以关键字参数的方式传入 Tensor

**详细描述：**

- 前向重计算是将在前向计算时，除了小部分必须存储在内存中的张量外，其他中间结果都将被删除；在反向计算中，首先重新计算一遍前向算子，以获得中间结果，再运行反向算子。简而言之，前向重计算出网络中间的激活值，从而达到节省显存的目的。用户使用 Paddle 定义网络时，可以使用 `recompute`函数定义网络中进行重计算的层，具体使用方法可见 [Paddle 前向重计算原理及使用方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/data_parallel/recompute_cn.html)。`recompute`函数内部（`recompute`函数位于 `python/paddle/distributed/fleet/recompute/recompute.py`）由 `use_reentrant`参数控制，会采用两种方法实现重计算功能：`use_reentrant == True`时会使用 `PyLayer`来实现。但 `PyLayer`目前不支持以 dict 形式传入 Tensor 类型参数（因为以 dict 形式传入的 Tensor 不会创建反向节点、反向边） 。
- 为了提升分布式训练的易用性，你的任务是：支持 `recompute`函数在 `use_reentrant == True`时能以关键字参数的方式传入 tensor，同时要求启用重计算后性能不下降。详情可见需求描述：https://github.com/PaddlePaddle/Paddle/issues/62363

**提交内容：**

- 实现代码 & PR 描述：实现上述功能，并在 PR 描述中说明实现方法和性能数据。
- 单测代码：在 `test/collective/fleet/`目录下添加单测，测试功能的正确性

**技术要求：**

- 熟悉动态图前向重计算的基本原理
- 熟练掌握 C++ 和 Python

**参考资料：**

- [Paddle 前向重计算原理及使用方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/data_parallel/recompute_cn.html)

## 【开源贡献个人挑战赛-框架其他】任务详情

### NO.50 将 PyLayer 机制迁移至 PIR 体系下

**详细描述：**

飞桨的动态图 PyLayer 向用户提供了一种「高度灵活且便利」的自定义网络层前、反向计算的机制，比如存在一些用户自定义的计算逻辑，无法通过飞桨 现有的某个 API 或某些 API 组合实现，故可以借助 PyLayer 来实现用户的「idea」。在[之前的工作中](https://github.com/PaddlePaddle/Paddle/issues/54120)，飞桨的动态图中的 [PyLayer 机制](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayer_cn.html)能够与飞桨的[动转静@to_staitc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/index_cn.html) 互通，支持模型中 PyLayer 的自定义层能够被 @to_static 感知并正确地生成静态图 Program，支撑[转静训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/basic_usage_cn.html#erdongzhuanjingxunlian)和[导出推理](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/basic_usage_cn.html#sandongzhuanjingmoxingbaocunhejiazai)。

`PyLayer` 支持 `@to_static` 的思路是使用 Paddle 基于 AST 的动转静转写工具，获取动态图 `PyLayer` 的正反向逻辑，分别创建前向 block 和反向 block，并创建静态图 PyLayer Op。这工作的基础便是设计并实现了静态图 PyLayer Op，其功能是能够运行用户创建的前向 block 和反向 block。

目前静态图 PyLayer Op 仍是基于旧 IR，但从 Paddle 2.6 版本开始，Paddle 团队引入 PIR 作为静态图 IR。随着 PIR 底层机制的日益完善，我们可以把  PyLayer 机制迁移至 PIR 体系下，进一步支持动转静等相关工作，保证 Paddle 的完备性和高效性。

在这个任务中你将收获：

- 学习 PyLayer 旧 IR 体系代码和 PIR 体系核心机制代码，掌握开源套件开发基本知识
- 熟悉 Paddle 中控制流算子相关知识
- 熟悉 Paddle 中自动微分机制
- 熟悉 Paddle 中动转静原理和工具的相关知识

主框架将获得收益：

- 完善动转静机制
- 完善控制流算子正反向机制

**提交内容：**

- 完成 PyLayer Op 的「前向」PIR 适配
- 完成 PyLayer Op 的「反向」PIR 适配
- PIR 体系下的 PyLayer Op 能支持动转静训练
- PIR 体系下的 PyLayer Op 能支持导出推理

**技术要求：**

- 熟悉 Paddle 动转静的基本原理
- 熟悉 PyLayer 执行机制
- 熟悉控制流算子执行机制
- 熟悉 PIR 基本组件与 API 层次 
- 熟练掌握 C++, Python

**参考资料：**
https://github.com/PaddlePaddle/Paddle/issues/60688

### NO.51 PIR 计算图支持可视化

**详细描述：**

- 在模型的开发调试过程中，计算图的可视化是一个很重要的基础设施，飞桨当前的 pir program 不支持可视化。
- 你的任务是增加一个工具函数，利用飞桨已有的 networkx 库，根据 pir program，可视化展示计算图，并且提供输出为.dot文件的方法

**提交内容：**

- 实现代码 & PR 描述：实现上述功能，并在 PR 描述中说明实现方法和性能数据。

**技术要求：**

- 熟悉模型的 pir 表示
- 熟悉 python

**参考资料：**

- [networkx](https://networkx.org/documentation/latest/tutorial.html)
