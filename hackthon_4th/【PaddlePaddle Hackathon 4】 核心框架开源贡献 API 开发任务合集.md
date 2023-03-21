# 【PaddlePaddle Hackathon 4】核心框架开源贡献 API 开发任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

注：为飞桨框架新增一系列 API，提交流程请参考 [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项在任务列表后：

### No.1：为 Paddle 新增 finfo API <a name='task1'></a>

- 任务难度：基础
- 详细描述：finfo计算浮点数类型的数值限制，输入参数为 Paddle 浮点数类型(paddle.float16/paddle.float32/paddle.float64/paddle.complex64/paddle.complex128)，返回包含下表属性对象。此任务目标是为 Paddle 新增 finfo API，调用路径为 paddle.finfo。要求通过 pybind 方式直接将 C++ 层实现绑定到 Python，无需开发 Paddle Kernel，可以参考 [paddle/fluid/pybind/pybind.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/pybind.cc) 中代码。更详细内容可以参考 [numpy.finfo](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html) 。API 设计文档可参考 [api_design_for_finfo.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220330_api_design_for_finfo.md)。
- 提交内容：
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 目录；
  - C ++ 实现代码，在 Paddle repo 的 [paddle/fluid/pybind](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/pybind) 目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求：
  - 熟练掌握 C++ （如有 C++ 开发）、Python；
  - 熟悉 C++ 标准库 std::numeric_limits。


### No.2：为 Paddle 新增 cdist API <a name='task2'></a>

- 任务难度：基础
- 详细描述：cdist API 是 [dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist) 的拓展。dist API 用于计算两个输入 Tensor 的 p 范数（p-norm），计算结果为形状为 [1] 的 Tensor，而 cdist API 则用于计算两个输入 Tensor 的所有行向量对的 p 范数（p-norm），输出结果的形状和两个 Tensor 乘积的形状一致。此任务的目标是在 Paddle 框架中，新增 cdist API，调用路径为：paddle.cdist，设计文档可参考 [api_design_for_cdist.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220316_api_design_for_cdist.md)。
- 提交内容
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py) 文件；
  - C ++ /CUDA 实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels) 目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 p 范数（p-norm）计算原理；
  - 熟悉 Paddle 动静态图下数学计算过程；
  - 熟练掌握 C++、CUDA、Python。


### No.3：为 Paddle 新增 trapezoid API <a name='task3'></a>

- 任务难度：基础
- 详细描述：实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 的算法，在输入N 维 Tensor 指定的某一维计算 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 结果。比如输入数据 y = paddle.to_tensor([[2, 4, 8], [3, 5, 9]])，x = paddle.tensor([[1, 2, 3], [3, 4, 5]])，则 paddle.trapezoid(y, x, axis=-1)) 或 y.trapezoid(x, axis=-1) 得到 [9, 11]，同时 paddle.trapezoid(y, x, axis=0)) 或 y.trapezoid(x, axis=0) 得到 [5, 9, 17] 。此 API 需支持的调用路径为：paddle.trapezoid 和 Tensor.trapezoid，设计文档可参考 [community#pull/173](https://github.com/PaddlePaddle/community/pull/173)。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 目录，并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py) 中，添加 trapezoid API，以支持 Tensor.trapezoid 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 trapezoid 算法逻辑；
  - 熟练掌握 Python，及 Tensor 切片操作。

### No.4：为 Paddle 新增 cumulative_trapezoid  API <a name='task4'></a>

- 任务难度：基础
- 详细描述：在输入N 维 Tensor 指定的某一维计算 累积的 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 结果。和trapezoid的区别是：trapezoid 在计算加和时使用sum，而 cumulative_trapezoid 计算加和时使用 cumsum 。比如输入数据 y = paddle.to_tensor([[2, 4, 8], [3, 5, 9]])，x = paddle.tensor([[1, 2, 3], [3, 4, 5]])，则 paddle.cumulative_trapezoid(y, x, axis=-1)) 或 y.cumulative_trapezoid(x, axis=-1) 得到 [[3, 9], [4, 11]]，同时 paddle.cumulative_trapezoid(y, x, axis=0)) 或 y.cumulative_trapezoid(x, axis=0) 得到 [[5, 9, 17]]，注意这个结果的shape=(1, 3) 。此 API 需支持的调用路径为：paddle.cumulative_trapezoid 和 Tensor.cumulative_trapezoid
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 目录，并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py) 中，添加 cumulative_trapezoid API，以支持 Tensor.cumulative_trapezoid 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 trapezoid 算法逻辑，熟悉 cumsum 的逻辑；
  - 熟练掌握 Python，及 Tensor 切片操作。

### No.5：为 Paddle 新增 nextafter API <a name='task5'></a>

- 任务难度：基础
- 详细描述：得到输入在特定方向的下一个浮点数值，输入和 方向均支持N 维 Tensor，两者的shape 必须是可广播的。此任务的目标是在 Paddle 框架中，新增 nextafter API，调用路径为：paddle.nextafter 和 Tensor.nextafter
- 提交内容
  - 设计文档可参考 [api_design_for_nextafter.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220902_api_design_for_nextafter.md)，但需要补充完善文档内容，包括更多的业界方案调研(pytorch/tf等)，增加C++算子的实现逻辑描述等
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 nextafter API，以支持 Tensor.nextafter 的调用方式；
  - C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 nextafter 函数的算法原理和适用场景；
  - 熟练掌握 Python。

### No.6：为 Paddle 新增 ldexp API <a name='task6'></a>

- 任务难度：基础
- 详细描述：通过将输入中的尾数与从另一个输入中的指数创建的2的整数幂相乘来构造浮点数。此任务的目标是在 Paddle 框架中，新增 ldexp API，调用路径为：paddle.ldexp 和 Tensor.ldexp。
- 提交内容
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 ldexp API，以支持 Tensor.ldexp 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 ldexp 函数的算法原理和适用场景；
  - 熟练掌握 Python。


### No.7：为 Paddle 新增 Unflatten API <a name='task7'></a>

- 任务难度：基础
- 详细描述：将输入Tensor的某一个维度，扩展成多个维度。此任务的目标是在 Paddle 框架中，添加以下调用方式：

​       paddle.unflatten 、Tensor.unflatten以及paddle.nn.Unflatten

- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，分别在 Paddle repo 的 [python/paddle/tensor/manipulation.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/tensor/manipulation.py) 和[python/paddle/nn/layer/common.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/nn/layer/common.py)；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 unflatten API，以支持 Tensor.unflatten 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，以及[docs/api/paddle/nn](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/nn)目录，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 和[docs/api/paddle/nn/Overview_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/Overview_cn.rst)文件中添加 API 介绍。
- 技术要求
  - 熟悉 unflatten 函数的算法原理和适用场景；
  - 熟练掌握 Python。

### No.8：为 Paddle 新增 xlogy API <a name='task8'></a>

- 任务难度：基础
- 详细描述：使用分段函数计算 input * log(other)。此任务的目标是在 Paddle 框架中，新增 xlogy API，调用路径为：paddle.xlogy 和 Tensor.xlogy。
- 提交内容
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 xlogy API，以支持 Tensor.xlogy 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和[docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 及文件中添加 API 介绍。
- 技术要求
  - 熟悉 xlogy 函数的算法原理和适用场景；
  - 熟练掌握 Python。

### No.9：为 Paddle 新增 pca_lowrank API <a name='task9'></a>

- 任务难度：基础
- 详细描述：对低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析 ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis))。此任务的目标是在 Paddle 框架中，新增 pca_lowrank  API，调用路径为：paddle.pca_lowrank 和 paddle.Tensor.pca_lowrank 。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 pca_lowrank  API，以支持 paddle.Tensor.pca_lowrank 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 PCA 原理和适用场景；
  - 熟练掌握 Python、C++。


### No.10：为 Paddle 新增 copysign API <a name='task10'></a>

- 任务难度：基础
- 详细描述：根据两个输入逐元素地计算结果张量，其结果由第一个输入的绝对值大小及第二个输入的符号组成。此任务的目标是在 Paddle 框架中，新增 copysign API，调用路径为：paddle.copysign 和 Tensor.copysign。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.4/python/paddle/tensor/math.py)文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 copysign  API，以支持 paddle.Tensor.copysign 的调用方式；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 copysign 函数的计算原理和适用场景；
  - 熟练掌握 Python。

### No.11：为 Paddle 新增 Geometric API <a name='task11'></a>

- 任务难度：基础

  详细描述：新增 paddle.distribution.Geometric，用于 Geometric 分布的概率统计与随机采样，至少包括如下方法：

  - `mean`计算均值；
  - `variance`计算方差 ；
  - `sample`随机采样；
  - `rsample` 重参数化采样；
  - `prob` 概率密度；
  - `log_prob`对数概率密度；
  - `entropy`  熵计算；
  -  kl散度计算(python/paddle/distribution/kl.py)

  上述方法可能无法全部支持，需要设计中说明不支持原因，抛出`NotImplementedError`异常即可。

  类签名及各个方法签名，请通过调研 Paddle 及业界实现惯例进行决策。要求代码风格及设计思路与已有概率分布保持一致，参考 [python/paddle/distribution/beta.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/beta.py)。

  【提交内容】

  - API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/distribution](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distribution)目录
  - 单测代码，在 Paddle repo 的 python/paddle/fluid/tests/unittests/distribution 目录
  - 中文API文档，在 docs repo 的 [docs/api/paddle/distribution](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/distribution) 目录

  【技术要求】

  - 熟悉概率分布基本原理；
  - 熟练掌握Python；
  - 能通过阅读源码，了解Paddle现有概率分布设计与实现原理，并基于已有设计，扩展新的概率分布，参考 [python/paddle/distribution](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distribution)。


### No.12：为 Paddle 新增 Cauchy API <a name='task12'></a>

- 任务难度：基础

  详细描述：新增 paddle.distribution.Cauchy，用于 Cauchy 分布的概率统计与随机采样，至少包括如下方法：

  - `mean`计算均值；
  - `variance`计算方差 ；
  - `sample`随机采样；
  - `rsample` 重参数化采样；
  - `prob` 概率密度；
  - `log_prob`对数概率密度；
  - `entropy`  熵计算；
  -  kl散度计算 (python/paddle/distribution/kl.py)

  上述方法可能无法全部支持，需要设计中说明不支持原因，抛出`NotImplementedError`异常即可。

  类签名及各个方法签名，请通过调研 Paddle 及业界实现惯例进行决策。要求代码风格及设计思路与已有概率分布保持一致，参考 [python/paddle/distribution/beta.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/beta.py)。

  【提交内容】

  - API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/distribution](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distribution)目录
  - 单测代码，在 Paddle repo 的 python/paddle/fluid/tests/unittests/distribution 目录
  - 中文API文档，在 docs repo 的 [docs/api/paddle/distribution](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/distribution) 目录

  【技术要求】

  - 熟悉概率分布基本原理；
  - 熟练掌握Python；
  - 能通过阅读源码，了解Paddle现有概率分布设计与实现原理，并基于已有设计，扩展新的概率分布，参考 [python/paddle/distribution](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distribution)。


### No.13：为 Paddle 新增 Bernoulli API <a name='task13'></a>

- 任务难度：基础

  详细描述：新增 paddle.distribution.Bernoulli，用于 Bernoulli 分布的概率统计与随机采样，至少包括如下方法：

  - `mean`计算均值；
  - `variance`计算方差 ；
  - `sample`随机采样；
  - `rsample` 重参数化采样；
  - `prob` 概率密度；
  - `log_prob`对数概率密度；
  - `entropy`  熵计算；
  -  kl散度计算 (python/paddle/distribution/kl.py)

  上述方法可能无法全部支持，需要设计中说明不支持原因，抛出`NotImplementedError`异常即可。

  类签名及各个方法签名，请通过调研 Paddle 及业界实现惯例进行决策。要求代码风格及设计思路与已有概率分布保持一致，参考 [python/paddle/distribution/beta.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/beta.py)。

  【提交内容】

  - API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/distribution](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distribution)目录
  - 单测代码，在 Paddle repo 的 python/paddle/fluid/tests/unittests/distribution 目录
  - 中文API文档，在 docs repo 的 [docs/api/paddle/distribution](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/distribution) 目录

  【技术要求】

  - 熟悉概率分布基本原理；
  - 熟练掌握Python；
  - 能通过阅读源码，了解Paddle现有概率分布设计与实现原理，并基于已有设计，扩展新的概率分布，参考 [python/paddle/distribution](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distribution)。


### No.14：为 Paddle 新增 polar  API <a name='task14'></a>

- 任务难度：基础
- 详细描述：通过输入模和相位角，elementwise 构造复数 tensor。可以参考 paddle.complex 函数（通过输入实部和虚部构造复数 tensor)。
- 提交内容
  - API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录。
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/creation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py) 文件；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 polar 函数的算法原理和适用场景，对复数有一定了解；
  - 熟练掌握 Python。

### No.15：为 Paddle 新增 GaussianNLLLoss API <a name='task15'></a>

- 任务难度：基础
- 详细描述：GaussianNLLLoss是指真实标签服从高斯分布的负对数似然损失，神经网络的输出作为高斯分布的均值和方差。此任务的目标是在 Paddle 框架中，新增 GaussianNLLLoss API，调用路径为：paddle.nn.GaussianNLLLoss 和 paddle.nn.functional.gaussian_nll_loss
- 提交内容
  - API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/nn/layer/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/loss.py) 文件中增加GaussianNLLLoss类，并在 [python/paddle/nn/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/__init__.py) 中，添加 GaussianNLLLoss ，以支持 paddle.nn.GaussianNLLLoss 的调用方式；在[python/paddle/nn/functional/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/loss.py) 中增加gaussian_nll_loss计算函数，并在[python/paddle/nn/functional/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/__init__.py) 中添加gaussian_nll_loss，以支持 paddle.nn.functional.gaussian_nll_loss的调用方式。
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/nn/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 和  [docs/api/paddle/nn/functional](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/nn/functional) 新增对应中文API文档
- 技术要求
  - 熟悉 GaussianNLLLoss 原理和使用场景；
  - 熟练掌握 Python 以及 Tensor操作；

### No.16：为 Paddle 新增 PoissonNLLLoss API <a name='task16'></a>

- 任务难度：基础
- 详细描述：PoissonNLLLoss是指真实标签服从泊松分布的负对数似然损失，神经网络的输出作为泊松分布的参数λ。此任务的目标是在 Paddle 框架中，新增 PoissonNLLLoss API，调用路径为：paddle.nn.PoissonNLLLoss 和 paddle.nn.functional.poisson_nll_loss
- 提交内容
  - API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/nn/layer/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/loss.py) 文件中增加PoissonNLLLoss类，并在 [python/paddle/nn/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/__init__.py) 中，添加 PoissonNLLLoss ，以支持 paddle.nn.PoissonNLLLoss 的调用方式；在[python/paddle/nn/functional/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/loss.py) 中增加对应poisson_nll_loss计算函数，并在[python/paddle/nn/functional/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/__init__.py) 中添加poisson_nll_loss，以支持 paddle.nn.functional.poisson_nll_loss的调用方式。
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/nn/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 和  [docs/api/paddle/nn/functional](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/nn/functional) 新增对应中文API文档
- 技术要求
  - 熟悉 PoissonNLLLoss 原理和使用场景；
  - 熟练掌握 Python 以及 Tensor操作；

### No.17：为 Paddle 新增 cummax / cummin API <a name='task17'></a>

- 任务难度：进阶
- 详细描述：cummax/cummin API 是一个按轴寻找累计最大值/最小值所在位置的 API。此任务的目标是在 Paddle 框架中，新增 cummax/cummin API，调用路径为：paddle.cummax / paddle.cummin 和 paddle.Tensor.cummax / paddle.Tensor.cummin，设计文档可参考 [api_design_for_cummax.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220316_api_design_for_cummax.md)。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；
  - C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录；
  - yaml 文件，cummax / cummin前反向分别添加到 [paddle/phi/api/yaml/ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/ops.yaml)、[paddle/phi/api/yaml/backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/backward.yaml) 文件中；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉 cummax/cummin 计算原理；
  - 熟练掌握 C++、CUDA、Python。


### No.18：为 Paddle 新增 matrix_exp API <a name='task18'></a>

- 任务难度：进阶
- 详细描述：方阵的指数函数，类似 exp 函数（注意与 elementwise 的 exp 函数的区别）。设*X*为*n*×*n*的[实数](https://zh.wikipedia.org/wiki/实数)或[复数](https://zh.wikipedia.org/wiki/复数_(数学))[矩阵](https://zh.wikipedia.org/wiki/矩阵)。*X*的指数，用*eX*或exp(*X*)来表示，是由以下[幂级数](https://zh.wikipedia.org/wiki/幂级数)所给出的*n*×*n*矩阵：

<img width="267" alt="1c0fd4d8b22a08b8c616e3303a8069c4" src="https://user-images.githubusercontent.com/117967927/223112176-737e65f9-694a-49de-b6c9-8e159ab30d81.png">

- 此任务的目标是在 Paddle 框架中，新增 matrix_exp API，调用路径为：paddle.linalg.matrix_exp，设计文档可参考 [api设计文档模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)。
- 提交内容
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py) 文件；
  - C ++ /CUDA 实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels) 目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉级数，复数微分数学原理；
  - 阅读和分析代码的能力，对 torch (linalg.matrix_exp)，scipy(linalg.expm) 中的对应 API 进行调研分析的能力；
  - 熟练掌握 C++、CUDA、Python。

### No.19：为 Paddle 新增 polygamma API <a name='task19'></a>

- 任务难度：进阶
- 详细描述：对于输入张量，其 digamma 函数的 n 阶导，称为多伽马函数（polygamma）。此任务的目标是在 Paddle 框架中，新增 polygamma API，调用路径为：paddle.polygamma 和 paddle.Tensor.polygamma。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 i0/ i0e API，以支持 Tensor.i0/ i0e 的调用方式；
  - C ++ 前反向kernel实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉polygamma函数计算原理；
  - 熟练掌握 C++、Python。


### No.20：为 Paddle 新增 i0 / i0e API <a name='task20'></a>

- 任务难度：进阶
- 详细描述：根据输入的tensor，计算其每个元素的第一类零阶修正贝塞尔函数（对应api：i0）和第一类指数缩放的零阶修正贝塞尔函数（对应api：i0e）（[贝塞尔函数](https://en.wikipedia.org/wiki/Bessel_function)、[修正贝塞尔函数](https://elec424.fandom.com/wiki/Modified_Bessel_Functions)）。此任务的目标是在Paddle 框架中，新增 i0 和 i0e API，调用路径为：paddle.i0 / paddle.i0e 和 paddle.Tensor.i0 / paddle.Tensor.i0e 。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 i0/ i0e API，以支持 Tensor.i0/ i0e 的调用方式；
  - C ++ 实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉贝塞尔函数计算原理；
  - 熟练掌握 C++、Python。


### No.21：为 Paddle 新增 i1/ i1e API <a name='task21'></a>

- 任务难度：进阶

- 详细描述：根据输入的tensor，计算其每个元素的第一类一阶修正贝塞尔函数（对应api：i1）和第一类指数缩放的一阶修正贝塞尔函数（对应api：i1e）（[贝塞尔函数](https://en.wikipedia.org/wiki/Bessel_function)、[修正贝塞尔函数](https://elec424.fandom.com/wiki/Modified_Bessel_Functions)）。此任务的目标是在Paddle 框架中，新增 i1 和 i1e API，调用路径为：paddle.i1 / paddle.i1e 和 paddle.Tensor.i1 / paddle.Tensor.i1e 。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 i1/ i1e API，以支持 Tensor.i1/ i1e 的调用方式；
  - C ++ 实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉贝塞尔函数计算原理；
  - 熟练掌握 C++、Python。


### No.22：为 Paddle 新增 lu_solve API <a name='task22'></a>

- 任务难度：进阶

- 详细描述：使用 LU分解 来求解线性方程组 AX=B，A为方阵，B为矩阵，A和B已知，通过LU分解方阵A来求解X。即 LU, pivots =paddle.linalg.lu(A);  X = paddle.linalg.lu_solve(LU, pivots, B) 与 使用 X=paddle.linalg.lu_solve(A, B) 直接求解线性方程组的结果一样。此任务的目标是在 Paddle 框架中，新增 lu_solve  API，调用路径为：paddle.linalg.lu_solve 和 Tensor.lu_solve
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；并在 [python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274) 中，添加 lu_solve API，以支持 Tensor.lu_solve 的调用方式；
  - C ++ 实现代码，在 Paddle repo 放置。其中头文件在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu) 目录 和 cu文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu) 目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉矩阵 LU 分解原理，了解当前 paddle.linalg.lu 的逻辑；
  - 熟悉 lapack/cublas 库；
  - 熟练掌握 Python。

### No.23：为 Paddle 新增 vander API <a name='task23'></a>

- 任务难度：基础

- 详细描述：根据输入构造 [范德蒙矩阵](https://en.wikipedia.org/wiki/Vandermonde_matrix)（各列为几何级数的矩阵）。此任务的目标是在 Paddle 框架中，新增 vander API，调用路径为：paddle.vander 和 paddle.Tensor.vander。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py) 文件；
  - C ++ 实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求
  - 熟悉范德蒙矩阵计算原理；
  - 熟练掌握 C++、Python。


### No.24：为 Paddle 新增 paddle.sparse.is_nan 稀疏 API <a name='task24'></a>

- 技术标签：深度学习框架，Python，C++，CUDA

- 任务难度：基础

- 详细描述：针对 Paddle 的稀疏 Tensor 格式 COO，需要新增 is_nan 的计算逻辑，一共需要新增 1个 kernel 的前向与反向，其中参数 axis 可支持任意维度，注意只需新增 coo 格式的逻辑，csr 格式的已经实现，此次无需实现。

- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/unary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/unary.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录的 softamx_kernel.h/cc/cu 三个文件中，分别补充 coo 的计算 kernel；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_is_nan.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/test_sparse_is_nan.py) 文件；
  - yaml 文件，前反向分别添加到[python/paddle/utils/code_gen/sparse_api.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/utils/code_gen/sparse_api.yaml)、[python/paddle/utils/code_gen/sparse_bw_api.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/utils/code_gen/sparse_bw_api.yaml) 文件中。
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/incubate/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/incubate/sparse) 目录。

- 技术要求
  - 熟悉稀疏 COO 存储格式，Paddle 的 SparseCooTensor 数据结构；
  - 熟悉稀疏 Tensor 的 is_nan 在 COO 存储格式下的计算逻辑；
  - 熟练掌握 Python、C++、CUDA 代码编写。

### No.25：为 Paddle 新增 paddle.sparse.any 稀疏 API <a name='task25'></a>

- 任务难度：基础
- 详细描述：针对 Paddle 的两种稀疏 Tensor 存储格式 COO 与 CSR，需要新增 any 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor，动静态图都需要支持。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/unary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/unary.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的 [paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_any_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/) 文件；
  - yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。
- 技术要求
  - 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
  - 熟悉稀疏 Tensor的 any 计算逻辑；
  - 熟练掌握 Python、C++、CUDA 代码编写。

### No.26：为 Paddle 新增 paddle.sparse.nn.Softmax 稀疏 API 的 coo 格式计算逻辑 <a name='task26'></a>

- 任务难度：进阶
- 详细描述：针对 Paddle 的稀疏 Tensor 格式 COO，需要新增 softmax 的计算逻辑，一共需要新增 1个 kernel 的前向与反向，其中参数 axis 可支持任意维度，注意只需新增 coo 格式的逻辑，csr 格式的已经实现，此次无需实现。动静态图都需要支持。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/nn/functional/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/functional/activation.py) 文件和 [python/paddle/sparse/nn/layer/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/nn/layer/activation.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录的 softamx_kernel.h/cc/cu 三个文件中，分别补充 coo 的计算 kernel；
  - 单测代码，在 Paddle repo 新当前的 [python/paddle/fluid/tests/unittests/test_sparse_softmax_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/test_sparse_softmax_op.py) 文件中新增测试case；
  - yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。
- 技术要求
  - 熟悉稀疏 COO 存储格式，Paddle 的 SparseCooTensor 数据结构；
  - 熟悉稀疏 Tensor 的 softmax 计算逻辑；
  - 熟练掌握 Python、C++、CUDA 代码编写。


### No.27：为 Paddle 新增 paddle.sparse.concat 稀疏 API <a name='task27'></a>

- 任务难度：进阶
- 详细描述：针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 concat 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。动静态图都需要支持。动静态图都需要支持。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 新建 [python/paddle/sparse/multiary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/multiary.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_concat_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/) 文件；
  - yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。
- 技术要求
  - 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
  - 熟悉稀疏 Tensor的 concat 计算逻辑；
  - 熟练掌握 Python、C++、CUDA 代码编写。

### No.28：为 Paddle 新增 paddle.sparse.index_select 稀疏 API <a name='task28'></a>

- 任务难度：进阶
- 详细描述：针对 Paddle 的两种稀疏 Tensor 存储格式 COO 与 CSR，都需新增 index_select 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的Tensor。动静态图都需要支持。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/binary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/binary.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_index_select_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/) 文件；
  - yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。
- 技术要求
  - 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
  - 熟悉稀疏 Tensor 的 index_select 计算逻辑；
  - 熟练掌握 Python、C++、CUDA 代码编写。

### No.29：为 Paddle 新增 paddle.sparse.slice 稀疏 API <a name='task29'></a>

- 任务难度：进阶
- 详细描述：针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 slice 的计算逻辑，一共需要新增 2个 kernel 的前向与反向。动静态图都需要支持。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/unary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/unary.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录中；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_slice_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/) 文件；
  - yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。
- 技术要求
  - 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
  - 熟悉稀疏 Tensor 的 slice 计算逻辑；
  - 熟练掌握 Python、C++、CUDA 代码编写。

### No.30：为 Paddle 新增 paddle.sparse.sum 稀疏 API <a name='task30'></a>

- 任务难度：进阶
- 详细描述：针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，需新增 sum 的计算逻辑，sum为一种指定维度上累加求和操作。一共需要新增 2个 kernel 的前向与反向，其中 coo 格式的 axis 支持任意维度，csr 格式的 axis 可只支持-1，即按行读取。另外当 axis=None 时所有元素相加。动静态图都需要支持。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/sparse/unary.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/sparse/unary.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_sum_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 文件；
  - yaml 文件，前反向分别添加到 [paddle/phi/api/yaml/sparse_ops.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_ops.yaml)、[paddle/phi/api/yaml/sparse_backward.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/sparse_backward.yaml) 文件中；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/sparse) 目录。
- 技术要求
  - 熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
  - 熟悉稀疏 Tensor 的 sum 计算逻辑；
  - 熟练掌握 Python、C++、CUDA代码编写。


### No.31：部分API发生除0、空指针、堆栈溢出等问题的修复 <a name='task31'></a>

- 任务难度：基础
- 详细描述：
  - 现状：Paddle 中的 API/算子存在鲁棒性不足问题，通过一些简单的 case可以制造出除0、空指针、堆栈溢出等错误，需要进行修复。
  - 除0错误case如下：
     - [Case6:paddle.nn.functional.softmax_with_cross_entropy](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386643789)
     - [Case8:paddle.nn.functional.npair_loss](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386644960)
     - [Case17:paddle.incubate.graph_reindex](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386649810)
     - [Case18:paddle.logsumexp](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386650112)
     - [Case21:paddle.mode](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386651168)
     - [Case26:paddle.all](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386653198)
     - [Case30:DataNormKernel](https://github.com/PaddlePaddle/Paddle/issues/49919#issuecomment-1386654610)
  - 空指针错误 case 如下：
     - [Case4:paddle.incubate.graph_khop_sampler](https://github.com/PaddlePaddle/Paddle/issues/49922#issuecomment-1386662756)
     - [Case5:paddle.incubate.graph_khop_sampler](https://github.com/PaddlePaddle/Paddle/issues/49922#issuecomment-1386663874)
     - [Case7:paddle.distribution.Beta](https://github.com/PaddlePaddle/Paddle/issues/49922#issuecomment-1386664355)
  - 堆栈溢出错误 case 如下：
     - [Case12:paddle.transpose](https://github.com/PaddlePaddle/Paddle/issues/49925#issuecomment-1386710704)
- 任务提交：
  - 文档参考：[Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/49927) 中提供了修复鲁棒性不足的PR示例。
- 技术要求
  - 了解 Paddle API 及算子含义
  - 熟悉 Python 和 C++ 开发



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 合入标准

-  按 [API 设计规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) 完成 API设计文档；
- 按 [API 验收标准](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) 完成 API功能实现、单测、API文档；
- 稀疏 API 任务需符合稀疏 OP 的特殊开发规范（如有）：
  * 【yaml规则】：写到同一个yaml api，不要写多个，yaml需支持调度
  * 【kernel名规则】：[计算名] + 异构后缀，例如 matmul_csr_dense、softmax_csr、softmax_coo
  * 【文件名规则】：sparse/xx_kernel.cc，sparse/xx_kernel.cu，以目录区分，文件名与dense保持一致

### 参考内容

- [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)
- [新增 API 设计模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)
- [飞桨API Python 端开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html)
- [C++ 算子开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
- [飞桨API文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
- [API单测开发及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)


### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&微信群的通知，及时参与。
