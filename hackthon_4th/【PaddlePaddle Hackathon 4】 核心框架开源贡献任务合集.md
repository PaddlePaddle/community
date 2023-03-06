# 【PaddlePaddle Hackathon 4】核心框架开源贡献任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/50629)）

注：为飞桨框架新增一系列 API，提交流程请参考 [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项在任务列表后：

### No.1：为 Paddle 新增 finfo API <a name='task1'></a>

- 任务难度**：**基础
- 详细描述**：**finfo计算浮点数类型的数值限制，输入参数为 Paddle 浮点数类型(paddle.float16/paddle.float32/paddle.float64/paddle.complex64/paddle.complex128)，返回包含下表属性对象。此任务目标是为 Paddle 新增 finfo API，调用路径为 paddle.finfo。要求通过 pybind 方式直接将 C++ 层实现绑定到 Python，无需开发 Paddle Kernel，可以参考 [paddle/fluid/pybind/pybind.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/pybind.cc) 中代码。更详细内容可以参考 [numpy.finfo](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html) 。API 设计文档可参考 [api_design_for_finfo.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220330_api_design_for_finfo.md)。
- 提交内容**：**
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 目录；
  - C ++ 实现代码，在 Paddle repo 的 [paddle/fluid/pybind](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/pybind) 目录；
  - 单测代码，在 Paddle repo 的 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录；
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录和 [docs/api/paddle/Tensor_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Tensor_cn.rst) 文件，同时需要在 [docs/api/paddle/Overview_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/Overview_cn.rst) 文件中添加 API 介绍。
- 技术要求**：**
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


### No.13：为 Paddle 新增 LogNormal API <a name='task13'></a>

- 任务难度：基础

  详细描述：新增 paddle.distribution.LogNormal，用于 LogNormal 分布的概率统计与随机采样，至少包括如下方法：

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

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6d943ac21929411583796230ef58e4a2&docGuid=NXFlS7Ad4WBi83)

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

- 任务难度：进阶

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

- 任务难度：进阶

- 详细描述：针对 Paddle 的稀疏 Tensor 格式 COO，需要新增 softmax 的计算逻辑，一共需要新增 1个 kernel 的前向与反向，其中参数 axis 可支持任意维度，注意只需新增 coo 格式的逻辑，csr 格式的已经实现，此次无需实现。

- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs)  目录；
  - Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/incubate/sparse/nn/functional/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/incubate/sparse/nn/functional/activation.py) 文件和 [python/paddle/incubate/sparse/nn/layer/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/incubate/sparse/nn/layer/activation.py) 文件；
  - C++ kernel 实现代码，在Paddle repo 的[paddle/phi/kernels/sparse/](https://github.com/PaddlePaddle/Paddle/blob/develop//paddle/phi/kernels/sparse) 目录的 softamx_kernel.h/cc/cu 三个文件中，分别补充 coo 的计算 kernel；
  - 单测代码，在 Paddle repo 新建 [python/paddle/fluid/tests/unittests/test_sparse_softmax_op.py](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests/test_sparse_softmax_op.py) 文件；
  - yaml 文件，前反向分别添加到[python/paddle/utils/code_gen/sparse_api.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/utils/code_gen/sparse_api.yaml)、[python/paddle/utils/code_gen/sparse_bw_api.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/utils/code_gen/sparse_bw_api.yaml) 文件中。
  - 中文 API 文档，在 docs repo 的 [docs/api/paddle/incubate/sparse](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/incubate/sparse) 目录。

- 技术要求
  - 熟悉稀疏 COO 存储格式，Paddle 的 SparseCooTensor 数据结构；
  - 熟悉稀疏 Tensor 的 softmax 在 COO 存储格式下的计算逻辑；
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

### No.32：为 Paddle 优化 expand_as 前向&反向 op 在 GPU 上的计算性能 <a name='task32'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 expand_as 前向和反向算子的 GPU 实现采用 Eigen 组合的模式，缺少 GPU Kernel，性能相对不足；
  - 目标：请实现高性能的 GPU 计算 Kernel，为 Paddle 优化 expand_as op 在 GPU 上的计算性能，性能至少提升6倍，对性能极差的 case 提升达700倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/expand_as_grad_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/expand_as_grad_kernel_impl.h)，[paddle/phi/kernels/impl/expand_as_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/expand_as_kernel_impl.h) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.33：为 Paddle 优化 Histogram 在GPU上的性能 <a name='task33'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 Histogram 算子的计算调用了eigen，kernel实现中存在原子操作。GPU kernel的性能有待提升；
  - 目标：请优化计算实现，为 Paddle 优化 Histogram OP 在 GPU 上的计算性能，性能平均提升2x。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf) 目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/histogram_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/histogram_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.34：为 Paddle 优化 Lerp OP在GPU上的性能 <a name='task34'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 lerp 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 lerp op 在 GPU 上的计算性能，性能至少提升20%，针对性能差的case，性能至少提升4+倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/lerp_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/lerp_kernel_impl.h) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)

### No.35：为 Paddle 优化 Prelu OP在GPU上的性能 <a name='task35'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 lerp 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 lerp op 在 GPU 上的计算性能，性能至少提升20%，针对性能差的case，性能至少提升4+倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/prelu_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/prelu_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.36：为 Paddle 优化 Tile OP在GPU上的性能 <a name='task36'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 Tile 算子的 GPU 计算和CPU采用相同的逻辑，GPU性能仍有提升空间；
  - 目标：请优化计算实现，为 Paddle 优化 Tile op 在 GPU 上的计算性能，性能平均提升70%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf) 目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/tile_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/tile_kernel_impl.h)目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.37：为 Paddle 优化 matrix_rank op 在 GPU 上的计算性能 <a name='task37'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 matrix_rank 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 matrix_rank op 在 GPU 上的计算性能，性能至少提升3倍，针对性能差的case，性能提升120+倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/matrix_rank_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/matrix_rank_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.38：为 Paddle 优化 p_norm op 在 GPU 上的计算性能 <a name='task38'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 p_norm 算子 Reduce_Any Kernel 的性能较竞品存在差异；
  - 目标：请优化计算实现，为 Paddle 优化 p_norm op 在 GPU 上的计算性能，性能平均提升2.5倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/p_norm_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/p_norm_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.39：为 Paddle 优化 p_norm_grad op 在 GPU 上的计算性能 <a name='task39'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 p_norm_grad 算子 GPU 计算采用了 CUDA Kernel 与 Eigen 混合的模式，用现有的 Reduce OP 等取代 Eigen 可以提升计算性能，减少数据 HtoD 拷贝等开销；
  - 目标：请优化计算实现，为 Paddle 优化 p_norm_grad op 在 GPU 上的计算性能，性能平均提升3倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/p_norm_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/p_norm_grad_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.40：为 Paddle 优化 kthvalue op 在 GPU 上的计算性能 <a name='task40'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 kthvalue 算子 GPU 计算采用了cub库实现，性能仍有不足；
  - 目标：请优化计算实现，为 Paddle 优化 kthvalue op 在 GPU 上的计算性能，性能平均提升2.7倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/kthvalue_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/kthvalue_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.41：为 Paddle 优化 cumprod_grad op 在 GPU 上的计算性能 <a name='task41'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 cumprod_grad 算子 GPU 计算采用了采用GPU Kernel等拼接实现，性能仍有提升空间；
  - 目标：请优化计算实现，为 Paddle 优化 cumprod_grad op 在 GPU 上的计算性能，性能平均提升30%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/cumprod_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/cumprod_grad_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.42：为 Paddle 优化 FminGrad OP在GPU上的性能 <a name='task42'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内  FminGrad 在 GPU 的计算性能有待提升；
  - 目标：请优化计算实现，为 Paddle 优化  FminGrad OP 在 GPU 上的计算性能，性能平均提升40%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.43：为 Paddle 优化 GeluGrad OP在GPU上的性能 <a name='task43'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内  GeluGrad OP 在GPU的计算性能有待提升；
  - 目标：请优化计算实现，为 Paddle 优化  GeluGrad 在 GPU 上的计算性能，性能平均提升60%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/gelu_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/gelu_grad_kernel.cu)目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.44：为 Paddle 优化 logsumexp OP在GPU上的性能 <a name='task44'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内  logsumexp OP 的 GPU 计算调用了 eigen，性能较差，有较大的提升空间；
  - 目标：请优化计算实现，为 Paddle 优化 logsumexp OP 在 GPU 上的计算，性能平均提升10x。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/logsumexp_grad_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/logsumexp_grad_kernel_impl.h)目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.45：为 Paddle logical 算子实现 float16 数据类型支持 <a name='task45'></a>

- 技术标签：深度学习框架，C++
- 任务难度：基础
- 详细描述：logical 类的算子未支持 float16 类型，因此该功能要求为这类算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。本任务共包含 4 个具体算子：logical_and，logical_or，logical_xor，logical_not
- 任务提交：
  - C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/kps/logical_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/kps/logical_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_logical_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_logical_op.py)；
  - API文档中增加数据类型支持说明，以logical_and为例英文文档在[python/paddle/tensor/logic.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/logic.py)，中文文档在[docs/api/paddle/logical_and_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/logical_and_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.46：为 Paddle gumbel_softmax 算子实现 float16 数据类型支持 <a name='task46'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：gumbel_softmax 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/gumbel_softmax_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/gumbel_softmax_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/gumbel_softmax_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/gumbel_softmax_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_gumbel_softmax_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_gumbel_softmax_op.py)；
  - API文档中增加数据类型支持说明，英文文档在[python/paddle/nn/functional/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/activation.py)，中文文档在[docs/api/paddle/nn/functional/gumbel_softmax_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/functional/gumbel_softmax_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.47：为 Paddle cross 算子实现 float16 数据类型支持 <a name='task47'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：cross 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/cross_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/cross_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/cross_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/cross_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_cross_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_cross_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在 [python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py)，中文文档在 [docs/api/paddle/cross_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/cross_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.48：为 Paddle assign_value、meshgrid、kthvalue、determinant 算子实现 float16 数据类型支持 <a name='task48'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：assign_value 前向，meshgrid、kthvalue、determinant 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/assign_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/assign_kernel.cc)、[paddle/phi/kernels/gpu/meshgrid_kernel.cu.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/meshgrid_kernel.cu.cc)、[paddle/phi/kernels/gpu/kthvalue_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/kthvalue_kernel.cu)、[paddle/phi/kernels/gpu/determinant_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/determinant_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/meshgrid_grad_kernel.cu.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/meshgrid_grad_kernel.cu.cc)、[paddle/phi/kernels/gpu/kthvalue_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/kthvalue_grad_kernel.cu)、[paddle/phi/kernels/gpu/determinant_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/determinant_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_assign_value_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_assign_value_op.py)、[python/paddle/fluid/tests/unittests/test_meshgrid_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_meshgrid_op.py)、[python/paddle/fluid/tests/unittests/test_kthvalue_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_kthvalue_op.py)、[python/paddle/fluid/tests/unittests/test_determinant_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_determinant_op.py)；
  - API文档中增加数据类型支持说明，英文文档在[python/paddle/tensor/creation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py)、[python/paddle/tensor/search.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/search.py)、[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py)，中文文档在[docs/api/paddle/assign_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/assign_cn.rst)、[docs/api/paddle/meshgrid_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/meshgrid_cn.rst)、[docs/api/paddle/kthvalue_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/kthvalue_cn.rst)、[docs/api/paddle/linalg/det_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/linalg/det_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.49：为 Paddle bce_loss 算子实现 float16 数据类型支持 <a name='task49'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：bce_loss 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/bce_loss_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/bce_loss_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/bce_loss_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/bce_loss_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_bce_loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_bce_loss.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/nn/layer/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/loss.py)，中文文档在[docs/api/paddle/nn/BCELoss_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/BCELoss_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.50：为 Paddle lerp 算子实现 float16 数据类型支持 <a name='task50'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：lerp 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/lerp_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/lerp_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/lerp_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/lerp_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_lerp_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_lerp_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)，中文文档在 [docs/api/paddle/lerp_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/lerp_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.51：为 Paddle maxout 算子实现 float16 数据类型支持 <a name='task51'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：maxout 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/maxout_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/maxout_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/maxout_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/maxout_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_maxout_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_maxout_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/nn/functional/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/activation.py)，中文文档在[docs/api/paddle/nn/functional/maxout_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/functional/maxout_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.52：为 Paddle dist 算子实现 float16 数据类型支持 <a name='task52'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：dist 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/dist_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/dist_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/dist_grad_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/dist_grad_kernel.cc)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_dist_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_dist_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py)，中文文档在[docs/api/paddle/dist_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/dist_cn.rst)； 
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.53：为 Paddle label_smooth 算子实现 float16 数据类型支持 <a name='task53'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：label_smooth 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/label_smooth_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/label_smooth_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/label_smooth_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/label_smooth_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_label_smooth_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_label_smooth_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/nn/functional/common.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/python/paddle/nn/functional/common.py)，中文文档在[docs/api/paddle/nn/functional/label_smooth_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/functional/label_smooth_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.54：为 Paddle allclose、isclose 算子实现 float16 数据类型支持 <a name='task54'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：allclose、isclose 算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/allclose_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/allclose_kernel.cu)、[paddle/phi/kernels/gpu/isclose_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/isclose_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_allclose_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_allclose_op.py)、[python/paddle/fluid/tests/unittests/test_isclose_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/python/paddle/fluid/tests/unittests/test_isclose_op.py)；
  - API文档中增加数据类型支持说明，英文文档在[python/paddle/tensor/logic.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/logic.py)，中文文档在[docs/api/paddle/allclose_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/allclose_cn.rst)、[docs/api/paddle/isclose_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/isclose_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.55：channel_shuffle 等算子FP16/BF16算子及单测完善 <a name='task55'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 argsort、channel_shuffle、pixel_unshuffle、pixel_shuffle、fmax、fmin、erf、erfinv 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在**任务表单**中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单：
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.56：set_value 等算子 FP16/BF16算子及单测完善 <a name='task56'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 fill、fill_diagonal_tensor、diag、diagonal、bernoulli、poisson、trunc、searchsorted 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - - -
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.57：gaussian 等算子 FP16/BF16算子及单测完善 <a name='task57'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 gaussian、cross、dot、conv3d、conv3d_transpose、max_pool2d_with_index、max_pool3d_with_index 、flip算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.58：linear_interp 等算子 FP16/BF16算子及单测完善 <a name='task58'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 logcumsumexp、logsumexp、empty、empty_like、kthvalue、exponential 、atan2、set_value、pad算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.59：addmm 等算子 FP16/BF16算子及单测完善 <a name='task59'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 addmm、bmm、angle、put_along_axis、take_along_axis、index_sample、index_add、hardtanh 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单：
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.60：angle 等算子 FP16/BF16算子及单测完善 <a name='task60'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 prelu、multinomial、multi_dot、overlap_add、clip_by_norm、randperm、sign、split、split_with_num 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单：
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.61：unfold 等算子 FP16/BF16算子及单测完善 <a name='task61'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 unfold、min、mode、logsigmoid、uniform_inplace、segment_pool、update_loss_scaling、remainder 算子的FP16算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.62：masked_select 等算子 FP16/BF16算子及单测完善 <a name='task62'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 masked_select、dirichlet、lgamma、dropout_nd、digamma、margin_cross_entropy、broadcast_tensors 、pool3d、transfer_layout算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.63：complex 等算子 FP16/BF16算子及单测完善 <a name='task63'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 complex、eye、lerp、frame、embedding、nanmedian、temporal_shift、conj 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单：
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.64：trace 等算子 FP16/BF16算子及单测完善 <a name='task64'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 trace、elementwise_heaviside、huber_loss、logspace、full_batch_size_like、unbind、einsum、matmul_with_flatten、trace 算子的FP16算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性。
- 任务提交：
  - 任务表单：
    - [低精度算子及单测完善任务表单](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/iFa3RSPAew3RIQ)
  - 文档参考：
    - [低精度算子支持开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/usBqLmAfWjVloW)
    - [低精度算子单测开发规范](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/35xykUZW3L/jWsmHAi730t-_V)
  - 提交代码至 Paddle 代码仓库

### No.65：为 Paddle matmul_with_flatten/matmul 前向算子实现 int8 数据类型支持 <a name='task65'></a>

- 技术标签：深度学习框架，C++，CUDA, 量化
- 任务难度：基础
- 详细描述：为了更好的支持量化训练和原生量化推理，需要对 matmul 和matmul_with_flatten 算子添加 int8 数据类型的支持。推荐使用框架内接口调用 cublasLt 作为 int8 矩阵乘的计算后端并根据实际情况适配，也可使用其他方式实现。要求精度无误，算子性能在常见 case 下快于 FP16。
- 提交内容：
  - 修改前向算子注册: paddle/phi/kernels/gpu/matmul_kernel.cu
  - 修改前向算子GPU kernel：paddle/phi/kernels/impl/matmul_kernel_impl.h
  - 修改 cublasLt int8 计算后端：paddle/fluid/operators/fused/cublaslt.h
  - 单元测试脚本中增加测试样例：
    - python/paddle/fluid/tests/unittests/test_mul_op.py
    - python/paddle/fluid/tests/unittests/test_matmul_op.py
    - python/paddle/fluid/tests/unittests/test_matmul_v2_op.py 
  - 提交PR，在PR描述中说明做法并附上FP16/32的性能对比
- 技术要求：
  - 熟练掌握 Python、C++、CUDA 代码编写
  - 了解量化原理和量化推理执行流程

### No.66：为Paddle FC 前向算子实现量化计算 <a name='task66'></a>

- 技术标签：深度学习框架，C++，CUDA, 量化
- 任务难度：基础
- 详细描述：为了更好的支持原生量化推理，需要对FC算子添加量化支持。算子的 Input/Bias/Out 类型保持不变，要求W支持int8输入；需要额外添加量化kernel，反量化 kernel，和 int8 类型的矩阵乘运算；要求为OP添加量化开关作为算子属性，同时添加属性 max_bound/min_bound, round_type；算子已有quant_in_scale 属性，可以直接使用该属性作为量化输入的参数(layer_wise)，添加 weight_scale(channel_wise) 作为输入Tensor用于计算输出的反量化参数。INT8矩阵乘推荐使用框架内接口调用 cublasLt 作为int8矩阵乘的计算后端并根据实际情况适配，也可使用其他方式实现。要求精度无误，算子性能在常见 case 下快于 FP16。
- 提交内容：
  - 修改前向算子声明: paddle/fluid/operators/fc_op.cc
  - 修改前向算子GPU kernel：paddle/fluid/operators/fc_op.h
  - 修改cublasLt int8计算后端：paddle/fluid/operators/fused/cublaslt.h
  - 单元测试脚本中增加测试样例：python/paddle/fluid/tests/unittests/test_fc_op.py
  - 提交PR，在PR描述中说明做法并附上 FP16/32 的性能对比
- 技术要求：
  - 熟练掌握 Python、C++、CUDA 代码编写
  - 熟悉量化原理和量化推理执行流程
  - 了解 Paddle 算子体系

### No.67：解耦 PHI 算子库对 operator.h 头文件的依赖 <a name='task67'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/framework/operator.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/framework/operator.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。
- 任务提交：
  - 文档参考：可参考该 [Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/47615) ,这个里边提供了PHI算子库独立编译相关文档及头文件解耦依赖PR示例。
  - 解耦头文件代码 PR，提交目录至：[paddle/phi](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi) 和 [paddle/fluid](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid)
  - 时间：3月6日前完成 PR 提交，3月13日前完成 PR 合入
- 技术要求
  - 了解 Paddle PHI 算子库
  - 熟悉使用 C++ 开发及 cmake 编译

### No.68：解耦 PHI 算子库对 utils.h 头文件的依赖 <a name='task68'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/operators/utils.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/operators/utils.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。
- 任务提交：
  - 文档参考：可参考该 [Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/47615) ,这个里边提供了PHI算子库独立编译相关文档及头文件解耦依赖PR示例。
  - 解耦头文件代码 PR，提交目录至：[paddle/phi](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi) 和 [paddle/fluid](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid)
  - 时间：3月6日前完成 PR 提交，3月13日前完成 PR 合入
- 技术要求
  - 了解 Paddle PHI 算子库
  - 熟悉使用 C++ 开发及 cmake 编译

### No.69：解耦 PHI 算子库对 device_wrapper.h 头文件的依赖 <a name='task69'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/platform/device/device_wrapper.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/platform/device/device_wrapper.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。
- 任务提交：
  - 文档参考：可参考该 [Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/47615) ,这个里边提供了PHI算子库独立编译相关文档及头文件解耦依赖PR示例。
  - 解耦头文件代码 PR，提交目录至：[paddle/phi](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi) 和 [paddle/fluid](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid)
  - 时间：3月6日前完成 PR 提交，3月13日前完成 PR 合入
- 技术要求
  - 了解 Paddle PHI 算子库
  - 熟悉使用 C++ 开发及 cmake 编译

### No.70：解耦 PHI 算子库对 kernels.h 头文件的依赖 <a name='task70'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/operators/jit/kernels.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/operators/jit/kernels.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。
- 任务提交：
  - 文档参考：可参考该 [Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/47615) ,这个里边提供了PHI算子库独立编译相关文档及头文件解耦依赖PR示例。
  - 解耦头文件代码 PR，提交目录至：[paddle/phi](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi) 和 [paddle/fluid](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid)
  - 时间：3月6日前完成 PR 提交，3月13日前完成 PR 合入
- 技术要求
  - 了解 Paddle PHI 算子库
  - 熟悉使用 C++ 开发及 cmake 编译

### No.71：为 Paddle-TRT 添加 pad3d 算子 <a name='task71'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 pad3d 算子 TRT Layer映射实现，为了让含有 pad3d 的算子以全图形式执行 TensorRT engine，需添加该算子实现。
  - 目标：
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 pad3d TRT算子映射，并提交 PR
  - 任务要求：
    - 完成pad3d功能实现代码
    - 单测python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_pad3d.py 验证通过
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.72：为 Paddle-TRT 添加 flip 算子 <a name='task72'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 flip 算子 TRT Layer 映射实现。
  - 目标：完成 flip 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 flip TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 flip 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.73：为 Paddle-TRT 添加 temporal_shift 算子 <a name='task73'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏temporal_shift 算子 TRT Layer 映射实现。
  - 目标：完成 temporal_shift 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 temporal_shift TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 temporal_shift 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.74：为 Paddle-TRT 添加 grid_sampler 算子 <a name='task74'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 grid_sampler 算子 TRT Layer 映射实现，TRT 8.5 已提供 IGridSampleLayer 实现，基于该 Layer 完成 OP 映射工作。
  - 目标：完成 grid_sampler 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 grid_sampler TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 temporal_shift 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.75：为 Paddle-TRT 添加 expand_as_v2 算子 <a name='task75'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 expand_as_v2 算子 TRT Layer 映射实现。
  - 目标：完成 expand_as_v2 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 expand_as_v2 TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 expand_as_v2 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.76：为 Paddle-TRT 添加elementwise_mod 算子 <a name='task76'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 elementwise_mod 算子 TRT Layer 映射实现。
  - 目标：完成 elementwise_mod 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 elementwise_mod TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 elementwise_mod 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.77：为 Paddle-TRT 添加 inverse 算子 <a name='task77'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 inverse 算子 TRT 映射实现，基于 [通用plugin方案](https://github.com/PaddlePaddle/Paddle/pull/45355) 完成TRT映射。
  - 目标：完成 inverse 算子 TRT Layer 映射
  - 通用 plugin PR 参考示例见[PR47003](https://github.com/PaddlePaddle/Paddle/pull/47003)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 inverse TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 inverse 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.78：为 Paddle-TRT 添加 cumsum 算子 <a name='task78'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：进阶
- 详细描述：
  - 背景：Paddle-TRT 缺乏 cumsum 算子 TRT Layer 映射实现。
  - 目标：完成 cumsum 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 cumsum TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 cumsum 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.79：为 Paddle-TRT 添加 while 算子 <a name='task79'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：进阶
- 详细描述：
  - 背景：Paddle-TRT 缺乏 while 算子 TRT Layer 映射实现。
  - 目标：完成 while 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 while TRT 算子映射，并提交 PR
  - 任务要求：
    - 完成 while 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉TensorRT，熟悉推理优化

### No.80：为 Paddle-TRT 添加 conditional_block 算子 <a name='task80'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：进阶
- 详细描述：
  - 背景：Paddle-TRT 缺乏 conditional_block 算子TRT Layer 映射实现。
  - 目标：完成 conditional_block 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 conditional_block TRT算子映射，并提交 PR
  - 任务要求：
    - 完成 conditional_block 功能实现代码
    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.81：为神经网络编译器 CINN 增加 bitcast_convert 算子 <a name='task81'></a>

- 任务难度：基础
- 详细描述：
  - 现状：在对接上层框架时，编译器会将上层的框架复杂算子进一步拆分为若干基础算子便于编译器执行算子融合，同时减少开发成本。
  - 目标：通过 extern call 的方式，实现 bitcast_convert 算子。在不改变底层存储的情况下，强制转换数据类型。若转换前后数据类型的字节大小不相同，则形状会改变。比如一个 shape=[10] 的 float32 类型数据被强制转换为 float16 类型后，其 shape 应为[10, 2]。
- 任务提交：
  - 参考示例 [PR1018](https://github.com/PaddlePaddle/CINN/pull/1018)。
  - 算子接口应有一个输入，一个 string 类型属性指定要转换的类型，以及一个输出。
- 技术要求：
  - 了解 AI 编译器 Compute 和 Schedule 含义
  - 熟练使用 C++ 开发以及 cmake 编译

### No.82：为神经网络编译器 CINN 增加 triangular_solve 算子 <a name='task82'></a>

- 任务难道：基础
- 详细描述：
  - 现状：在对接上层框架时，编译器会将上层的框架复杂算子进一步拆分为若干基础算子便于编译器执行算子融合，同时减少开发成本。
  - 目标：通过 custom call 的方式，实现 triangular_solve 算子。triangular_solve 算子用于计算具有唯一解的线性方程组，其中系数方阵A为上（下）三角系数矩阵。若系数方阵A不可逆，则线性方程不可解。
- 任务提交：
  - 参考示例 [#1133](https://github.com/PaddlePaddle/CINN/pull/1133)。当前仅需考虑CUDA环境，可调用 cuslover 库。
  - 算子接口应有两个输入，分别指定矩阵A（形状[*, M, M]）和B（形状[*, M, K]），两矩阵batch维度大小相同。四个 bool 类型的属性，left_side 用于指定系数方阵是位于求解矩阵的左侧还是右侧，upper 指定对系数方阵取上三角还是下三角，transpose_a 指定是否对系数方阵进行转置，unit_diagonal 指定是否假定系数方阵对角线上的元素都为1。输出只有一个，且形状为[*, M, K]。
- 技术要求：
  - 了解 cuda
  - 熟练使用 C++ 开发以及 cmake 编译

### No.83：为神经网络编译器 CINN 增加 resize 算子 <a name='task83'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：在对接上层框架时，编译器会将上层的框架复杂算子进一步拆分为若干基础算子便于编译器执行算子融合，同时减少开发成本。
  - 目标：resize 算子是图像处理领域中的常见算子，resize 算子将输入图片通过指定插值方法调整为指定大小。可通过 CINN IR + extern call 或者 custom call 的方式实现该算子。
- 任务提交：
  - 参考示例 [PR1018](https://github.com/PaddlePaddle/CINN/pull/1018)，以及 [PR1133](https://github.com/PaddlePaddle/CINN/pull/1133)。当前仅需考虑CUDA环境。
  - 算子接口应有一个输入，该输入应为 4-D 张量，形状为[N, C, H, W]。有两属性 out_shape 和 mode，其中 out_shape 指定调整后的张量大小，且形状为[out_H, out_W]；mode 指定插值方法，当前只需实现 bilinear、bicubic、nearest 方法。输出只有一个，且形状为[N, C, out_H, out_W]。
- 技术要求：
  - 有 gpu 算子或 AI 编译器算子开发经验
  - 了解 AI 编译器 Compute 和 Schedule 含义
  - 熟练使用 C++ 开发以及 cmake 编译

### No.84：为神经网络编译器 CINN 增加 ReverseComputeInline 原语 <a name='task84'></a>

- 任务难度：基础
- 详细描述：
  - 现状：Schedule 原语是 CINN 编译器优化算子计算实现的接口，目前已经实现了Split、Fuse、Reorder等常用原语，其中 ComputeInline 原语操作是将一个 tensor 的计算过程内联到其消费者中完成，简化计算过程。
  - 目标：参考已有的 ComputeInline 操作和 [CINN 调度原语开发说明文档](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/CINN/CINN_ir_schedule.md)，添加 ReverseComputeInline 原语，实现将一个 tensor 的计算内联到其生产者中。
- 任务提交：
  - ComputeInline 原语：分别添加接口及实现至[ cinn/ir/ir_schedule.h](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/ir_schedule.h)、[cinn/ir/ir_schedule.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/ir_schedule.cc)，单测添加至 [cinn/backends/ir_schedule_test.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/backends/ir_schedule_test.cc)
  - 支持新增原语 Trace 重放：在 [cinn/ir/schedule_desc.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/schedule_desc.cc) 中使用CINN_BUILD_STEP_KIND 注册 ComputeInline 原语的重放函数，单测添加至 [cinn/ir/schedule_desc_test.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/schedule_desc_test.cc)
- 技术要求：
  - 了解 AI 编译器 Schedule 作用及含义，熟悉 CINN IR 结构
  - 熟练掌握 C++

### No.85：【开源贡献对比研究】以 CI 流水线为例 <a name='task85'></a>

- 任务难度：基础
- 详细描述：请以社区开发者的视角，通过与其他的知名开源项目（不局限在 AI 框架领域）的对比分析，来产生一份分析报告。报告内容中，请首先进行针对开源贡献当中涉及到的因素和流程做出概要分析，然后，针对 CI 流水线进行专项的细致分析。报告要能够通过与其他开源项目相比，就飞桨框架的 CI 流水线的使用体验、自动化程度、对项目的保护情况，等等方面（不限于前述三个方面）给出有论据支撑的分析结论。
- 提交内容
  - markdown 格式（推荐）或者 pdf 格式的分析报告，请提交 PR至：https://github.com/PaddlePaddle/community/tree/master/reports 。
  - 报告是否合格的判断，由组委会选定的不少于三名的专家在阅读报告后，通过综合评分来给出。如有多人提交报告，会选出评分最高的一份报告发放奖励金。
- 技术要求
  - 有 hands-on 的开源项目贡献经历。
  - 对 CI 流水线及其工作原理有一定了解。

### No.86：【开源贡献对比研究】以贡献文档为例 <a name='task86'></a>

- 任务难度：基础
- 详细描述：请以社区开发者的视角，通过与其他的知名开源项目（不局限在 AI 框架领域）的对比分析，来产生一份分析报告。报告内容中，请首先进行针对开源贡献当中涉及到的因素和流程做出概要分析，然后，针对贡献文档（面向项目开发者的文档）进行专项的细致分析。报告要能够通过与其他开源项目相比，就飞桨框架的贡献文档的丰富程度、文档组织、更新及时性等等方面（不限于前述三个方面）给出有论据支撑的分析结论。
- 提交内容
  - markdown 格式（推荐）或者 pdf 格式的分析报告，请提交 PR至：https://github.com/PaddlePaddle/community/tree/master/reports 。
  - 报告是否合格的判断，由组委会选定的不少于三名的专家在阅读报告后，通过综合评分来给出。如有多人提交报告，会选出评分最高的一份报告领取奖励金。
- 技术要求
  - 有 hands-on 的开源项目贡献经历。

### No.87：【开源贡献对比研究】以代码组织为例 <a name='task87'></a>

- 任务难度：基础
- 详细描述：请以社区开发者的视角，通过与其他的知名开源项目（不局限在AI框架领域）的对比分析，来产生一份分析报告。报告内容中，请首先进行针对开源贡献当中涉及到的因素和流程做出概要分析，然后，针对代码仓库的组织进行专项的细致分析。报告要能够通过与其他开源项目相比，就飞桨框架的代码组织的逻辑结构、可维护性、测试覆盖情况等等方面（不限于前述三个方面）给出有论据支撑的分析结论。
- 提交内容
  - markdown 格式（推荐）或者 pdf 格式的分析报告，请提交 PR至：https://github.com/PaddlePaddle/community/tree/master/reports 。
  - 报告是否合格的判断，由组委会选定的不少于三名的专家在阅读报告后，通过综合评分来给出。如有多人提交报告，会选出评分最高的一份报告发放奖励金。
- 技术要求
  - 有 hands-on 的开源项目贡献经历。

### No.88：飞桨框架API文档发布的工具链的分析 <a name='task88'></a>

- 任务难度：进阶
- 详细描述：将飞桨框架的API文档（含英文文档与中文文档）发布到飞桨官网，是通过一系列散落在[Paddle仓库](https://github.com/PaddlePaddle/Paddle)与[Docs仓库](https://github.com/PaddlePaddle/Docs)的脚本组成的工具链来完成的，且缺少完善的材料介绍这一整套的工具链。本任务，希望你能通过一份详细完整的报告，首先分析所开发的 API 文档从开发到发布至官网的过程中遇到的可能的导致最终产出的 API 文档不符合预期的问题（例如：文档格式有误、展示的 API 文档的 API 签名与实际的 API 的实现不一致、中英文文档不一致、示例代码运行出错、等等）；接下来通过报告详细完整的讲解这一系列的工具链；最后对如何改进这些工具链提出思路和技术方案（通过一个或者多个proof of concept 的 PR 来展示你的方案）。
- 提交内容
  - markdown 格式（推荐）或者 pdf 格式的分析报告，请提交 PR至：https://github.com/PaddlePaddle/community/tree/master/reports。 POC 的PR 请提交至[Paddle仓库](https://github.com/PaddlePaddle/Paddle)与[Docs仓库](https://github.com/PaddlePaddle/Docs)。
  - 报告是否合格的判断，由组委会选定的不少于三名的专家在阅读报告后，通过综合评分来给出。如有多人提交报告，会选出评分最高的一份报告发放奖励金。
- 技术要求
  - 有 hands-on 的开源项目贡献经历。
  - 熟悉 python 和 shell 编程。

### No.89：清理动态  import语句，解决circle import 问题 <a name='task89'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前飞桨框架 python 目录下部分模块函数存在动态 import 语句，即 import 语句并未放到文件的最开始地方。这种动态 import 语句存在两个问题：
    - 可能因为存在 circle import 而临时添加的，但不合规范
    - 每次调用函数都会解释执行 import，影响性能
- 任务提交：
  - 设计文档：在 [paddle/community](https://github.com/PaddlePaddle/community/tree/master/rfcs) 仓库下新建 RFC，并提供技术设计思路文档
  - 提交 PR 至Paddle代码仓库。在保持 API 不变的情况下，可适当调整函数或文件位置，以实现「分层管理」

### No.90：JITLayer C++ 端暴露AnaLysisConfig 给用户，提升易用性 <a name='task90'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：paddle/fluid/jit 目录下的Layer是孵化中的项目，旨在提供一个与Python端nn.Layer相同使用方式的后端数据结构，底层封装了预测执行引擎：AnalysisPredictor——推理部署的核心引擎。目前存在如下问题：
    - jit/engine/predictor_engine.h 里的 PredictorEngine 数据结构并未向用户提供灵活 AnaLysisConfig 选项
    - AnaLysisConfig 选项可用于设置 GPU、CPU、MKLDNN、以及自定义优化策略，对提升 Layer 易用性有重要意义。
- 任务提交：
  - 设计文档：在 [paddle/community](https://github.com/PaddlePaddle/community/tree/master/rfcs) 仓库下新建 RFC，并提供技术设计思路文档
  - 提交PR代码至 paddle/fluid/jit 目录下

### No.91：TensorHook支持动转静 <a name='task91'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：动态图下 Tensor 提供了 [register_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#register-hook-hook) 方法，支持用户注册一个hook函数自定义反向grad的数据计算逻辑，此接口仅支持动态图，静态图下缺失对 Tensor 的反向grad同等的逻辑实现，属于动静行为不统一。当用户模型代码包含 register_hook 的用法时，动转静会报错。
- 任务提交：
  - 设计文档：在 [paddle/community](https://github.com/PaddlePaddle/community/tree/master/rfcs) 仓库下新建 RFC，并提供技术设计思路文档
  - 提交PR代码至 python/paddle/jit 动转静目录下

### No.92：ppocr det&rec 全量化模型在 tim-vx（晶晨/瑞芯微） 等设备上的精度提升 <a name='task92'></a>

- 技术标签：深度学习，C++、压缩量化
- 任务难度：进阶
- 详细描述：
  - PP-OCRv3 rec FP32 精度为 76.87，使用PaddleSlim auto-compress全量化后 CPU 侧 eval 精度为75.43，但 GPU、NPU 等目前精度较低。需要通过调整量化工具参数、微调模型等手段，让 PP-OCRv3 rec 的全量化模型在 NPU 上的精度趋近 FP32，目标精度为70.0。
  - 硬件平台为 tim-vx（晶晨/瑞芯微）等任一芯原NPU，如瑞芯微RV1126、1109、晶晨A311D等，系统 OS 建议使用 Linux。
- 提交内容：
  - 完成PP-OCRv3 rec的全量化模型提交，提交至 [Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo) repo下 Paddle-Lite-Demo/[ocr](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/ocr)/assets/ 目录中，会有 RD 同学整理归档。
  - 中文文档，修改 PaddleSlim auto-compress 的文档，描述对该模型在使用auto-compress量化过程中做了哪些修改（量化配置修改、模型修改等）以达到精度在端侧的提升。提交至 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/ocr) repo下PaddleSlim/[example](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example)/[auto_compression](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression)/ocr/README.md 文档文件。
  - 验收标准：先提交该全量化模型以及模型在 tim-vx（晶晨/瑞芯微） 等任一芯原NPU上的精度结果（开源数据集icdar2015，精度70.0以上），待RD验证通过后，提交 PR 到Paddle-Lite-Demo 和 PaddleSlim 仓库。
- 技术要求：部署
  - 熟练掌握 C++、Python 开发。
  - 了解 AI 模型及全量化。
  - 了解 OCR 算法。
  - 掌握 PaddleSlim auto-compress量化工具、PP-OCRv3 模型的修改、Paddle-Lite + TIM-VX 芯原 NPU 部署。

### No.93：增加 linux 下 cpu tensor file_descriptor 传输方案 <a name='task93'></a>

- 技术标签：深度学习，C++
- 任务难度：基础
- 详细描述：
  - 背景：Multiprocessing 是支持进程间 Tensor 传输的一种方式。#[37302](https://github.com/PaddlePaddle/Paddle/pull/37302)  初步支持了paddle的tensor进程间传输，需要继续完善，可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
  - 目前 paddle 支持了 file_system 的 cpu 传输方式，以文档形式存储传输tensor 的中间态。file_descriptor 打开文件句柄之后立即删除，更加安全，不容易发生文件残留。
- 提交内容：
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - 完成 file_descriptor 的支持。提交到 Paddle 主 repo
  - file_descriptor的功能对齐竞品，全面且完善支持，切换为默认传输方式。
  - 验收标准：
    - 自测传输10000次，不发生文件残留；
    - 传输速度与竞品差距10%内
- 技术要求：
  - 熟练掌握 C++、Python 开发。

### No.94：GPU tensor 全局引用计数 <a name='task94'></a>

- 技术标签：深度学习，C++
- 任务难度：进阶
- 详细描述：
  - 背景：Multiprocessing 是支持进程间 Tensor 传输的一种方式。#[37302](https://github.com/PaddlePaddle/Paddle/pull/37302)  初步支持了paddle的tensor进程间传输，需要继续完善，可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
  - 传输过程是生产者消费者场景，为了维护 tensor 的生命周期，需要将cuda 传输的 tensor 与文件绑定，实现全局引用计数。
  - 目前已有初步实现，需要继续完善：[ZHUI/Paddle/commit/d1ec460](https://github.com/ZHUI/Paddle/commit/d1ec460388c9c8efbbaf0bff3abca492d1b81a12) [ZHUI/Paddle/commits/multiprocessing_gpu_ref_count](https://github.com/ZHUI/Paddle/commits/multiprocessing_gpu_ref_count)
- 提交内容：
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - 支持`CudaIPCSentData`，`CudaIPCRefcountedFiles`等功能，将ipc 传输后的Tensor与`CudaIPCSentData`使用`UniqueVoidPtr`绑定。全局引用计数。
  - 验收标准：
    - 自测传输10000次，不发生文件残留
    - 传输速度与竞品差距10%内
- 技术要求：
  - 熟练掌握 C++、Python、CUDA 代码编写。

### No.95：CPU tensor mac/win32 传输 + 适配 DataLoader <a name='task95'></a>

- 技术标签：深度学习，C++
- 任务难度：进阶
- 详细描述：
  - 背景：Multiprocessing 是支持进程间 Tensor 传输的一种方式。#[37302](https://github.com/PaddlePaddle/Paddle/pull/37302)  初步支持了paddle的tensor进程间传输，需要继续完善，可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
  - 支持 mac/win32 平台上cpu tensor进程间传输，并打通DataLoader支持。
- 提交内容：
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - 支持 mac/win32 平台上cpu tensor 进程间传输，并打通 DataLoader 支持。
  - 验收标准：
    - 自测传输10000次，不发生文件残留
    - 传输速度与竞品差距10%内
- 技术要求：
  - 熟练掌握 C++、Python 开发。
  - 熟悉 mac/win 文件系统

### No.96：基于 Paddle 的数据并行DataParallel 添加 join 接口，满足数据流的不均衡输入 <a name='task96'></a>

- 任务难度：进阶
- 详细描述：构造上下文管理器，结合 paddle.DataParallel，使得参与的进程使用不均匀的输入进行训练。这个上下文管理器，将跟踪相关的 DP 进程，并通过通信操作来“隐藏”前向和反向计算，以便匹配未加入的 DP 进程。 这将确保每个通信调用，都有一个由已加入的进程进行调用，从而防止在跨进程输入不均匀的情况下，训练时发生 hang等错误。 或者，设置某个环境变量throw_on_early_termination，一旦某个 rank 用完输入数据，所有训练进程都会抛出错误，从而允许程序捕获和处理这些错误。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录
  - API 功能，提交至 [python/paddle/fluid/dygraph/parallel.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dygraph/parallel.py)
  - 提交代码至 Paddle 代码仓库：[python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests)
- 技术要求
  - 熟悉数据并行的计算原理
  - 熟悉掌握 c++、cuda、python
  - 熟悉掌握集合通信的基本原理，使用集合通信方式

### No.97：基于 Paddle 实现异构集群数据并行训练自动负载均衡 <a name='task97'></a>

- 任务难度：进阶
- 详细描述：异构集群（P40，K40，V100，A100）的设备有不同的显存大小，算力吞吐。在异构集群上进行分布式数据并行，需要考虑不同硬件的显存和算力，来实现在所有硬件显存不溢出的前提下达到最高的整体训练吞吐。参赛者需要通过 Cost Model 对不同异构硬件的显存和算力、任务模型进行建模，并实现一套负载均衡的算法； 将建模信息作为均衡算法输入，计算出每个设备的上 local batch size 等具体训练参数。评价指标是：任务模型使用均衡算法得到的训练参数，在异构集群上数据并行整体吞吐。
- 提交内容
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - Cost Model 需要新增的模块提交到 [auto_parallel/cost](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distributed/auto_parallel/cost) 目录（目录下已有较完备的cost model 基础设施可以直接使用）；
  - 负责均衡算法的实现需要新增的模块提交到 [auto_parallel/tuner](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distributed/auto_parallel/tuner) 目录
  - 实际模型负责均衡脚本提交到 [PaddleFleetx/example](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/projects/gpt) 目录
- 技术要求
  - 熟悉数据并行的计算原理
  - 熟悉掌握 Cost Model 和 负载均衡
  - 熟悉掌握 c++、cuda、python



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
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&QQ群的通知，及时参与。