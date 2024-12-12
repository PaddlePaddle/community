此文档展示 **PaddlePaddle Hackathon 第七期活动——开源贡献个人挑战赛框架开发方向任务** 详细介绍

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

---

### NO.18 为稀疏计算添加复数支持

**详细描述：**

为提高框架对复数的支持能力，对以下的稀疏计算添加复数支持：add_coo_coo/add_coo_dense/add_csr_csr/subtract_coo_coo/subtract_csr_csr /multiply_coo_coo/multiply_csr_csr/divide_coo_coo/divide_csr_csr

**提交内容：**

- 在 op 对应的前向以及反向 kernel 增加复数运算逻辑，且注册相应的 complex64, complex128 数据类型。
- 在对应 op 的单测中增加复数类型
- 在对应 api 的类型校验中增加复数

**技术要求：**

- 熟悉 Python、C++，CUDA；
- 了解 Paddle 算子开发流程；
- 了解稀疏矩阵的表示方式；

**参考资料：**

- https://github.com/PaddlePaddle/Paddle/issues/61975

### NO.19 为 Paddle 新增 load_state_dict_from_url API

**详细描述：**

从给定 URL 加载 Paddle 序列化对象。如果下载的文件是 zip 文件，它将自动解压缩。如果对象已存在于 model_dir 中，则将其反序列化并返回。此任务的目标是在 Paddle 框架中，新增 load_state_dict_from_url API，调用路径为：paddle.hub.load_state_dict_from_url 。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/hapi/hub.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/hapi/hub.py) 文件
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/hub](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/hub) 目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的映射关系文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python

### NO.20 为 Paddle 新增 Tensor.set\_ / Tensor.resize\_ API

**详细描述：**

设置 Tensor shape 和 stride。source 设置的张量将与源张量共享相同的存储空间，并可以设置为相同或不同的 shape 和 stride。张量的共享可以参考 paddle 已有 api *share_buffer_to。paddle.Tensor.resize*可以通过 Tensor.set*view/reshape*/set*来组合实现。此任务的目标是在 Paddle 框架中，新增 set* 和 resize* API，调用路径为：paddle.Tensor.set* / paddle.Tensor.resize\_ 。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/tensor/creation.py 文件
- 中文 API 文档，在 docs repo 的 docs/api/paddle/Tensor 目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 了解 Paddle 算子开发流程
- 熟练掌握 Python、C++

### NO.21 为 Paddle 新增 reset_peak_memory_stats/reset_max_memory_allocated/memory_stats API

**详细描述：**

- 在 paddle.device.cuda 包中，增加对 CUDA 张量类型的以下三个支持：重置 CUDA 内存分配器跟踪的“峰值”统计信息，在 paddle 框架中新增 reset_peak_memory_stats API，调用路径为 paddle.device.cuda.reset_peak_memory_stats；
- 重置跟踪给定设备上张量占用的最大 GPU 内存的起点，在 paddle 框架中新增 reset_max_memory_allocated API，调用路径为 paddle.device.cuda.reset_max_memory_allocated；
- 返回给定设备的 CUDA 内存分配器统计信息字典，在 paddle 框架中新增 memory_stats API，调用路径为 paddle.device.cuda.memory_stats。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/device/cuda/\_\_init\_\_.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/device/cuda/__init__.py) 文件
- C++ /CUDA 实现代码，在 Paddle repo 的 [paddle/phi/core/memory](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/core/memory) 目录
- 单测代码，在 Paddle repo 的 [test/cpp/fluid/memory](https://github.com/PaddlePaddle/Paddle/tree/develop/test/cpp/fluid/memory) 目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/device/cuda](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/device/cuda) 目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的映射关系文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉 CUDA 内存分配
- 熟练掌握 C++, CUDA, Python

### NO.22 在 paddle.audio.functional.get_window 中支持 bartlett 、 kaiser 和 nuttall 窗函数

**详细描述：**

当前 paddle.audio.functional.get_window 中已支持 hamming，hann，blackman 等窗函数，需扩充支持 bartlett 、 kaiser 和 nuttall 窗函数

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/audio/functional/window.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/audio/functional/window.py) 文件
- 单测代码，在 Paddle repo 的 [test/legacy_test/test_audio_functions.py](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/test_audio_functions.py) 文件中添加
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/audio/functional/get_window_cn.rst](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/audio/functional/get_window_cn.rst) 中补充新的窗函数用法，并扩充示例代码
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的映射关系文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉窗函数的作用和原理
- 熟练掌握 Python

### NO.23 为 Paddle 新增 ParameterDict API

**详细描述：**

paddle.nn.ParameterDict 提供参数字典容器。此容器的行为类似于 Python 字典，但它包含的参数将被正确地注册和添加。使用方式为：

```python
import paddle
import paddle.nn as nn

class MyLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'p1': nn.Parameter(paddle.create_parameter(shape=[2, 2], dtype='float32')),
            'p2': nn.Parameter

(paddle.create_parameter(shape=[2, 2], dtype='float32'))
        })

    def forward(self, x, px):  # px can use 'p1' or 'p2'
        x = self.params[px].add(x)
        return x
```

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/nn/layer/container.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/container.py)文件
- 单测代码，在 Paddle repo 的 [test/legacy_test/](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/) 目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/nn](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/nn) 目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的映射关系文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉 Python
- 了解 paddle.nn.Layer 中参数的组织方式

**参考资料：**

- [ParameterList](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ParameterList_cn.html#parameterlist)

### NO.24 为 Paddle 新增 EmbeddingBag API

**详细描述：**

EmbeddingBag 是 [Embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Embedding_cn.html#embedding) 的拓展，在功能上相当于 Embedding + 求和/求均值/求最大值的操作，相比直接组合，EmbeddingBag 会有更高的计算效率和更小的内存消耗。此任务的目标是在 Paddle 框架中，新增 EmbeddingBag 和 embedding_bag API，调用路径为：paddle.nn.EmbeddingBag 和 paddle.nn.functional.embedding_bag。可以在之前开发者[未开发完的 PR](https://github.com/PaddlePaddle/Paddle/pull/49000)基础上进行开发。

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/nn/layer/common.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/common.py) 文件 和 [python/paddle/nn/functional/input.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/input.py) 文件
- C++ /CUDA 实现代码，在 Paddle repo 的 [paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels) 目录
- 单测代码，在 Paddle repo 的[ python/paddle/test/legacy_test](https://github.com/PaddlePaddle/Paddle/tree/develop/test/legacy_test) 目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/nn](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/nn) 目录和 [docs/api/paddle/nn/functional/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle/nn/functional)
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉 EmbeddingBag 算法原理和适用场景
- 熟悉 Paddle 动静态图下数学计算过程
- 熟练掌握 C++, CUDA, Python

### NO.25 为 Paddle 新增 is_coalesced/sparse_dim/dense_dim API

**详细描述：**

Paddle Tensor 可以存储在连续内存中，以便快速访问，实现高效算法。但一些用户可能使用张量来表示数据，例如图邻接矩阵、修剪权重或点云，这些张量的元素大多数为零值，即稀疏张量。Paddle Tensor 也需支持稀疏存储格式，如 COO、CSR、CSC 等。is_coalesced/sparse_dim/dense_dim 即用于稀疏 Tensor 的一组方法。

- Tensor.is_coalesced：如果 Tensor 是一个已合并的稀疏张量，则返回 True，否则返回 False
- Tensor.sparse_dim：返回稀疏张量中稀疏维度的数量
- Tensor.dense_dim：返回稀疏张量中密集维度的数量

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/tensor/sparse.py 文件
- 并在 python/paddle/tensor/\_\_init\_\_.py 中，添加 is_coalesced/sparse_dim/dense_dim API，以支持 Tensor.is_coalesced/sparse_dim/dense_dim 的调用方式
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系 文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉稠密 Tensor/稀疏 Tensor/非连续内存等
- 熟练掌握 Python

### NO.26 为 Paddle 新增 lu_solve API

**详细描述：**

使用 LU 分解 来求解线性方程组 AX=B，A 为 1 个或多个矩阵，A.shape=[m, n] or [batch, m, n]，B.shape=[m, k] or [batch, m, k]，A 和 B 已知，通过 LU 分解方阵 A 来加速求解 X。需要满足 LU, pivots, info = paddle.linalg.lu(A); X = paddle.linalg.lu_solve(B, LU, pivots) 与 使用 X = paddle.linalg.solve(A, B) 直接求解线性方程组的结果一样。

此任务的目标是在 Paddle 框架中，新增 lu_solve API，调用路径为：paddle.linalg.lu_solve 和 Tensor.lu_solve

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)文件
- 并在[python/paddle/tensor/\_\_init\_\_.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 lu_solve API，以支持 Tensor.lu_solve 的调用方式
- C++ 实现代码，在 Paddle repo 放置。头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录
- 单测代码，在 Paddle repo 的 [test/](https://github.com/PaddlePaddle/Paddle/tree/develop/test)目录
- 中文 API 文档，在 docs repo 的[docs/api/paddle](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle)目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系 文件，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉矩阵 LU 分解原理，了解当前 paddle.linalg.lu 的逻辑
- 熟悉 lapack/cublas 库
- 熟练掌握 Python

### NO.27 为 Paddle 新增 baddbmm API

**详细描述：**

为 Paddle 新增 baddbmm API，以实现 $\beta * input + \alpha (A @ B)$的功能。

此任务的目标是在 Paddle 框架中，新增 baddbmm API，调用路径为：paddle.baddbmm 和 Tensor.baddbmm

**提交内容：**

- API 的设计文档，并提 PR 至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 rfcs/APIs 目录
- Python 实现代码 & 英文 API 文档，在 Paddle repo 的 [python/paddle/nn/utils](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 目录下新建文件
- 并在[python/paddle/tensor/\_\_init\_\_.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L274)中，添加 API，以支持 Tensor.baddbmm 的调用方式
- C++ 实现代码，在 Paddle repo 放置。头文件在 Paddle repo 的[paddle/phi/kernels](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels)目录，cc 文件在[paddle/phi/kernels/cpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/cpu)目录 和 cu 文件在[paddle/phi/kernels/gpu](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/gpu)目录
- 单测代码，在 Paddle repo 的[ python/paddle/test/legacy_test](https://github.com/PaddlePaddle/Paddle/tree/develop/test/legacy_test) 目录
- 中文 API 文档，在 docs repo 的 [docs/api/paddle/nn/utils](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录
- API 映射文档：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系 文件，请务必遵守[《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟悉 Paddle 的参数初始化方式
- 熟练掌握 C++, CUDA, Python

### NO.28 为 `paddle.clip` 进行功能增强

**详细描述：**

为 paddle.clip 进行功能增强，支持 min 和 max 传入 Tensor，从而按照 elementwise 的方式进行上下界裁剪。

**提交内容：**

- **API/OP 修改代码** 和 **API 英文文档**，代码提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- **API 单测代码**，需自行寻找相应的 API/OP 单测，如无现成测试，则需新增对应的测试 case
- **API 中文文档**，如果有 API 签名的改动，需修改 API 文档，提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中
- **API 映射文档**，描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中修改对应的 API 映射文档，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

- Pytorch 对应 API 参考： `torch.clip`

### NO.29 为 `paddle.grad` 进行功能增强

**详细描述：**

为 `paddle.grad` 进行功能增强，支持 is_grads_batched 参数，按 batch 计算 vjp；支持 only_inputs 参数。

**提交内容：**

- **API/OP 修改代码** 和 **API 英文文档**，代码提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- **API 单测代码**，需自行寻找相应的 API/OP 单测，如无现成测试，则需新增对应的测试 case
- **API 中文文档**，如果有 API 签名的改动，需修改 API 文档，提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中
- **API 映射文档**，描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中修改对应的 API 映射文档，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

- Pytorch 对应 API 参考： `torch.autograd.grad`

### NO.30 为 `paddle.divide` 进行功能增强

**详细描述：**

为 `paddle.divide` 支持 rounding_mode 参数：支持 None、'trunc' 、 'floor' 三种类型的设置，其中 'trunc' 为向 0 取整，'floor'为向负无穷取整；None 时相当于 true_divide，此时还涉及到输入 int 时类型提升，否则无法表达出正确结果。目前 `paddle.divide` 的逻辑相当于 rounding_mode=None，但还缺少 int 时的类型提升，导致计算结果不准确。

**提交内容：**

- **API/OP 修改代码** 和 **API 英文文档**，代码提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- **API 单测代码**，需自行寻找相应的 API/OP 单测，如无现成测试，则需新增对应的测试 case
- **API 中文文档**，如果有 API 签名的改动，需修改 API 文档，提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中
- **API 映射文档**，描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中修改对应的 API 映射文档，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

- Pytorch 对应 API 参考： `torch.divide`

### NO.31 为 `paddle.sparse.sparse_csr_tensor`进行功能增强

**详细描述：**

为 `paddle.sparse.sparse_csr_tensor` 支持 shape 的自动推导，可不输入 shape，此时会自动推导 shape。

**提交内容：**

- **API/OP 修改代码** 和 **API 英文文档**，代码提交到[Paddle Repo](https://github.com/PaddlePaddle/Paddle)中
- **API 单测代码**，需自行寻找相应的 API/OP 单测，如无现成测试，则需新增对应的测试 case
- **API 中文文档**，如果有 API 签名的改动，需修改 API 文档，提交到[Docs Repo](https://github.com/PaddlePaddle/docs)中
- **API 映射文档**，描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中修改对应的 API 映射文档，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

- Pytorch 对应 API 参考： `torch.sparse_csr_tensor`

### NO.32 为 `paddle.nn.functional.scaled_dot_product_attention` 进行功能增强

**详细描述：**

为 `paddle.nn.functional.scaled_dot_product_attention` 完成以下几个方面的功能增强：

- 支持 float32 数据类型输入
- 当前要求卡的计算能力不低于 8，需调研分析是否可以进一步降低对硬件 GPU 的需求。（torch 对硬件要求更低，可参考）
- 支持 math 后端，在不使用 flash_attn 加速时，也可正常计算

**提交内容：**

- **API/OP 修改代码** 和 **API 英文文档**，代码提交到 [Paddle Repo](https://github.com/PaddlePaddle/Paddle) 中
- **API 单测代码**，需自行寻找相应的 API/OP 单测，如无现成测试，则需新增对应的测试 case
- **API 中文文档**，如果有 API 签名的改动，需修改 API 文档，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs) 中
- **API 映射文档**，描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中修改对应的 API 映射文档，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

- Pytorch 对应 API 参考： `torch.nn.functional.scaled_dot_product_attention`

### NO.33 为 `paddle.nn.MaxPool1D/MaxPool2D/MaxPool3D` 及其对应 functional API 增加 dilation 参数

**详细描述：**

为 `paddle.nn.MaxPool1D/MaxPool2D/MaxPool3D` 及其对应的 `paddle.nn.functional.max_pool1d/max_pool2d/max_pool3d` 增加 dilation 参数。

**提交内容：**

- **API/OP 修改代码** 和 **API 英文文档**，代码提交到 [Paddle Repo](https://github.com/PaddlePaddle/Paddle) 中
- **API 单测代码**，需自行寻找相应的 API/OP 单测，如无现成测试，则需新增对应的测试 case
- **API 中文文档**，如果有 API 签名的改动，需修改 API 文档，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs) 中
- **API 映射文档**，描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中修改对应的 API 映射文档，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)，提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)

**技术要求：**

- 熟练掌握 C++，Python

**参考内容：**

- Pytorch 对应 API 参考： `torch.nn.MaxPool1D/MaxPool2D/MaxPool3D`

### NO.34 为 Paddle 代码转换工具新增 API 转换规则（第 1 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 1 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的映射关系 文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.35 为 Paddle 代码转换工具新增 API 转换规则（第 2 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 2 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.36 为 Paddle 代码转换工具新增 API 转换规则（第 3 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 3 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下

- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.37 为 Paddle 代码转换工具新增 API 转换规则（第 4 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 4 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.38 为 Paddle 代码转换工具新增 API 转换规则（第 5 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 5 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.39 为 Paddle 代码转换工具新增 API 转换规则（第 6 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 6 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.40 为 Paddle 代码转换工具新增 API 转换规则（第 7 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 7 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.41 为 Paddle 代码转换工具新增 API 转换规则（第 8 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 8 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异

### NO.42 为 Paddle 代码转换工具新增 API 转换规则（第 9 组）

**详细描述：**

为了能自动化将其它深度学习代码转写成 Paddle 代码，从而提升模型代码迁移的效率，我们建设了 [**代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。

此次需要你完成 [**第七期黑客松 API 转换名单**](https://doc.weixin.qq.com/sheet/e3_AakAbwboADEk5j1q1ABTf0V9nYWD1?scode=AHAA0Qc9AFooltcxprAakAbwboADE&tab=BB08J2) **中第 9 组** 的 API 转换规则开发。

**提交内容：**

- **API 映射文档**：描述 Paddle API 与 Pytorch API 之间的映射关系，请在 [API 映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 中为每个 API 新增对应的 映射关系文件，文件名为 PyTorch API 名，请务必遵守 [《API 映射关系-格式规范》](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) ，代码提交到 [Docs Repo](https://github.com/PaddlePaddle/docs)下
- **API 转换规则**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#如何贡献代码)，代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下
- **API 转换单测**：具体开发步骤，请参考 [《PaConvert：如何贡献代码》](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#步骤5编写单元测试)中步骤 5，注意单测规范与要求。代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 下

**注意事项：**
- 务必按照 [映射文档模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 编写，一比一对照编写，提交review前先自查问题
- 本次放出来的API，内部已初步分析过，基本上都有相关功能，功能缺失不太可能。尽可能去找组合替代实现，判定为功能缺失要慎重，除非paddle完全无相类似功能
- 先写好映射文档，再根据文档来实现Matcher，注意不要出现文档与Matcher的diff。如果后面实现Matcher时，发现文档有误，需返工及时更正文档

**技术要求：**

- 熟练掌握 Python
- 熟悉 Pytorch、Paddle 两者 API 的使用，善于捕捉并分析细节差异
