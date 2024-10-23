# 飞桨适配 torch-scatter

|              |                    |
| ------------ | -----------------  |
| 提交作者      | NkNaN             |
| 提交时间      | 2024/10/22   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop 版本 (https://github.com/PaddlePaddle/Paddle/pull/68780 之后的版本)       |
| 文件名        | 20241022_torchscatter.md  |

## 1. 概述

### 1.1 相关背景

torch-scatter 是一个小型扩展库，用于 PyTorch 中高度优化的稀疏（分散和分段）操作。分散和分段操作可以大致描述为基于给定“组索引”张量的规约操作。分段操作需要有序的“组索引”张量，而分散操作不做限制。

飞桨适配 torch-scatter： https://github.com/PaddlePaddle/PaddleScience/issues/1000

### 1.2 功能目标

1. 使用 paddle.geometric 下的基础 python API 等价组合实现 torch-scatter 的公开 API 功能，如果无法使用基础 API 进行实现，则需参考 pytorch C++ 自定义算子代码，通过 paddle 的 C++ 自定义算子实现对应功能
2. 参考 pytorch 后端已有代码，撰写飞桨后端的单测文件，保证能够完全通过

### 1.3 意义

为 PaddleScience 增加能够支持的稀疏计算以及分段计算 API

## 2. PaddleScience 现状

- PaddleScience 无对应 API；
- paddle 有 torch.scatter/torch.gather 对应的基础 API：paddle.put_along_axis/paddle.take_along_axis；
- paddle.geometric 有 DenseTensor 的分段计算 API：paddle.geometric.segment_max/paddle.geometric.segment_min/paddle.geometric.segment_mean/paddle.geometric.segment_sum

## 3. 目标调研 - torch-scatter 公开 API 梳理

> 共 25 个公开 API，可分为 4 大类：scatter/segment_coo/segment_csr/gather

scatter 类 API 图示：
![scatter_api](scatter.png)

segment 类 API 图示：
![segment_api](segment.png)

### A. scatter
1. torch_scatter.scatter 分散计算 - 将 src 按照指定的 index 延 dim 轴进行 reduce 规约合并，输出到 out。
```py
def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1, 
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum"
) -> torch.Tensor:
"""
    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
"""
```

2. torch_scatter.scatter_sum 分散求和
```py
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1, 
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
3. torch_scatter.scatter_add 分散求和（等价于 scatter_sum ）
```py
def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
4. torch_scatter.scatter_mul 分散求乘积
```py
def scatter_mul(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
5. torch_scatter.scatter_mean 分散求均值（可以由 scatter_sum 组合实现）
```py
def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
6. torch_scatter.scatter_min 分散求最小值
```py
def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
7. torch_scatter.scatter_max 分散求最大值
```py
def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
8. torch_scatter.composite.scatter_std 分散求标准差
```py
def scatter_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True
) -> torch.Tensor
```
9. torch_scatter.composite.scatter_logsumexp 分散求 logsumexp
```py
def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    eps: float = 1e-12
) -> torch.Tensor
```
10. torch_scatter.composite.scatter_softmax 分散求 softmax
```py
def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None
) -> torch.Tensor
```
11. torch_scatter.composite.scatter_log_softmax 分散求 logsoftmax
```py
def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
    dim_size: Optional[int] = None
) -> torch.Tensor
```

### B. segment_csr
1. torch_scatter.segment_csr 分段计算 csr 类型的稀疏 tensor - 将 src 沿着 indptr 的最后一维并按照 indptr 指定的范围进行分段 reduce 规约合并，输出到 out。
```py
def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    reduce: str = "sum") -> torch.Tensor:
"""

    .. math::
        \mathrm{out}_i =
        \sum_{j = \mathrm{indptr}[i]}^{\mathrm{indptr}[i+1]-1}~\mathrm{src}_j.

    Due to the use of index pointers, :math:`segment_csr` is the fastest
    method to apply for grouped reductions.

    :param src: The source tensor.
    :param indptr: The index pointers between elements to segment.
        The number of dimensions of :attr:`index` needs to be less than or
        equal to :attr:`src`.
    :param out: The destination tensor.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mean"`,
        :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
"""
```
2. torch_scatter.segment_sum_csr 分段计算 csr tensor 求和
```py
def segment_sum_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```
3. torch_scatter.segment_add_csr 分段计算 csr tensor 求和
```py
def segment_add_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```
4. torch_scatter.segment_mean_csr 分段计算 csr tensor 求均值
```py
def segment_mean_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```
5. torch_scatter.segment_min_csr 分段计算 csr tensor 求最小值
```py
def segment_min_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```
6. torch_scatter.segment_max_csr 分段计算 csr tensor 求最大值
```py
def segment_max_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

### C. segment_coo
1. torch_scatter.segment_coo 分段计算 coo 类型的稀疏 tensor - 将 src 延index最后一维，按照指定的 index 进行分散 reduce 规约合并，输出到 out。
```py
def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum"
) -> torch.Tensor:
"""

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    In contrast to :meth:`scatter`, this method expects values in :attr:`index`
    **to be sorted** along dimension :obj:`index.dim() - 1`.
    Due to the use of sorted indices, :meth:`segment_coo` is usually faster
    than the more general :meth:`scatter` operation.

    :param src: The source tensor.
    :param index: The sorted indices of elements to segment.
        The number of dimensions of :attr:`index` needs to be less than or
        equal to :attr:`src`.
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :obj:`index.dim() - 1`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mean"`,
        :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
"""
```
2. torch_scatter.segment_sum_coo 分段计算 csr tensor 求和
```py
def segment_sum_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor
```
3. torch_scatter.segment_add_coo 分段计算 csr tensor 求和
```py
def segment_add_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor
```
4. torch_scatter.segment_mean_coo 分段计算 csr tensor 求均值
```py
def segment_mean_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor
```
5. torch_scatter.segment_min_coo 分段计算 csr tensor 求最小值
```py
def segment_min_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor
```
6. torch_scatter.segment_max_coo 分段计算 csr tensor 求最大值
```py
def segment_max_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor
```

### D. gather
1. torch_scatter.gather_csr 聚集操作 - 将 src 按照 indptr 指定的范围取出到 out 
```py
def gather_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```
2. torch_scatter.gather_coo 聚集操作 - 将 src 按照指定的 index 取出到 out 
```py
def gather_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

## 4. 设计思路与实现方案

1. scatter 类: 用 `paddle.scatter/paddle.put_along_axis` 以及 `paddle.std/paddle.logsumexp/paddle.nn.functional.softmax` 组合实现；
2. segment_csr 类：可以用 `paddle.diff` 和 `paddle.repeat_interleave` 以及 `paddle.geometric.segment_*` 组合实现；
3. segment_coo 类：用 `paddle.geometric.segment_*` 实现；
4. gather 类：gather_coo 用 `paddle.gathehr/paddle.take_along_axis` 组合实现，gather_csr 用 `paddle.diff` 和 `paddle.repeat_interleave` 以及 `paddle.gathehr/paddle.take_along_axis` 组合实现；

### 4.1 补充说明[可选]

torch-scatter[文档](https://pytorch-scatter.readthedocs.io/en/latest/index.html)

## 5. 测试和验收的考量

参考 pytorch-scatter 仓库中已有的单测代码，撰写飞桨后端的单测文件，并自测通过。

单测文件：

    1. test/test_broadcasting.py
    2. test/test_gather.py
    3. test/test_multi_gpu.py
    4. test/test_scatter.py
    5. test/test_segment.py
    6. test/test_zero_tensors.py
    7. test/composite/test_logsumexp.py
    8. test/composite/test_softmax.py
    9. test/composite/test_std.py

- 使用 pytest 测试框架；
- scatter/segment/gather API 共有的测试：包含前向，反向，输入包含out参数，以及src/index/indptr是非连续存储时的情况；
- scatter 还需要测试 index 的广播机制；
- scatter/segment/gather 还需要测试 zero-element tensor ；
- scatter/segment 还需要测试多个 GPU 的情况；
- scatter_logsumexp/scatter_std/scatter_softmax 测试前向反向结果。

## 6. 可行性分析和排期规划

2024/10 完成文档设计
2024/11 完成API开发以及跑通单测

## 7. 影响面

为 PaddleScience 添加稀疏计算支持
