# 飞桨适配 torch-scatter

> RFC 文档相关记录信息

|              |                          |
| ------------ |--------------------------|
| 提交作者      | Z-NAVY                   |
| 提交时间      | 2024-12-24               |
| RFC 版本号    | v1.0                     |
| 依赖飞桨版本  | develop                  |
| 文件名        | 20241224_torchscatter.md |

## 1. 概述

### 1.1 相关背景

torch-scatter 是一个用于在 PyTorch 中进行高效张量分散（scatter）操作的库，提供了一些高效的实现，用于在不同维度上对张量进行分散操作。

### 1.2 功能目标

1. 整理 torch-scatter 的所有公开 API
2. 尽量使用 paddle.geometric 下的基础 python API 等价组合实现 torch-scatter 公开 API 的功能，如果无法使用基础 API 进行实现，通过 paddle 的 C++ 自定义算子实现对应功能
3. 参考 pytorch 后端代码，撰写飞桨后端单测文件，并自测通过
4. 整理代码并提交PR至 PaddleScience 官方仓库，完成代码修改与合入。

### 1.3 意义

扩展 PaddleScience 库功能，通过api接口实现高效调用。

## 2. PaddleScience 现状

1. PaddleScience 中暂无 torch-scatter 中的所有公开API；
2. paddle.geometric 下的基础 python API 能够初步组合实现 torch-scatter 中的功能；
3. 大量科学项目需要 torch-scatter 中的 API 功能支持。

## 3. 目标调研

根据 torch-scatter 中实现文件的划分，将25个公开api分为四大类：scatter、segment_coo、segment_csr、composite，这四类api有着不同的实现特性。

### 3.1 scatter

1. scatter
```python
def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor
```
通用接口，根据reduce参数选择具体的操作，支持sum, add, mul, mean, min, max这些reduction方式。
2. scatter_sum
```python
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor
```
将源张量src中对应相同索引index的数据累加到目标张量out中。

3. scatter_add
```python
def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor
```
同scatter_sum。
4. scatter_mul
```python
def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor
```
将源张量src中对应相同索引index的数据累乘到目标张量out中。
5. scatter_mean
```python
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor
```
先调用scatter_sum对源张量src进行求和，然后计算每个index对应的均值。
6. scatter_min
```python
def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]
```
返回源张量src中相同索引index中的最小值。
7. scatter_max
```python
def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]
```
返回源张量src中相同索引index中的最大值。
### 3.2 segment_coo

索引数组是排序好的(详见4.5部分pytorch-scatter文档)，1-6每个函数的作用都能在前述scatter中找到对应函数解释。

1. segment_coo
```python
def segment_coo(src: torch.Tensor, index: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                reduce: str = "sum") -> torch.Tensor
```
2. segment_sum_coo
```python
def segment_sum_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor
```
3. segment_add_coo
```python
def segment_add_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor
```
4. segment_mean_coo
```python
def segment_mean_coo(src: torch.Tensor, index: torch.Tensor,
                     out: Optional[torch.Tensor] = None,
                     dim_size: Optional[int] = None) -> torch.Tensor
```
5. segment_min_coo
```python
def segment_min_coo(
        src: torch.Tensor, index: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]
```
6. segment_max_coo
```python
def segment_max_coo(
        src: torch.Tensor, index: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]
```
7. gather_coo
```python
def gather_coo(src: torch.Tensor, index: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor
```
假设源张量src中每个元素对应一个索引，根据索引数组扩展源张量形成新张量返回。
### 3.3 segment_csr

索引数组改成了位置指针(详见4.5部分pytorch-scatter文档)，1-6每个函数的作用都能在前述scatter中找到对应函数解释。

1. segment_csr
```python
def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor
```
2. segment_sum_csr
```python
def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor
```
3. segment_add_csr
```python
def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor
```
4. segment_mean_csr
```python
def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor
```
5. segment_min_csr
```python
def segment_min_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```
6. segment_max_csr
```python
def segment_max_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```
7. gather_csr
```python
def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor
```
假设源张量src中每个元素对应一个索引，根据位置数组扩展源张量形成新张量返回。
### 3.4 composite

1. scatter_std
```python
def scatter_std(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                unbiased: bool = True) -> torch.Tensor
```
将源张量src中对应相同索引index的数据分组，计算每组的标准差。

2. scatter_softmax
```python
def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor
```
将源张量src中对应相同索引index的数据分组，计算每组的softmax值。
3. scatter_log_softmax
```python
def scatter_log_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                        eps: float = 1e-12,
                        dim_size: Optional[int] = None) -> torch.Tensor
```
将源张量src中对应相同索引index的数据分组，计算每组的log_softmax值。
4. scatter_logsumexp
```python
def scatter_logsumexp(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                      out: Optional[torch.Tensor] = None,
                      dim_size: Optional[int] = None,
                      eps: float = 1e-12) -> torch.Tensor
```
将源张量src中对应相同索引index的数据分组，计算每组的logsumexp值。
## 4. 设计思路与实现方案

### 4.1 简单函数

1. scatter_sum 和 scatter_add 通过 paddle.scatter_nd_add 或者 paddle.geometric.segment_sum 实现；
2. scatter_mul 暂定通过循环遍历实现；
3. scatter_min 和 scatter_max 通过 paddle.geometric.segment_min 和 paddle.geometric.segment_max 实现；
4. segment_sum_coo、segment_add_coo、segment_mean_coo、segment_min_coo、segment_max_coo 五个函数和前面的实现类似，index变为有序；
5. segment_sum_csr、segment_add_csr、segment_mean_csr、segment_min_csr、segment_max_csr 五个函数和前面的实现类似，索引数组变为分组位置；
6. gather_coo 和 gather_csr 使用 paddle.gather 进行实现。

### 4.2 汇总函数
1. scatter、segment_coo、segment_csr 三个函数在实现其他函数的基础上添加根据参数调用。

### 4.3 组合实现
1. scatter_std、scatter_softmax、scatter_log_softmax、scatter_logsumexp 四个函数组合其他函数实现即可。

### 4.4 C++扩展优化
1. 考虑到部分操作的实现高效性，参考 pytorch-scatter 中的 cuda 实现进行性能优化。

### 4.5 参考链接

1. pytorch-scatter 仓库：https://github.com/rusty1s/pytorch_scatter/tree/2.1.2
2. PaddleScience 仓库：https://github.com/PaddlePaddle/PaddleScience
3. paddle 仓库：https://github.com/PaddlePaddle/Paddle
4. pytorch-scatter 文档：https://pytorch-scatter.readthedocs.io/en/latest/
5. paddle api 文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

## 5. 测试和验收的考量

1. 参考 pytorch-scatter 中的测试代码，完成以下测试文件自测通过
```text
    test_broadcasting.py
    test_gather.py
    test_multi_gpu.py
    test_scatter.py
    test_segment.py
    test_zero_tensors.py
    test_logsumexp.py
    test_softmax.py
    test_std.py
```
2. 参考 pytorch-scatter 中的benchmark代码，完成以下性能测试文件并对比性能
```text
    gather.py
    scatter_segment.py
```

## 6. 可行性分析和排期规划

1. 通过利用 paddle.geometric 中的 api 能够完成 pytorch-scatter 中25个公开api的初步实现；
2. 使用 gpu cuda 函数的实现进一步优化性能存在难点。

**具体排期：**
1. 24-12-30 前完成所有公开api的初步实现；
2. 25-01-02 前完成所有测试以及benchmark实现；
3. 25-01-07 前利用 cuda 算子部分优化已经实现的函数；
4. 25-01-09 最终完善，提交PR。

## 7. 影响面

1. 对原有功能无影响；
2. 扩展当前库的公开API；
3. 提高当前库的可用性。
