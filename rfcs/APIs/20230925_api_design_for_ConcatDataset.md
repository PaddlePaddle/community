# paddle.index_fill 设计文档

| API名称                                                      | paddle.io.ConcatDataset |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者   | NetPunk                   |
| 提交时间| 2023-09-25                 |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名                                                       | 20230925_api_design_for_ConcatDataset.md |

# 一、概述

## 1、相关背景

ConcatDataset可以将多个数据集连接在一起，形成一个大的数据集，适用于需要同时处理多个数据集的情况。

## 2、功能目标

实现ConcatDataset，将多个数据集连接在一起，调用路径为：

- paddle.io.ConcatDataset

## 3、意义

完善Paddle API丰富度

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有ConcatDataset操作 API（https://pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset）

在 PyTorch 文档中，介绍为：

```
Dataset as a concatenation of multiple datasets.

This class is useful to assemble different existing datasets.

Parameters:
datasets (sequence) – List of datasets to be concatenated
```
输入datasets组成的列表，返回连接后的Dataset

### 实现方法

PyTorch采用的是python端实现，封装为类

```python
class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
```

# 四、对比分析

可以直接参考的实现是pytorch，因为paddle和pytorch的dataset在行为上比较相似，因此大致逻辑可以套用pytorch实现

# 五、方案设计

## 命名与参数设计

API的实现为一个类，方法组成和io中其它Dataset类相似

paddle.io.ConcatDataset
----------------------
参数
:::::::::

- datasets (sequence) - 用于连接的dataset列表

:::::::::

- datasets - 连接后的dataset

## 底层OP设计

python端API组合实现

## API实现方案

`__init__`方法中将输入的多个dataset转为列表，`__getitem__`方法中将输入的idx索引到不同的dataset上

# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：
  - 不同 shape；
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入dataset类型不能是`IterableDataset`。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

# 八、影响面

为独立新增API，对其他模块没有影响