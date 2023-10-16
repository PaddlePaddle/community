# paddle.utils.data.SubsetRandomSampler API 增强设计文档

| API名称      | paddle.utils.data.SubsetRandomSampler|
| ------------ | -------------------------------------- |
| 提交作者     | Asthestarsfalll                                    |
| 提交时间     | 2023-09-25                             |
| 版本号       | V1.0                                   |
| 依赖飞桨版本  |  develop                                |
| 文件名       | 20230925_api_design_for_SubsetRandomSampler.md |


 # 一、概述
 ## 1、相关背景

`SubsetRandomSampler`子集随机采样器，支持从数据集的指定子集中随机选择样本，可以用于将数据集分成训练集和验证集等子集。

 ## 2、功能目标

增加`paddle.utils.data.SubsetRandomSampler`，实现对给定子集的随机采样。

 ## 3、意义

飞桨支持`SubsetRandomSampler`。

 # 二、飞桨现状

目前paddle缺少相关功能实现，但是有类似功能的API，只需要继承`Sampler` 基类，并重写`__iter__`和`__iter__`方法实现相关功能即可。


 # 三、业内方案调研

 ## PyTorch

Pytorch中有API`torch.utils.data.SubsetRandomSampler(indices, generator)`.在pytorch中，介绍为：

```
Samples elements randomly from a given list of indices, without replacement.
```

 ### 实现方法

实现方法较为简单，其子集通过给定的`indices`确定，采样时只需要从`indices`中采样便可达到相应的效果。

```python
class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

```

 ## MindSpore

MindSpore 中`Sampler`的整体设计与`Paddle`并不相同

 ``` python
class SubsetRandomSampler(SubsetSampler):
    """
    Samples the elements randomly from a sequence of indices.

    Args:
        indices (Iterable): A sequence of indices (Any iterable Python object but string).
        num_samples (int, optional): Number of elements to sample. Default: ``None`` , which means sample all elements.

    Raises:
        TypeError: If elements of `indices` are not of type number.
        TypeError: If `num_samples` is not of type int.
        ValueError: If `num_samples` is a negative value.

    Examples:
        >>> import mindspore.dataset as ds
        >>> indices = [0, 1, 2, 3, 7, 88, 119]
        >>>
        >>> # create a SubsetRandomSampler, will sample from the provided indices
        >>> sampler = ds.SubsetRandomSampler(indices)
        >>> data = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8, sampler=sampler)
    """

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SubsetRandomSamplerObj(self.indices, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        return True

    def parse_for_minddataset(self):
        """Parse the sampler for MindRecord."""
        c_sampler = cde.MindrecordSubsetSampler(self.indices, ds.config.get_seed())
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(self.get_num_samples())
        return c_sampler
 ```

 ## API实现方案

pytorch的sampler整体设计与paddle类似，因此考虑参考pytorch的方案实现
在 python\paddle\io\sampler.py 中添加对应类。但是由于paddle并没有完全支持`generator`，因此将该参数移除。

 # 六、测试和验收的考量

 测试考虑的 case 如下：
 - 确保结果符合预期:一次遍历中遍历所有的 `index` 一次且仅有一次，确保不重复不遗漏.

 # 七、可行性分析和排期规划

 方案实施难度可控，工期上可以满足在当前版本周期内开发完成。

 # 八、影响面

 为已有 API 的增强，对其他模块没有影响

 # 名词解释

 无

 # 附件及参考资料

 无
