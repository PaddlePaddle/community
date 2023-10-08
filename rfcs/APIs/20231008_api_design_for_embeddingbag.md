# paddle.nn.EmbeddingBag  设计文档

 | API名称      | paddle.nn.EmbeddingBag                      |
 | ------------ | --------------------------------------- |
 | 提交作者     | mhy                                 |
 | 提交时间     | 2023-10-08                              |
 | 版本号       | V1.0                                    |
 | 依赖飞桨版本  |  develop                                 |
 | 文件名       | 20221008_api_design_for_embeddingbag.md |

 # 一、概述

 ## 1、相关背景

EmbeddingBag 是 Embedding 的拓展，在功能上相当于 Embedding + 求和/求均值/求最大值的操作，相比直接组合，EmbeddingBag 会有更高的计算效率和更小的内存消耗。

 ## 2、功能目标

新增 EmbeddingBag 和 embedding_bag API，调用路径为：

- paddle.nn.EmbeddingBag
- paddle.nn.functional.embedding_bag。

 ## 3、意义

 完善Paddle API丰富度

 # 二、飞桨现状

 目前paddle缺少相关功能实现。

 # 三、业内方案调研

 ## PyTorch

 PyTorch 中有 EmbeddingBag API（https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag）

 在 PyTorch 文档中，介绍为：

 ```
    Computes sums or means of ‘bags’ of embeddings, without instantiating the intermediate embeddings.
 ```

该API等价于 Embedding + Sum/Mean/Max：

```
with mode="sum" is equivalent to Embedding followed by torch.sum(dim=1),

with mode="mean" is equivalent to Embedding followed by torch.mean(dim=1),

with mode="max" is equivalent to Embedding followed by torch.max(dim=1).
```


# 四、对比分析

可以直接参考的实现是pytorch，涉及到的API在Paddle中均有实现，可以想到用Paddle API组合实现相同的逻辑


# 五、方案设计

## 命名与参数设计

 API设计为`paddle.nn.functional.embedding_bag(num_embeddings, embedding_dim, padding_idx=None, sparse=False, weight_attr=None, name=None, mode='mean')`和`paddle.nn.EmbeddingBag(num_embeddings, embedding_dim, padding_idx=None, sparse=False, weight_attr=None, name=None, mode='mean')`

 paddle.nn.functional.embedding_bag
 ----------------------
 参数
 :::::::::
- num_embeddings (int) - 嵌入字典的大小，input 中的 id 必须满足 0 =< id < num_embeddings 。
- embedding_dim (int) - 每个嵌入向量的维度。
- padding_idx (int|long|None，可选) - padding_idx 的配置区间为 [-weight.shape[0], weight.shape[0]，如果配置了 padding_idx，那么在训练过程中遇到此 id 时，其参数及对应的梯度将会以 0 进行填充。
- sparse (bool，可选) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
- weight_attr (ParamAttr|None，可选) - 指定嵌入向量的配置，包括初始化方法，具体用法请参见 ParamAttr，一般无需设置，默认值为 None。
- name  (str) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
- mode(str) - “sum”，“mean”或“max”。指定规约的方法。“sum”计算总和，“mean”计算包中值的平均值，“max”计算每个包的最大值。默认值:“mean”。

 :::::::::

 - Tensor 返回组合拼接而成的张量。数据类型和输入`x`一致。

paddle.nn.EmbeddingBag 调用 paddle.nn.functional.embedding_bag，两者是相同的API

 ## 底层OP设计

 python端API组合实现

 ## API实现方案
 参考pytorch逻辑，实现初版代码如下

 ~~~python
class EmbeddingBag(Layer):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        sparse=False,
        weight_attr=None,
        name=None,
        mode='mean',
    ):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._sparse = sparse
        self._is_distributed = False
        self._padding_idx = padding_idx
        self._mode = mode

        if self._num_embeddings <= 0:
            raise ValueError("num_embeddings must be gather than 0")

        if self._embedding_dim <= 0:
            raise ValueError("embedding_dim must be gather than 0")

        padding_idx = (
            -1
            if padding_idx is None
            else padding_idx
            if padding_idx >= 0
            else (num_embeddings + padding_idx)
        )

        if padding_idx >= num_embeddings or padding_idx < -num_embeddings:
            raise ValueError(
                "padding_idx must be within [-{}, {})".format(
                    num_embeddings, num_embeddings
                )
            )

        self._dtype = self._helper.get_default_dtype()
        self._size = [self._num_embeddings, self._embedding_dim]

        self._weight_attr = weight_attr
        self._remote_prefetch = False
        self._name = name
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False,
        )

        if in_dynamic_mode() and padding_idx != -1:
            with paddle.no_grad():
                self.weight[padding_idx] = 0.0

    def forward(self, x):
        out = F.embedding(
            x,
            weight=self.weight,
            padding_idx=self._padding_idx,
            sparse=self._sparse,
            name=self._name,
        )
        if self._mode == "sum":
            return paddle.sum(out, axis=1)
        elif self._mode == "mean":
            return paddle.mean(out, axis=1)
        elif self._mode == "max":
            return paddle.max(out, axis=-1)
        else:
            raise ValueError("Not supported mode")
 ~~~

 # 六、测试和验收的考量

 测试考虑的case如下：
 - 正确性验证：
   - 前向计算和反向计算；
 - 错误检查：输入参数类型、形状的有效性校验。


 # 七、可行性分析及规划排期
 技术可行性：参考同类项目和相似的 API，无重大难点；

 # 八、影响面
 为独立新增API，对其他模块没有影响
