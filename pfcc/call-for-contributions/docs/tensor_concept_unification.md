# 飞桨 DenseTensor 概念统一


> This project will be mentored by [@chenwhql](http://github.com/chenwhql) and [@Ligoml](https://github.com/ligoml)

## 1. 背景与意义

飞桨基础框架的 fluid 目录中原先有 `paddle::framework::Tensor` 和 `paddle::framework::LoDTensor` 两个底层 Dense Tensor 概念，而在 Python 端的 `paddle.Tensor` 实际上对应的是更上层的概念，在静态图对应的 `paddle::framework::Variable` ，在动态图对应的是的 `paddle::imperative::VarBase` 。

原先的 `paddle::framework::LoDTensor` 继承自 `paddle::framework::Tensor` ，在其基础上新增了 LoD 描述信息，LoD信息简介如下：

```c++
/*
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
```

LoD 信息主要是为了处理的语言类模型而引入的机制设计，实际上在 Paddle fluid 中主要是 sequence_xxx 系列的 Op 对其有依赖，其他的 Op 里一般需要传递相应的 LoD 信息。后来随着 Paddle 基础框架的演进，LoD 相关机制由于适配复杂，特殊情况多，相比传统的 Padding 机制维护成本高，Paddle 逐渐废弃了对该机制的维护，也不再新增相应的 Op，原先的 sequence_xxx 系列 Op 在将来也会逐渐移除。

在这样的背景下，后续我们在实施算子库和动态图重构相关工作时，弱化了 LoD 的概念，将原先的 `paddle::framework::Tensor` 和 `paddle::framework::LoDTensor` 合二为一，改为 `phi::DenseTensor` 。在重构之后的新动态图下，Python 端的 `paddle.Tensor` 即为 C++ 端的 `paddle::Tensor` ，作为一个泛化的 Tensor 概念存在，在 `paddle::Tensor` 之下，可以支持多种异构的 Tensor 实现，包括 DenseTensor、SparseCooTensor、SparseCsrTensor 等。

但由于 `paddle::framework::Tensor` 和 `paddle::framework::LoDTensor` 在原 fluid 框架下使用极多，目前并未完全替换，导致代码里基础的 DenseTensor 数据结构使用目前比较混乱，同时，在一些官方 API 的文档中，也仍然在使用 LoDTensor 的概念，这可能也会给用户带来一些困惑。

因此从代码与文档两个方面，统一飞桨基础框架的 DenseTensor 使用，能够使代码更容易理解与维护。


## 2. 目标

本专项的目标就是移除所有 Paddle 基础框架中 `paddle::framework::Tensor` 和 `paddle::framework::LoDTensor` 的使用以及Python API 文档中对 `LoDTensor` 概念的使用。

## 3. 工作及执行思路

具体地，本专项包括两个方面的工作：

### 3.1 代码清理

该部分工作主要是移除框架代码中 `paddle::framework::Tensor` 和 `paddle::framework::LoDTensor` 的使用以及各处 `using Tensor = phi::DenseTensor` 和 `using LoDTensor = phi::DenseTensor` 的使用。

该部分清理工作之前已经进行了一部分，包括：

- 将底层的 fluid/framework/tensor.h 中的 `using Tensor = phi::DenseTensor` 移除（[PR46342](https://github.com/PaddlePaddle/Paddle/pull/46432)）
- 将底层的 fluid/framework/lod_tensor.h 中的 `using LoDTensor = phi::DenseTensor` 移除（[PR46663](https://github.com/PaddlePaddle/Paddle/pull/46663), [PR46953](https://github.com/PaddlePaddle/Paddle/pull/46953)）

但仍然还有很多代码实现在使用 LoDTensor 的概念，截至本文档撰写之时（2022年11月1日），在 C++ 端还可以搜索到1000余处 LoDTensor 的使用。

具体执行时，需要搜索 `LoDTensor` 的使用，替换为 `phi::DenseTensor` 。

### 3.2 文档清理

该部分工作主要是移除官方文档中对 `LoDTensor` 的使用，特别是 2.x API 文档（除fluid目录之外） 中对其的使用。例如在 paddle.flip API文档中：

```python
def flip(x, axis, name=None):
    """
    Reverse the order of a n-D tensor along given axis in axis.

    Args:
        x (Tensor): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor x
            should be float32, float64, int32, int64, bool.
        axis (list|tuple|int): The axis(axes) to flip on. Negative indices for indexing from the end are accepted.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor or LoDTensor calculated by flip layer. The data type is same with input x.
    """
```

这里需要将 `or LoDTensor` 的描述从文档中移除，仅保留 Tensor 的介绍。此类问题需要统一搜索并逐一修改。
