# paddle.cartesian_prod 设计文档

| API名称      | paddle.cartesian_prod                     |
| ------------ | --------------------------------------- |
| 提交作者     | NKNaN                                 |
| 提交时间     | 2024-06-04                              |
| 版本号       | V1.1                                    |
| 依赖飞桨版本 | develop                                 |
| 文件名       | 20240604_api_design_for_cartesian_prod.md |

# 一、概述

## 1、相关背景

计算给定的一组 Tensor 的笛卡尔积。python 的 itertools 中有 product 方法，用于计算集合的笛卡尔积，其数学表示可以为：

若集合 $ A = \{a\}$ ， 集合 $B = \{b\}$ ，则 $A$ 和 $B$ 的笛卡尔积为 $A \times B = \{(a, b)|a \in A \wedge b \in B\}$ ;

若集合 $ X_i = \{x_i\}$ ，则 $n$ 个集合的笛卡尔积为 $\prod_{i=1}^n X_i = X_1 \times ... \times X_n = \{(x_1, ... ,x_n)|x_1 \in X_1 \wedge ... \wedge x_n \in X_n\}$

一组 1 维 Tensor 的笛卡尔积则相当于将 Tensor 视为集合，然后求集合的笛卡尔积。

## 2、功能目标

paddle.cartesian_prod 作为独立的函数调用，对给定的张量序列进行笛卡尔积。相当于把所有输入的张量转成列表，对这些列表做itertools.product，最后再把得到的列表转成张量。

## 3、意义

完善Paddle API丰富度

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有cartesian_prod API（https://pytorch.org/docs/1.12/generated/torch.cartesian_prod.html?highlight=cartesian_prod#torch.cartesian_prod）

在 PyTorch 文档中，介绍为：

```
Do cartesian product of the given sequence of tensors. The behavior is similar to python’s itertools.product.
```

对于给定任意数量的1D tensor，计算它们的笛卡尔积，行为类似于 python 的 itertools.product

### 实现方法

在实现方法上, PyTorch采用的是API组合实现

```cpp
Tensor cartesian_prod(TensorList tensors) {
  for(const Tensor &t : tensors) {
    TORCH_CHECK(t.dim() == 1, "Expect a 1D vector, but got shape ", t.sizes());
  }
  if (tensors.size() == 1) {
    return tensors[0];
  }
  std::vector<Tensor> grids = at::meshgrid(tensors, "ij");
  for(Tensor &t : grids) {
    t = t.flatten();
  }
  return at::stack(grids, 1);
}
```

可以看出实现思路比较清晰：

* 通过meshgrid方法构造grids
* 将grids内tensor展开
* 最后将结果stack起来


## TensorFlow_MRI

TensorFlow_MRI 中有cartesian_prod API（https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/cartesian_product/）

在 TensorFlow_MRI 文档中，介绍为：

```
Cartesian product of input tensors.
```

### 实现方法

在实现方法上, TensorFlow_MRI采用的是API组合实现

```python
def cartesian_product(*args):
  """Cartesian product of input tensors.

  Args:
    *args: `Tensors` with rank 1.

  Returns:
    A `Tensor` of shape `[M, N]`, where `N` is the number of tensors in `args`
    and `M` is the product of the sizes of all the tensors in `args`.
  """
  return tf.reshape(meshgrid(*args), [-1, len(args)])

def meshgrid(*args):
  """Return coordinate matrices from coordinate vectors.

  Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
  fields over N-D grids, given one-dimensional coordinate arrays
  `x1, x2, ..., xn`.

  .. note::
    Similar to `tf.meshgrid`, but uses matrix indexing and returns a stacked
    tensor (along axis -1) instead of a list of tensors.

  Args:
    *args: `Tensors` with rank 1.

  Returns:
    A `Tensor` of shape `[M1, M2, ..., Mn, N]`, where `N` is the number of
    tensors in `args` and `Mi = tf.size(args[i])`.
  """
  return tf.stack(tf.meshgrid(*args, indexing='ij'), axis=-1)
```

## Numpy

numpy 暂无相应的 API 实现

# 四、对比分析

pytorch 和 tensorflow_mri 都是用 API 组合实现，虽然实现的方式略有不同，但核心都是需用到 meshgrid、stack 以及 reshape 或 flatten。二者的实现区别主要在于 pytorch 使用了循环语句，而 tensorflow_mri 则没有用循环。其次 pytorch 有对输入 tensor 的维度检验，以及如果输入只有一个 tensor 的情况单独返回。

所以建议将二者结合，对输入 tensor 的维度检验，以及如果输入只有一个 tensor 的情况参考 pytorch，主体部分参考 tensorflow_mri，用 Paddle API 组合实现。



# 五、方案设计

## 命名与参数设计

API设计为`paddle.cartesian_prod(x, name)`

paddle.cartesian_prod
----------------------

参数
:::::::::

- x (list[Tensor]|tuple[Tensor]) - 输入的一个至多个 1-D Tensor，`x` 的数据类型可以是 bfloat16、float16、float32、float64、int32、int64、complex64、complex128。
- name  (str) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。

:::::::::

- Tensor 返回输入张量序列的笛卡尔积。数据类型和输入`x`一致。



## 底层OP设计

python端API组合实现

## API实现方案

参考pytorch逻辑，实现初版代码如下

```python
def cartesian_prod(x, name=None):
    for tensor in x:
        if len(tensor.shape) != 1:
            raise ValueError("Expect a 1D vector, but got shape {}".format(tensor.shape))

    if len(x) == 1:
        return x[0]

    coordinates = paddle.stack(paddle.meshgrid(x), axis=-1)
    return paddle.reshape(coordinates, [-1, len(x)])
```


# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：可以与 itertools.product 的结果对齐；
  - 只输入 1 个 tensor
  - 不同 shape，包括 shape 中有 0 的情况；
  - 动态图静态图；
  - 计算dtype类型：验证 `bfloat16`，`float16`，`float32`，`float64`，`int32`，`int64`，`complex64`，`complex128`；

- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：形状的有效性校验。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

# 八、影响面

stack API 需添加支持 shape 中有 0 的情况