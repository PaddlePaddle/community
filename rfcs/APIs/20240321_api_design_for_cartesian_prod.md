# paddle.cartesian_prod 设计文档

| API名称      | paddle.cartesian_prod                     |
| ------------ | --------------------------------------- |
| 提交作者     | NetPunk                                 |
| 提交时间     | 2023-09-26                              |
| 版本号       | V1.0                                    |
| 依赖飞桨版本 | develop                                 |
| 文件名       | 20220926_api_design_for_cartesian_prod.md |

# 一、概述

## 1、相关背景

计算给定Tensor的长度为r的组合

## 2、功能目标

paddle.cartesian_prod 作为独立的函数调用，对给定的张量序列进行笛卡尔积。该行为类似于 python 的 itertools.product 。相当于把所有输入的张量转成列表，对这些列表做itertools.product，最后把得到的列表转成张量。

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



# 四、对比分析

可以直接参考的实现是pytorch，涉及到的API在Paddle中均有实现，可以想到用Paddle API组合实现相同的逻辑



# 五、方案设计

## 命名与参数设计

API设计为`paddle.cartesian_prod(*tensors, name)`

paddle.cartesian_prod
----------------------

参数
:::::::::

- tensors (Tensor|list(Tensor)) - 输入的一个至多个 1-D Tensor，`x` 的数据类型可以是 float32，float64，int32，int64
- name  (str) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。

:::::::::

- Tensor 返回各个张量的笛卡尔积。数据类型和输入`x`一致。



## 底层OP设计

python端API组合实现

## API实现方案

参考pytorch逻辑，实现初版代码如下

~~~python
def cartesian_prod(*tensors):
    for tensor in tensors:
        if len(tensor.shape) != 1:
            raise ValueError("Expect a 1D vector, but got shape {}".format(tensor.shape))

    if len(tensors) == 1:
        return tensors[0]

    grids = paddle.meshgrid(*tensors)
    for i, grid in enumerate(grids):
        if 0 in grid.shape:
            return paddle.empty([0])
        grids[i] = paddle.flatten(grid)
    return paddle.stack(grids, axis=1)
~~~

因为 paddle.flatten 并不支持当shape中有0存在的情况，因此对该情况做特殊处理


# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算和反向计算；
  - 计算dtype类型：验证 `float64`，`int32`等；

- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：形状的有效性校验。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

# 八、影响面

为独立新增API，对其他模块没有影响
