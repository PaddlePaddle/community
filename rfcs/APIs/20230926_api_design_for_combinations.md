# paddle.combinations 设计文档

| API名称      | paddle.combinations                     |
| ------------ | --------------------------------------- |
| 提交作者     | NetPunk                                 |
| 提交时间     | 2023-09-26                              |
| 版本号       | V1.0                                    |
| 依赖飞桨版本 | develop                                 |
| 文件名       | 20220926_api_design_for_combinations.md |

# 一、概述

## 1、相关背景

计算给定Tensor的长度为r的组合

## 2、功能目标

实现combinations API，计算给定Tensor的长度为r的组合，调用路径为：

- paddle.combinations为独立的函数调用
- Tensor.combinations做为 Tensor 的方法使用

## 3、意义

完善Paddle API丰富度

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有combinations API（https://pytorch.org/docs/stable/generated/torch.combinations.html?highlight=combination）

在 PyTorch 文档中，介绍为：

```
Compute combinations of length r of the given tensor. The behavior is similar to python’s itertools.combinations when with_replacement is set to False, and itertools.combinations_with_replacement when with_replacement is set to True.
```

计算给定张量长度为 r 的组合。当 with_replacement 设置为 False 时，其行为类似于 python 的 itertools.combinations；当 with_replacement 设置为 True 时，其行为类似于 itertools.combinations_with_replacement。

其中输入参数信息有：

* input必须为一维向量
* r不能小于0，且必须为整数
* with_replacement默认为False

### 实现方法

在实现方法上, PyTorch采用的是API组合实现

```cpp
Tensor _triu_mask(int64_t n, int64_t dims, bool diagonal, TensorOptions opt) {
  // get a mask that has value 1 whose indices satisfies i < j < k < ...
  // or i <= j <= k <= ... (depending on diagonal)
  Tensor range = at::arange(n, opt.dtype(kLong));
  std::vector<Tensor> index_grids = at::meshgrid(std::vector<Tensor>(dims, range), "ij");
  Tensor mask = at::full(index_grids[0].sizes(), true, opt.dtype(kBool));
  if(diagonal) {
    for(int64_t i = 0; i < dims - 1; i++) {
      mask *= index_grids[i] <= index_grids[i+1];
    }
  } else {
    for(int64_t i = 0; i < dims - 1; i++) {
      mask *= index_grids[i] < index_grids[i+1];
    }
  }
  return mask;
}

Tensor combinations(const Tensor& self, int64_t r, bool with_replacement) {
  TORCH_CHECK(self.dim() == 1, "Expect a 1D vector, but got shape ", self.sizes());
  TORCH_CHECK(r >= 0, "Expect a non-negative number, but got ", r);
  if (r == 0) {
    return at::empty({0}, self.options());
  }
  int64_t num_elements = self.numel();
  std::vector<Tensor> grids = at::meshgrid(std::vector<Tensor>(r, self), "ij");
  Tensor mask = _triu_mask(num_elements, r, with_replacement, self.options());
  for(Tensor &t : grids) {
    t = t.masked_select(mask);
  }
  return at::stack(grids, 1);
}
```

可以看出实现思路比较清晰：

* 通过meshgrid方法构造grids，r种分布
* 通过meshgrid方法构造index_grids，表示r种索引
* 通过index_grids构造mask，表示r个位置
* 通过masked_select方法，得到grid选择后的结果
* 最后将结果stack起来



# 四、对比分析

可以直接参考的实现是pytorch，涉及到的API在Paddle中均有实现，可以想到用Paddle API组合实现相同的逻辑



# 五、方案设计

## 命名与参数设计

API设计为`paddle.combinations(x, r, with_replacement, name)`

paddle.combinations
----------------------

参数
:::::::::

- x (Tensor) - 1-D Tensor，`x` 的数据类型可以是 float16, float32，float64，int32，int64
- r (int) - 组合内元素的个数，数据类型为 int，默认值为2
- with_replacement (bool) - 是否允许组合内有重复数，数据类型为 bool，默认值为`False`
- name  (str) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。

:::::::::

- Tensor 返回组合拼接而成的张量。数据类型和输入`x`一致。

paddle.Tensor.combinations指向paddle.combinations，两者是相同的API



## 底层OP设计

python端API组合实现

## API实现方案

参考pytorch逻辑，实现初版代码如下

~~~python
def combinations(x, r=2, with_replacement=False):
    if len(x.shape) != 1:
        raise TypeError("Expect a 1-D vector, but got x shape {}".format(x.shape))
    if not isinstance(r, int) or r <= 0:
        raise ValueError("Expect a non-negative int, but got r={}".format(r))

    if r == 0:
        return paddle.empty([0], dtype=x.dtype)

    if r > 1:
        t_l = [x for i in range(r)]
        grids = paddle.meshgrid(t_l)
    else:
        grids = [x]
    num_elements = x.numel()
    t_range = paddle.arange(num_elements, dtype='long')
    if r > 1:
        t_l = [t_range for i in range(r)]
        index_grids = paddle.meshgrid(t_l)
    else:
        index_grids = [t_range]
    mask = paddle.full(index_grids[0].shape, True, dtype='bool')
    if with_replacement:
        for i in range(r - 1):
            mask *= index_grids[i] <= index_grids[i + 1]
    else:
        for i in range(r - 1):
            mask *= index_grids[i] < index_grids[i + 1]
    for i in range(r):
        grids[i] = grids[i].masked_select(mask)

    return paddle.stack(grids, 1)
~~~



# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算和反向计算；
  - 计算dtype类型：验证 `float64`，`int32`等；

- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入参数类型、形状的有效性校验。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

# 八、影响面

为独立新增API，对其他模块没有影响