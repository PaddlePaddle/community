# paddle.sparse.is_nan 设计文档

| API名称                                                      | paddle.sparse.is_nan                                   |
| ------------------------------------------------------------ | ------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                               |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-05                                        |
| 版本号                                                       | V1.0                                              |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                           |
| 文件名                                                       | 20230305_api_design_for_paddle_sparse_isnan.md<br> |


# 一、概述

## 1、相关背景

is_nan 是一个检查输入稀疏 Tensor  的每一个值是否为 +/-NaN 。目前在 PaddlePaddle 种，没有稀疏涨量的is_nan计算逻辑。
针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 reshape 的计算逻辑，一共需要新增 2个 kernel 的前向与反向。

## 2、功能目标

在飞桨中增加 paddle.sparse.is_nan API。

## 3、意义

飞桨将支持 paddle.sparse.is_nan API。

# 二、飞桨现状

目前飞桨还没有稀疏张量的 is_nan 操作。


# 三、业内方案调研

Pytorch, Tensorflow, Scipy都没有直接实现对稀疏张量的 is_nan　的一元操作。

# 四、对比分析

Pytorch, Tensorflow, Scipy虽然对于稠密张量支持is_nan操作，但都没有直接实现对稀疏张量的 is_nan　的一元操作。
在　PaddlePaddle　中将对这个算子操作进行支持。

# 五、设计思路与实现方案

## 命名与参数设计

sparse is_nan 这个稀疏张量上的方法的命名和参数不需要额外设计，在 paddle/phi/api/yaml 下新增注册该算子的前向和反向。

## 底层OP设计

新增两个 Kernel：

```    cpp
SparseCooTensor IsnanCoo(const Context& dev_ctx,const SparseCooTensor& x) {
    SparseCooTensor coo;
    IsnanCooKernel<T, Context>(dev_ctx, x, shape, &coo);
    return coo;
}
```

```cpp
SparseCsrTensor IsnanCsr(const Context& dev_ctx,const SparseCsrTensor& x) {
    SparseCsrTensor csr;
    IsnanCsrKernel<T, Context>(dev_ctx, x, shape, &csr);
    return csr;
}
```

在前向推理中，一元操作的实现较简单，取出 DenseTensor 类型的 non_zero_elements() 后，逐元素进行 is_nan 操作，并创建新的稀疏张量即可。
在反向梯度传播里, 只对+/-NaN的元素进行梯度为1.０的数值进行反向传播，对于非NaN数值的元素设置梯度为0.0.

## API实现方案

在 Paddle repo 的 python/paddle/sparse/unary.py 文件中新增api:

```cpp
@dygraph_only
def is_nan(x, name=None):
    return _C_ops.sparse_is_nan(x)
```

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性

# 七、可行性分析和排期规划

前两周实现代码、文档和测试。

第三周进行 Code Review 和继续迭代。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料
