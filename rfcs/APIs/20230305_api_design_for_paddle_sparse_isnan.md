# paddle.sparse.isnan 设计文档

| API名称                                                      | paddle.sparse.isnan                                   |
| ------------------------------------------------------------ | ------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                               |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-05                                        |
| 版本号                                                       | V1.0                                              |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                           |
| 文件名                                                       | 20230305_api_design_for_paddle_sparse_isnan.md<br> |


# 一、概述

## 1、相关背景

isnan 检查输入Tensor  的每一个值是否为 +/-NaN, 并返回布尔型结果。目前在 PaddlePaddle 中，对于稀疏Ｔensor还没有支持isnan的API。
针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 isnan 的计算逻辑，一共需要新增 2个 kernel 的前向。

## 2、功能目标

在飞桨中增加 paddle.sparse.isnan API。

## 3、意义

飞桨将支持 paddle.sparse.isnan API。

# 二、飞桨现状

目前飞桨还没有稀疏张量的 isnan 操作。


# 三、业内方案调研

Tensorflow, Scipy都没有直接实现对稀疏张量的 isnan　的一元操作。

Pytorch对于很多zero-preserving unary function支持COO/CSR/CSC/BSR/CSR等多种数据格式的稀疏张量计算，其中也包括isnan，可直接使用dense　tensor中的torch.isnan API。

```    python
import torch
t = torch.tensor([[[0., 0], [1., 2.]], [[0., 0], [3., float('nan')]]])
x = t.to_sparse(sparse_dim=2)
torch.isnan(x)
```

# 四、对比分析

Tensorflow, Scipy虽然对于稠密张量支持isnan操作，但都没有直接实现对稀疏张量的 isnan　的一元操作。
在　PaddlePaddle　中可以参考Ｐytorch将对这个算子操作进行支持。


# 五、设计思路与实现方案

## 命名与参数设计

sparse isnan 这个稀疏张量上的方法的命名和参数不需要额外设计，由于在判断isnan时，此处不可导，所以不考虑反向算子的实现。在 paddle/phi/api/yaml 下新增注册该算子的前向。

API命名为paddle.sparse.isnan, 接口参数支持两个参数，x (Tensor) - 输入的 Tensor，数据类型为：float16、float32、float64、int32、int64。
name (str，可选) - 具体用法请参见 Name，一般无需设置，默认值为 None。

```    yaml
  - op : isnan
  args : (Tensor x)
  output : Tensor(out)
  infer_meta :
    func : UnchangedInferMeta
    param: [x]
  kernel :
    func : isnan_coo{sparse_coo -> sparse_coo},
      isnan_csr{sparse_csr -> sparse_csr}
    layout : x
```

## 底层OP设计

新增两个 Kernel：

```    cpp
IsnanCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    SparseCooTensor* out)

IsnanCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   SparseCsrTensor* out)
```

在前向推理中，一元操作的实现较简单，取出 DenseTensor 类型的 non_zero_elements() 后，逐元素进行 isnan 操作，并创建新的稀疏张量即可。



## API实现方案

参考unary_kernel.cc中其他一元算子的实现，可以批量注册复用dense kernel。

在 Paddle repo 的 python/paddle/sparse/unary.py 文件中新增api, 支持静态图和动态图:

```cpp
def isnan(x, name=None):
    return _C_ops.sparse_isnan(x)
```

# 六、测试和验收的考量

新增单测代码　python/paddle/fluid/tests/unittests/test_sparse_isnan.py, 测试考虑的case如下：

- 数值正确性
- COO和CSR数据格式
- 不同输入tensor的数据类型下检查输出结果
- 计算结果与dense tensor进行比较

# 七、可行性分析和排期规划

前两周实现代码、文档和测试。

第三周进行 Code Review 和继续迭代。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料
