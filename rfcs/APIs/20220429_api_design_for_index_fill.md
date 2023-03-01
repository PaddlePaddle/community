# paddle.Tensor.index_fill 设计文档



|API名称 | paddle.Tensor.index_fill                         | 

|---|------------------------------------------|

|提交作者<input type="checkbox" class="rowselector hidden"> | thunder95                                | 

|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-04-29                               | 

|版本号 | V1.0                                     | 

|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                  | 

|文件名 | 20220429_api_design_for_index_fill.md<br> | 



# 一、概述



## 1、相关背景

为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要实现API`paddle.index_fill功能需求。

## 2、功能目标

增加API`paddle.index_fill`, `paddle.index_fill_`，`tensor.index_fill`，`tensor.index_fill_`, 通过按index中给定的顺序, 在指定轴上用固定值填充输入的张量。

## 3、意义

飞桨支持index_fill算子进一步满足用户需求。



# 二、飞桨现状

目前paddle缺少相关功能实现, 且用其他API来组合实现也较为困难， 因为axis可以是任意轴而且index也不一定连续，`paddle.index_select`和`paddle.slice`也无法直接达到目的。

简单场景下计算逻辑如下：





```Python

import paddle

import numpy as np

np.random.seed(102)



np_data = np.random.rand(4, 3)

pd_tensor = paddle.to_tensor(np_data)

np_res = np_data.copy()

fill_val = 9.0

index = [0, 2]

np_res[index, :] = 9.0

for i in range(pd_tensor.shape[1]):

    if i in index:

        pd_tensor[i, ...] = fill_val



print(np.allclose(np_res, pd_tensor.numpy()))

```



# 三、业内方案调研

## Numpy 

### 实现方法

Numpy没有对该功能有特定的API进行支持，但是Numpy有非常完善的切片操作和广播机制，可以很好的实现。示例如下：



```Python

import numpy as np

axis = 0

index = [0, 2]

data = np.random.rand(4, 3)

print(data[[0, 2],:])

```



## TensorFlow

tensorflow目前也没有特定的API支持类似功能，

但在tensorflow里也可以通过tf.exprimental.numpy直接调用numpy函数



## Pytorch

Pytorch中有API`Tensor.index_fill_(dim, index, value)`和`Tensor.index_fill(dim, index, value)`， 

其中dim是选取index所在的轴， value是待填充的值, index_fill_是对原输入张量的修改。

index_fill对应的out of palce



在pytorch中，[文档地址](https://pytorch.org/docs/stable/generated/torch.Tensor.index_fill_.html#torch.Tensor.index_fill_), 介绍为：

```

Fills the elements of the self tensor with value value by selecting the indices in the order given in index.

```

在底层分别通过c++和cuda和函数实现上



### 实现方法



cuda核函数实现的主要位置：

```c++

void index_fill_kernel_impl(

  TensorIterator& iter,

  int64_t dim,

  int64_t self_dim_size,

  int64_t self_dim_stride,

  scalar_t fill_val)；

}

```

cpu核函数主要基于loop方式实现，

```c++

void index_fill_kernel(

  TensorIterator& iter,

  int64_t dim,

  int64_t self_dim_size,

  int64_t self_dim_stride,

  const Scalar& source)；

}

```



# 四、对比分析

- Numpy基于切片操作和广播机制功能上更灵活更自由。

- Pytorch支只支持一个axis，不仅支持cpu还支持gpu。



# 五、方案设计

## 命名与参数设计

新增API设计为:

`paddle.index_fill(x, axis, index, fill_value)`

`paddle.index_fill_(x, axis, index, fill_value)`

`Tensor.index_fill(axis, index, fill_value)`

`Tensor.index_fill_(axis, index, fill_value)`



index_fill_支持inplace方式修改输入张量。

axis是index索引选择的轴, 支持int以及0维的Tensor参数类型。

index在指定轴上含索引下标的list of int, tuple of int 或者 1-D Tensor。

fill_value是待填充的数据，参数类型支持bool, int, float以及0维的Tensor。



## 底层OP设计

参考飞桨现有算子，分别实现cpu和cuda的算子kernel。
对于fill_value是Tensor和非Tensor的两种不同情况各自使用单独的OP。



## API实现方案

在 python/paddle/tensor/manipulation.py 中增加index_fill以及index_fill_函数，分别通过_C_ops调用底层算子

计算正确的stride之后，参考index_select算子进行逻辑修改

在指定轴上指定索引的输入元素梯度为0.0，其他未被选中的元素梯度是1.0

若fill_value是0维的Tensor，其反向传播的梯度是对应选中的输出梯度的总和sum。


## 代码实现文件路径



CPU中正向和反向计算： 
paddle/phi/kernels/cpu/index_fill_scalar_kernel.cc paddle/phi/kernels/cpu/index_fill_scalar_grad_kernel.cc
paddle/phi/kernels/cpu/index_fill_tensor_kernel.cc paddle/phi/kernels/cpu/index_fill_tensor_grad_kernel.cc

GPU中正向和反向计算:
paddle/phi/kernels/gpu/index_fill_scalar_kernel.cu paddle/phi/kernels/gpu/index_fill_scalar_grad_kernel.cu
paddle/phi/kernels/gpu/index_fill_tensor_kernel.cu paddle/phi/kernels/gpu/index_fill_tensor_grad_kernel.cu



```c++

template <typename T, typename Context>
void IndexFillScalarKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     float fill_value,
                     int axis,
                     DenseTensor* output);

template <typename T, typename Context>
void IndexFillTensorKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     const DenseTensor& fill_value,
                     int axis,
                     DenseTensor* output);
```

算子注册路径：

paddle/fluid/operators/index_fill_scalar_op.cc
paddle/fluid/operators/index_fill_tensor_op.cc


函数API实现路径: python/paddle/tensor/manipulation.py

单元测试路径： python/paddle/fluid/tests/unittests/test_index_fill_op.py



# 六、测试和验收的考量

测试考虑的case如下：



- 和numpy结果的数值的一致性, `paddle.index_fill`和numpy切片操作结果是否一致；

- 参数`axis`校验参数类型int以及0维的tensor，判断axis合法，并进行边界检查；

- 校验参数`index`的正确性，索引边界检查，输出结果的正确性；

- 校验参数fill_value的正确性， 是否是支持的数据类型，当fill_value是0维tensor时梯度正确回传

- 测试在进行反向梯度计算时结果的正确性；

- 错误检查：输入`x`不是Tensor时,能否正确抛出错误；



# 七、可行性分析及规划排期



方案实施难度可控，工期上可以满足在当前版本周期内开发完成。



# 八、影响面

为独立新增API，对其他模块没有影响



# 名词解释

无

# 附件及参考资料

无




