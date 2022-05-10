# paddle.Tensor.index_add 设计文档



|API名称 | paddle.Tensor.index_add                       | 

|---|------------------------------------------|

|提交作者<input type="checkbox" class="rowselector hidden"> | SmirnovKol                             | 

|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-05-09                            | 

|版本号 | V1.0                                     | 

|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                  | 

|文件名 | 20220509_api_design_for_index_add.md<br> | 



# 一、概述



## 1、相关背景

为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要实现API`paddle.index_add功能需求。

## 2、功能目标

增加 API `paddle.index_add`, `paddle.index_add_`，`tensor.index_add`，`tensor.index_add_`, 在指定轴上, 通过按index中给定的顺序切片, 对每个切片的张量加上固定的常量。

## 3、意义

飞桨支持index_add算子进一步满足用户需求。



# 二、飞桨现状

目前paddle缺少相关功能实现, 且用其他API来组合实现也较为困难， 因为axis可以是任意轴而且index也不一定连续，`paddle.index_select`和`paddle.slice`也无法直接达到目的。

简单场景下计算逻辑如下：





```Python

import paddle
import numpy as np
np.random.seed(102)


x_np = np.random.rand(4, 3)
x_tensor = paddle.to_tensor(x_np)

result_np = x_np.copy()

added_value = 9.0

index = [0, 2]
axis = 0
result_np[index] += added_value

length = x_tensor.shape[axis]
for i in index:
  if i < 0 or i >= length:
    raise ValueError('index is wrong: {}'.format(index))
  else:
       result_tensor[i] += added_value

print(np.allclose(result_np, result_tensor.numpy()))

```



# 三、业内方案调研

## Numpy 

### 实现方法

Numpy没有对该功能有特定的API进行支持，但是Numpy有非常完善的切片操作和广播机制，可以很好的实现。示例如下：



```Python

import numpy as np

axis = 2

index = [0, 2, 3]

added_value = 97

data = np.random.rand(4, 3, 7, 9)

data[:, :, index, :] += added_value


```



## TensorFlow

tensorflow目前也没有特定的API支持类似功能，

但在tensorflow里也可以通过tf.exprimental.numpy直接调用numpy函数



## Pytorch

Pytorch中有API`Tensor.index_add_(dim, index, source, *, alpha=1)`和`Tensor.index_add(dim, index, source, *, alpha=1)`， 


[文档地址](https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html), 介绍为：



# 四、对比分析

- Numpy基于切片操作和广播机制功能上更灵活更自由。

- Pytorch支只支持一个axis，不仅支持cpu还支持gpu。



# 五、方案设计

## 命名与参数设计

新增API设计为:

`paddle.index_add(x, axis, index, added_value)`

`paddle.index_add_(x, axis, index, added_value)`

`Tensor.index_add(axis, index, added_value)`

`Tensor.index_add_(axis, index, added_value)`



index_add_支持inplace方式修改输入张量。

axis是index索引选择的轴, 支持int类型。

index在指定轴上含索引下标的list of int, tuple of int 或者 1-D Tensor。

added_value是待相加的数据，参数类型支持bool, int, float。



## 底层OP设计

参考飞桨现有算子，分别实现cpu和cuda的算子kernel。



## API实现方案

在 python/paddle/tensor/manipulation.py 中增加index_add以及index_add_函数，分别通过_C_ops调用底层算子

计算正确的stride之后，参考index_select算子进行逻辑修改

输入tensor的所有元素梯度是1.0


## 代码实现文件路径



CPU中正向和反向计算： 

paddle/phi/kernels/cpu/index_add_kernel.cc  
paddle/phi/kernels/cpu/index_add_grad_kernel.cc

GPU中正向和反向计算:

paddle/phi/kernels/gpu/index_add_kernel.cu
paddle/phi/kernels/gpu/index_add_grad_kernel.cu



```c++

template <typename T, typename Context>
void IndexAddKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    int axis,
                    float added_value,
                    DenseTensor* output);


template <typename T, typename Context>
void IndexAddGradKernel(const Context& ctx,
                        const DenseTensor& out_grad,
                        int axis,
                        float added_value,
                        DenseTensor* x_grad);                   
```

算子注册路径：

paddle/fluid/operators/index_add_op.cc

函数API实现路径: python/paddle/tensor/manipulation.py

单元测试路径： python/paddle/fluid/tests/unittests/test_index_add_op.py



# 六、测试和验收的考量

测试考虑的case如下：



- 和numpy结果的数值的一致性, `paddle.index_add`和numpy切片操作结果是否一致；

- 参数`axis`校验参数类型int，判断axis合法，并进行边界检查；

- 校验参数`index`的正确性，索引边界检查，输出结果的正确性；

- 校验参数added_value的正确性， 是否是支持的数据类型

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




