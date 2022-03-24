# paddle.frac设计文档

| API名称                                                      | paddle.frac                     |
| ------------------------------------------------------------ | ------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                 |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-19                      |
| 版本号                                                       | V1.0                            |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                         |
| 文件名                                                       | 20200319_api_design_for_frac.md |

# 一、概述

## 1、相关背景

frac用于计算输入中每个元素的分数部分。

## 2、功能目标

为 Paddle 新增 frac 数学计算API。

## 3、意义

Paddle支持frac数学计算API。

# 二、飞桨现状

飞桨目前并无此API，相关API有`paddle.trunc`，将输入 Tensor 的小数部分置0，返回置0后的 Tensor ，如果输入 Tensor 的数据类型为整数，则不做处理。

因此`paddle.frac`的实现可在Python端调用`trunc`和`elementwise_sub`的C++ OP组合实现。



# 三、业内方案调研

## Pytorch

Pytorch中有API `torch.frac()`和`Tensor.frac()`，介绍为：

```
Computes the fractional portion of each element in input.
```

### 实现方法

Pytorch中实际上使用输入tensor减去trunc后的结果得到输出，一些相关代码如下：

[代码位置](https://github.com/pytorch/pytorch/blob/2d110d514f9611dd00bf63ae5ef7d5ce017c900f/torch/csrc/jit/codegen/cuda/runtime/helpers.cu)

```c++
__device__ double frac(double x) {
  return x - trunc(x);
}

__device__ float frac(float x) {
  return x - trunc(x);
}
```

[代码位置](https://github.com/pytorch/pytorch/blob/3a0c680a14d2f1211adc4dfcc7ab0be5d1f1f214/torch/csrc/jit/codegen/fuser/cpu/resource_strings.h#L46)

```c++
double frac(double x) {
  return x - trunc(x);
}
float fracf(float x) {
  return x - truncf(x);
}
```

[代码位置](https://github.com/pytorch/pytorch/blob/f64906f470916c3edc1937155a8a37e77c35f393/aten/src/ATen/test/vec_test_all_types.h)

```c++
template <typename T>
T frac(T x) {
  return x - std::trunc(x);
}
```

## TensorFlow

未找到相关实现，在[该文件](https://github.com/tensorflow/tensorflow/blob/a0192a3285e2d010ae57a76cc8e8981632655cb9/tensorflow/compiler/mlir/tensorflow/transforms/legalize_hlo_patterns.td#L256)注释中的有所提及，如下：

```
 frac = x - floor(x)
```



# 四、对比分析

Pytorch与TensorFlow思路一致。

# 五、设计思路与实现方案

## 命名与参数设计

API设计为`paddle.frac(x, name=None)`和`Tensor.frac(x, name=None)`

## 底层OP设计

使用现有C++ Op组合完成，无需设计底层OP

## API实现方案

使用`trunc`与`elementwise_sub`组合实现，实现位置为`paddle/tensor/math.py`与`trunc`，`sum`和`nansum`等方法放在一起：

1. 首先使用`trunc`获取输入tensor的整数部分；
2. 再用输入x减去上一步得到的整数部分即可获取小数部分。

# 六、测试和验收的考量

测试考虑的case如下：

- 在动态图、静态图下，与numpy结果的一致性。

# 七、可行性分析和排期规划

方案主要依赖现有Paddle C++ Op组合而成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无