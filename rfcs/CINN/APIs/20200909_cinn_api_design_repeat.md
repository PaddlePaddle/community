# CINN repeat 设计文档
|API名称 | repeat | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-09-09 | 
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20200909_cinn_api_design_repeat.md<br> | 


# 一、概述

## 1、相关背景
CINN 框架实现了许多算子，为了丰富 CINN 的算子功能，本次任务计算增加 `repeat` 算子。

## 2、名词解释
tensor：张量，形式为多维数组。  
axis：轴，指示 tensor 的某个维度。

## 3、功能目标
实现 `repeat` 算子，算子输入包含一个张量 x、以及两个整数 repeats 和 axis。

算子的功能是重复张量 x 中的一些元素，参数 repeats 为重复次数，axis 指定操作 tensor 的哪个轴。

例子如下：
```
x = [[1, 2], [3, 4]]

repeat(x, repeats=2) = [1, 1, 2, 2, 3, 3, 4, 4]

repeat(x, repeats=2, axis=1) = [[1, 1, 2, 2],
                                [3, 3, 4, 4]]
```

## 4、意义
实现 `repeat` 算子，将能进一步完善CINN的基础算子库。

# 二、CINN现状
CINN框架暂不支持 `repeat` 算子，需要实现。

# 三、业内方案调研
1. tvm 的`repeat` 算子

算子的输入参数为 x、repeats 和 axis，算子功能是将张量 x 的 axis 轴进行重复，重复次数为 repeats 。

实现 `repeat` 的实现方法是先构造新 tensor 的 shape，再用 compute 生成新的 tensor。

核心代码如下：

```c++
inline Tensor repeat(const Tensor& x, int repeats, int axis, std::string name = "T_repeat",
                     std::string tag = kBroadcast) {
  int ndim = static_cast<int>(x->shape.size());
  ICHECK(-ndim - 1 <= axis && axis <= ndim)
      << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  ICHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                       << ", but got repeats = " << repeats;
  if (axis < 0) {
    // Calculate offset from last dimension
    axis += ndim;
  }
  Array<PrimExpr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(x->shape[i]);
  }
  new_shape.push_back(repeats * x->shape[axis]);
  for (size_t i = axis + 1; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(
      new_shape,
      [&](const Array<Var>& indices) {
        Array<PrimExpr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        idx.push_back(indexdiv(indices[axis], repeats));
        for (size_t i = axis + 1; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }
        return x(idx);
      },
      name, tag);
}
```

2. xla 的`repeat` 算子

算子的输入参数为 input 和 repeats，算子功能是将张量 input 中的元素进行重复，数组 repeats 指示了各个维度上的重复次数。

核心代码如下：

```c++
xla::XlaOp BuildRepeat(xla::XlaOp input, absl::Span<const int64_t> repeats) {
  const auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_GE(repeats.size(), input_sizes.size())
      << "Number of dimensions of repeat dims can not be smaller than number "
         "of dimensions of tensor";
  size_t broadcast_dims = repeats.size() - input_sizes.size();
  xla::XlaOp repeated = input;
  for (size_t dim = 0; dim < input_sizes.size(); ++dim) {
    std::vector<xla::XlaOp> repeated_inputs(repeats[broadcast_dims + dim],
                                            repeated);
    repeated = xla::ConcatInDim(input.builder(), repeated_inputs, dim);
  }
  if (repeats.size() > input_sizes.size()) {
    std::vector<int64_t> remaining_repeats(repeats.begin(),
                                           repeats.begin() + broadcast_dims);
    repeated = xla::Broadcast(repeated, remaining_repeats);
  }
  return repeated;
}
```

# 四、对比分析
tvm 与 xla 的 `repeat` 算子的输入参数有所不同，tvm 的 `repeat` 算子一次只能将 tensor 的一个维度进行重复，而 xla 的 `repeat`算子一次可将 tensor 的多个维度进行重复。tvm 算子的实现风格与 CINN 更接近。

# 五、设计思路与实现方案

## 命名与参数设计
```c++
Variable Repeat(const Variable& in, int repeats, int axis)
```
`in` 为输入的 tensor。

`repeats` 为重复次数。

`axis` 为重复操作的轴。

## 底层OP设计
在 `cinn/hlir/op/contrib` 中新增 `repeat` 算子。
```c++
std::vector<ir::Tensor> Repeat(const ir::Tensor &in_tensor,
                              int repeats,
                              int axis, 
                              std::string &output_name)
```
实现 `repeat` 的 strategy：`StrategyForRepeat`、`InferDtypeForRepeat` 和 `InferShapeForRepeat`，并注册算子。

## API实现方案
在 `cinn/frontend` 中的 `NetBuild` 类中增加 `Repeat` 函数。
```c++
Variable NetBuilder::Repeat(const Variable& in, int repeats, int axis)
```


# 六、测试和验收的考量。
1. 在 `cinn/hlir/op/contrib/repeat_test.cc` 中添加对底层 OP `repeat` 的测试，测试代码生成的结果是否正确。
2. 在 `cinn/frontend/net_builder_test.cc` 中添加对前端使用 `repeat` 的测试，测试算子的实现是否正确。

# 七、可行性分析和排期规划
- 可行性分析

CINN已实现 Builder、Expr IR、算子注册等模块，在 CINN 已有的框架基础上能够很好地增加算子功能。

- 排期规划

9月9日 ~ 9月14日完成 API 的开发与调试。

9月15日 ~ 9月18日完成测试代码的开发。

# 八、影响面
本次任务影响模块如下，
1. `cinn/hlir/op/contrib`

增加 `repeat.h`、 `repeat.cc`和 `repeat_test.cc` 文件。

2. `cinn/frontend`

修改 `NetBuild` 类，修改`net_builder_test.cc`文件。

# 附件及参考资料
1. [tvm 的 repeat 实现代码](https://github.com/apache/tvm/blob/111169c7df2831ab8ee40d5388ebcfcf551fd86f/include/tvm/topi/transform.h)  
2. [xla 的 repeat 实现代码](https://github.com/pytorch/xla/blob/f72dcc655d8adbdef36e1f5c724a7dc8c2610fce/torch_xla/csrc/data_ops.h)
3. [深度学习框架开发指南-飞桨黑客松3.0](https://aistudio.baidu.com/aistudio/course/introduce/26351?directly=1&shared=1)  
4. [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
5. [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
6. [CINN IR DSL在C++的matmul写法例子](https://github.com/PaddlePaddle/CINN/blob/develop/tutorials/matmul.cc) 
7. [CINN算子开发示例：pool2d_grad算子](https://github.com/PaddlePaddle/CINN/pull/858)  
