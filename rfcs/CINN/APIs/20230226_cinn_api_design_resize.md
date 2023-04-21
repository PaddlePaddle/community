# CINN resize 设计文档
|API名称 | resize | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-02-26 | 
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230226_cinn_api_design_resize.md<br> | 


# 一、概述

## 1、相关背景
CINN是一种在不改变模型代码的条件下加速飞桨模型运行速度的深度学习编译器。在对接上层框架时，编译器会将上层的框架算子进一步拆分为若干基础算子，这样做的目的一方面是为了减少算子开发的工作量，仅实现有限的基础算子便可以组合出大量的上层框架算子；另一方面便于算子融合技术在编译器中可以实现跨算子自动融合，减少最终执行时的kernel数目和访存开销，达到更好的性能。

为了丰富 CINN 的基础算子，本次任务计算增加 `resize` 算子。

## 2、名词解释
NCHW ：一种图的数据格式。N 指 Batch，C 指 Channel，H 指 Height，W 指 width。

## 3、功能目标
实现 `resize` 算子，将输入图片通过指定插值方法调整为指定大小，输入图片应该是 4-D 张量，且形状为`[N, C, H, W]`，注意调整仅适用于H、W对应维度。


## 4、意义
实现 `resize` 算子，将能进一步完善CINN的基础算子库。

# 二、CINN现状
CINN框架暂不支持 `resize` 算子，需要实现。

# 三、业内方案调研
**TVM 的 `resize` 算子**

在 TVM 中，与本次任务将要实现的算子对应的是 `resize2d` 算子，核心代码如下：
```c++
Expr MakeResize2D(Expr data, Expr size, Expr roi, String layout, String method,
                  String coordinate_transformation_mode, String rounding_method, double cubic_alpha,
                  double cubic_exclude, double extrapolation_value, DataType out_dtype) {
  auto attrs = make_object<Resize2DAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  attrs->rounding_method = rounding_method;
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("dyn.image.resize2d");
  return Call(op, {data, size, roi}, Attrs(attrs), {});
}
```
代码在 C++ 侧通过 Call 调用 python 侧的 `resize2d` 实现，python 侧已经以 te 形式实现了算子。

[resize2d compute的核心代码](https://github.com/apache/tvm/blob/5e652c1a7aa173cec6f9e68207b410ad06b2fcec/python/tvm/topi/image/resize.py#L531)


# 四、对比分析
TVM 的 `resize2d` 算子实现详细，可作为参考。本次任务计划以 extern call 的方式实现 `resize` 算子，使用 CINN IR 实现 Compute。

# 五、设计思路与实现方案

## 命名与参数设计
**算子参数：**

|   类别    |    类型     |   名称    |        Shape         |                                                                             描述                                                                              |
| :-------: | :---------: | :-------: | :------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Input   | Tensor\<T\> |     x     |  [N, C, in_H, in_W]  |                                                                           输入张量                                                                            |
| Attribute | vector<int> | out_shape |    [out_H, out_W]    |                                                         调整后的张量大小，只需指定H、W两个维度上的值                                                          |
| Attribute |   string    |   mode    |          -           | 指定插值方法，可选项包括：<br>**nearest**(最近邻插值，选取H和W维上最近的值);<br>**bilinear**(双线性插值，选取H和W维上相邻四个点做线性插值);<br>**bicubic**(二次立方插值).<br>默认值bilinear |
|  Output   | Tensor\<T\> |    out    | [N, C, out_H, out_W] |                                                              输出张量，数据类型与输入张量相同同                                                               |

**支持的数据类型:**
 
`uint8`、`int32`

## 底层OP设计
在 `cinn/hlir/op/contrib` 中新增 `resize` 算子。
```c++
ir::Tensor Resize(const ir::Tensor &x,
                  vector out_shape,
                  std::string mode, 
                  std::string &output_name)
```
实现 `resize` 的 strategy：`StrategyForResize`、`InferDtypeForResize` 和 `InferShapeForResize`，并注册算子。

## API实现方案
- c++ 接口
  
在 `cinn/frontend` 中的 `NetBuild` 类中增加 `Resize` 函数。

- python 接口
  
在 `cinn/pybind/frontend.cc` 中增加 `resize` 算子的接口。


# 六、测试和验收的考量。
在 `python/tests/ops/test_resize_op.py` 中添加 `resize` 算子的测试。测试内容覆盖所有 resize 模式，所有支持的数据类型，以及常见的 shape。 

# 七、可行性分析和排期规划
- 可行性分析

CINN中已经实现了大量的基础算子，在现有的框架基础上能够很好地增加算子功能。

- 排期规划

2月27日 ~ 3月11日完成 API 的开发与调试。

3月12日 ~ 3月19日完成测试代码的开发。

# 八、影响面
本次任务影响模块如下，

`cinn\backends`，`cinn\frontend`，`cinn\hlir`，`cinn\pybind`，`cinn\runtime`。

均是在原模块内增加代码，不影响原模块的已有功能。

# 附件及参考资料
1. [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
2. [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
3. [CINN IR DSL在C++的matmul写法例子](https://github.com/PaddlePaddle/CINN/blob/develop/tutorials/matmul.cc) 
4. [CINN算子开发示例：pool2d_grad算子](https://github.com/PaddlePaddle/CINN/pull/858)  
