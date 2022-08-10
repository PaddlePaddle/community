# CINN arange 设计文档
|API名称 | arange | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-08-02 | 
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20200802_cinn_api_design_arange.md<br> | 


# 一、概述

## 1、相关背景
`arange`是神经网络编译器中基础的算子，向算子输入一个数值区间的边界，以及步长`step`，算子输出一个间隔相等的序列

## 2、名词解释
tensor：张量，形式为多维数组  
step：步长 ，序列中相邻的两个元素的差值

## 3、功能目标
实现`arange`算子。  
算子输入起始 $start$ ，终点 $stop$ ，以及步长 $step$   
算子输出序列 $(x_0,x_1,...,x_n)$ ，序列长度 $n=\left [ (start-stop)/step \right ]$，序列满足 $x_0=start$ ， $x_{i+1} - x_i = step$ $(0 \leqslant i < n)$  
算子的输入参数可能有异常情况，并且部分输入参数可缺省，需考虑处理。
## 4、意义
实现`arange`算子，将进一步完善CINN的基础算子库。

# 二、CINN现状
CINN框架暂不支持`arange`算子，需要实现。

# 三、业内方案调研
1. tvm实现`arange`的核心代码如下，使用了lambda表达式实现功能
```c++
inline Tensor arange(const PrimExpr& start, const PrimExpr& stop, const PrimExpr& step,
                     DataType dtype, std::string name = "T_arange", std::string tag = kInjective) {
  PrimExpr num_elem = tvm::cast(
      tvm::DataType::Int(32), tvm::ceil(tvm::cast(tvm::DataType::Float(32), stop - start) / step));
  Array<PrimExpr> shape;
  return compute(
      {num_elem},
      [&](const Array<Var>& indices) { return tvm::cast(dtype, start + step * indices[0]); }, name,
      tag);
}
```
2. xla实现`arange`的核心代码如下
```c++
torch::lazy::NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
                            const at::Scalar& step,
                            at::ScalarType scalar_type) {
  xla::PrimitiveType type = MakeXlaPrimitiveType(scalar_type,
                                                 /*device=*/nullptr);
  XLA_CHECK_NE(step.toDouble(), 0.0);
  XLA_CHECK(!std::isnan(start.toDouble()) && !std::isnan(end.toDouble()))
      << "unsupported range: " << start.toDouble() << " -> " << end.toDouble();
  XLA_CHECK((start.toDouble() <= end.toDouble() && step.toDouble() > 0.0) ||
            (start.toDouble() >= end.toDouble() && step.toDouble() < 0.0));
  xla::Literal values;
  switch (type) {
    case xla::PrimitiveType::BF16:
      values = XlaHelpers::Range<tensorflow::bfloat16>(
          static_cast<tensorflow::bfloat16>(start.toFloat()),
          static_cast<tensorflow::bfloat16>(end.toFloat()),
          static_cast<tensorflow::bfloat16>(step.toFloat()));
      break;
    case xla::PrimitiveType::F16:
      values =
          XlaHelpers::Range<xla::half>(static_cast<xla::half>(start.toHalf()),
                                       static_cast<xla::half>(end.toHalf()),
                                       static_cast<xla::half>(step.toHalf()));
      break;
    case xla::PrimitiveType::F32:
      values = XlaHelpers::Range<float>(start.toFloat(), end.toFloat(),
                                        step.toFloat());
      break;
    case xla::PrimitiveType::F64:
      values = XlaHelpers::Range<double>(start.toDouble(), end.toDouble(),
                                         step.toDouble());
      break;
    case xla::PrimitiveType::U8:
      values = XlaHelpers::Range<uint8_t>(start.toByte(), end.toByte(),
                                          step.toByte());
      break;
    case xla::PrimitiveType::S8:
      values = XlaHelpers::Range<int8_t>(start.toChar(), end.toChar(),
                                         step.toChar());
      break;
    case xla::PrimitiveType::S16:
      values = XlaHelpers::Range<int16_t>(start.toShort(), end.toShort(),
                                          step.toShort());
      break;
    case xla::PrimitiveType::U16:
      values =
          XlaHelpers::Range<uint16_t>(start.toInt(), end.toInt(), step.toInt());
      break;
    case xla::PrimitiveType::S32:
      values =
          XlaHelpers::Range<int32_t>(start.toInt(), end.toInt(), step.toInt());
      break;
    case xla::PrimitiveType::U32:
      values = XlaHelpers::Range<uint32_t>(start.toLong(), end.toLong(),
                                           step.toLong());
      break;
    case xla::PrimitiveType::S64:
      values = XlaHelpers::Range<int64_t>(start.toLong(), end.toLong(),
                                          step.toLong());
      break;
    case xla::PrimitiveType::U64:
      values = XlaHelpers::Range<uint64_t>(start.toLong(), end.toLong(),
                                           step.toLong());
      break;
    default:
      XLA_ERROR() << "XLA type not supported: " << type;
  }
  return torch::lazy::MakeNode<Constant>(std::move(values));
}
```
主要函数XlaHelpers::Range的实现
```c++
template <typename T>
static xla::Literal Range(T start, T end, T step) {
  return xla::LiteralUtil::CreateR1<T>(xla::util::Range<T>(start, end, step));
}

//xla::util::Range
template <typename T>
std::vector<T> Range(T start, T end, T step = 1) {
  std::vector<T> result;
  result.reserve(static_cast<size_t>((end - start) / step));
  if (start < end) {
    for (; start < end; start += step) {
      result.push_back(start);
    }
  } else {
    for (; start > end; start += step) {
      result.push_back(start);
    }
  }
  return result;
}
```

# 四、对比分析
tvm与xla的arange实现方法基本类似。

# 五、设计思路与实现方案

## 命名与参数设计
start：区间起点（且区间包括此值），默认值为0。   
stop：区间终点（且通常区间不包括此值）。  
step：均匀分割的步长，默认值为1。  
dtype：输出`tensor`的数据类型，支持int32、int64、float32、float64。默认值为float32。  

## 底层OP设计
1. 在 `cinn/hlir/op/contrib/arange.h` 里声明`arange`算子。
2. 在 `cinn/hlir/op/contrib/arange.cc` 里实现`arange`算子和 `strategy`。
## API实现方案
1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::Arange`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `NetBuilder::Arange`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `NetBuilder` 添加 `arange` 接口，并绑定到 `NetBuilder::Arange`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

python通过Builder类的方法调用`arange`。
```python
builder = NetBuilder("test_basic")
b = builder.arange(1,10,1,"int32")
```
# 六、测试和验收的考量
1. 提供基础的 demo 文件。
2. 在`cinn/hlir/op/contrib/arange_test.cc`中添加对底层OP进行测试的代码。
3. 在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
4. 提交 API 说明到相应的文档中。
# 七、可行性分析和排期规划
- 可行性分析：CINN已实现Builder、Expr IR、算子注册等模块，在CINN已有的框架基础上能够很好地增加算子功能。
- 排期规划：预计9月1日完成算子实现、功能测试以及文档

# 八、影响面
对其他模块无影响。

# 附件及参考资料
1. [tvm的arange实现代码](https://github.com/apache/tvm/blob/111169c7df2831ab8ee40d5388ebcfcf551fd86f/include/tvm/topi/transform.h)  
2. [xla的arange实现代码](https://github.com/pytorch/xla/blob/f72dcc655d8adbdef36e1f5c724a7dc8c2610fce/torch_xla/csrc/ops/ops.cpp)  
3. [深度学习框架开发指南-飞桨黑客松3.0](https://aistudio.baidu.com/aistudio/course/introduce/26351?directly=1&shared=1)  
4. [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
5. [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
6. [CINN算子开发示例：pool2d_grad算子](https://github.com/PaddlePaddle/CINN/pull/858)  
7. [CINN IR DSL在C++的matmul写法例子](https://github.com/PaddlePaddle/CINN/blob/develop/tutorials/matmul.cc)  
