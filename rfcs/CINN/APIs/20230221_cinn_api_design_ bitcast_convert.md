# CINN bitcast_convert设计文档

|API名称 | 新增API名称 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | XDUWQ| 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-02-21 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 20230221_cinn_api_design_bitcast_convert.md<br> | 


# 一、概述
## 1、相关背景
`bitcast_convert` 是神经网络中的算子，实现的功能是在不改变底层存储的情况下，强制转换数据类型。
若转换前后数据类型的字节大小不相同，则形状会改变。比如一个 shape=[10] 的 float32 类型数据被强制转换为 float16 类型后，其 shape 应为[10, 2]。

## 2、功能目标
在不复制数据的情况下，将张量从一种类型转换为另一种类型。若转换前后数据类型的字节大小不相同，则形状会改变。
输入是`inputs` 和 目标转换的类型`dtype`，输出是`outputs`。

## 3、意义
实现`bitcast_convert` 算子，将进一步完善CINN的基础算子库。

# 二、CINN现状
CINN框架暂不支持`bitcast_convert`算子，需要实现。

# 三、业内方案调研
`tensorflow` 中有bitcast算子实现，核心代码如下：
```c++
static void BitcastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<BitcastOp*>(kernel);
  int dim_count = 0;

  TF_Tensor* tensor;
  TF_Status* status = TF_NewStatus();
  TF_GetInput(ctx, 0, &tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    dim_count = TF_NumDims(tensor);
    if (!(k->in_size >= k->out_size ||
          (dim_count > 0 &&
           TF_Dim(tensor, dim_count - 1) == k->out_size / k->in_size))) {
      std::ostringstream err;
      err << "Cannot bitcast from " << k->input_data_type << " to "
          << k->output_data_type;
      TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
    }
  }

  if (TF_GetCode(status) == TF_OK) {
    auto* dims = new int64_t[dim_count + 1];
    int new_dim_count = dim_count;
    for (int dim = 0; dim < dim_count; ++dim) {
      dims[dim] = TF_Dim(tensor, dim);
    }
    if (k->out_size < k->in_size) {
      dims[new_dim_count++] = static_cast<int64_t>(k->in_size / k->out_size);
    } else if (k->out_size > k->in_size) {
      --new_dim_count;
    }

    TF_Tensor* output = TF_AllocateTensor(k->output_data_type, dims, 0,
                                          TF_DataTypeSize(k->output_data_type));
    TF_TensorBitcastFrom(tensor, k->output_data_type, output, dims,
                         new_dim_count, status);
    if (TF_GetCode(status) == TF_OK) {
      TF_SetOutput(ctx, 0, output, status);
    }
    delete[] dims;
    TF_DeleteTensor(output);
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  TF_DeleteStatus(status);
  TF_DeleteTensor(tensor);
}
```
`xla` 中有详细实现，核心代码如下：
获取处理后的shape

```c++
/* static */ StatusOr<Shape> ShapeInference::InferBitcastConvertShape(
    const Shape& operand_shape, PrimitiveType new_element_type) {
  auto old_element_type = operand_shape.element_type();
  if (primitive_util::IsComplexType(old_element_type) !=
      primitive_util::IsComplexType(new_element_type)) {
    return InvalidArgument("Conversion between complex and real type %s => %s.",
                           ShapeUtil::HumanString(operand_shape),
                           PrimitiveType_Name(new_element_type));
  }
  if (!operand_shape.IsArray() ||
      !primitive_util::IsArrayType(new_element_type)) {
    // Note: we may want to support tuple conversions via this operation in the
    // future, by recursing into the tuple elements to check all sub-conversions
    // are valid. For now we just reject them, though.
    return InvalidArgument(
        "Cannot convert from or to tuple type; requested conversion: %s => %s.",
        ShapeUtil::HumanString(operand_shape),
        PrimitiveType_Name(new_element_type));
  }

  int input_bitwidth = primitive_util::BitWidth(old_element_type);
  int output_bitwidth = primitive_util::BitWidth(new_element_type);
  if (std::max(input_bitwidth, output_bitwidth) %
          std::min(input_bitwidth, output_bitwidth) !=
      0) {
    return InvalidArgument(
        "Cannot bitcast types with undivisible bit-widths: %s => %s.",
        PrimitiveType_Name(old_element_type),
        PrimitiveType_Name(new_element_type));
  }
  int ratio = std::max(output_bitwidth, input_bitwidth) /
              std::min(output_bitwidth, input_bitwidth);

  Shape new_shape = operand_shape;
  new_shape.set_element_type(new_element_type);
  if (input_bitwidth > output_bitwidth) {
    ShapeUtil::AppendMinorDimension(ratio, &new_shape);
  } else if (input_bitwidth < output_bitwidth) {
    int last_dimension_idx = operand_shape.dimensions_size() - 1;
    if (operand_shape.dimensions_size() < 1 ||
        operand_shape.dimensions(last_dimension_idx) != ratio) {
      return InvalidArgument(
          "Last dimension of input shape=%d is not equal to ratio of "
          "bit-widths=%d "
          "for bitcast-convert from %s to %s",
          operand_shape.dimensions(last_dimension_idx), ratio,
          ShapeUtil::HumanString(operand_shape),
          PrimitiveType_Name(new_element_type));
    }
    new_shape.DeleteDimension(last_dimension_idx);
  }
  return new_shape;
}
```

转换
```c++
XlaOp XlaBuilder::BitcastConvertType(XlaOp operand,
                                     PrimitiveType new_element_type) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferBitcastConvertShape(
                                         *operand_shape, new_element_type));
    return BitcastConvertTypeInternal(shape, operand);
  });
}

StatusOr<XlaOp> XlaBuilder::BitcastConvertTypeInternal(const Shape& shape,
                                                       XlaOp operand) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kBitcastConvert,
                        {operand});
}
```

# 四、对比分析
`xla` 的实现很详细，可以借鉴xla的实现。


# 五、设计思路与实现方案



## 命名与参数设计
* `A`：Tensor类型，表示输入张量
* `dytpe`: string类型，表示转换输出类型

## 底层OP设计
1. 在 `cinn/hlir/op/contrib/bitcast_convert.h` 里声明`bitcast_convert`算子。
2. 在 `cinn/hlir/op/contrib/bitcast_convert.cc` 里实现`bitcast_convert`算子和 `strategy`。

## API实现方案
1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::Bitcast_convert`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `NetBuilder::Bitcast_convert`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `NetBuilder` 添加 `bitcast_convert` 接口，并绑定到 `NetBuilder::Bitcast_convert`。
4. 在 `cinn/python/tests` 中添加 `test_bitcast_convert_op.py` 单测。

python通过Builder类的方法调用`bitcast_convert`。
```python
builder = NetBuilder("test_basic")
b = builder.bitcast_convert([10], "float32")
```

# 六、测试和验收的考量
1. 提供基础的 demo 文件。
2. 在 `cinn/python/tests` 中添加 `test_bitcast_convert_op.py` 单测。单测要求测试对各种类型的转换，至少包括`float32->float32`、`float32->int32`、`float32->float64`、`float64->float32`、`int64->bool`、`bool->int64`等。shape同样需要考虑至少一维、二维、四维等，数据数目应考虑1、1024 、2048等各类常见大小
3. 提交 API 说明到相应的文档中。

# 七、可行性分析和排期规划
- 可行性分析：CINN已实现Builder、Expr IR、算子注册等模块，在CINN已有的框架基础上能够很好地增加算子功能。
- 排期规划：预计3月10日前完成算子实现、功能测试以及文档

# 八、影响面
对其他模块无影响。

# 附件及参考资料
* [手把手教你为神经网络编译器CINN增加One-Hot算子](https://blog.csdn.net/PaddlePaddle/article/details/128509915)
* [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
* [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
* [CINN算子开发示例：pool2d_grad算子](https://github.com/PaddlePaddle/CINN/pull/858)  
* [CINN IR DSL在C++的matmul写法例子](https://github.com/PaddlePaddle/CINN/blob/develop/tutorials/matmul.cc)  
