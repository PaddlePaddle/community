# CINN cast 设计文档

| API名称                                                      | cast                                |
| ---------------------------------------------------------- | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                             |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-08-11                                       |
| 版本号                                                        | V1.0                                             |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                          |
| 文件名                                                        | 20220729_api_design_for_cast.md<br> |

# 一、概述

## 1、相关背景

`cast` 是众多神经网络编译器中基础的算子。转化输入的数据类型，例如将Int(32)变为Float(32)。
为了提升 CINN API 丰富度，需要扩充 API `cast`。

## 2、名词解释

- 张量/Tensor：指高维数组。
- cast：转化输入的数据类型。
- dtype：数据类型。

## 3、功能目标

实现cast功能，将输入转换为指定数据类型。例如，对于张量 $A$ = [1, 2, 3]，
cast( $A$, dtype = Float(32) 结果为 $[1.0f, 2.0f, 3.0f]$。

## 4、意义

为神经网络编译器 CINN 增加基础算子`cast`。

# 二、CINN现状

对CINN框架目前可以调用底层ir实现，即ir::Cast::Make，但没有相应的hlir实现，因此有必要实现 `cast` hlir API。

# 三、业内方案调研

- [TVM](https://github.com/apache/tvm/blob/main/src/relay/transforms/canonicalize_cast.cc)：
对张量中每个元素使用更底层的ir进行类型转化。
  
  ```cpp
  class CastCanonicalizer : public ExprMutator {
 public:
  CastCanonicalizer() : cast_op_(Op::Get("cast")) {}

  Expr VisitExpr_(const CallNode* call) {
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");

    if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto pattern = fpattern[GetRef<Op>(opnode)];
      if (pattern <= kBroadcast) {
        Array<Expr> call_args = call->args;
        bool unchanged = true;
        for (size_t i = 0; i < call_args.size(); ++i) {
          Expr arg = call_args[i];
          Expr new_arg = GetNewCallArg(arg);
          if (!arg.same_as(new_arg)) {
            call_args.Set(i, new_arg);
            unchanged = false;
          }
        }
        if (unchanged) {
          return GetRef<Expr>(call);
        }
        return Call(call->op, call_args, call->attrs, call->type_args);
      }
    }

    Expr new_expr = ExprMutator::VisitExpr_(call);
    return new_expr;
  }

 private:
  std::unordered_map<const Object*, size_t> ref_counter_;
  // cast op is frequently checked for equivalence. Therefore, we cache it to
  // reduce lookup overhead.
  const Op& cast_op_;

  Expr GetNewCallArg(const Expr& e) {
    // if e is a upcast and ref count > 1, create an copy; otherwise call the default visitor
    Expr new_expr = this->VisitExpr(e);

    if (const CallNode* call = e.as<CallNode>()) {
      if (call->op == cast_op_) {
        auto attrs = call->attrs.as<CastAttrs>();
        const auto* from_type = call->args[0]->type_as<TensorTypeNode>();
        ICHECK(from_type);

        if (from_type->dtype.bits() < attrs->dtype.bits()) {
          if (++ref_counter_[call] > 1) {
            const CallNode* new_call = new_expr.as<CallNode>();
            ICHECK(new_call);
            ICHECK(new_call->op == cast_op_);
            return Call(new_call->op, new_call->args, new_call->attrs, new_call->type_args);
          }
        }
      }
    }
    return new_expr;
  }
};

Expr CanonicalizeCast(const Expr& e) { return CastCanonicalizer().Mutate(e); }

namespace transform {

Pass CanonicalizeCast() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CanonicalizeCast(f));
      };
  return CreateFunctionPass(pass_func, 3, "CanonicalizeCast", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CanonicalizeCast").set_body_typed(CanonicalizeCast);

}  // namespace transform

  ```

- [XLA](https://github.com/pytorch/xla/blob/3d24d955b6121289a3c8bb86eda541fca7a0d69f/torch_xla/csrc/ops/cast.cpp)：与TVM类似。

```cpp
namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           xla::PrimitiveType type) {
  xla::Shape shape = GetXlaShape(input);
  shape.set_element_type(type);
  return shape;
}

}  // namespace

Cast::Cast(const torch::lazy::Value& input, xla::PrimitiveType type)
    : XlaNode(xla_cast, {input}, NodeOutputShape(input, type),
              /*num_outputs=*/1, torch::lazy::MHash(static_cast<int>(type))),
      type_(type) {}

Cast::Cast(const torch::lazy::Value& input, at::ScalarType dtype,
           c10::optional<at::ScalarType> stype)
    : XlaNode(xla_cast, {input},
              NodeOutputShape(input,
                              MakeXlaPrimitiveType(dtype, /*device=*/nullptr)),
              /*num_outputs=*/1,
              torch::lazy::MHash(101, static_cast<int>(dtype),
                                 torch::lazy::OptionalOr<int>(stype, -1))),
      type_(MakeXlaPrimitiveType(dtype, /*device=*/nullptr)),
      dtype_(dtype),
      stype_(stype) {}

torch::lazy::NodePtr Cast::Clone(torch::lazy::OpList operands) const {
  return dtype_ ? torch::lazy::MakeNode<Cast>(operands.at(0), *dtype_, stype_)
                : torch::lazy::MakeNode<Cast>(operands.at(0), type_);
}

XlaOpVector Cast::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType raw_from =
      stype_ ? TensorTypeToRawXlaType(*stype_) : input_shape.element_type();
  xla::PrimitiveType raw_to = dtype_ ? TensorTypeToRawXlaType(*dtype_) : type_;
  xla::XlaOp output =
      ConvertToRaw(input, input_shape.element_type(), raw_from, type_, raw_to,
                   /*device=*/nullptr);
  return ReturnOp(output, loctx);
}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", type=" << xla::primitive_util::LowercasePrimitiveTypeName(type_);
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  if (stype_) {
    ss << ", stype=" << *stype_;
  }
  return ss.str();
}

}  // namespace torch_xla
```

# 四、对比分析

TVM 与 XLA 实现方案类似。

# 五、设计思路与实现方案

## 命名与参数设计

- A：输入张量
- dtype：指定数据类型
- name：输出名称

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/cast.h` 里声明`cast`算子。
2. 在 `cinn/hlir/op/contrib/cast.cc` 里实现`cast`算子和 `strategy`。

## API实现方案

例如，对于张量 A = [1, 1, 1]，
cast( A, dtype = Float(32)) 结果为[1.0f, 1.0f, 1.0f]，
cast( A, dtype = bool) 结果为[True, True, True]。

1. 在 `cinn/frontend/net_build.h` 里声明 `BaseBuilder::Cast`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `BaseBuilder::Cast`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `BaseBuilder` 添加 `cast` 接口，并绑定到`BaseBuilder::Cast`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

通过使用 Builder 类的方法调用 Cast。

```python
builder = NetBuilder("test_basic")
a = builder.create_input(Int(32), (8, 24, 124), "A1")
b = builder.cast(a, Float(32))  # 输出类型变为Float(32)
```

# 六、测试和验收的考量

1. 在`cinn/hlir/op/contrib/cast_test.cc`和`cinn/hlir/op/contrib/cast_test.cc`中添加对底层OP进行测试的代码，在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
2. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：预计8月25日前完成

# 八、影响面

对其他模块无影响。

# 附件及参考资料

[TVM文档](https://github.com/apache/tvm/blob/main/src/relay/transforms/canonicalize_cast.cc)
[XLA文档](https://github.com/pytorch/xla/blob/3d24d955b6121289a3c8bb86eda541fca7a0d69f/torch_xla/csrc/ops/cast.cpp)
[CINN文档](https://paddlepaddle.github.io/CINN/)
