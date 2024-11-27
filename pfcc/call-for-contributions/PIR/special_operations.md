# PIR 组网插入隐式算子的场景

## 一、背景
PIR 组网过程中存在『插入隐式算子』的场景，所谓『隐式算子』是指：非用户直接调用 API 接口插入的对应算子，而是底层组网逻辑及执行逻辑中自动插入的算子，这里算子包括三类：

* 组网过程中：为可变 Attribute 插入的 pd_op.full / pd_op.full_int_array 算子
* 组网过程中：针对 pir::VectorType 插入的 builtin.combine / builtin.split 算子
* 执行过程中：为静态 kernel 选择插入的 pd_op.shadow_feed 算子

## 二、隐式算子介绍
|插入的隐式算子|含义|
|-|-|
|pd_op.full / pd_op.full_int_array|飞桨的算子定义包含可变 Attribute 的概念，对于可变 Attribute，用户的组网 API 可传入一个常量、也可传入一个 Tensor/Value。在 PIR 的算子定义体系下，可变 Attribute 都将被视为输入变量，因此，当用户 API 传入一个常量的时候，将在组网代码中通过自动插入 pd_op.full / pd_op.full_int_array 将输入的常量转换为变量，再构造对应的算子。 包含可变 Attribute 的算子集合：在 paddle/phi/ops/yaml/op_compat.yaml 中搜索 scalar 及 int_array 标记的属性。|
|builtin.combine / builtin.split|这两个算子针对 pir::VectorType 引入的辅助算子，用于将一组具有相同 Type 的 Value 拼接成一个 VectorType 的 Value，或者将 VectorType 的 Value 拆分成多个具有相同 Type 的 Value。 算子定义过程中，会出现上述内容的都是输入/输出包含 Tensor[] 类型的算子，例如：concat 算子的输入 Tensor[] x。|
|pd_op.shadow_feed|为执行流程中全静态选 Kernel 所引入的隐式算子，该算子的签名是：out = shadow_feed(x, dst_place_type)，作用是将输入 x 拷贝/共享到 dst_place_type，若 x 的 place 与 dst_place_type 不一致，则执行 memcpy，否则 out 直接与 x share data。 算子定义见：paddle/phi/ops/yaml/inconsistent/static_ops.yaml；Kernel 定义见：paddle/phi/kernels/impl/data_impl.h。|

##### 1、为可变 Attribute 隐式插入的 pd_op.full / pd_op.full_int_array：
> 备注：包含可变 Attribute 的算子集合：在 paddle/phi/ops/yaml/op_compat.yaml 中搜索 scalar 及 int_array 标记的属性。
以 Adam_OP 为例，参数 beta1_，beta2_，epsilon_ 等参数为可变 Attribute、在算子的 Build 接口中可以看到 FullOp 的隐式插入过程：

```
void Adam_Op::Build(pir::Builder &builder, pir::OperationArgument &argument, pir::Value param_, pir::Value grad_, pir::Value learning_rate_, pir::Value moment1_, pir::Value moment2_, pir::Value beta1_pow_, pir::Value beta2_pow_, pir::Value master_param_, pir::Value skip_update_, float beta1, float beta2, float epsilon, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {
  VLOG(4) << "Start build Adam_Op";
  // Generate scalar mutable attribute: beta1
  paddle::dialect::FullOp full_beta1_op = builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1}, beta1, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::Value beta1_ = full_beta1_op->result(0);
      // Generate scalar mutable attribute: beta2
  paddle::dialect::FullOp full_beta2_op = builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1}, beta2, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::Value beta2_ = full_beta2_op->result(0);
      // Generate scalar mutable attribute: epsilon
  paddle::dialect::FullOp full_epsilon_op = builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1}, epsilon, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::Value epsilon_ = full_epsilon_op->result(0);
    
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {param_, grad_, learning_rate_, moment1_, moment2_, beta1_pow_, beta2_pow_, master_param_, skip_update_, beta1_, beta2_, epsilon_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_lazy_mode = pir::BoolAttribute::get(pir::IrContext::Instance(), lazy_mode);
  argument_attributes.insert({"lazy_mode", attr_lazy_mode});
  pir::Attribute attr_min_row_size_to_use_multithread = pir::Int64Attribute::get(pir::IrContext::Instance(), min_row_size_to_use_multithread);
  argument_attributes.insert({"min_row_size_to_use_multithread", attr_min_row_size_to_use_multithread});
  pir::Attribute attr_multi_precision = pir::BoolAttribute::get(pir::IrContext::Instance(), multi_precision);
  argument_attributes.insert({"multi_precision", attr_multi_precision});
  pir::Attribute attr_use_global_beta_pow = pir::BoolAttribute::get(pir::IrContext::Instance(), use_global_beta_pow);
  argument_attributes.insert({"use_global_beta_pow", attr_use_global_beta_pow});

  std::vector<pir::Type> argument_outputs = Adam_Op::InferMeta(argument_inputs, &argument_attributes);
  argument.AddAttributes(argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}
```
以 CropOp 为例，参数 shape，offset 等参数为可变 Attribute，在算子的 Build 接口中可以看到 FullIntArrayOp 的隐式插入过程：

```
void CropOp::Build(pir::Builder &builder, pir::OperationArgument &argument, pir::Value x_, const std::vector<int64_t>& shape, const std::vector<int64_t>& offsets) {
  VLOG(4) << "Start build CropOp";

  // Generate int_array mutable attribute: shape
  paddle::dialect::FullIntArrayOp full_shape_op = builder.Build<paddle::dialect::FullIntArrayOp>(shape, phi::DataType::INT64, phi::CPUPlace());
  pir::Value shape_ = full_shape_op->result(0);
      // Generate int_array mutable attribute: offsets
  paddle::dialect::FullIntArrayOp full_offsets_op = builder.Build<paddle::dialect::FullIntArrayOp>(offsets, phi::DataType::INT64, phi::CPUPlace());
  pir::Value offsets_ = full_offsets_op->result(0);
    
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_, offsets_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};

  std::vector<pir::Type> argument_outputs = CropOp::InferMeta(argument_inputs, &argument_attributes);
  argument.AddAttributes(argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}
```

##### 2、为 VectorType 隐式插入的 builtin.combine / builtin.split
以 ConcatOp 为例：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=51d46f82fc224e489a33e931c80bf42c&docGuid=VluTZj7ya9nChk "")
```
(%0) = "pd_op.data" () {dtype:(pd_op.DataType)float64,name:"_jst.0.args.0",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[5,1,4,5],stop_gradient:[false]} : () -> builtin.tensor<5x1x4x5xf64>
(%1) = "pd_op.data" () {dtype:(pd_op.DataType)float64,name:"_jst.0.args.1",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[5,2,4,5],stop_gradient:[false]} : () -> builtin.tensor<5x2x4x5xf64>
(%2) = "pd_op.data" () {dtype:(pd_op.DataType)float64,name:"_jst.0.args.2",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[5,3,4,5],stop_gradient:[false]} : () -> builtin.tensor<5x3x4x5xf64>
(%3) = "pd_op.full" () {dtype:(pd_op.DataType)int32,place:(pd_op.Place)Place(cpu),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Double)1} : () -> builtin.tensor<1xi32>
(%4) = "builtin.combine" (%0, %1, %2) {stop_gradient:[false]} : (builtin.tensor<5x1x4x5xf64>, builtin.tensor<5x2x4x5xf64>, builtin.tensor<5x3x4x5xf64>) -> vec[builtin.tensor<5x1x4x5xf64>,builtin.tensor<5x2x4x5xf64>,builtin.tensor<5x3x4x5xf64>]
(%5) = "pd_op.concat" (%4, %3) {stop_gradient:[false]} : (vec[builtin.tensor<5x1x4x5xf64>,builtin.tensor<5x2x4x5xf64>,builtin.tensor<5x3x4x5xf64>], builtin.tensor<1xi32>) -> builtin.tensor<5x6x4x5xf64>
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2d649a03b27b42f58ab66b451ec30448&docGuid=VluTZj7ya9nChk "")
以 SplitOp 为例：

```
(%0) = "pd_op.data" () {dtype:(pd_op.DataType)float64,name:"_jst.0.args.0",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,5,6],stop_gradient:[false]} : () -> builtin.tensor<4x5x6xf64>
(%1) = "pd_op.full_int_array" () {dtype:(pd_op.DataType)int64,place:(pd_op.Place)Place(cpu),stop_gradient:[true],value:[(Int64)2,(Int64)1,(Int64)2]} : () -> builtin.tensor<3xi64>
(%2) = "pd_op.full" () {dtype:(pd_op.DataType)int32,place:(pd_op.Place)Place(cpu),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Double)1} : () -> builtin.tensor<1xi32>
(%3) = "pd_op.split" (%0, %1, %2) {stop_gradient:[false]} : (builtin.tensor<4x5x6xf64>, builtin.tensor<3xi64>, builtin.tensor<1xi32>) -> vec[builtin.tensor<4x2x6xf64>,builtin.tensor<4x1x6xf64>,builtin.tensor<4x2x6xf64>]
(%4, %5, %6) = "builtin.split" (%3) {stop_gradient:[false,false,false]} : (vec[builtin.tensor<4x2x6xf64>,builtin.tensor<4x1x6xf64>,builtin.tensor<4x2x6xf64>]) -> builtin.tensor<4x2x6xf64>, builtin.tensor<4x1x6xf64>, builtin.tensor<4x2x6xf64>
```
