# Kernel选择分发体系梳理与优化

| 分享内容     | Kernel选择分发体系梳理与优化                          |
| ------------ | ----------------------------------------------- |
| 提交作者     | Jia Hongyu(@jiahy0825)  |
| 提交时间     | 2022-11-30                                      |
| 版本号       | v1.0                                            |
| 依赖飞桨版本 | develop                                         |
| 文件名       | 20221130_kernel_selection.md |

### 一、基本概念梳理

#### 1.1 以linear运算为例

从一个最为基础的全链接层开始介绍

```python
# paddle/nn/functional/common.py
# Out = XW + b
def linear(x, weight, bias=None, name=None):
if in_dygraph_mode():    # 新动态图
    return _C_ops.linear(x, weight, bias)
else:
    if _in_legacy_dygraph():    # 老动态图
        pre_bias = _legacy_C_ops.matmul_v2(x, weight, 'trans_x', False, 'trans_y', False)
        if bias is None:
            return pre_bias
        return _legacy_C_ops.elementwise_add(pre_bias, bias)
    else:    # 静态图
        ......
        helper.append_op(type='matmul_v2',
                         inputs=inputs,
                         outputs={'Out': tmp},
                         attrs=attrs)
        ......
```

**从外部用户的角度，调用最为常见的`linear`运算，可以观察到：**

- 当前框架有新动态图、老动态图、静态图三种运算模式

- 老动态图中，`linear`运算由`matmul_v2`和`elementwise_add`运算组合而成

从框架开发者的角度，思考如何实现`matmul_v2`运算：

- 数据（输入、输出）：数据存储在哪个硬件上？在内存中的组织形式如何？数据类型是什么？

  - **Backend**：CPU、ONEDNN、GPU、XPU、**ALL_BACKEND**......

  - **Layout**：NCHW、NDHWC、ONEDNN、SPARSE_COO、**ALL_LAYOUT**、kNCHW......

  - **Datatype**：BOOL、INT8、FLOAT32、COMPLEX64、**ALL_DTYPE**......

- 计算（算子与kernel）：使用`matmul_v2`函数，如何处理不同类型的数据？

  - 算子：用户通用的计算接口，调用一个`matmul_v2`算子即可，BLD无关（BLD={backend, layout, datatype}）

  - kernel：内部特化的计算逻辑，适配多个`matmul_v2kernel`，BLD相关

  - kernel选择分发：连接两者之间的桥梁

#### 1.2 OP与kernel概念梳理

**如何定义一个OP？** 运算的名称是什么、输入输出是什么

——运算名称：matmul_v2；输入：x 和 weight；输出：output

**如何调用kernel？** kernel选择分发，贯穿了从OP注册到kernel调用的过程

##### 1.2.1 静/老动态图的OP概念梳理（剪枝版）

`# 前向op`

`OpMaker` # 构造算子，描述算子的输入、输出和属性

`OperatorWithKernel` # 算子类，包括从OP如何运行到kernel（几个重点概念）

- `InferShape` # 算子输出形状的推导函数

- `GetExpectedKernelType` # 算子需要的Kernel类型（kernel选择分发——通用选择）

- `GetKernelTypeForVar` #某个Var特定的Kernel类型（kernel选择分发——特化选择）

`# Op注册`

`REGISTER_OPERATOR` # 将算子注册到全局map中（模版元编程）

```C++
// OP——paddle/fluid/operators/matmul_v2_op.cc
class MatMulV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "tensor of shape (d0, d1 ... M, K)");
    AddInput("Y", "tensor of shape (d0, d1 ... K, N)");
    AddOutput("Out", "tensor of shape (d0, d1 ... M, N)");
    // 添加属性的同时，也会添加 AddAttrChecker，进行属性检查
    AddAttr<bool>("trans_x").SetDefault(false);
    AddAttr<bool>("trans_y").SetDefault(false);
  }
};

class MatMulV2Op : public framework::OperatorWithKernel {
 public:
  void InferShape(framework::InferShapeContext* ctx) const override { ... }
 protected:
  framework::OpKernelType GetExpectedKernelType(const framework::ExecutionContext& ctx) const override { ... }
  framework::OpKernelType GetKernelTypeForVar(const std::string& var_name, const phi::DenseTensor& tensor, const framework::OpKernelType& expected_kernel_type) const override { ... }
};

REGISTER_OPERATOR(matmul_v2,                                                // 算子名称
                  ops::MatMulV2Op,                                          // 算子类，分发信息
                  ops::MatMulV2OpMaker,                                     // 算子定义信息
                  ops::MatMulV2GradOpMaker<paddle::framework::OpDesc>,      // 反向算子定义信息，静态图
                  ops::MatMulV2GradOpMaker<paddle::imperative::OpBase>);    // 反向算子定义信息，动态图
```

##### 1.2.2 OP与kernel注册

###### 1.2.2.1 OP注册（无BLD信息）

全部在**类的构造函数**完成：

- 模版元编程 --> 模版结构体（登记员） --> 递归注册op和信息（OpInfo 保存了 op 和反向的信息）

```C++
// 注册代码
REGISTER_OPERATOR(matmul_v2,
                  ops::MatMulV2Op,
                  ops::MatMulV2OpMaker,
                  ops::MatMulV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2GradOpMaker<paddle::imperative::OpBase>);
```

```C++
// 第一次展开
// 检测当前处于全局命名空间
STATIC_ASSERT_GLOBAL_NAMESPACE( __reg_op__matmul_v2, "REGISTER_OPERATOR must be called in global namespace"); 
// 登记员模版结构体，注册op至全局单例
static ::paddle::framework::OperatorRegistrar<
    ops::MatMulV2Op, 
    ops::MatMulV2OpMaker, 
    ops::MatMulV2GradOpMaker<paddle::framework::OpDesc>, 
    ops::MatMulV2GradOpMaker<paddle::imperative::OpBase>
    > __op_registrar_matmul_v2__("matmul_v2"); 
// 构造函数中注册op，需要有函数调用，否则链接器生成的二进制文件会删除这部分动作
int TouchOpRegistrar_matmul_v2() { __op_registrar_matmul_v2__.Touch(); return 0; }
```

```C++
// paddle/fluid/framework/op_registry.h
template <typename... ARGS>
struct OperatorRegistrar : public Registrar {
  explicit OperatorRegistrar(const char* op_type) {
    OpInfo info;
    details::OperatorRegistrarRecursive<0, false, ARGS...>(op_type, &info);
    OpInfoMap::Instance().Insert(op_type, info);    // 全局单例
  }
};
// 递归注册 OP 信息
template <size_t I, typename... ARGS>
class OperatorRegistrarRecursive<I, false, ARGS...> {
 public:
  using T = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
  OperatorRegistrarRecursive(const char* op_type, OpInfo* info) {
    OpInfoFiller<T> fill;    // 匹配到对应的注册类型
    fill(op_type, info);
    constexpr auto size = sizeof...(ARGS);    // 【优化】不需要根据 size 做判断，ARGS 可以自动展开
    OperatorRegistrarRecursive<I + 1, I + 1 == size, ARGS...> reg(op_type, info);
    (void)(reg);
  }
};

// fill 以函数指针的形式，注册下列信息（info 与 op 通过 op 名来连接）
// info中没有注册 GetExpectedKernelType 和 GetKernelTypeForVar
struct OpInfoFiller<T, kOperator>                  // 注册 info->creator_ 和 info->infer_shape_
struct OpInfoFiller<T, kShapeInference>            // 注册 info->infer_shape_（新动态图覆盖老动态图）

DECLARE_INFER_SHAPE_FUNCTOR(bilinear_interp_v2,
                            BilinearInterpInferShapeFunctor,
                            PD_INFER_META(phi::InterpolateInferMeta))

// 【省略】其他注册类型，时间原因不再展开
struct OpInfoFiller<T, kOpProtoAndCheckerMaker>    // 注册 info->proto_ 和 info->checker_
struct OpInfoFiller<T, kGradOpDescMaker>           // 注册 info->grad_op_maker_和使用default、empty grad的标志
struct OpInfoFiller<T, kGradOpBaseMaker>           // 注册 info->dygraph_grad_op_maker_
struct OpInfoFiller<T, kVarTypeInference>          // 注册 info->infer_var_type_
struct OpInfoFiller<T, kInplaceOpInference>        // 注册 info->infer_inplace_
struct OpInfoFiller<T, kNoNeedBufferVarsInference> // 注册 info->infer_no_need_buffer_vars_
struct OpInfoFiller<void, kUnknown>                // 无注册逻辑
```

###### 1.2.2.2 Kernel注册（有BLD信息）
```C++
// Kernel——paddle/fluid/operators/matmul_op.cc
template <typename DeviceContext, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override { ... }
};

REGISTER_OP_KERNEL(matmul, 
                   CPU, 
                   ::paddle::platform::CPUPlace, 
                   ops::MatMulKernel<phi::CPUContext, float>, 
                   ops::MatMulKernel<phi::CPUContext, double>);
```

最终通过**函数**注册kernel：
- 登记员仿函数递归解析 --> `RegisterKernelClass`函数注册 kernel

```C++
// paddle/fluid/framework/op_registry.h
static ::paddle::framework::OpKernelRegistrar<        // 通过模版参数传递的信息
    ::paddle::platform::CPUPlace,                     // place 
    ops::MatMulKernel<phi::CPUContext, float>,        // datatype
    ops::MatMulKernel<phi::CPUContext, double>        // datatype
    > __op_kernel_registrar_matmul_CPU_DEFAULT_TYPE__(// 通过构造函数参数传递的信息
        "matmul",                                     // op 名称 
        "CPU",                                        // library_type
        ::paddle::framework::OpKernelType::kDefaultCustomizedTypeValue);     // custom_value


template <typename PlaceType, typename... KernelType>
class OpKernelRegistrar : public Registrar {
 public:
  explicit OpKernelRegistrar(const char* op_type, const char* library_type, int customized_type_value) {
    // 仿函数构造 kernel
    OpKernelRegistrarFunctor<PlaceType, false, 0, KernelType...> func;
    func(op_type, library_type, customized_type_value);
  }
};
```

```C++
// RegisterKernelClass函数
template <typename PlaceType, size_t I, typename... KernelTypes>
struct OpKernelRegistrarFunctor<PlaceType, false, I, KernelTypes...> {
  void operator()(const char* op_type, const char* library_type, int customized_type_value) const {
    // 通过 RegisterKernelClass 函数将 kernel_type 和 func 注入全局单例中
    RegisterKernelClass<PlaceType, T>(op_type, library_type, customized_type_value,
        // 将类中的 Compute 函数，封装为函数指针
        [op_type](const framework::ExecutionContext& ctx) {
          KERNEL_TYPE().Compute(ctx);
          CheckKernelLaunch<PlaceType>(op_type);
        });
    constexpr auto size = std::tuple_size<std::tuple<KernelTypes...>>::value;
    OpKernelRegistrarFunctor<PlaceType, I + 1 == size, I + 1, KernelTypes...> func;
    func(op_type, library_type, customized_type_value);
  }
};
template <typename PlaceType, typename T, typename Func>
inline void RegisterKernelClass(const char* op_type,
                                const char* library_type,
                                int customized_type_value,
                                Func func) {
  std::string library(library_type);
  std::string data_layout = "ANYLAYOUT";        // 老动态图默认注册所有的layout！实际上不根据 layout 做选择
  if (library == "MKLDNN") {                    // 此处的data_layout实际上是和mkldnn绑定的
    data_layout = "MKLDNNLAYOUT";
  }
  OpKernelType key(ToDataType(std::type_index(typeid(T))),
                   PlaceType(),
                   StringToDataLayout(data_layout),
                   StringToLibraryType(library_type),
                   customized_type_value);
  OperatorWithKernel::AllOpKernels()[op_type][key] = func;
}
```

> 新动态图的OP信息如何组织？Kernel如何选择？
>
> ——后文对比新动态图和静态图中概念的异同


### 二、老动态图调用逻辑梳理

> 本文详细梳理新/老动态图的调用逻辑（老动态图即将被废弃，为什么还需要详细介绍？）：
>
> 老动态图的调用过程，揉合了静态图、phi的逻辑
> 
> - 老动态图复用静态图的部分实现
>
> - 老动态图优先调用phi下的kernel
>
> 注：本文为了介绍的简洁，部分示例代码有删减，仅展示代码中的重要逻辑

构建一个静态图的OperatorBase --> 使用动态图的OpBase调用静态图op --> PreparedOp执行真实的计算逻辑

#### 2.1 python端调用

```python
# python/paddle/nn/functional/common.py
if _in_legacy_dygraph():
    pre_bias = _legacy_C_ops.matmul_v2(x, weight, 'trans_x', False, 'trans_y', False)
```

#### 2.2 解析参数（代码生成）：解析input、解析 attr 键值对、占位 output；最终调用TraceOp(...)
```C++
// C++函数的开始——imperative_matmul_v2
// paddle/fluid/pybind/op_function1.cc
static PyObject * imperative_matmul_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
    std::string op_type = "matmul_v2";
    // 返回类型：std::shared_ptr<imperative::VarBase>
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    // std::unordered_map<std::string, Attribute>
    framework::AttributeMap attrs;
    // 属性按照 {attr_name: data_type} 组织，解析出一个参数需要遍历两次
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    // 保存输入和输出参数，Out参数的VarBase名称使用atomic，不会重复
    // 此处只是声明了一个out变量，规定类型和分配空间在 
OpBase::Run 函数中
    // std::map<std::string, std::vector<std::shared_ptr<VarBase>>>
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    // 关键调用函数
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    // 返回结果
    return MakeReturnPyObject(outs["Out"][0]);
}
```

#### 2.3 构造OperatorBase（静态图）、attr属性检查、AMP类型转换；最终调用OpBase::Run(...)
```C++
// Tracer::TraceOpImpl——运算逻辑
// paddle/fluid/imperative/tracer.cc & tracer.h
template <typename VarType>
void Tracer::TraceOpImpl( ... ) {
  ......
  if (FLAGS_use_mkldnn) {    // 全局bool标志
    // 1. 根据 FLAGS_tracer_mkldnn_ops_on/off, 修改attrs["use_mkldnn"]属性
  }
  // 2. 复用静态图逻辑，构造OP，CreateOp 的形参为 VariableNameMap，无法传入
  OperatorBase op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  // 3. 【省略】attr 属性检查
  auto* attr_checker = op->Info().Checker();
  // 4. 【省略】获取默认参数
  const paddle::framework::AttributeMap& default_attrs = /*...*/ attr_checker->GetDefaultAttrMap()
  // 5. 【省略大段代码】AMP 模式下对 inputs 做转换
  // 6. 【省略】根据 GPU、XPU、NPU、MLU 设置设备ID 
  platform::SetDeviceId(place.device);
  // 7. 旧动态图兼容静态图，OperatorBase是静态图的实现
  OpBase::Run(*op, new_ins, outs, attrs, default_attrs, place);
  // 8. 【省略】旧动态图相关逻辑
  if (ComputeRequiredGrad(new_ins, outs, trace_backward)) {
    CreateGradOpNode(
      *op, new_ins, outs, attrs, default_attrs, place, inplace_map);
  }
}
```

#### 2.4 设置outs数据类型、准备PreparedOp、准备Transform后数据；最终调用PreparedOp::Run(...)

class PreparedOp的成员函数极其简单，只有Prepare函数（构造PreparedOp）、Run函数

```C++
// OpBase::Run & OpBaseRunImpl
// paddle/fluid/imperative/layer.cc
template <typename VarType>
static void OpBaseRunImpl( ... ) {
  // 1. 基类转派生类，象征着 kernel 选择分发的开始
  auto* op_kernel = static_cast<const framework::OperatorWithKernel*>(&op);
  if (info.infer_var_type_) {
    // NOTE：此处是 infer_var_type_ 推理 var 的shape，推导 outs 的数据类型，最终会调用 SetVarDataType
    RuntimeInferVarTypeContext<VarType> infer_var_type_ctx(ins, outs, attrs, default_attrs);
    info.infer_var_type_(&infer_var_type_ctx);
  }
  // 2. 【省略】设置 outs 的数据类型
  
  // 3. 准备 op ，此处进行 kernel 选择，输入的 kernel 类型为 OperatorWithKernel
  auto prepared_op = PreparedOp::Prepare(ins, outs, *op_kernel, place, attrs, default_attrs);
  // 4. 准备输入参数，将 ins 参数转换为 prepared_op.kernel_type()
  auto tmp_ins_ptr = PrepareData<VarType>(*op_kernel, ins, prepared_op.kernel_type());
  // 5. 准备好 op 和 数据之后，终于可以执行了！！！
  prepared_op.Run(*tmp_ins_ptr, outs, attrs, default_attrs);
  
  // 将 var->grad 的 type 设置为 output var 的 type
  SetForwardDataTypeOfGradVars<VarType>(outs);
}
```

##### 2.4.1 Prepare函数（构造、与phi兼容）：GetExpectedKernelType、kernel选择、构造PreparedOp对象；

**kernel 选择的优先级**：phi Xpu kernel > fluid Xpu kernel > phi cpu kernel > fluid cpu kernel

```C++
// PreparedOp::Prepare & PrepareImpl
// paddle/fluid/imperative/prepared_operator.cc

// kernels_: 二层map：{string kernel_name: {KernelKey key: Kernel}};       // kernel 的全局单例
const phi::KernelFactory& PreparedOp::phi_kernel_factory
// base_kernel_name_map_: {string old_name, string new_name}              // 函数名称的映射
// arg_mapping_fn_map_: {string name, ArgumentMappingFn func}             // 旧动态图的参数
const phi::OpUtilsMap& PreparedOp::phi_op_utils_map
// map_: {string, KernelSignature sig}                                    // 函数签名
const phi::DefaultKernelSignatureMap& PreparedOp::default_phi_kernel_sig_map

// 【优化】一个函数300多行代码，亟需优化
template <typename VarType>
PreparedOp PrepareImpl( ... ) {
  // 1. GetExpectedKernelType：获得期望运行的 Kernel 类型
  auto dygraph_exe_ctx = DygraphExecutionContext<VarType>(op, empty_scope, *dev_ctx, empty_ctx, ins, outs, attrs, default_attrs);
  auto expected_kernel_key = op.GetExpectedKernelType(dygraph_exe_ctx);

  // 2. 根据 sig 的有无判断是否有 phi 的kernel
  bool has_phi_kernel = false;
  const auto* arg_map_fn = phi_op_utils_map.GetArgumentMappingFn(op.Type());
  if (arg_map_fn) {
    has_phi_kernel = true;
    kernel_signature = (*arg_map_fn)(framework::ExecutionArgumentMappingContext(dygraph_exe_ctx));
  }
  // 3. 【省略大段代码】kernel选择，从全局的 phi_kernel_factory 进行两次查找
  // kernel 选择的优先级：phi Xpu kernel > fluid Xpu kernel > phi cpu kernel > fluid cpu kernel
  // 4. 根据 has_phi_kernel 判断调用 phi 还是 fluid 的构造函数
  // has_phi_kernel 的构造函数，内部设置了run_phi_kernel_
  return PreparedOp(op,                                // OperatorBase
                    empty_ctx,                         // RuntimeContext
                    expected_kernel_key,               // OpKernelType
                    arg_map_fn,                        // ArgumentMappingFn
                    default_kernel_signature,          // KernelSignature
                    std::move(kernel_signature),       // KernelSignature
                    phi_kernel,                        // Kernel
                    dev_ctx);                          // DeviceContext
  // fluid 的构造函数
  return PreparedOp(op,                                
                    empty_ctx,                         
                    expected_kernel_key,               
                    kernel_iter->second,    // OperatorWithKernel::OpKernelFunc, void(const ExecutionContext&)
                    arg_map_fn,
                    default_kernel_signature,
                    dev_ctx);
}
```

##### 2.4.2 PrepareData函数（数据）：GetKernelTypeForVar准备输入数据（数据BLD转换）

```C++
// PrepareData
// paddle/fluid/imperative/prepared_operator.h
template <typename VarType>
std::shared_ptr<NameVarMap<VarType>> PrepareData(
    const framework::OperatorWithKernel& op,
    const NameVarMap<VarType>& ins,
    const framework::OpKernelType& expected_kernel_key) {
  std::shared_ptr<NameVarMap<VarType>> tmp_ins_ptr = nullptr;
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) { // 处理每个Tensor
      if (tensor && tensor->IsInitialized() && (tensor->memory_size() != 0)) {
        // 1. 调用 GetKernelTypeForVar 函数，对每个 Var 特化进行 kernel 选择
        // 默认按照 {expected_kernel_type.data_type_, tensor.place(), tensor.layout()} 组织
        auto kernel_type_for_var = op.GetKernelTypeForVar(name_pair.first, *tensor, expected_kernel_key);
        // 2. BLD 均相同，不需要转换
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {    
          continue;
        } else { // 3. 如果var的kernel_type和期待的不一致，kernel_type_for_var -> expected_kernel_key
            phi::DenseTensor out;
            // 转换 layout、datatype、device
            TransformData(expected_kernel_key, kernel_type_for_var, *tensor, &out);
}}}}
```

##### 2.4.3 调用kernel（运算）：InferShape设置output，区分调用的是 fluid 还是 phi 下的 kernel

```C++
// PreparedOp::Run
void PreparedOp::Run(const NameVarMap<VarBase>& ins,
                     const NameVarMap<VarBase>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  // 根据 run_phi_kernel_标志判断用新/老动态图
  if (run_phi_kernel_) {
    PreparedOpRunPtImpl<VarBase>(op_, kernel_type_, arg_map_fn_, default_kernel_signature_,
                                 kernel_signature_, phi_kernel_,
                                 dev_ctx_, ins, outs, attrs, default_attrs);
    // op.Info().infer_shape_(&infer_shape_ctx);
    // PreparePhiData<VarType>(phi_kernel, kernel_signature, ins);
    // phi_kernel(&phi_kernel_context);                             调用 kernel 计算
    // 【省略】其他逻辑
  } else {
    PreparedOpRunImpl<VarBase>(op_, ctx_, kernel_type_, func_, arg_map_fn_, default_kernel_signature_,
                               dev_ctx_, ins, outs, attrs, default_attrs);
    // op.Info().infer_shape_(&infer_shape_ctx);
    // func(DygraphExecutionContext<VarType>(op, empty_scope, *dev_ctx, ctx, ins, outs, attrs, default_attrs));
    // 【省略】其他逻辑
  }
}
```

#### 2.5 总结

- 老动态图的前后向兼容
  - 前向：复用静态图构造OP的代码，节省了op迁移的成本

  - 后向：优先选择phi下的kernel

- 多个OP概念与数据处理逻辑相互穿插

  - 对于数据的处理，遍布整个调用流程：解析参数、参数检查、设置输出类型、DataTransform、设置输出形状。

    - 【优化】部分实现能否整合在一个函数中？

  - OperatorBase（静态图）、OpBase（老动态图）、PreparedOp（兼容新老动态图）

- GetKernelTypeForVar相比于GetExpectedKernelType，是更加特化的一种kernel 选择方式

- 历史背景：需要承前启后，代码为了兼容性牺牲了部分可读性，理解起来最为复杂

### 三、静态图调用逻辑梳理

> kernel选择分发子项，不依赖于对静态图组网等过程的调研，因此本章仅概述op的调用流程
> 
> 此处假设已经调用完**CreateOp**，创建完成OperatorWithKernel类型的op，即将调用kernel进行运算（相当于老动态图**2.4**的调用逻辑）
> 
> 注：为了介绍的简洁，部分示例代码有删减，仅展示代码中的重要逻辑

**kernel 选择的优先级**：phi Xpu kernel > fluid Xpu kernel > phi cpu kernel > fluid cpu kernel

```C++
// OperatorWithKernel::RunImpl
// paddle/fluid/framework/operator.cc
void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place,
                                 RuntimeContext* runtime_ctx) const {
  // 相当于老动态图 2.4.1 章节，kernel 选择
  // 静态图根据 op 名称判断是否有 phi kernel，更新 phi 所需的一系列信息
  if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(type_)) {
    kernel_signature_.reset(new phi::KernelSignature(std::move(GetExpectedPhiKernelArgs(exe_ctx))));  // 函数签名
    phi_kernel_name = kernel_signature_->name;                                                // kernel 名称
    // 额外解析 Attr<std::string>("op_device")，返回 OpKernelType
    kernel_type_.reset(new OpKernelType(std::move(InnerGetExpectedKernelType(exe_ctx))));     // kernel 类型
    phi_kernel_key = TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());                    // kernel 类型
    phi_kernel_.reset(new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(phi_kernel_name, phi_kernel_key)));
    run_phi_kernel_ = true;
  }
  // 选择 fluid 下的 kernel
  if (!run_phi_kernel_) { 
    // 更新 kernel_type_ 和 kernel_func_，如果没有找到 Xpu 下的 fluid kernel，则会 fallback 到 cpu
    ChooseKernel(exe_ctx);
  }
  
  // 相当于老动态图 2.4.2 章节
  // 【注】PrepareData 与老动态图的函数同名，但是函数签名和逻辑不同，需要用到静态图的 scope 特性，插入 Transform scope
  std::vector<std::string> transfered_inplace_vars;
  Scope* transfer_scope = nullptr;
  transfer_scope = PrepareData(scope, *kernel_type_, &transfered_inplace_vars, runtime_ctx);
  
  // 相当于老动态图 2.4.3 章节
  // InferShape设置output
  this->Info().infer_shape_(&infer_shape_ctx);
  // kernel 调用逻辑
  if (run_phi_kernel_) {
    phi::KernelContext phi_kernel_context;
      // 【此处省略 cache 优化】
      phi::KernelContext phi_kernel_context;
      BuildPhiKernelContext(*runtime_ctx, dev_ctx, &phi_kernel_context);
      (*phi_kernel_)(&phi_kernel_context);
  } else {
    (*kernel_func_)(ExecutionContext(*this, exec_scope, *dev_ctx, *runtime_ctx));
  }
  // 【省略】后续还有一些其他逻辑
}
```

### 四、新动态图调用逻辑梳理

> 新动态图中的op为函数式op，相比于老动态图，新动态图的调用路径更为简洁清晰、调度过程更为高效

#### 4.1 调用逻辑梳理

##### 4.1.1 python端调用

```python
# python端调用API——matmul
# python/paddle/tensor/math.py
if in_dygraph_mode():
    return _C_ops.matmul(input, mat2, False, False)
elif paddle.in_dynamic_mode():
    return _legacy_C_ops.matmul_v2(input, mat2)
```

##### 4.1.2 解析参数（代码生成）：解析input、解析 attr、获取输出 output；数据类型`Tensor`

```C++
// C++函数的开始——eager_api_matmul
// paddle/fluid/pybind/eager_op_function.cc
static PyObject * eager_api_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
    // 1. 解析输出，返回值为 Tensor
    auto x = GetTensorFromArgs("matmul", "x", args, 0, false);
    auto y = GetTensorFromArgs("matmul", "y", args, 1, false);
    // 2. 【性能优化】直接根据 kernel 签名解析 attribute，而非解析键值对
    PyObject* transpose_x_obj = PyTuple_GET_ITEM(args, 2);
    bool transpose_x = CastPyArg2Boolean(transpose_x_obj, "matmul", 2);
    PyObject* transpose_y_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose_y = CastPyArg2Boolean(transpose_y_obj, "matmul", 3);
    
    // 设置设备ID，仅支持GPU
    auto place = egr::Controller::Instance().GetExpectedPlace();
    phi::backends::gpu::SetDeviceId(place.device);
    
    // 3. 调用 dygraph 层获得输出
    decltype(::matmul_ad_func(x,y,transpose_x,transpose_y)) out = ::matmul_ad_func(x,y,transpose_x,transpose_y);
}
```

##### 4.1.3 dygraph层（代码生成）：AMP类型转换、attr属性检查（无）；数据类型`Tensor`

```C++
// matmul_ad_func
// paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.cc
paddle::experimental::Tensor matmul_ad_func(const paddle::experimental::Tensor& x, const paddle::experimental::Tensor& y, bool transpose_x, bool transpose_y) {
  // 【省略】ins 参数 AMP 类型转换
  // 【省略】Layout autotune，特定 op 专用逻辑

  // Forward API Call，调用的核心逻辑
  auto api_result = paddle::experimental::matmul(x, y, transpose_x, transpose_y);
  
  // 【省略大段代码】动态图构建和反向传播
}
```

##### 4.1.4 api层（代码生成）：Kernel选择（GetExpectedKernelType、GetKernelTypeForVar）、DataTransform、InferShape、Compute

**六次**kernel选择逻辑：GPUDNN > GPUDNN_all_layout > kernel_backend >kernel_backend_all_layout > CPU > CPU_all_layout

```C++
// matmul
// paddle/phi/api/lib/api.cc
PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y) {
  // 1. 相当于老动态图下的 GetExpectedKernelType
  // 首先将所有输入的信息放到一个 KernelKeySet 中
  auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
  // 然后，根据 KernelKeySet 获取优先级最高的 KernelKey = {Backend, DataLayout, DataType}
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  // 2. kernel 选择【此处可优化，依赖于准确指定layout】【与静/老动态图选择逻辑有差异】
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "matmul", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  // 3. 与静/老动态图的功能类似 PrepareData ，具有 GetKernelTypeForVar 和 TransformData 的逻辑
  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  // 4. InferShape 逻辑，设置 output 形状
  Tensor api_output;
  auto kernel_out = SetKernelOutput(&api_output);
  phi::MetaTensor meta_out(kernel_out);
  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), transpose_x, transpose_y, &meta_out);

  // 5. 调用 kernel
  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    bool,
                                    bool,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();    // 找到 kernel function
  (*kernel_fn)(*dev_ctx, *input_x, *input_y, transpose_x, transpose_y, kernel_out);
  return api_output;
}
```

#### 4.2 对应关系

> 在前向兼容的过程中，原有逻辑不会消失，无非是逻辑迁移或者逻辑覆盖
>
> 迁移新旧动态图算子时的每个步骤，究竟是在干什么？
> 
> phi是函数式kernel，实际上只有kernel，没有算子的概念。或者说yaml文件中的描述对应着 fluid 中的算子

##### 4.2.1 新动态图中的对应关系

`# 前向op`

`OpMaker——yaml函数签名` # 构造算子，描述算子的输入、输出和属性

`OperatorWithKernel——phi中为函数式概念` # 算子类，包括从OP如何运行到kernel

`- InferShape——InferMeta` # 算子输出形状的推导函数

`- GetExpectedKernelType——yaml中的data_type` # 算子需要的Kernel类型（kernel选择分发）

`- GetKernelTypeForVar——phi中op注册时添加` #某个Var特定的Kernel类型（kernel选择分发）

`# Op注册`

`REGISTER_OPERATOR` # 将算子注册到全局map中（模版元编程）

```YAML
# paddle/phi/api/yaml/legacy_ops.yaml
- op : matmul                      # 与 fliud 中的 op 对应
  args : (Tensor x, Tensor y, bool transpose_x = false, bool transpose_y = false)    # 与 OpMaker 对应
  output : Tensor
  infer_meta :                     # 与 InferShape 对应
    func : MatmulInferMeta
    data_type : ......             # 指定函数的输入
  kernel :
    func : matmul                  # 调用的 kernel 函数
    data_type : ......             # GetExpectedKernelType 根据某个输入返回结果
  backward : matmul_grad
```

##### 4.2.2 GetExpectedKernelType的对应关系

```C++
// GetExpectedKernelType对应关系
// 静/老动态图代码
framework::OpKernelType GetExpectedKernelType(const framework::ExecutionContext& ctx) const override {
  auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // 【省略 mkldnn 硬编码的逻辑】
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

// yaml 中的配置
  kernel :
    func : cast
    data_type : x
    
// 新动态图 api.cc 中对应的代码
kernel_data_type = ParseDataType(x);
```

##### 4.2.3 GetKernelTypeForVar的对应关系

```C++
// GetKernelTypeForVar
// 静态图和老动态图代码
framework::OpKernelType GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DenseTensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const override {
  // 不对这两个输入参数进行 DataTransform
  if (var_name == "SizeTensor" || var_name == "Scale") {
    return expected_kernel_type;
  }
  return framework::OpKernelType(expected_kernel_type.data_type_, tensor.place(), tensor.layout());
}
// 新动态图中的 “GetKernelTypeForVar” 逻辑
PD_REGISTER_KERNEL(bilinear_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   float,
                   double,
                   uint8_t) {
  // 跳过对于 "SizeTensor" 和 "Scale" 的转换
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
  // fluid 注册的基本都是ALL_LAYOUT，因此不跳过 Layout 也没问题
}
```

**新动态图PrepareData**中的GetKernelTypeForVar逻辑

```C++
// 新动态图的PrepareData函数
std::unique_ptr<std::vector<phi::DenseTensor>> PrepareData(
    const std::vector<Tensor>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag) {
  for (const auto& input : inputs) {
    const auto& tensor_in = input.impl();
    if (!transform_flag.NeedTransform() || !tensor_in->initialized() ||
        (!NeedTransformPlace(tensor_in->place(), target_args_def.backend, transform_flag) &&
         !NeedTransformDataType(tensor_in->dtype(), target_args_def.dtype, transform_flag) &&
         !NeedTransformLayout(tensor_in->layout(), target_args_def.layout, tensor_in->place(),transform_flag))){
      pt_tensors->emplace_back(*std::dynamic_pointer_cast<phi::DenseTensor>(tensor_in));
    } else {
      pt_tensors->emplace_back(TransformData((static_cast<phi::DenseTensor*>(tensor_in.get())), target_args_def, transform_flag));
    }
  }
}

TransformFlag(bool stop_transform = false,
              bool trans_dtype = false,   // 此处默认关闭
              bool trans_backend = true,  // 此处默认打开
              bool trans_layout = true)   // 此处默认打开
    : stop_transform_(stop_transform),
      trans_data_type_(trans_dtype),
      trans_backend_(trans_backend),
      trans_layout_(trans_layout) {}
```

**phi 中的 kernel 注册**：较为复杂，仅了解 ConstructKernel 函数将 kernel 注册到了 KernelFactory 

```C++
// ConstructKernel
// paddle/phi/core/kernel_registry.h
void ConstructKernel(RegType reg_type,
                      const char* kernel_name_cstr,
                      const char* backend_cstr,
                      DataLayout layout,
                      DataType dtype,
                      KernelArgsParseFn args_parse_fn,
                      KernelArgsDefFn args_def_fn,
                      KernelFn kernel_fn,
                      void* variadic_kernel_fn) {
  std::string kernel_name(kernel_name_cstr);
  KernelKey kernel_key(paddle::experimental::StringToBackend(backend_cstr), layout, dtype);
  Kernel kernel(kernel_fn, variadic_kernel_fn);
  args_parse_fn(kernel_key, kernel.mutable_args_def());
  args_def_fn(kernel_key, &kernel);
  if (reg_type == RegType::INNER) {
    KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
  } else {
    CustomKernelMap::Instance().RegisterCustomKernel(kernel_name, kernel_key, kernel);
  }
}
```

### 4.3 总结

#### 4.3.1 优点（相比于静/老动态图）

- 性能提升：解析attr参数时不再采用key:value键值对，而是按照输入顺序进行解析（依赖于代码生成）

- 性能提升：节省了很多数据结构重复构造的过程（例如Context、Operator）

- 算子复用：kernel 层算子方便复用，可以进行组合运算提升性能（例如 linear = matmul + add，减少 dispatch 次数）

- 理解成本：dygraph、api、kernel三个调用层次清晰，学习成本低

- 高效开发：大量运用代码生成技术，减少重复的代码copy工作

#### 4.3.2 与静/老动态图的异同

- 静/老动态图的kernel选择机制：在 kernel 选择时，未利用layout信息，注册时全部设置为ALL_LAYOUT

- KernelKey的简化：静/老动态图
  
  - class OpKernelType {proto::VarType::Type, DataLayout, Place, LibraryType, customized_type_value_}
  
  - class KernelKey {Backend, DataLayout, DataType}

- fallback 机制

  - 静 / 老动态图： CUDA 不会 fallback 到 CPU，会直接报错

    
    - 老动态图：phi 和 fluid 下没有 MKLDNN 的 fallback 逻辑
    
    - 静态图中：fluid 下 MKLDNN 会 fallback 到 plain cpu；phi 下没有 MKLDNN 的 fallback 逻辑
  
  - 新动态图：CUDA 会 fallback 到 CPU，无 MKLDNN fallback 逻辑

##### 4.3.3 可能存在的问题
- **性能优化**：kernel选择顺序和次数

  - 新动态图kernel选择时最多需要搜索6次：GPUDNN > GPU > CPU（因为会fallback到ALL_LAYOUT，3*2=6次）

  - 静/老动态图最多只会选择4次（还是在老动态图兼容新动态图的情况下，phi kernel > fluid kernel）

- 算子复用层次：算子组合时可以在dygraph、api、kernel三个层次添加，如果不遵守新增算子的原则，未来框架可能变得臃肿

### 五、个人感悟

- 框架开发要守原则：很多设计最初都是好的设计，但是中间遇到问题时会不断的妥协，导致框架变得冗杂（例如 kernel 选择时直接用MKLDNN 硬编码）

- 框架开发要勤同步：有些代码已经考虑到为以后留出了接口，但是其他人不了解的话，很容易用别的方式实现，污染框架（例如op_registry 中，mkldnn 注册时已经同步了 layout 和 library_type ）；自身也要常读代码，常读常新

- 框架开发要有边界：如何实现一个功能有很多种方案、可以在很多地方实现。仅仅实现是不够的，需要遵守每个模块的边界（例如 kernel 选择体系中的 DataTransform 是框架做的事情，就不能交给 Kernel 来做）

### 六、讨论问题整理与答复

- kernel选择方案支持之后，静态图也是都支持的吗？动静一致方面有何考虑？

kernel选择分发体系的最终目标是，统一静态图/老动态图/新动态图的选择分发逻辑。因此kernel选择方案支持后，静态图也是支持的，并且动静态图之间的选择动作相同。

当前状态：正在清理kernel选择过程中，每个opGetExpectedKernelType函数中的mkldnn硬编码，这部分op很多都是静态图和老动态图共用的op，因此这部分的清理过程首先是一致的

后续的考虑：静态图的kernel选择逻辑向新动态图的yaml靠拢，最终使得动静态图在选择kernel时以yaml为标准

另外，目前框架内静态图和老动态图的选择逻辑其实是有稍许偏差的，例如静态图实际上没有找到mkldnn的kernel时，会fallback到CPU的plain kernel上，而目前老动态图并没有这一机制。在后续的实现过程中我也会注意fallback的情形。

- Tensor中的MKLDNN有了解过吗？为什么不能新建一个MKLDNN的Tensor？

当前 phi 下的 DenseTensor 类有一段被 MKLDNN hack 的逻辑，加入了对 MKLDNN tensor 格式的描述，从这个 Tensor 也可以看出来，MKLDNN 的硬编码几乎已经侵入了框架的每个层次。

新建一个MKLDNN Tensor指的是，可以新建一个MKLDNN类，继承自DenseTensor，在这个MKLDNN Tensor中持有特化的信息，而不污染 paddle 内的其他Tensor。出于如下考虑，是没有办法实现的：

1. 转化或者新建一个MKLDNN Tensor，需要知道选择出来的kernel是一个MKLDNN kernel；然而kernel的选择过程，也依赖于Tensor中持有MKLDNN的信息，这就变成了一个先有鸡还是先有蛋的问题。

2. 如果加入MKLDNN的Tensor，要在所有需要转化的位置加上static_cast，这样做的改动成本过高，同时框架的性能也可能受影响。

- 框架内 fallback 的实现机制？

Paddle 目前在kernel选择分发时，遵循着如下的选择优先级：phi kernel > fluid kernel；Xpu kernel > CPU kernel（此处的Xpu指除了CPU以外的所有硬件）fallback指的是在kernel选择时，如果没有找到Xpu的kernel，则会fallback到CPU的kernel，保障代码的正常运行。

然而并不是所有的Xpu都可以fallback到CPU中，特例是GPU的kernel不会fallback到CPU，如果GPU的kernel没有找到，则会直接报错。从代码调试的角度出发，如果支持GPU fallback 到CPU，那么当模型的性能下降时，会很难排查，需要确定有没有找到对应kernel，还是其他的问题。
