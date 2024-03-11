# 六、执行流程
> 本文文档作者： @Aurelius84

- [六、执行流程](#六执行流程)
  - [1. JitKernelOp](#1-jitkernelop)
  - [2. ToKernelDialect](#2-tokerneldialect)
  - [3. Instruction构造](#3-instruction构造)
  - [4. host/device 函数调用链](#4-hostdevice-函数调用链)


第3、4、5章节分别介绍了「前端表示」、「Lower过程」、「CodeGen」，这里我们重新回顾下在pir::Program层面的前后变化：

```c++
===-------------------------------------------------------------------------===
                               Origin Program
===-------------------------------------------------------------------------===
{
 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"_jst.0.x.0",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[64,128],stop_gradient:[false]} : () -> pd_op.tensor<64x128xf32>
 (%1) = "cinn_op.reduce_max" (%0) {dim:[(Int64)-1],keep_dim:true,stop_gradient:[false]} : (pd_op.tensor<64x128xf32>) -> pd_op.tensor<64x1xf32>
 (%2) = "pd_op.subtract" (%0, %1) {stop_gradient:[false]} : (pd_op.tensor<64x128xf32>, pd_op.tensor<64x1xf32>) -> pd_op.tensor<64x128xf32>
 (%3) = "pd_op.exp" (%2) {stop_gradient:[false]} : (pd_op.tensor<64x128xf32>) -> pd_op.tensor<64x128xf32>
 (%4) = "cinn_op.reduce_sum" (%3) {dim:[(Int64)-1],keep_dim:true,stop_gradient:[false]} : (pd_op.tensor<64x128xf32>) -> pd_op.tensor<64x1xf32>
 (%5) = "pd_op.divide" (%3, %4) {stop_gradient:[false]} : (pd_op.tensor<64x128xf32>, pd_op.tensor<64x1xf32>) -> pd_op.tensor<64x128xf32>
 () = "builtin.set_parameter" (%5) {parameter_name:"output_0"} : (pd_op.tensor<64x128xf32>) ->
}

===-------------------------------------------------------------------------===
                               Final Program
===-------------------------------------------------------------------------===
{
 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"input_0",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[64,128],stop_gradient:[true]} : () -> pd_op.tensor<64x128xf32>
 (%1) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"input_1",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[64,128],stop_gradient:[true]} : () -> pd_op.tensor<64x128xf32>
 (%2) = "cinn_runtime.jit_kernel" (%1, %0) {kernel_info:(0x7fac3cc3a7c0)} : (pd_op.tensor<64x128xf32>, pd_op.tensor<64x128xf32>) -> pd_op.tensor<64x128xf32>
 () = "builtin.set_parameter" (%2) {parameter_name:"output_2"} : (pd_op.tensor<64x128xf32>) ->
}
```

## 1. JitKernelOp

从上面的Final Program可以看出，最后每个GroupOp都会替换为CINN Runtime Dialect中的JitKernelOp，具有如下特点：

* 多输入、多输出；
* 只含一个 Attribute

```c++
// paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h

class JitKernelOp : public ::pir::Op<JitKernelOp> {
 public:
  using Op::Op;
  static const char* name() { return "cinn_runtime.jit_kernel"; }
  // TODO(Aurelius84): Think deeply what should contains
  static constexpr uint32_t attributes_num = 1;
  static constexpr char* kAttrName = "kernel_info";
  static const char* attributes_name[attributes_num];

  static void Build(::pir::Builder& builder,             // NOLINT
                    ::pir::OperationArgument& argument,  // NOLINT
                    const std::vector<::pir::Value>& x,
                    const ::pir::AttributeMap& attributes,
                    const std::vector<::pir::Type>& out_types);

  const hlir::framework::pir::CINNKernelInfo& cinn_kernel_info();

  void VerifySig();
};
```

> [!NOTE]
> 小提示：在Operator Dialect中，所有的算子都是单Value，比如Concat，其输入是pir::Value x，并非是std::vector<pir::Value>& x。但这里形态是后者。


大家只需要关心这里的Attribute，其类型是CINNKernelInfoAttribute，具体的信息由CINNKernelInfoAttributeStorage来管理。从接口可以看出，这里真正保存了「第5章节」中backend::Compiler编译出的 fn_ptr。

```c++
// paddle/cinn/hlir/dialect/operator/ir/attribute_storage.h
struct CINNKernelInfoAttributeStorage : public pir::AttributeStorage {
  // ====================== ParamKey ======================
  using ParamKey = cinn::hlir::framework::pir::CINNKernelInfo;

  explicit CINNKernelInfoAttributeStorage(const ParamKey& key) : data_(key) {}

  static CINNKernelInfoAttributeStorage* Construct(const ParamKey& key) {
    return new CINNKernelInfoAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey& key) {
    return std::hash<int64_t>()(*(reinterpret_cast<int64_t*>(key.fn_ptr)));
  }

  bool operator==(const ParamKey& key) const {
    return data_.fn_ptr == key.fn_ptr;
  }

  const ParamKey& GetAsKey() const { return data_; }

 private:
  ParamKey data_;
};
```

CINNKernelInfo目前成员很简单，主要包含：

* fn_ptr，后端编译器返回的函数指针；
* int_args_map：表示函数入参中shape语义的int值从何而来。比如2: {0, 3}表示函数下标第2个入参来自于第0个Tensor的第3维度

```c++
// paddle/cinn/hlir/framework/pir/utils.hpaddle/cinn/hlir/framework/pir/utils.h
struct CINNKernelInfo {
  void* fn_ptr;

  struct ArgDimIdx {
    int arg_idx;
    int dim_idx;
  };
  // int_args_map records the int_args_map.key argument (dtype is Int) in the
  // kernel parameter taken from the dim_idx dimension of the shape of the
  // ArgDimIdx.arg_idx argument.
  // Examples:
  //   a func like: foo(tensor A, tensor B, int S1, int S2)
  //   S1 = A.shape[3]
  //   S2 = B.shape[2]
  //   int_args_map will be like
  //   {
  //     2: {0, 3},
  //     3: {1, 2}
  //   }
  std::map<int, ArgDimIdx> int_args_map;
};
```


> [!NOTE]
> 小问题：CINNKernelInfo的信息是在哪个模块构建的？

答案：是在PirCompiler里的BuildCUDAJITInfo接口中构建的。
```c++
// paddle/cinn/hlir/framework/pir_compiler.cc
std::vector<pir::CINNKernelInfo> PirCompiler::BuildCUDAJITInfo(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<pir::CINNKernelInfo> cinn_kernel_info_vecs(groups.size());

    // Step 1: Lower to AST
    auto op_lowerer = CreateOpLowerer<pir::GroupPtr>(target_);
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
    for (int i = 0; i < groups.size(); ++i) {
      lowered_funcs.emplace_back(op_lowerer.Lower(groups[i]));
    }

    for (auto&& lowered_func : lowered_funcs) {
      ProcessFunction(lowered_func);
    }
    // Step 2: CodeGen and Compile them
    compiler_ = backends::Compiler::Create(target_);
    auto build_module = m_builder_.Build();
    compiler_->Build(build_module, "");

    // Step 3： Construct CINNKernelInfo
    auto fn_ptrs = compiler_->GetFnPtr();
    for (int idx = 0; idx < groups.size(); ++idx) {
      pir::CINNKernelInfo cinn_kernel_info;
      auto fn_name = groups[idx]->FuncName();
      auto fn_ptr = compiler_->Lookup(fn_name);
      cinn_kernel_info.fn_ptr = fn_ptr;
      cinn_kernel_info.int_args_map = groups[idx]->int_args_map;

      cinn_kernel_info_vecs[idx] = cinn_kernel_info;
    }

  return cinn_kernel_info_vecs;
}
```

## 2. ToKernelDialect

至此，关于CINN相关的pir::Program已经介绍完毕。这里针对「多层级 Dialect」 之间的转换进一步介绍些细节。在飞桨的新 IR 体系下：

* Operator Dialect是 High-Level IR，其角色与之前的OpDes集合是类似的；
* Kernel Dialect：新增的 Dialect，比上面更第一层，主要面向执行器

在传递给执行器时，会调用PdOpLowerToKernelPass，进行Operator Dialect→ Kernel Dialect的转换。

> [!NOTE]
> 小提示：虽然这里叫PdOpLowerToKernelPass，但未继承pir::Pass，属于待规范的模块代码；

对于前面的softmax的样例，经过此Pass前后的Program变化如下：

```c++
 // ===============  IR before lowering   ===============
 {
 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"_jst.0.x.0",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[64,128],stop_gradient:[false]} : () -> pd_op.tensor<64x128xf32>
 (%1) = "cinn_runtime.jit_kernel" (%0) {kernel_info:(0x7f437760d7c0)} : (pd_op.tensor<64x128xf32>) -> pd_op.tensor<64x128xf32>
 () = "builtin.set_parameter" (%1) {parameter_name:"output_0"} : (pd_op.tensor<64x128xf32>) ->
}


 // ===============  IR after lowering   ===============
{
 (%0) = "data(phi_kernel)" () {dtype:(pd_op.DataType)float32,kernel_key:<backend:Undefined|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"data",name:"_jst.0.x.0",op_name:"pd_op.data",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[64,128],stop_gradient:[false]} : () -> undefined_tensor<64x128xf32>
 (%1) = "shadow_feed(phi_kernel)" (%0) {kernel_key:<backend:GPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"shadow_feed",op_name:"pd_op.shadow_feed"} : (undefined_tensor<64x128xf32>) -> gpu_tensor<64x128xf32>
 (%2) = "cinn_runtime.jit_kernel" (%1) {kernel_info:(0x7f437760d7c0)} : (gpu_tensor<64x128xf32>) -> gpu_tensor<64x128xf32>
 () = "builtin.set_parameter" (%2) {parameter_name:"output_0"} : (gpu_tensor<64x128xf32>) ->
}
```

* 跳过一些 Op 的处理；比如pd_op.data和pd_op.feed重合了，则跳过后者；
* 处理一些特殊 Op 的 Lowering。比如builtin_op、控制流 Op、JitKernelOp等；
* BuildOutputType：针对每个Op的输出，将其DenseTensorType等转换为AllocatedDenseTensorType等；后者包含Place信息；
* BuildInputs：包含同上逻辑，但会额外分析插入必要DataTransform相关的Op
* BuildKernelOp：创建对应的Kernel Dialect中的Op

```c++
pir::Operation* BuildKernelOp(...){
  // ....
  pir::Operation* op = nullptr;
  if (IsLegacyOp(op_item->name())) {
    op = pir::Operation::Create(
        vec_inputs, op_attribute, op_output_types, legacy_kernel_op_info);
  } else {
    op = pir::Operation::Create(
        vec_inputs, op_attribute, op_output_types, phi_kernel_op_info);
  }
  // ....
}
```

> [!NOTE]
> 小问题：在Operator Dialect中我们定义了几百个Op，那是不是意味着Kernel Dialect中也有对应的几百个 Op？

答案：目前Kernel Dialect中只有 2 个 Op。

```c++
// paddle/fluid/pir/dialect/kernel/ir/kernel_op.h
class PhiKernelOp : public pir::Op<PhiKernelOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_kernel.phi_kernel"; }
  static constexpr uint32_t attributes_num = 3;
  static const char *attributes_name[attributes_num];
  std::string op_name();
  std::string kernel_name();
  phi::KernelKey kernel_key();
  void VerifySig();
};

class LegacyKernelOp : public pir::Op<LegacyKernelOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_kernel.legacy_kernel"; }
  static constexpr uint32_t attributes_num = 3;
  static const char *attributes_name[attributes_num];
  std::string op_name();
  std::string kernel_name();
  phi::KernelKey kernel_key();
  void VerifySig();
};
```

## 3. Instruction构造

接下来，我们将正式进入新执行器。为了更好与旧Program的执行器做隔离，并尽可能统一后续的接口。我们在新增Pir下的执行器之前，先对当前的执行器进行了抽象，并派生出了ProgramInterpreter和PirInterpreter。

```c++
//paddle/fluid/framework/new_executor/pir_interpreter.h
/// @brief InterpreterBaseImpl is a abstract Base Class and define necessary
/// interface with virtual keywords for Derived class.
class InterpreterBaseImpl {
 public:
  virtual ~InterpreterBaseImpl() = default;
  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors,
      bool need_fetch = true,
      bool enable_job_schedule_profiler = false) = 0;
}

class ProgramInterpreter : public InterpreterBaseImpl {
  // .....
}


class PirInterpreter : public InterpreterBaseImpl {
// ...
}

```

大家目前只需要重点关注BuildInstruction的构建过程即可，逻辑实现是针对不同Dialect 的分发:

```c++
// paddle/fluid/framework/new_executor/pir_interpreter.cc
void PirInterpreter::BuildInstruction() {
  vec_instruction_base_.clear();
  size_t op_idx = 0;
  for (auto& op : *ir_block_) {
    if (op.dialect()->name() == "builtin") {
        //...
    } else if (op.dialect()->name() == "cf") {
        //...
    } else if (op.dialect()->name() == "pd_op") {
      if (op.isa<paddle::dialect::IfOp>()) {
         //...
      } else if (op.isa<paddle::dialect::WhileOp>()) {
        //...
      }
    } else if (op.dialect()->name() == "pd_kernel") {
      if (op.isa<paddle::dialect::LegacyKernelOp>()) {
        CREATE_INSTR(LegacyKernelInstruction);
      } else {
        CREATE_INSTR(PhiKernelInstruction);
      }
#ifdef PADDLE_WITH_CINN
    } else if (op.dialect()->name() == "cinn_runtime") {
      CREATE_INSTR(CinnJitInstruction);
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Now only support pd_kernel and cinn dialect."));
    }
  }
}
```

## 4. host/device 函数调用链

此处我们关注下CinnJitInstruction的构造和执行流程。由前面章节可知，JitKernelOp 核心的资源是fn_ptr，我们就顺着fn_ptr的流转来梳理整个过程，首先看下在 CinnJitInstruction是否存储了：

```c++
class CinnJitInstruction : public InstructionBase {
 public:
  CinnJitInstruction(size_t id,
                     const platform::Place& place,
                     ::pir::Operation* op,
                     const ValueExecutionInfo* value_exec_info);

  void Run() override;

 private:
  class FnPtrImpl;
  std::shared_ptr<FnPtrImpl> fn_ptr_impl_{nullptr};   // <========= 构造了新的

  platform::Place place_;
  phi::DeviceContext* dev_ctx_;
  phi::DenseTensor* out_tensor_;
  std::vector<phi::DenseTensor*> tensor_args_;
  ::pir::Operation* op_{nullptr};  // not owned
};
```


在构造函数里可以看出fn_ptr的传递，本质性上只使用了cinn_kernel_info来构造了新的FnPtrImpl对象。

```c++
// paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc
CinnJitInstruction::CinnJitInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place) {

  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  fn_ptr_impl_ = std::make_shared<FnPtrImpl>(jit_kernel_op.cinn_kernel_info());
  // .... 省略
 }
```

先略过FnPtrImpl的设计。CinnJitInstruction是怎么来执行的？

* 解析出 DeviceContext;
* 申请输出Tensor的内存；
* 直接调用fn_ptr_impl_->Run

> [!NOTE]
> 小问题：这里目前仅适配的CUDA，未来还有一些其他多硬件的支持工作要做。

```c++
// paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc
void CinnJitInstruction::Run() {
#if defined(PADDLE_WITH_CUDA)
  auto gpu_ctx = static_cast<phi::GPUContext*>(dev_ctx_);

  auto stream = gpu_ctx->stream();

  for (size_t i = 0; i < tensor_args_.size(); ++i) {
    // TODO(6clc): template infer shape from tensor_args_[0].
    // After supporting symbolic calculation, perfect the code to query shape
    // of output tensor
    if (FLAGS_cinn_bucket_compile) {
      tensor_args_[i]->Resize(tensor_args_[0]->dims());
    }
    gpu_ctx->Alloc(tensor_args_[i], tensor_args_[i]->dtype());
  }

  fn_ptr_impl_->Run(tensor_args_, static_cast<void*>(stream));
#else
  VLOG(phi::FATAL) << "Not Supported: cinn jit instruction currently does not "
                      "support non-CUDA kernel";
#endif
}
```

因为fn_ptr_impl_其实承担了PHI Kernel Function 的角色，我们回过头来看 FnPtrImpl的设计实现：

* Step 1: 入参准备。因为在CINN里，使用的是原生指针，所以不能像PHI Kernel那样使用DenseTensor *
* Step 2：调用fn_ptr ，拉起 Cuda Kernel

```c++
// paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc
class CinnJitInstruction::FnPtrImpl {
  using CINNKernelInfo = cinn::hlir::framework::pir::CINNKernelInfo;

 public:
  explicit FnPtrImpl(const CINNKernelInfo& cinn_kernel_info)
      : cinn_kernel_info_(cinn_kernel_info) {}

  void Run(const std::vector<phi::DenseTensor*>& kernel_args, void* stream) {
    func_args_.clear();

    // 1. Convert the phi::DenseTensor type to cinn_pod_value_t
    for (size_t i = 0; i < kernel_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      buffer->memory = reinterpret_cast<uint8_t*>(kernel_args[i]->data());
      func_args_.emplace_back(buffer);
    }
    // 2. Convert arg's data about shape of Tensor to cinn_pod_value_t
    for (const auto& int_arg_mp : cinn_kernel_info_.int_args_map) {
      func_args_.emplace_back(kernel_args[int_arg_mp.second.arg_idx]->dims().at(
          int_arg_mp.second.dim_idx));
      func_args_.emplace_back(static_cast<int64_t>(
          kernel_args[int_arg_mp.second.arg_idx]->dims().at(
              int_arg_mp.second.dim_idx)));
    }

    // 3. Launch host kernel
    ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
        static_cast<void*>(func_args_.data()), func_args_.size(), stream);
  }

 private:
  CINNKernelInfo cinn_kernel_info_;   // <======== copy 存储了一份
  std::vector<cinn_pod_value_t> func_args_;  // <==== 函数入参
};
```

> [!NOTE]
> 小问题：Kernel 拉起所需要<<<gridDim, blockDim>>>>是在哪里指定的？


答：backend::Compiler在编译生成Host fn_ptr时，就已经内嵌在函数源码里了。

```c++
define void @fn_reduce_max_broadcast_to_subtract_exp_reduce_sum_broadcast_to_0_divide(i8* %0, i32 %1, i8* %2) #12 {
entry:
  %fn_reduce_max_broadcast_to_subtract_exp_reduce_sum_broadcast_to_0_divide_kernel_ptr_load =  \
          load i8*, i8** @fn_reduce_max_broadcast_to_subtract_exp_reduce_sum_broadcast_to_0_divide_kernel_ptr_, align 8
  call void @cinn_call_cuda_kernel(i8* %fn_reduce_max_broadcast_to_subtract_exp_reduce_sum_broadcast_to_0_divide_kernel_ptr_load,
                                   i8* %0,   // 输入指针
                                   i32 %1,   // 输入指针
                                   i32 64,   // gridDim.x
                                   i32 1,    // gridDim.y
                                   i32 1,    // gridDim.z
                                   i32 128,  // blockDim.x
                                   i32 1,    // blockDim.x
                                   i32 1,    // blockDim.z
                                   i8* %2   // 输出指针
                                   )
  ret void
}
```
