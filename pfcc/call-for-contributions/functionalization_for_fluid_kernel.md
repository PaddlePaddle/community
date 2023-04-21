# 飞桨算子体系重构 - Fluid算子函数式改造

> This project will be mentored by [@From00](http://github.com/from00) and [@AndSonder](https://github.com/AndSonder)

## 1. 背景及计划
为了解决飞桨原Fluid算子体系存在的规范性、一致性、易用性、可维护性等诸多问题，我们设计开发了新的PHI算子体系，并在2022年进行了三期规模化迁移，将关联Python API的必要kernel从Fluid迁移至PHI。

> 关于 PHI 的设计可参考 [《飞桨高可复用算子库 PHI 设计文档》](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)。

经过几期迁移工作，截至2022年09月20日，基于paddle develop分支commit bcef827统计飞桨框架算子总数701个，其中347个已迁移至PHI，另外354个算子未迁移。

这个统计数据放到现在是过时的，但并不影响对算子整体状态的判断：当前Fluid下仍有大量算子没有迁移到PHI。Fluid下剩余算子的迁移，是一项长期而艰巨的任务，不仅代码量巨大，且存在许多后续计划废弃的算子，这些算子原则上是没必要做迁移的，但短期上却无法被删除，且可能为了兼容保留很长时间。因此，我们探索了一种低成本统一Fluid和PHI kernel的方案，通过对PHI注册体系进行扩展，让原Fluid算子可以在不改写kernel的情况下注册到PHI，从而达到统一注册体系的目的（相关PR [#49328](https://github.com/PaddlePaddle/Paddle/pull/49328)）。

低成本统一的方案将一些Fluid算子"原封不动"地注册到PHI，虽然在注册机制的层面实现了两套算子体系的统一，但由于Fluid kernel仍保留了原先的class类形式实现，丧失了PHI下函数式算子注册时所应具备的“记录自身输入输出属性“的能力。
> Kernel 需要将自身全部关键信息暴露给框架，记录其输入、输出和属性的信息，否则将导致框架调度与 Kernel 计算之间界限不清。
> 
> 现有 fluid Kernel 注册时仅记录了 Kernel 的 place，layout，dtype，输入输出等统一由 ExecutionContext 管理，没有相应的信息记录，现在 kernel 要改成函数式，每一个函数的输入输出和属性都是明确的，我们希望在这里记录每一个输入输出的信息，也是为了兼容 paddle-lite 的调度。
> 
> 摘自[《飞桨高可复用算子库 PHI 设计文档》2.3.4.4 Kernel注册](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md#2344-kernel-%E6%B3%A8%E5%86%8C)


kernel注册时记录输入输出的信息，是PHI算子注册机制的一个重要实现目标，也是框架调度与kernel计算界限清晰的前置要求。相关注册信息的缺失，使得PHI算子库无法达到实质上统一所有算子的理想状态，同时也给基础框架调度体系的一些优化工作带来较大的负担，如单机执行器全静态选kernel优化工作（相关PR [#50670](https://github.com/PaddlePaddle/Paddle/pull/50670)）依赖于kernel注册时记录的输入输出信息，在算子未正确注册时将无法正确地实现全静态选kernel。

针对已迁移到PHI，但在注册时未规范记录输入输出信息的kernel，飞桨社区已启动注册信息的补全工作（issue [#51292](https://github.com/PaddlePaddle/Paddle/issues/51292)）。但未迁移到PHI下的算子，需要先做迁移，将kernel改造成PHI体系下的函数式形式，才能对其输入输出信息做标记。因而，针对这些算子，我们需要建立专项对其进行迁移，并在注册时正确记录输入输出参数的信息。

考虑到Fluid下存量算子数量庞大，其中包含一定数量的废弃算子，且大部分kernel输入输出参数的属性与kernel本身是一致的，针对这部分算子可以根据kernel的信息自动生成填充每个输入输出的信息，不需要在注册时专门做特殊记录；因而，本期迁移只针对未废弃且必须在注册时记录输入输出参数信息的算子，其余算子当前不迁移不会对框架的调度产生影响，暂不纳入迁移范围。

飞桨研发工程师已经梳理出了本期需要迁移的算子清单（共12个）：
1. cudnn_lstm
2. dequantize
3. distributed_fused_lamb
4. fused_batch_norm_act
5. fused_batch_norm_act_grad
6. fused_attention
7. fused_attention_grad
8. fusion_group
9. pow2_decay_with_linear_warmup
10. sequence_mask
11. sequence_pool
12. stft

希望感兴趣的社区开发者共同来完成这些算子的迁移改造工作。



## 2. 核心工作内容

在开始算子的函数式迁移改造工作之前，大家可以先了解一下PHI的整体设计，有助于更准确地迁移算子。其中，可以重点了解下目录结构、Kernel相关设计以及InferMeta相关设计，这些内容与本次迁移工作直接相关。
设计文档见：[《飞桨高可复用算子库 PHI 设计文档》](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)


本次迁移工作需要大家迁移的是原先Op的两个重要部分： 1) Opkernel， 2) InferShape函数。

以trace op为例（PR[#39227](https://github.com/PaddlePaddle/Paddle/pull/39227/files)），要迁移的内容如下：

**1) trace Op的kernel（trace op共包含cpu的正、反向kernel，gpu的正、反向kernel，共4个计算kernel）**

```cpp
template <typename DeviceContext, typename T>
class TraceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int64_t offset = context.Attr<int>("offset");
    const int64_t dim1 = context.Attr<int>("axis1");
    const int64_t dim2 = context.Attr<int>("axis2");

    auto output_dims = out->dims();

    T* out_data = out->mutable_data<T>(context.GetPlace());

    const framework::Tensor diag =
        Diagonal<DeviceContext, T>(context, input, offset, dim1, dim2);
    if (diag.numel() > 0) {
      auto x = framework::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      auto output = framework::EigenVector<T>::Flatten(*out);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({1});
      output.device(place) = x.sum(reduce_dim);
      out->Resize(output_dims);
    } else {
      std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
    }
  }
};
```

**2) trace op的InferShape （前反向Op各有一个InferShape， 部分检查代码省略，方便大家查看）**

```cpp
class TraceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
   
    int dim1 = ctx->Attrs().Get<int>("axis1");
    int dim2 = ctx->Attrs().Get<int>("axis2");

    auto x_dims = ctx->GetInputDim("Input");

    int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
    int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;
   

    auto sizes = vectorize(x_dims);
    if (x_dims.size() == 2) {
      sizes.clear();
      sizes.push_back(1);
    } else {
      sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
      sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(sizes));
  }
};

```

> 注：有高阶微分支持的算子，其高阶反向Kernel也需要一并迁移

迁移的核心是改变kernel的形式，不需要了解kernel的实现细节，最核心的变化是：

- **Functor+ExecutionContext** -> **Funtion+具体参数**：原先从ExecutionContext中获取kernel的input，output， attribute信息，新写法是直接使用函数传入的参数即可。对于旧的compute函数内部，大家主要需要关注的就是跟ExecutionContext相关的信息，其他的信息不需要详细了解。

以trace op为例，跟ExecutionContext相关的操作一共以下几个地方，在改造的时候需要重点关注这些地方

```C++
    auto* input = context.Input<framework::Tensor>("Input");    // 获取 input
    auto* out = context.Output<framework::Tensor>("Out");       // 获取 output

    const int64_t offset = context.Attr<int>("offset");         // 获取attribute offset
    const int64_t dim1 = context.Attr<int>("axis1");            // 获取attribute axis1
    const int64_t dim2 = context.Attr<int>("axis2");            // 获取attribute axis2
  
    auto& place =
          *context.template device_context<DeviceContext>().eigen_device();  // 获取eigen device
```


以下分别详细介绍OpKernel迁移和InferShape迁移两部分内容，实际操作时，也建议大家分两步完成。

## 3. OpKernel迁移

OpKernel迁移共包含以下5个步骤，建议从前到后按顺序来执行：

1. 声明新Kernel函数
2. 迁移并改写原OpKernel实现
3. 注册新Kernel
4. 实现OpMaker参数与Kernel参数映射函数
5. 移除原OpKernel实现及注册

### 3.1 声明新Kernel函数

#### 3.1.1 声明前向Kernel函数

**文件创建：**

- 以trace op为例，首先在`paddle/phi/kernels`目录下新建`trace_kernel.h`文件，用于放置前向Kernel函数声明。
- 对于 fused 这一类算子，则需要在 `paddle/phi/kernels/fused` 目录下创建相关文件。

`trace_kernel.h` 内容如下：

```cpp
// 模板为固定写法
template <typename T, typename Context> 
// Kernel 的命名统一加Kernel后缀
void TraceKernel(const Context& ctx, 
                 const DenseTensor& x, // 输入的 Tensor
                 // Trace op的输入属性参数
                 int offset,
                 int axis1,
                 int axis2,
                 // 输出Tensor的指针
                 DenseTensor* out);
```

> 注：所有的kernel声明，统一放在namespace phi中，缩短函数的调用前缀。但fused类算子的声明统一放在phi::fusion命名空间中。

说明如下：

1. 模板为固定写法，第一个模板参数为数据类型`T`，第二个模板参数为设备上下文`Context`，`template <typename T, typename Context>` 
2. 函数命名：Kernel 的命名统一加Kernel后缀。即：Kernel名称+Kernel后缀，驼峰式命名，例如：AddKernel
3. 参数顺序：Context， InputTensor …, Attribute …, OutTensor* 。即：第一位参数为Context， 后边为输入的Tensor， 接着是输入的属性参数， 最后是输出的Tensor的指针参数。如果Kernel没有输入Tensor或者没有属性参数，略过即可
4. 第1个函数参数，类型为`const Context&`的dev_ctx
5. 第2个函数参数，输入Tensor，类型一般为`const DenseTensor&`，多个input可以参考OpMaker定义的顺序，变量命名对齐OpMaker
6. 第3-5个函数参数，均为attribute（根据具体的含义，选择特定的int、float，vector<int>等类型），多个attribute可以参考OpMaker定义的顺序，变量命名对齐OpMaker
7. 第6个函数参数，输出Tensor，类型一般为`DenseTensor*`，多个output 可以参考OpMaker定义的顺序， 变量命名对齐OpMaker

> 一般情况下参数的命名和顺序应与python API对齐，但本次迁移的Op大多没有对应的Python API，因而可直接参考OpMaker的定义。 

以trace op的OpMaker为例，共定义了1个input， 1个output，3个 attribute：

```cpp
 AddInput("Input",
             "(Tensor) The input tensor, from which the diagonals are taken.");
    AddOutput("Out", "(Tensor) the sum along diagonals of the input tensor");
    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis1",
        R"DOC((int, default 0), the first axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis2",
        R"DOC((int, default 1), the second axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 1.
        )DOC")
        .SetDefault(1);
```

>   OpMaker可以直接全局搜索：算子名+OpMaker得到（例如：TraceOpMaker）

trace op实现比较规范，OpMaker和Python API的参数个数、顺序都是一致的。

> **特殊情况补充：**
>
> 1. **特殊模板参数**：对于某些Kernel （如reshape ，copy ），这些kernel不关注数据类型T， 可以省去第一个模板参数，即为：`template <typename Context>`
> 2. **特殊输入类型**：对于某些特殊Kernel （如concat 和split kernel）的部分输入或输出是数组类型的DenseTensor（OpMaker中有`AsDuplicable`标记）, 此时输入类型为：`const std::vector<const DenseTensor*>&`; 输出类型为：`std::vector<DenseTensor*>`。一般情况下，int、float等C++内置类型通过值拷贝的方式传入，vector、optional等非内置类型通过const引用的方式输入

注意迁移后的命名风格，OpMarker 里面都是驼峰式命名，但迁移的时候需要其它kernel统一命名风格。比如某个参数名在OpMarker下是 ISTest，那么迁移过去之后就要变成 is_test。

#### 3.1.2 声明反向Kernel函数

**文件创建：**

仍然以trace op为例，首先在`paddle/phi/kernels`目录下新建`trace_grad_kernel.h`文件，用于放置反向Kernel函数声明。与前向 Kernel 函数类似，我们参照 trace op 的 GradOpMaker 的实现编写即可。

> 注：为了更好地服务于推理编包裁剪，phi设计上前向和反向kernel分离放置，这可能会导致文件数有一定膨胀，但可以减少推理在编包时不编译反向kernel的实现阻力.

反向kernel没有对应的Python API，其声明需要参考GradOpMaker的定义，下面为trace op的GradOpMaker的实现，其中定义了2个input， 3个attribute（正向的时候就是3个attribute），1个output:

```cpp
 void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("trace_grad");
    grad_op->SetInput("Input", this->Input("Input"));      // 需要正向的Input作为输入 
    // 正向输出的梯度，也作为输入
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));  
    // input的梯度作为输出
    grad_op->SetOutput(framework::GradVarName("Input"),                          
                       this->InputGrad("Input"));
    // 同时需要正向所有的attribute
    grad_op->SetAttrMap(this->Attrs());                                          
  }
```

相应地，`TraceGradKernel` 声明为：

```cpp
template <typename T, typename Context>
void TraceGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     int offset,
                     int axis1,
                     int axis2,
                     DenseTensor* in_grad);
```

>   GradOpMaker里`framework::GradVarName`等价于给参数命名添加后缀`_grad`

函数声明同样存在格式要求，基本与前向kernel函数声明要求一致，按照输入、参数、输出的顺序排列参数，不同之处包括：

1. 反向Kernel输入参数，需按以下顺序排列：
	- 最前面是前向的输入（如果需要的话），参数名称和前向参数的名称一样，例如 `x`
	- 然后是前向的输出（如果需要的话），参数名称和前向参数的名称一样，例如 `out`
	- 最后是前向输出的梯度，参数名称为前向参数名称加 `_grad` 后缀，例如 `x_grad`
2. `attribute` 的顺序和前向参数保持一致

> 注：如果有二阶反向、三阶反向Kernel，需要一并迁移，参数定义顺序参考一阶的原则

### 3.2 迁移并改写原OpKernel实现

迁移并改写原OpKernel实现可以分为6个步骤进行：

1. 剪切OpKernel实现
2. 修改ExecutionContext相关的逻辑
3. 替换对象类型或函数
4. 迁移依赖的函数
5. 添加头文件
6. 编译调试

以trace op的正向cpu kernel为例

**文件创建：**

仍然以trace op为例，首先在`paddle/phi/kernels/cpu`目录下新建`trace_kernel.cc`文件，用于放置反向Kernel函数实现。

#### 3.2.1 剪切OpKernel实现

直接将原来的Compute函数的内容全部剪切，粘贴到新的TraceKernel中（注意，所有的kernel实现，必须放在namespace phi中，对于 fused 这类的算子需要把算子放在 namespace phi::fusion下）

> 注：
> 1. cpu设备的Compute代码一般在paddle/fluid/operators/xxx_op.h 里，gpu设备对应的Compute代码一般在paddle/fluid/operators/xxx_op.cu里
>
> 2. 全局搜索：REGISTER_OP_CPU_KERNEL(trace 或者 REGISTER_OP_CUDA_KERNEL(trace 可以判断 trace_op 是否需要实现对应设备的算子。以 fused_attention 前向算子为例，全局搜索 REGISTER_OP_CPU_KERNEL(fused_attention) 就无法搜索到，说明该算子不需要在 CPU 设备上实现

```cpp
namespace phi {
template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out){
    auto* input = context.Input<framework::Tensor>("Input");   
    auto* out = context.Output<framework::Tensor>("Out");      

    const int64_t offset = context.Attr<int>("offset");
    const int64_t dim1 = context.Attr<int>("axis1");
    const int64_t dim2 = context.Attr<int>("axis2");

    auto output_dims = out->dims();

    T* out_data = out->mutable_data<T>(context.GetPlace());

    const framework::Tensor diag =
        Diagonal<DeviceContext, T>(context, input, offset, dim1, dim2);
    if (diag.numel() > 0) {
      auto x = framework::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      auto output = framework::EigenVector<T>::Flatten(*out);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({1});
      output.device(place) = x.sum(reduce_dim);
      out->Resize(output_dims);
    } else {
      std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
    }
  }     
 }               
 
 } // namespace phi      
```

#### 3.2.2 修改ExecutionContext相关的逻辑

将与ExecutionContext相关的逻辑全部改掉

```cpp
template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out){
    //~~auto* input = context.Input<framework::Tensor>("Input");~~   // 删除
    //~~auto* out = context.Output<framework::Tensor>("Out");~~      // 删除

    //~~const int64_t offset = context.Attr<int>("offset");~~         //删除
    //~~const int64_t dim1 = context.Attr<int>("axis1");~~            //删除
    //~~const int64_t dim2 = context.Attr<int>("axis2");~~            //删除

    auto output_dims = out->dims();

    T* out_data = out->mutable_data<T>(context.GetPlace());

    const framework::Tensor diag =
        Diagonal<DeviceContext, T>(context, &x, offset, dim1, dim2);  // 将input换成 &x
    if (diag.numel() > 0) {
      auto x = framework::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      auto output = framework::EigenVector<T>::Flatten(*out);
      //~~auto& place =
      //    *context.template device_context<DeviceContext>().eigen_device();~~
      // 改成下面的写法
      auto& place = *ctx.eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({1});
      output.device(place) = x.sum(reduce_dim);
      out->Resize(output_dims);
    } else {
      std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
    }
  }     
 }     
```

> 注：  
> 1. 在迁移的过程中，有一部分输入参数为可选参数，可选的参数在OpMaker中会使用 AsDispensable 标注，比如 `AddInput("LnScale", "...").AsDispensable()`。 可选参数迁移过来时需要声明成 `optional`类型: `const paddle::optional<DenseTensor>& LnScale` 。 输出带有 `AsDispensable` 的就可以不用管
> 2. 在迁移的时候，我们会发现老的 Compute 函数中都是声明指针指向各个 Input，对于可选的参数我们可以使用 `get_ptr()` 来获得指针，如果该参数没有传入 `get_ptr()` 会返回 `nullptr` 。具体使用代码为： `auto *p = ln_scale.get_ptr()`。对于普通的 `const DenseTensor&` 类输入只需使用 `&` 获取其指针即可
> 3. 如果所有的可选参数都设置正确，仍然有类似于 `Expected input_names.size() == input_defs.size(), but received input_names.size():3 != input_defs.size():11` 这样的错误，请添加 sig 文件（后续会介绍）后重试
> 4. 对于反向的Op（xxx_grad），有部分输入并不会使用 `AsDispensable` 而是写在 `if` 判断里的，这种情况也需要给在 if 里添加的输入添加 `optional` 参数

#### 3.2.3 替换对象类型或函数

将fluid中原先的对象类型或定义，替换为phi中对应对象定义或函数定义，替换的映射关系如下：

| fluid写法 | phi写法 |
|---|---|
| `farmework::Tensor` | `DenseTensor` |
| `farmework::LoDTensor` | `DenseTensor` |
| 模板`DeviceContext` | 模板`Context` |
| `out->mutbale_data(ctx.GetPlace()/place)` | `dev_ctx.template Alloc(out)` |
| `auto* ptr = out->mutbale_data()` | `auto* ptr = out->data()` |
| `out->mutbale_data(dims, place)` | `out->Resize(dims); dev_ctx.template Alloc(out)` |
| `out->mutbale_data(place, dtype)` | `dev_ctx.Alloc(out, dtype)` |
| `platform::erros::XXX` | `erros::XXX` |
| `platform::float16/bfloat16/complex64/complex128` | `dtype::float16/bfloat16/complex64/complex128` |
| `framework::Eigen***` | `Eigen***` |
| `platform::XXXPlace` | `phi::XXXPlace` |
| `framework::DefaultCPUGenerator()` | `dev_ctx.GetGenerator()->GetCPUEngine()` |
| `framework::LoD` | `phi::LoD` |
| `framework::TensorCopy/TensorCopySync` | `phi::Copy` |
| `platform::is_xxx_place` | `place.GetType() == phi::AllocationType::XXX` |

```C++
template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out){ 
    auto output_dims = out->dims();
    //~~T* out_data = out->mutable_data<T>(context.GetPlace());~~
    #换成以下实现
    T* out_data = dev_ctx.template Alloc<T>(out);
    
    const DenseTensor diag =
        Diagonal<Context, T>(dev_ctx, &x, offset, dim1, dim2);  
    if (diag.numel() > 0) {
      // auto x = framework::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      auto x = phi::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      // auto output = framework::EigenVector<T>::Flatten(*out);
      auto output = phi::EigenVector<T>::Flatten(*out);
   
      auto& place = *dev_ctx.eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({1});
      output.device(place) = x.sum(reduce_dim);
      out->Resize(output_dims);
    } else {
      std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
    }
  }     
 }     
```

> 注：
> 1. phi最终是要作为独立的库编译，然后服务于fluid、infrt、自定义算子等上层应用的，因此phi中的文件不能include fluid的头文件，迁移时注意尽可能不要include多余的fluid头文件
> 
> 2. 可以使用如下正则表达式在 IDE 寻找需要替换的部分（需要开启IDE搜索的正则匹配功能）
>
>`farmework::Tensor|farmework::LoDTensor|DeviceContext|mutbale_data|platform::erros|platform::float16|platform::bfloat16|platform::complex64|platform::complex128|framework::Eigen|platform::.*?Place|framework::DefaultCPUGenerato|framework::LoD|framework::TensorCopy|platform::is_.*?_place`

#### 3.2.4 迁移依赖函数

迁移Kernel时，kernel调用的非本文件内的相关function及functor也需要一并迁移到phi，根据所依赖function或者functor使用场景不同，可以分为以下几种情况：

1. 仅有当前所迁移kernel使用的辅助函数（具体到设备，比如trace的cpu kernel），一律和kernel实现放到同一个设备文件夹中

    - 如果辅助函数相关代码较少，就直接和kernel实现放到同一个`.cc/cu`中
    - 如果辅助函数相关代码较多，就在kernel所在的设备目录创建`.h`管理代码
2. 有同设备多个kernel使用的辅助函数，在kernel所在的设备目录创建`.h`放置代码
3. 有跨设备多个kernel使用的辅助函数，在`kernels/funcs`目录下创建`.h/cc/cu`管理代码
4. 如果当前依赖的辅助函数可以直接归类到`kernels/funcs`目录下已有的文件中，则直接放过去，不用创建新的文件
5. 只有fused类算子使用的辅助函数，放到`kernels/fusion`目录下

从第3.2.3的代码来看，trace 依赖了Diagonal的函数，需要将Diagonal函数迁移到kernels中，Diagonal函数同时用于trace的cpu和gpu kernel中，即有跨设备的多个kernel使用，因此它需要在phi/funcs中创建对应的文件放置代码，这里将其放置到`phi/funcs/diagonal.h`中

> 注：
> 1. 迁移时注意看phi目录下是否已经有实现了对应的函数组件，如果已经有了，直接使用phi下的即可。可以通过在phi下搜索其它kernel是否有使用到相同的组件，看它们引用哪个位置的代码来做判断
> 2. 迁过来之后的文件位置和命名尽量保持和fluid对应，比如paddle/fluid/operators/fused/fmha_ref.h迁移到paddle/phi/kernels/fusion/gpu/fmha_ref.h
> 3. 迁移时以函数组件为基本单位迁移，对于依赖较复杂的文件，可以只迁移其中需要使用到的一部分，另一部分保留在原来的位置。在迁移函数组件时，若fluid下原先使用到的代码不多，可以在迁移后把fluid下的使用都替换成phi下的，并把原先fluid下的实现删除。若fluid下使用到的代码较多，可以通过`using fluid::xxx = phi::xxx`的方式做重定向
> 4. 迁移的基本原则是不给phi引入新的fluid依赖。如果原先phi下其它kernel已经使用了对应的fluid依赖，则允许新迁移的kernel也使用这个fluid依赖，后续[PHI算子库独立编译专项](https://github.com/PaddlePaddle/Paddle/issues/47615)会一并做解耦。
> 5. 如果有较复杂的依赖无法迁移到phi，则可以将kernel改造成函数式之后，直接放在fluid下按phi函数式kerenl的方式注册，而不迁移到phi中。具体例子可参考`save_combine_op`

#### 3.2.5 添加头文件

第5步，添加依赖的头文件，为了防止编译体积较大，编译速度较慢，建议只添加最小依赖

对于xxx_kernel.h，一般只需要tensor等基础数据结构头文件，例如trace_kernel.h

```cpp
#include "paddle/phi/core/dense_tensor.h"
```

对于xxx_kernel.cc/cu，其中设备Context与kernel注册逻辑相关的头文件必须添加，例如trace的cpu kernel需要

```cpp
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
```

#### 3.2.6 编译调试

至此，一个kernel的主体代码迁移工作已经完成，这里建议，先重新cmake，make确认下有没有语法错误或者其他编译错误，根据错误的提示进行问题的修复，解决后再继续后面的步骤。 

常见错误参考最后章节的 FAQ。


### 3.3 注册新Kernel

注册kernel的方式比较简单，直接使用宏，字段说明：
1. trace: kernel名称，和c++的名称一致（补充如果python api的名称和op名称不一致的情况）
2. CPU: backend名称， 这次迁移大多数算子主要是CPU和GPU kernel，如果遇到有XPU kernel的算子，也按类似的方式迁移。
3. phi::TraceKernel: kernel的函数名称，记得带上namespace phi
4. 剩余的均为数据类型，注册的数据类型对齐旧的kernel注册即可
```cpp
PD_REGISTER_KERNEL(trace,
                   CPU,
                   ALL_LAYOUT,
                   phi::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
```

> 注：
> 1. 如果忘记添加注册相关的头文件，会编译错误，如果出现编译错误，请检查include的头文件
> 2. 当新的kernel注册之后，旧kernel的compute kernel所在类和注册函数均需要删除，必须是在代码文件中直接删除，**不能注释**，否则会有链接错误
> 3. phi下的注册宏后边是带函数体{}，不是直接加分号
> 4. 注册kernel的宏声明需要在global namespace


本次迁移需要判断出kernel需要标记的输入输出参数信息并在函数体{}中进行设置，对于标记信息的设置可参考PR [#51233](https://github.com/PaddlePaddle/Paddle/pull/51233)。除了输出的DataType，如果kernel输入需要注册，或Backend需要注册，也应一同注册上。

对于一些输出信息无法静态确定的kernel，可以在注册宏花括号里推导输出信息，如`batch_norm`的gpu kernel:
```C++
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
```

对于一些无法在注册宏里推导的参数，则需要对该参数注册`UNDEFINED`，然后通过`InferMeta`推导。

### 3.4 实现OpMaker参数与Kernel参数映射函数

为了保证迁移后的kernel兼容原体系，我们需要对op注册参数映射函数。

这里分为两种情况：

1. OpMaker中输入、属性、输出参数**个数和顺序**和迁移后的Kernel一致的，我们会从OpProto中读取相关参数，确保其能够正确匹配，不需要实现此映射函数，对于trace前向op来讲就是如此，不需要关注这个环节

	```cpp
	 AddInput("Input",
	             "(Tensor) The input tensor, from which the diagonals are taken.");
	    AddOutput("Out", "(Tensor) the sum along diagonals of the input tensor");
	    AddAttr<int>(
	        "offset",
	        R"DOC((int, default 0), offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
	        )DOC")
	        .SetDefault(0);
	    AddAttr<int>(
	        "axis1",
	        R"DOC((int, default 0), the first axis of the 2-D planes from which the diagonals should be taken. 
	        Can be either positive or negative. Default: 0.
	        )DOC")
	        .SetDefault(0);
	    AddAttr<int>(
	        "axis2",
	        R"DOC((int, default 1), the second axis of the 2-D planes from which the diagonals should be taken. 
	        Can be either positive or negative. Default: 1.
	        )DOC")
	        .SetDefault(1);
	```

	- 具体来说，可以忽略此环节的条件如下：
		- Input和Output，除去Dispensable、Extra、Quant之外的参数和Python端API顺序、类型一致
		- Attr，除去Extra、Quant以及use_mkldnn，use_cudnn之外的参数和Python端API顺序、类型一致
	- 如果迁移Kernel之后，没有写这个映射函数，试着跑一下，报类似 `Expected input_names.size() == input_defs.size(), but received input_names.size():3 != input_defs.size():11` 这样的错误的话，就说明需要加一下这个函数了


2. OpMaker中输入、属性、输出参数**个数和顺序**和迁移后的Kernel不一致的，及所有反向Op（没有Proto信息），需要实现此函数，这里trace grad op就需要

- 文件创建：仍然以trace op为例，首先在`paddle/phi/ops/compat`目录下新建`trace_sig.cc`文件，用于放置这里的映射函数。

- 由于函数式kernel的一个最重要的特别就是**参数顺序和类型**（顺序和类型是关键，变量名称不影响），我们需要**定义一个函数来做一个从OpMaker中如何获取信息**，并且按照顺序传递给新的kernel函数； 这个模块就是`OpArgumentMapping`， trace反向op的OpArgumentMapping定义如下， KernelSignature共包含4个内容
	1. kernel名称，这个是我们给kernel注册的时候的名称
	2. input list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致
	3. attribute list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致
	4. output list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致

Trace op 的映射函数如下：


```cpp
 #include "paddle/phi/core/compat/op_utils.h"
 
 namespace phi {
 
 KernelSignature TraceGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
   return KernelSignature("trace_grad",
                          {GradVarName("Out"), "Input"},
                          {"offset", "axis1", "axis2"},
                          {GradVarName("Input")});
 }
 
 }  // namespace phi
 
 PD_REGISTER_ARG_MAPPING_FN(trace_grad, phi::TraceGradOpArgumentMapping);
```

当与OpMake的关联建立之后，可以重新cmake，编译，然后可以运行旧的Python op单测来测试正确性。

>注：没有input list或attribute list的，相应花括号内留空，不能省略花括号


### 3.5 移除原OpKernel实现及注册

前序步骤完成之后，可以移除原先OpKernel的实现所在文件，以及相关Kernel的注册声明。

一般来讲，可以直接删除原先fluid operators目录以下相应文件：

`paddle/fluid/operators/xxx_op.h`
`paddle/fluid/operators/xxx_op.cu`

以及删除 `paddle/fluid/operators/xxx_op.cc` 中相关的 `REGISTER_OP_CPU_KERNEL `和`REGISTER_OP_CUDA_KERNEL` 声明

仍然以trace op为例，可以移除trace_op.h，trace_op.cu，以及trace_op.cc中相关的`REGISTER_OP_CPU_KERNEL`和`REGISTER_OP_CUDA_KERNEL`声明。

当然，如果原先的`xxx_op.h`有其他op依赖，也可以暂时保留相应头文件，或者将相应的公共函数按照3.2.4的步骤迁移到phi中，更新一下原先的依赖。

> 注：如果出现找不到符号的报错，可能需要将部分单测中`USE_OP(xxx)`手动改为`USE_OP_ITSELF(xxx)`。

### 3.6 xpu kernel迁移
如果迁移的算子有实现xpu kerenl，则需要参照上述流程对xpu kernel也进行迁移。在迁移xpu kernel时，有以下注意点：
1. xpu尽量和cpu/gpu算子分开迁移，不要合并在一个PR中提交
2. 在自己的开发机上可以完成xpu kernel的编译，使用xpu编译选项：`cmake .. -DPY_VERSION=3.8 -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_XPU=ON -DON_INFER=ON -DWITH_PYTHON=ON -DWITH_AVX=ON -DWITH_MKL=ON -DWITH_MKLDNN=ON -DWITH_XPU_BKCL=ON -DWITH_DISTRIBUTE=ON -DWITH_NCCL=OFF`
3. 迁移xpu算子的PR提交时，需要在commit message里添加`test=kunlun`触发xpu的CI测试
4. xpu算子的实现大多都是直接调用库接口，代码逻辑简单，较难出现迁移错误，一般迁移后都能通过CI测试。若CI执行失败且发现不了问题，需要xpu环境调试，可找项目指导同学沟通。

### 3.7 迁移后Kernel文件位置总结

预计存在以下几种情况，迁移时大家可以对号入座：

#### 3.7.1 迁移与设备无关的Kernel（很少）

该类Kernel 实现与所有硬件设备无关，只需要一份代码实现，可参考reshape kernel。其新增文件及目录包括：

`paddle/phi/kernels/xxx_kernel.h`
`paddle/phi/kernels/xxx_kernel.cc`

#### 3.7.2 迁移与设备相关、但CPU&GPU实现一致的Kernel

部分Kernel的实现在不同硬件上不一致，但是在常用的CPU， GPU上代码实现一致。为了减少重复代码，便于维护，此类kernel 需要抽出共有的实现逻辑，放置于`paddle/phi/kernels/impl` 目录下，可参考sign，full kernel等。其新增文件及目录包括：

`paddle/phi/kernels/xxx_kernel.h`
`paddle/phi/kernels/impl/xxx_kernel_impl.h`
`paddle/phi/kernels/cpu/xxx_kernel.cc` ：include`xxx_kernel_impl.h`，一般仅放置kernel注册代码
 `paddle/phi/kernels/gpu/xxx_kernel.cu` ：include`xxx_kernel_impl.h`，一般仅放置kernel注册代码

#### 3.7.3 迁移与设备相关、且CPU&GPU分别实现的Kernel

还有部分Kernel的实现，CPU 和GPU 上逻辑不同，此时没有共同实现的代码，需要区分CPU和GPU 硬件。
CPU 的实现位于`paddle/phi/kernels/cpu` 目录下； GPU的实现位于`paddle/phi/kernels/gpu` 下，可参考dot ，cast kernel等。其新增文件及目录包括：

`paddle/phi/kernels/xxx_kernel.h`
`paddle/phi/kernels/cpu/xxx_kernel.cc` 
 `paddle/phi/kernels/gpu/xxx_kernel.cu` 

### 3.7.4 迁移xpu kernel
xpu的代码位于`paddle/phi/kernels/xpu`目录下，新增文件`paddle/phi/kernels/xpu/xxx_kernel.cc` 

#### 3.7.5 迁移Kernel有include其他公共头文件

有些较为复杂的kernel，其实现可能不是全部在对应的xxx_op.h/cu中。如果include了其它头文件，建议根据具体情况，将其放置到合适位置，这里参考3.2.4节的内容。


## 4. InferShape迁移

InferShape迁移相比Kernel迁移要简单，包括以下2个步骤：

1. 迁移改写原Op的InferShape
2. 声明InferShapeFunctor并删除原先InferShape

### 4.1 迁移改写原Op的InferShape

InferShape迁移为InferMeta的文件放置规则（以Tensor输入个数为判定标准）：

- `nullary.h`：没有输入Tensor参数的函数
- `unary.h`：仅有一个输入Tensor参数的函数
- `binary.h`：有两个输入Tensor参数的函数
- `ternary.h`：有三个输入Tensor参数的函数
- `multiary.h`：有三个以上输入Tensor或者输入为`vector<Tensor>`的函数
- `backward.h`：反向op的InferMeta函数一律在此文件中，不受前序规则限制


仍然以trace为例，根据trace的输入Tensor个数，确定其InferShape应该迁移到`phi/infermeta`目录中的哪个文件，这里trace是一个输入Tensor，因此迁移至`phi/infermeta/unary.*`中

第一步，在`unary.h`中声明TraceInferMeta，其参数列表与前述TraceKernel命名顺序均一致，不同之处如下：

1. InferMeta函数不需要模板参数
2. InferMeta函数不需要首个Context参数
3. 输入Tensor的类型需要改为MetaTensor
4. 如果需要在函数内部判断is_runtime，则在函数最后添加参数`MetaConfig config = MetaConfig()`

示例如下：

```cpp
void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out);
```


第二步，将原先trace op的InferShape拷贝至`unary.cc`中，和Kernel迁移类似，需要：

1. 移除ctx相关的语句
2. 原先ctx相关设置输出的方法改为使用MetaTensor的方法

方法修改映射表（3.2.3节中的映射表在此处依然生效，以下为追加内容）：

| fluid写法 | phi写法 |
|---|---|
| ctx->SetOutputDim | out->set_dims |
| ctx->ShareLod | out->share_lod |

> 注意：
> 1. InferMeta范围要比原先的InferShape广，InferMeta推断的是tensor中的meta信息，包括推断dtype，所以在迁移时顺便加一下`set_dtype()`
> 2. 在迁移之前，可以看一下已有的InferShape函数是否能够直接复用，如果有现成InferMeta可以复用的话，就不建议迁移了，我们原本的op在InferShape上也有很多重复代码，比如很多简单算子都可以直接复用UnchangedInferMeta
> 3.  InferMeta函数在文件里推荐按照字母序放置，一个是为了便于查看，另一个是为了减少规模化迁移算子带来的代码冲突

下面是 `TraceInferMeta` 的实现 （移除了异常判断的代码）：

```cpp
void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  auto sizes = vectorize(x_dims);
  if (x_dims.size() == 2) {
    sizes.clear();
    sizes.push_back(1);
  } else {
    sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
    sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
  }
  out->set_dims(phi::make_ddim(sizes));
  out->set_dtype(x.dtype());
}
```


### 4.2 声明InferShapeFunctor并删除原先InferShape

将原先trace op override的InferShape函数删除，在最下方声明TraceInferShapeFunctor，并注册到Op中，示例如下：

```cpp
DECLARE_INFER_SHAPE_FUNCTOR(trace, TraceInferShapeFunctor,
                            PD_INFER_META(phi::TraceInferMeta));
REGISTER_OPERATOR(trace, ops::TraceOp, ops::TraceOpMaker,
                  ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  TraceInferShapeFunctor);
```

然后编译、调试完成后，单测test_trace_op执行通过即表明迁移完成。

以上InferShape的迁移示例详细可以参考PR [#39517](https://github.com/PaddlePaddle/Paddle/pull/39517)

> 注：对于一些InferShape较复杂，迁移难度较大的情况，如果kernel的输出类型不需要通过InferMeta进行推导（即没有注册UNDEFINED），则在本期计划中可以暂时先不迁移InferShape。

## 5. 注意事项

1. Fused kernel迁移至phi/kernels/fusion目录（可以参考该目录下README），PR [#45802](https://github.com/PaddlePaddle/Paddle/pull/45802)中有一个迁移fuse算子的例子供参考
2. 迁移的时候，尽可能避免对原Kernel实现的逻辑改动，如果觉得它原来写得不好，想优化，可以拆分PR进行（担心出现性能变化，CI又发现不了，后续导致CE模型下降）
3. 因为InferShape函数一般简短，所以没有拆过多文件，但这样会有一定概率冲突，建议大家，Kernel和Infershape分两个PR迁移，比如先一个PR把几个Kernel一起迁了，再去另一个PR迁对应的InferShape函数
4. kernel.cc和.cu文件中对应的kernel.h放在最前面。paddle遵循谷歌代码格式要求，头文件的include顺序是有规范的，参考[链接](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/headers/#include)。大家可以像code style中的示例一样，不同类别的头文件用空行分割一下，就不会被pre-commit重新调整了
5. 注意迁移Kernel之后，DenseTensor*的输出参数，仍然是返回值的定位，所以在kernel内注意只能写该参数的成员值，而不能读它的值用作逻辑判断，可能会出错


## 6. FAQ

> 本章节用于记录算子迁移中的常见问题及解决方法，会在算子迁移过程中持续更新。
> 如果遇到文档中未收录的问题，可直接添加到文档中。

 1. 问题描述：移除原Op下`REGISTER_OP_CPU_KERNEL`或`REGISTER_OP_CUDA_KERNEL`出现类似`undefined reference to 'TouchOpKernelRegistrar_xxx_CUDA_DEFAULT_TYPE()'`的报错提示。
   - 问题原因：由于在某些地方使用了该Kernel的注册符号，删除后找不到对应的注册符号便会报错。
   - 解决办法：全局搜索`USE_OP(op_name)`，并替换为`USE_OP_ITSELF(op_name)`。除`USE_OP`宏外，`USE_OP_DEVICE_KERNEL`宏也会导致此错误，若搜索到`USE_OP_DEVICE_KERNEL(op_name,`，可直接删除。另外，如果旧OP的`REGISTER_OP_CPU_KERNEL`和`REGISTER_OP_GPU_KERNEL`注册宏没有直接删除，而是直接注释掉，因Pybind模块编译时会在代码文本中扫描注册宏并自动生成USE_OP代码，亦会导致此错误

 2. 问题描述：把`T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());`改成了`T* out_data = dev_ctx.Alloc<T>(out);`后编译报错。
  - 问题原因：按照C++语言标准，当`.`和`->`操作符后接显式模板化的模板类成员（Alloc<T>）时，需要用`template`关键字显式指定，否则编译器将直接假定Alloc不是模板类成员，见[标准](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4296.pdf)P337 14.2节
  - 解决方法：改成`T* out_data = dev_ctx.template Alloc<T>(out); `

 3. 问题描述：按照新的命名规范，op命名需要和Python API名字保持一致，如果需要迁移的算子是V2版本(例如expand_v2)，在与原来的OpMaker进行关联、注册新的phi Kernel时需要注意什么地方？
  - 由于迁移过来，将`expend_v2`规范化为`expend`，会和原先已有的`expend` op产生冲突，这里原先的op一般是deprecated的版本，这种情况需要额外在`phi/core/compat/op_utils.h`中进行标记

 4. 问题描述：Scalar和ScalarArray什么时候使用？
  - 当原先Op有动态Attribute时需要使用，比如同时有`shape` attr和`ShapeTensor` input，或者同时有`axis` attr和`AxisTensor` input，可以参考reshape、scale、full等已有kernel的写法以及相应的映射函数。

 5. 问题描述：带有optional的参数什么时候使用？
  - 当原先Op的OpMaker中，输入输出标记有AsDispensable()时候使用，可以参考dropout、elementwise_multiply_grad等已有kernel的写法。
