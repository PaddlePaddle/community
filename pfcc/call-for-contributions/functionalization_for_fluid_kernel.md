# 飞桨算子体系重构 - Fluid算子函数式改造

> This project will be mentored by [@From00](http://github.com/from00)

[TOC]

## 1. 背景及计划
为了解决飞桨原Fluid算子体系存在的规范性、一致性、易用性、可维护性等诸多问题，我们设计开发了新的PHI算子体系，并在2022年进行了三期规模化迁移，将关联Python API的必要kernel从Fluid迁移至PHI。
经过几期迁移工作，截至2022年09月20日，基于paddle develop分支commit bcef827统计飞桨框架算子总数701个，其中347个已迁移至PHI，另外354个算子未迁移。这个统计数据放到现在是过时的，但并不影响对算子整体状态的判断：当前Fluid下仍有大量算子没有迁移到PHI。
Fluid下剩余算子的迁移，是一项长期而艰巨的任务，不仅任务量巨大，且存在许多后续计划废弃的算子，这些算子原则上是没必要做迁移的，但短期上却无法被删除，且可能为了兼容保留很长时间。因此，




> 注：关于 PHI 的设计可以参考 [《飞桨高可复用算子库 PHI 设计文档》](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)。

由于paddle旧的算子体系存在诸多的问题，我们设计实现了新的训推一体函数式算子体系，期望解决框架长久以来存在的诸多“根深蒂固的症结”，主要包括以下目标：

1. 解决由于原框架算子及算子Kernel之间无法复用，导致框架存在大量重复代码开发的问题，降低框架开发与维护成本；
2. 解决由于原框架算子开发方式复杂，导致内外算子开发学习理解成本高，难以推广的问题；
3. 解决由于算子体系复杂导致的动态图执行调度性能差，在小模型上与竞品存在数倍差距的问题；
4. 解决由于算子Kernel粒度粗，耦合调度功能，导致框架调度优化受限及其带来的性能问题；
5. 解决由于框架没有C++ API体系，导致基于飞桨进行外部二次开发困难，使生态拓展受限的问题；
6. 解决由于C++端算子开发规范薄弱，导致的C++端算子定义与Python端API定义差异较大的问题；
7. 解决由于Paddle与PaddleLite分别维护算子体系，导致升级同步困难、维护人力成本加倍的问题。

> 该项目的其他诸多称谓（以下简称phi）：新算子库、函数式算子库、Tensor计算库、phi（Paddle High-performance Operation Library）

目前phi已经能够支持运算类OpKernel及InferShape迁移，并在新旧体系下兼容适配，因此，我们需要对2.x模型使用的高频算子的相关组件进行规模化迁移，以支持新动态图架构及新的推理runtime架构infrt在2.3发布。

在开始迁移工作之前，大家可以先了解一下phi的整体设计，有助于大家更准确地迁移算子，设计文档见：

- [Paddle Tensor Operation Library (phi) 设计文档](http://agroup.baidu.com/paddlepaddle/md/article/4708213)

其中，可以重点了解下目录结构，Kernel相关设计、及InferMeta相关设计，这些内容与本次迁移工作直接相关。

![图片](http://agroup.baidu-int.com/file/stream/bj/bj-49580f79f76a2326a360eecb267c909590ba9ba9)

 此外，整体的迁移计划可以参考：[算子改造迁移计划](http://agroup.baidu.com/paddlepaddle/md/article/4693234)

#2. 核心迁移工作

本次迁移工作需要大家先迁移的是原先Op的两个重要部分： 1) Opkernel， 2) InferShape函数。最后还有个yaml补充工作：3) yaml文件补充

以trace op为例，要迁移的内容如下：

**1) trace Op的kernel的如下（trace op共包含cpu的正、反向kernel，gpu的正、反向kernel，共4个计算kernel）**

```
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

```
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

- **Functor+ExecutionContext** -> **Funtion+具体参数**：原先从ExecutionContext中获取kernel的input，output， attribute信息，新写法是直接使用函数传入的参数即可，对于旧的compute函数内部，大家主要需要关注的就是跟ExecutionContext相关的信息，其他的信息不需要详细了解。

以trace op为例，跟ExecutionContext相关的操作一共以下几个地方，在改造的时候需要重点关注这些地方

```
    auto* input = context.Input<framework::Tensor>("Input");    // 获取 input
    auto* out = context.Output<framework::Tensor>("Out");       // 获取 output

    const int64_t offset = context.Attr<int>("offset");         // 获取attribute offset
    const int64_t dim1 = context.Attr<int>("axis1");            // 获取attribute axis1
    const int64_t dim2 = context.Attr<int>("axis2");            // 获取attribute axis2
  
    auto& place =
          *context.template device_context<DeviceContext>().eigen_device();  // 获取eigen device
```

**3) 在yaml配置文件里添加前反向trace kernel的信息：**
```
- api : trace
  args : (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1)
  output : Tensor
  infer_meta :
    func : TraceInferMeta
  kernel :
    func : trace
  backward : trace_grad
```
```
- backward_api : trace_grad
  forward : trace (Tensor x, int offset, int axis1, int axis2) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, int offset, int axis1, int axis2)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : trace_grad
    data_type : out_grad
  no_need_buffer : x
```
这里分别配置了trace op的前向kernel和反向kernel的信息，配置此信息的目的是为了框架根据这些信息自动生成动态图调用的c++ api。配置好yaml后还需要修改单测进行测试，具体yaml配置说明和单测补充后续有详细介绍。

整体上我们分为三个部分介绍，OpKernel的迁移、InferShape的迁移及yaml和单测的补充，实际操作时，也建议大家分三步完成。

# 3 OpKernel迁移

OpKernel迁移共包含以下5个步骤，建议从前到后的按顺序来执行：

1. 声明新Kernel函数
2. 迁移并改写原OpKernel实现
3. 注册新Kernel
4. 实现OpMaker参数与Kernel参数映射函数
5. 移除原OpKernel实现及注册

##3.1 声明新Kernel函数

###3.1.1 声明前向Kernel函数

**文件创建：**

- 以trace op为例，首先在`paddle/phi/kernels`目录下新建`trace_kernel.h`文件，用于放置前向Kernel函数声明。

参考Python API，其Python API声明为：

```python
def trace(x, offset=0, axis1=0, axis2=1, name=None):
   """
    **trace**

    This OP computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.

    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
    of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    - Note that if offset is out of input's shape indicated by axis1 and axis2, 0 will be returned.

    Args:
        x(Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: the output data type is the same as input data type.
```

可以看到其参数有4个，那么对应的Kernel声明为：

```
template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out);
```

> 注：所有的kernel声明，统一放在namespace phi中，缩短函数的调用前缀，将来要给外部用户使用

说明如下：

1. 模板为固定写法，第一个模板参数为数据类型`T`，第二个模板参数为设备上下文`Context`，`template <typename T, typename Context>` 
2. 函数命名：Kernel 的命名统一加Kernel 后缀。即：Kernel名称+Kernel 后缀，驼峰式命名，例如：AddKernel
3. 参数顺序：Context， InputTensor …, Attribute …, OutTensor* 。即：第一位参数为Context， 后边为输入的Tensor， 接着是输入的属性参数， 最后是输出的Tensor的指针参数。如果Kernel没有输入Tensor 或者没有属性参数，略过即可
2. 第1个函数参数，类型为`const Context&`的dev_ctx
3. 第2个函数参数，输入Tensor，类型一般为`const DenseTensor&`，多个input可以参考python端API定义的顺序，变量命名对齐python api
4. 第3-5个函数参数，均为attribute（根据具体的含义，选择特定的int，float，vector<int>等类型），多个attribute 可以参考python端API定义的顺序，变量命名对齐python api
5. 第6个函数参数，输出Tensor，类型一般为`DenseTensor*`，多个output 可以参考python端API定义的顺序， 变量命名对齐python api

本次迁移的Op应该均有对应的Python API，如果没有对应的Python API，则参考OpMaker的定义， 以trace op的 OpMaker为例，共定义了1个input， 1个output，3个 attribute

```
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

trace op实现比较规范，OpMaker和Python API的参数个数、顺序都是一致的。

> **特殊情况补充：**
> 1. **特殊模板参数**：对于某些Kernel （如reshape ，copy ），这些kernel不关注数据类型T， 可以省去第一个模板参数，即为：`template <typename Context>`
> 2. **特殊输入类型**：对于某些特殊Kernel （如concat 和split kernel）的部分输入或输出是数组类型的DenseTensor（OpMaker中有`AsDuplicable`标记）, 此时输入类型为：`const std::vector<const DenseTensor*>&`; 输出类型为：`std::vector<DenseTensor*>`

### 3.1.2 声明反向Kernel函数

**文件创建：**

- 仍然以trace op为例，首先在`paddle/phi/kernels`目录下新建`trace_grad_kernel.h`文件，用于放置反向Kernel函数声明。

> 注：为了更好地服务于推理编包裁剪，phi设计上前向和反向kernel分离放置，这可能会导致文件数有一定膨胀，但可以减少推理在编包时不编译反向Kernel的实现阻力

反向kernel没有对应的Python API，其声明需要参考GradOpMaker的定义，下面为trace op的GradOPMaker的实现，其中定义了2个input， 3个attribute（正向的时候就是3个attribute），1个output；

```
 void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("trace_grad");
    grad_op->SetInput("Input", this->Input("Input"));                            // 需要正向的Input作为输入 
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));   // 正向输出的梯度，也作为输入
    grad_op->SetOutput(framework::GradVarName("Input"),                          // input的梯度作为输出，
                       this->InputGrad("Input"));
    grad_op->SetAttrMap(this->Attrs());                                          // 同时需要正向所有的attribute
  }
```

相应地，TraceGradKernel声明为：

```
template <typename T, typename Context>
void TraceGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     int offset,
                     int axis1,
                     int axis2,
                     DenseTensor* in_grad);
```

函数声明同样存在格式要求，基本与前向Kernel函数声明要求一致，按照输入、参数、输出的顺序排列参数，不同之处包括：

1.  反向Kernel输入参数，建议按以下顺序排列：
	- 最前面是前向的输入（如果需要的话），参数名称和前向参数的名称一样，例如`x`
	- 然后是前向的输出（如果需要的话），参数名称和前向参数的名称一样，例如`out`
	- 最后是前向输出的梯度，参数名称为前向参数名称加`_grad`后缀，例如`x_grad`
2. attribute的顺序和前向参数保持一致

> 注：如果有二阶反向、三阶反向Kernel，需要一并迁移，参数定义顺序参考一阶的原则

## 3.2 迁移并改写原OpKernel实现

迁移并改写原OpKernel实现可以分为6个步骤进行：

1. 剪切OpKernel实现
2. 修改ExecutionContext相关的逻辑
3. 替换对象类型或函数
4. 迁移依赖的函数
5. 添加头文件
6. 编译调试

以trace op的正向cpu kernel为例

**文件创建：**

- 仍然以trace op为例，首先在`paddle/phi/kernels/cpu`目录下新建`trace_kernel.cc`文件，用于放置反向Kernel函数实现。

### 3.2.1 剪切OpKernel实现

直接将原来的Compute函数的内容全部剪切，粘贴到新的TraceKernel中（注意，所有的kernel实现，必须放在namespace phi中）

```
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

###3.2.2 修改ExecutionContext相关的逻辑

将于ExecutionContext相关的逻辑全部改掉

```
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

###3.2.3 替换对象类型或函数

将fluid中原先的对象类型或定义，替换为phi中对应对象定义或函数定义，替换的映射关系如下：

> 注：phi最终是要作为独立的库编译，然后服务于fluid、infrt、自定义算子等上层应用的，因此phi中的文件不能include fluid的头文件，迁移时注意尽可能不要include多余的fluid头文件

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

```
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

###3.2.4 迁移依赖函数

迁移Kernel时，kernel调用的非本文件内的相关function及functor也需要一并迁移到phi，根据所依赖function或者functor使用场景不同，可以分为以下几种情况：

1. 仅有当前所迁移kernel使用的辅助函数（具体到设备，比如trace的cpu kernel），一律和kernel实现放到同一个设备文件夹中
- 如果辅助函数相关代码较少，就直接和kernel实现放到同一个`.cc/cu`中
- 如果辅助函数相关代码较多，就在kernel所在的设备目录创建`.h`管理代码
2. 有同设备多个kernel使用的辅助函数，在kernel所在的设备目录创建`.h`放置代码
3. 有跨设备多个kernel使用的辅助函数，在`kernels/funcs`目录下创建`.h/cc/cu`管理代码
4. 如果当前依赖的辅助函数可以直接归类到`kernels/funcs`目录下已有的文件中，则直接放过去，不用创建新的文件

从第3.2.3的代码来看，trace 依赖了Diagonal的函数，需要将Diagonal函数迁移到kernels中，Diagonal函数同时用于trace的cpu和gpu kernel中，即有跨设备的多个kernel使用，因此它需要在phi/funcs中创建对应的文件放置代码，这里将其放置到`phi/funcs/diagonal.h`中

> 注：如果觉得依赖的函数组件过于复杂，也可以保留在原位置，include对应头文件后使用，我们后续统一再迁移

###3.2.5 添加头文件

第5步，添加依赖的头文件，为了防止编译体积较大，编译速度较慢，建议只添加最小依赖

对于xxx_kernel.h，一般只需要tensor等基础数据结构头文件，例如trace_kernel.h

```
#include "paddle/phi/core/dense_tensor.h"
```

对于xxx_kernel.cc/cu，其中设备Context与kernel注册逻辑相关的头文件必须添加，例如trace的cpu kernel需要

```
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
```

### 3.2.6 编译调试

至此，一个kernel的主体代码迁移工作已经完成，这里建议，先重新cmake，make确认下有没有语法错误或者其他编译错误，根据错误的提示进行问题的修复，解决后再继续后面的步骤。

常见错误参考最后章节的FAQ。


##3.3 注册新Kernel

注册kernel的方式比较简单，直接使用宏，字段说明：
1. trace: kernel名称，和c++的名称一致（补充如果python api的名称和op名称不一致的情况）
2. CPU: backend名称， 这次迁移会遇到的主要就是CPU和GPU
3. phi::TraceKernel: kernel的函数名称，记得带上namespace phi
4. 剩余的均为数据类型，注册的数据类型对齐旧的kernel注册即可。
```
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
> 1. 如果忘记添加注册相关的头文件，会曝出一个xx的错误，如果遇到，请检查include的头文件 
> 2. 当新的kernel注册之后，旧kernel的compute kernel所在类和注册函数均需要删除，必须是在代码文件中直接删除，不能注释，否则会有链接错误
> 3. phi下的注册宏后边是带函数体{ }，不是直接加分号，此处与旧的注册宏有小区别 
> 4. 注册kernel的宏声明需要在global namespace


##3.4 实现OpMaker参数与Kernel参数映射函数

由于新的函数式Kernel参数是建议和Python api对齐的，而原先部分op的参数十分冗杂，比Python api参数多出许多，为了保证迁移后的kernel兼容原体系，我们需要对于这些原先不太规范的op注册参数映射函数。

这里分为两种情况：

1. OpMaker中输入、属性、输出参数**个数和顺序**和Python API及迁移后的Kernel一致的，我们会从OpProto中读取相关参数，确保其能够正确匹配，不需要实现此映射函数，对于trace前向op来讲就是如此，不需要关注这个环节

	```
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
	- 如果迁移Kernel之后，没有写这个映射函数，试着跑一下，报类似这样的错误的话，就说明需要加一下这个函数了
		- ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-47cd0de943222ad16716a49ea84d141a45bf2d9f)


2. OpMaker中输入、属性、输出参数**个数和顺序**和Python API及迁移后的Kernel不一致的，及所有反向Op（没有Proto信息），需要实现此函数，这里trace grad op就需要

- **文件创建：**仍然以trace op为例，首先在`paddle/phi/ops/compat`目录下新建`trace_sig.cc`文件，用于放置这里的映射函数。

- 由于函数式kernel的一个最重要的特别就是参数顺序和类型（顺序和类型是关键，变量名称不影响），我们需要定义一个函数来做一个从OpMaker中如何获取信息，并且按照顺序传递给新的kernel函数； 这个模块就是OpArgumentMapping， trace反向op的OpArgumentMapping定义如下， KernelSignature共包含4个内容
	1. kernel名称，这个是我们给kernel注册的时候的名称
	2. input list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致
	3. attribute list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致
	4. output list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致


	```
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


## 3.5 移除原OpKernel实现及注册

前序步骤完成之后，可以移除原先OpKernel的实现所在文件，以及相关Kernel的注册声明。

一般来讲，可以直接删除原先fluid operators目录以下相应文件：

`paddle/fluid/operators/xxx_op.h`
`paddle/fluid/operators/xxx_op.cu`

以及删除`paddle/fluid/operators/xxx_op.cc`中相关的`REGISTER_OP_CPU_KERNEL`和`REGISTER_OP_CUDA_KERNEL`声明

仍然以trace op为例，可以移除trace_op.h，trace_op.cu，以及trace_op.cc中相关的`REGISTER_OP_CPU_KERNEL`和`REGISTER_OP_CUDA_KERNEL`声明。

当然，如果原先的`xxx_op.h`有其他op依赖，也可以暂时保留相应头文件，或者将相应的公共函数按照3.2.4的步骤迁移到phi中，更新一下原先的依赖。

> 注意，如果出现找不到符号的报错，可能需要将部分单测中`USE_OP(xxx)`手动改为`USE_OP_ITSELF(xxx)`。

## 3.6 迁移后Kernel文件位置总结

预计存在以下几种情况，迁移时大家可以对号入座：

### 3.6.1 迁移与设备无关的Kernel（很少）

该类Kernel 实现与所有硬件设备无关，只需要一份代码实现，可参考reshape kernel。其新增文件及目录包括：

`paddle/phi/kernels/xxx_kernel.h`
`paddle/phi/kernels/xxx_kernel.cc`

###3.6.2 迁移与设备相关、但CPU&GPU实现一致的Kernel

部分Kernel 的实现在不同硬件上不一致，但是在常用的CPU， GPU 上代码实现一致。为了减少重复代码，便于维护，此类kernel 需要抽出共有的实现逻辑，放置于`paddle/phi/kernels/impl` 目录下，可参考sign，full kernel等。其新增文件及目录包括：

`paddle/phi/kernels/xxx_kernel.h`
`paddle/phi/kernels/impl/xxx_kernel_impl.h`
`paddle/phi/kernels/cpu/xxx_kernel.cc` ：include`xxx_kernel_impl.h`，一般仅放置kernel注册代码
 `paddle/phi/kernels/gpu/xxx_kernel.cu` ：include`xxx_kernel_impl.h`，一般仅放置kernel注册代码

### 3.6.3 迁移与设备相关、且CPU&GPU分别实现的Kernel

还有部分Kernel的实现，CPU 和GPU 上逻辑不同，此时没有共同实现的代码，需要区分CPU和GPU 硬件。
CPU 的实现位于`paddle/phi/kernels/cpu` 目录下； GPU的实现位于`paddle/phi/kernels/gpu` 下，可参考dot ，cast kernel等。其新增文件及目录包括：

`paddle/phi/kernels/xxx_kernel.h`
`paddle/phi/kernels/cpu/xxx_kernel.cc` 
 `paddle/phi/kernels/gpu/xxx_kernel.cu` 

### 3.6.4 迁移Kernel有include其他公共头文件

有些较为复杂的kernel，其实现可能不是全部在对应的xxx_op.h/cu中，如果include了其他的头文件，建议根据具体情况，将其放置到合适位置，这里参考3.2.4节的内容。


# 4. InferShape迁移

InferShape迁移相比Kernel迁移要简单，包括以下2个步骤：

1. 迁移改写原Op的InferShape
2. 声明InferShapeFunctor并删除原先InferShape

## 4.1 迁移改写原Op的InferShape

InferShape迁移为InferMeta的文件放置规则（以Tensor输入个数为判定标准）：

- `nullary.h`：没有输入Tensor参数的函数
- `unary.h`：仅有一个输入Tensor参数的函数
- `binary.h`：有两个输入Tensor参数的函数
- `ternary.h`：有三个输入Tensor参数的函数
- `multiary.h`：有三个以上输入Tensor或者输入为`vector<Tensor>`的函数
- `backward.h`：反向op的InferMeta函数一律在此文件中，不受前序规则限制


仍然以trace为例，根据trace的输入Tensor个数，确定其InferShape应该迁移到`phi/infermeta`目录中的哪个文件，这里trace是一个输入Tensor，因此迁移至`phi/infermeta/unary.*`中

第一步，在`unary.h`中声明TraceInferMeta，其参数列表与前述TraceKernel命名顺序均一致，不同之处如下：

1. InferMeta函数不需要模板参数
2. InferMeta函数不需要首个Context参数
3. 输入Tensor的类型需要改为MetaTensor
4. 如果需要在函数内部判断is_runtime，则在函数最后添加参数`MetaConfig config = MetaConfig()`

示例如下：

```
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

示例如下：

```
void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::OutOfRange(
          "Input's dim is out of range (expected at least 2, but got %ld).",
          x_dims.size()));
  PADDLE_ENFORCE_LT(
      dim1_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim1));
  PADDLE_ENFORCE_LT(
      dim2_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim2));
  PADDLE_ENFORCE_NE(
      dim1_,
      dim2_,
      phi::errors::InvalidArgument("The dimensions should not be identical "
                                    "%ld vs %ld.",
                                    dim1,
                                    dim2));

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


## 4.2 声明InferShapeFunctor并删除原先InferShape

将原先trace op override的InferShape函数删除，在最下方声明TraceInferShapeFunctor，并注册到Op中，示例如下：

```
DECLARE_INFER_SHAPE_FUNCTOR(trace, TraceInferShapeFunctor,
                            PD_INFER_META(phi::TraceInferMeta));
REGISTER_OPERATOR(trace, ops::TraceOp, ops::TraceOpMaker,
                  ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  TraceInferShapeFunctor);
```

然后编译、调试完成后，单测test_trace_op执行通过即表明迁移完成。

以上InferShape的迁移示例详细可以参考PR：https://github.com/PaddlePaddle/Paddle/pull/39517/files

# 5. Yaml文件及单测补充

phi算子库的Op通过yaml配置定义，相当于原先的Operator类和OpMaker类。在kernel迁移到Phi库后，也需要添加相应的Op定义，即需要补充yaml文件的内容，也是为了能让新动态图调用到我们迁移后的kernel。

yaml文件是一个使用yaml语法记录kernel信息的文件，paddle框架在编译阶段使用该文件自动生成新动态图调用的API，这个生成的API会间接调用我们迁移的Kernel。补充yaml文件后还需要将Python API接口配置为新动态图模式，并补充新动态图执行的单测，以验证补充的yaml文件的正确性。

## 5.1 补充yaml文件

yaml文件里补充的内容是迁移kernel的信息，其中前向kernel信息有关的yaml文件位置：
「*PaddleRoot/paddle/phi/api/yaml/api.yaml*」 
「*PaddleRoot/paddle/phi/api/yaml/legacy_api.yaml*」
反向kernel信息有关的yaml文件位置：
「*PaddleRoot/paddle/phi/api/yaml/backward.yaml*」
 「*PaddleRoot/paddle/phi/api/yaml/legacy_backward.yaml*」
其中，**带legacy**的yaml是我们**需要补充**的yaml文件，不带legacy的yaml文件是当前补充测试框架新功能用的文件，当前我们不必处理。以label_smooth这个kernel为例，迁移后前反向函数声明如下：

```
template <typename T, typename Context>
void LabelSmoothKernel(const Context& ctx,
                       const DenseTensor& label,
                       paddle::optional<const DenseTensor&> prior_dist,
                       float epsilon,
                       DenseTensor* out);
             
template <typename T, typename Context>
void LabelSmoothGradKernel(const Context& ctx,
                           const DenseTensor& out_grad,
                           float epsilon,
                           DenseTensor* label_grad);
```
api.yaml中label_smooth前向kernel的配置如下：
```
- api : label_smooth
  args : (Tensor label, Tensor prior_dist, float epsilon)
  output : Tensor
  infer_meta :
    func : UnchangedInferMeta
    param : [label]
  kernel :
    func : label_smooth
    data_type : label
  optional : prior_dist
  backward : label_smooth_grad
```
backward.yaml中label_smooth反向kernel的配置如下：
```
- backward_api : label_smooth_grad
  forward : label_smooth (Tensor label, Tensor prior_dist, float epsilon) -> Tensor(out)
  args : (Tensor out_grad, float epsilon)
  output : Tensor(label_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [out_grad]
  kernel :
    func : label_smooth_grad
  optional : prior_dist
```

该yaml文件每一项的含义如下：

| yaml配置项 | 含义 |
|----|----|
| api | 根据yaml产生的前向api的名字，一般为op名 |
| backward_api | 根据yaml产生的反向api的名字 |
| args | api的输入参数，和kernel中的「输入」及「属性」是一一对应的 |
| output | api的输出类型，多个输出需要加逗号隔开，可参考top_k的输出写法 |
| infer_meta | 用来指定使用的infermeta函数，「func」指定函数名，param指定传入的参数。如果传入参数和api的args配置项一样，param可以省略 |
| kernel | 用来指定该api调用的kernel函数，「func」指定函数名。如果调用kernel的类型，需要根据某些输入参数来决定，可以使用data_type指定这些输入参数 |
| optional | kernel输入参数中，有optional类型的输入，需要使用这个配置项说明 |
| backward | 前向api对应的反向api的名字 |
| forward | 反向api中用于配置前向api名字/参数列表/返回值 |
>1， **关于kernel下data_type项的补充说明：**
> data_type的设置，一般是没有输入Tensor，或者是有多个输入Tensor，但是每个输入Tensor的类型都不相同的时候设置的，比如有俩个输入参数，a参数tensor类型是Int，b参数类型是float这种情况。具体kernel类型以哪个参数的类型为准，可以参考xxx_op.cc文件里，有GetExpectedKernelType函数，这个函数里说明了需要从哪些参数里推断出kernel类型，如下以dropout为例，其GetExpectedKernelType如下，表示其从参数`X` 中进行推断kernel类型：
> ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-40b42cc79d70cab8755889f68454fe7c2a4e507c)
> 2， **output配置项注意事项：**
> 如果output中需要配置vector类型的输出，则需要使用花括号指定其vector大小，具体输出的大小需要根据python api里的实现逻辑来判断，比如einsum的输出vector大小和输入x一样，就这样指定：
> ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-7ede00ef34a43dae58577a170a5a16a371f7a107)
>  3， **反向yaml配置项注意事项1：**
>  反向梯度的输出命名要注意和前向一致，比如下图是错误写法：
>  ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-b3d5bd698421004e075e7dddfd5dfa83a3e569db)
>  正确的命名应该是`x_grad`，和前向的x要对应起来。
>   4， **反向yaml配置项注意事项2：**
>   args里是由输入Tensor+ Attr构成的，对于输入Tensor来说，需要满足如下顺序：*前向输入，前向输出，输出梯度*，如下所示，`roi_pool_grad`的args中，x/boxes/boxes_num是前向输入，arg_max是前向输出，out_grad是前向输出梯度。
>   ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-49908545999218a3afe9a360a1adb3ada0a99779)
>   为了满足反向yaml这个参数顺序，有时候也需要调整kernel的参数顺序以适应yaml的顺序，他俩的参数顺序要保持一致，同时，调整kernel参数顺序后，也要核查3.4 节中配置的映射文件，需要将映射输入的顺序和kernel的顺序调整一致保持对应。

## 5.2 Python API接口切换成新动态图
找到迁移op所对应的python api接口，添加切换新动态图模式的代码，以label_smooth为例：
label_smooth对应的python api定义在 *PaddleRoot/python/paddle/nn/functional/common.py* ，如下是其实现代码：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-b08b38fe029373c4b743eecbdec9628acbbd5c7a)
红色框内容是我们新添加的代码，其中in_dygraph_mode如果为true表示执行新动态图，**final_state_xxx** 表示调用yaml生成的新动态图api接口。
蓝色框是原有代码，表示执行调用旧动态图接口。
## 5.3 单测补充
### 5.3.1 一般单测的补充
添加了新动态图执行代码后，还需要添加单测进行验证。以label_smooth为例，找到label_smooth的单测文件 *test_label_smooth_op.py*：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-78e36e1d38e35b427e8b4dd7caf52e158b6ce504)
这个单测中，TestLabelSmoothOp继承了OpTest，对于这种单测，先设定好`self.python_api`（要调用的Python API接口），然后在check函数里边，添加`check_eager=True`，即可让单测测试新动态图。
> 注意：
>如果**Kernel有多个输出的，或者是有一个输出但是是vector类型输出**的，除了添加self.python_api，还需要添加`self.python_out_sig = [name1,name2,name3....] `，将新动态图输出与真值(self.outputs)进行匹配对比，这里name1，name2...表示self.outputs中的key值，详情使用可参考test_softmax_with_cross_entropy_op文件

另外还有一种单测，继承自TestCase，其测试代码是直接调用PythonAPI来实现的，如下图所示的例子：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-d64b4ff6bf1fb578d83ffaad001df3acd7abc640)
这种类型的单测，我们在执行单测时候打印一下Log：`GLOG_vmodule=api=6 ctest -R test_prelu_op -V`，如果是在api.cc里出现xx kernel，则说明添加的yaml在新动态图下是执行到了，如下所示：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-e4975cbc8cd1deadab64fa3e6d3e959e5c9070ce)
### 5.3.2 带有高阶微分kernel的单测补充
有的op带有二阶或者三阶反向Kernel函数，对于二阶和三阶反向执行逻辑在新动态图下的验证，和一般单测的修改方法略有不同，如下所示是一个二阶反向kernel的单测：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-7690e973437cc5ea16e573c112ad745781d96fd6)
首先定义一个函数，将paddle api封装起来，比如这里的subtract_wrapper，然后调用`gradient_checker.double_grad_check_for_dygraph` 并传入subtract_wrapper验证新动态图的高阶微分执行结果，并且在文件codegen_utils.py中补充该高阶反向api的名字：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-95c13f425b814488b0ee52e8078d75917d2847a8)
高阶微分单测添加更多示例可参考[PR#42361](https://github.com/PaddlePaddle/Paddle/pull/42361) 

更多yaml配置的详细说明，可参考官方文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html

# 6 注意事项

1. 迁移的时候，尽可能避免对原Kernel实现的逻辑改动，如果觉得它原来写得不好，想优化，可以拆分PR进行（担心出现性能变化，CI又发现不了，后续导致CE模型下降）
2. 迁移监控，在该卡片中添加子卡片的方式监控  https://console.cloud.baidu-int.com/devops/icafe/issue/DLTP-44313/show?cid=5&from=email&noticeStatistics=78391593  ； 在算子分配列表中，添加icafe卡片![图片](http://agroup.baidu-int.com/file/stream/bj/bj-a0a081a2fa79f50256f38ed38bcb3709e91d12ea)
3. 因为InferShape函数一般简短，所以没有拆过多文件，但这样会有一定概率冲突，建议大家，Kernel和Infershape分两个PR迁移，比如先一个PR把几个Kernel一起迁了，再去另一个PR迁对应的InferShape函数
4. kernel.cc/cu文件的include顺序问题，建议kernel.cc/cu文件中对应kernel.h在最前面
- 有一个共性的细节code style问题，这两天几乎大家都会碰到，这里简单说下，就是头文件的include顺序按照我们paddle遵循的规范，是有格式的
- ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-38df7266bca847d04c0e6d1703071a92ff3625c9)
- ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-066e0982431affbcbffb13be0439e12f07a461b3)
- 参考：https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/headers/#include
- 然后大家可以像code style中的示例一样，不同类别的头文件用空行分割一下，就不会被pre-commit重新调整了
5. 注意迁移Kernel之后，DenseTensor*的输出参数，仍然是返回值的定位，所以在kernel内注意只能写该参数的成员值，而不能读它的值用作逻辑判断，可能会出错


# 7 FAQ

> 本章节用于记录算子迁移中的常见问题及解决方法，会在算子迁移过程中持续更新。
> 如果遇到文档中未收录的问题，可直接添加到文档中（备注添加人）。

 1. 问题描述：移除原Op下`REGISTER_OP_CPU_KERNEL`或`REGISTER_OP_CUDA_KERNEL`如果出现类似`undefined reference to 'TouchOpKernelRegistrar_xxx_CUDA_DEFAULT_TYPE()'`的报错提示
   - 报错示例：![图片](http://agroup.baidu-int.com/file/stream/bj/bj-09144226801d5eeefb13ed6552b3adccc0323565)
   - 问题原因：由于在某些地方使用了该Kernel的注册符号，删除后找不到对应的注册符号便会报错。
   - 解决办法：
	   - 全局搜索`USE_OP(op_name)`，并替换为`USE_OP_ITSELF(op_name)`
   - 添加人：@云飞
 
	- 补充：除`USE_OP`宏外，`USE_OP_DEVICE_KERNEL`宏也会导致此错误，若搜索到`USE_OP_DEVICE_KERNEL(op_name,`，可直接删除。另外，如果旧OP的`REGISTER_OP_CPU_KERNEL`和`REGISTER_OP_GPU_KERNEL`注册宏没有直接删除，而是直接注释掉，因Pybind模块编译时会在代码文本中扫描注册宏并自动生成USE_OP代码，亦会导致此错误 @锐彪
 
 2. 问题描述：按照指南把`T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());`改成了`T* out_data = dev_ctx.Alloc<T>(out);`后编译报错。
	- 报错示例：![图片](http://agroup.baidu-int.com/file/stream/bj/bj-2044c403ed6317bc10d68226603ac2b24ecb7011)
	- 问题原因：暂不清楚
	- 解决方法：按照@元日升 指示改成`T* out_data = dev_ctx.template Alloc<T>(out); `编译通过。
	- 添加人：@陈鑫
	- 补充原因：按照C++语言标准，当`.`和`->`操作符后接显式模板化的模板类成员（Alloc<T>）时，需要用`template`关键字显式指定，否则编译器将直接假定Alloc不是模板类成员，见[标准](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4296.pdf)P337 14.2节 @锐彪

 3. 问题描述：按照新的命名规范，op命名需要和Python API名字保持一致，如果需要迁移的算子是V2版本(例如expand_v2)，在与原来的OpMaker进行关联、注册新的phi Kernel时需要注意什么地方？
 - 由于迁移过来，将`expend_v2`规范化为`expend`，会和原先已有的`expend` op产生冲突，这里原先的op一般是deprecated的版本，这种情况需要额外在`phi/core/compat/op_utils.h`中进行标记
 - ![图片](http://agroup.baidu-int.com/file/stream/bj/bj-22155bdda5b512b4f8c1b9493a289472dbab8b01)

 4.  问题描述：Scalar和ScalarArray什么时候使用？
 - 当原先Op有动态Attribute时需要使用，比如同时有`shape` attr和`ShapeTensor` input，或者同时有`axis` attr和`AxisTensor` input，可以参考reshape、scale、full等已有kernel的写法以及相应的映射函数。
 
 5.  问题描述：带有optional的参数什么时候使用？
 - 当原先Op的OpMaker中，输入输出标记有AsDispensable()时候使用，可以参考dropout、elementwise_multiply_grad等已有kernel的写法。
 
 6.  问题描述：注册kernel出现如下bug怎么修复？
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-8b7091e251a0f8d0fd6c84d868bf4c54ea047d45)
解决方式：先检查是否有语法问题，然后检查注册kernel的参数中，复杂类型是否使用了const &，如下是一个错误示例：
![图片](http://agroup.baidu-int.com/file/stream/bj/bj-f80131b3b64acb32ae6930341a7e8ec05db4f059)
红框部分缺少const &，注册后会报如上错误