# Paddle + NeuralPDE.jl 求解二维泊松方程

|API名称   |   | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden">   | songjhaha | 
|提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-22 | 
|版本号| V1.0 | 
|依赖飞桨版本 <input type="checkbox" class="rowselector hidden">   | develop版本 | 
|文件名 | 20220322_ChainRules_NeuralPDE.md<br> | 


# 一、概述
## 1、相关背景
[使用PaddlePaddle + Julia求解2D Poisson方程](https://github.com/X4Science/INFINITY/issues/1)

[NeuralPDE.jl](https://neuralpde.sciml.ai/dev/)为求解PDE提供了许多基于神经网络的算法实现，该库的神经网络模块主要基于[DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl)。

## 2、功能目标

在Julia中封装paddle的神经网络，使得NeuralPDE基于paddle的神经网络模块，实现PDE的求解。在[NeuralPDE example](https://github.com/SciML/NeuralPDE.jl#example-solving-2d-poisson-equation-via-physics-informed-neural-networks)中，可以将网络模块部分直接替换成封装后的paddle模块，如：

```julia
# full connected Neural Network, return a julia wrapper of paddle's network
paddlewrap = PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; dtype=Float32, activation='sigmoid')

# get the initial parameters of Neural network
initθ = Optimisers.destructure(paddlewrap)[1]

discretization = PhysicsInformedNN(paddlewrap, QuadratureTraining(), init_params = initθ)
```

## 3、意义

使得Paddle能够作为支持[NeuralPDE.jl](https://neuralpde.sciml.ai/dev/)的神经网络后端，扩宽在AI+科学计算领域的应用。

# 二、飞桨现状

目前paddle在Julia可以直接使用[PyCall.jl](https://github.com/JuliaPy/PyCall.jl)调用，但缺少相关封装可以和Julia生态直接结合。

# 三、业内方案调研

## [PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)
[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)实现了对torch和jax的封装，定义了ChainRules的反向传播的规则，通过[DLPack.jl](https://github.com/pabloferz/DLPack.jl)，支持在CPU和GPU上对tensor数据的无需拷贝的使用。其中定义`ChainRulesCore.rrlue`的代码为：

```julia
function ChainRulesCore.rrule(wrap::TorchModuleWrapper, args...; kwargs...)
    T = typeof(first(wrap.params))
    params = wrap.params
    pyparams = Tuple(map(x -> DLPack.share(x, PyObject, pyfrom_dlpack).requires_grad_(true), params))
    pyargs = fmap(x -> DLPack.share(x, PyObject, pyfrom_dlpack).requires_grad_(true), args)

    torch_primal, torch_vjpfun = functorch.vjp(py"buffer_implicit"(wrap.torch_stateless_module, wrap.buffers), pyparams, pyargs...; kwargs...)
    project = ProjectTo(args)
    function TorchModuleWrapper_pullback(Δ)
        cΔ = fmap(x->Adapt.adapt(PyAdaptor{T}(), x), Δ)
        pycΔ = fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), cΔ)
        torch_tangent_vals = torch_vjpfun(pycΔ)
        jlparams_tangents = map(x -> DLPack.wrap(x, pyto_dlpack), torch_tangent_vals[1])
        args_tangents = project(fmap(x -> DLPack.wrap(x, pyto_dlpack), torch_tangent_vals[2:end]))
        return (Tangent{TorchModuleWrapper}(; torch_stateless_module = NoTangent(), dtype = NoTangent(), params = jlparams_tangents, buffers = NoTangent()), args_tangents...)
    end
    res = fmap(x->DLPack.wrap(x, pyto_dlpack), torch_primal)
    return res, TorchModuleWrapper_pullback
end
```

## [Torch.jl](https://github.com/FluxML/Torch.jl)
[Torch.jl](https://github.com/FluxML/Torch.jl)实现了端到端的封装。有更细粒度的封装，包括基本的运算符，广播运算等。


# 四、对比分析

- [PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)利用了DLPack协议实现了python里tensor数据和julia内array数据的共享，[Torch.jl](https://github.com/FluxML/Torch.jl)则是为tensor封装了julia里的api， 前者的实现会更加简单方便些，paddle也支持[DLPack协议](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/utils/dlpack.py)
- [PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)使用了[functorch](https://github.com/pytorch/functorch)作为torch的函数式实现，并实现了`ChainRulesCore.rrlue`。[Torch.jl](https://github.com/FluxML/Torch.jl)则是定义了相关运算的`@adjoint`。

[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)的实现较为快捷，但封装的粒度不如[Torch.jl](https://github.com/FluxML/Torch.jl)。本方案目标主要是使得封装的paddle神经网络能够支持NeuralPDE的求解，理论上仅需实现对`Zygote.gradient`的支持，因此先采用[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)的方案进行实现。

# 五、设计思路与实现方案

## 命名与参数设计

实现PaddleModuleWrap类型包装paddle的神经网络。为了方便`ChainRulesCore.rrlue`的实现，参考[functorch](https://github.com/pytorch/functorch)的模式，将神经网络分为存储结构的`stateless_module`和对应的参数`params`：
```julia
struct PaddleModuleWrapper
    NN::PaddleStatelessModule
    dtype::PyObject
    params::Tuple
end
```

实现全连接神经网络的构造函数，能够返回对应的PaddleModuleWrap实例，如：
```julia
function PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; dtype=Float32, activation='sigmoid')
```

实现前向传播：
```julia
function (wrap::PaddleModuleWrapper)(args...; kwargs...)
```

实现函数式的`vjp`:
```julia
function vjp(stateless_module::PaddleStatelessModule, pyparams, pyargs...; kwargs...)
```

实现`ChainRulesCore.rrule`:
```julia
function ChainRulesCore.rrule(wrap::PaddleModuleWrapper, args...; kwargs...)
```

## API实现方案

为完成以上api，需要：
- 使用DLPack.jl对tensor和array进行转换
- 在实现`PaddleStatelessModule`的运算功能时，使用paddle的运算符，如`paddle.matmul()`，`paddle.add()`和`paddle.nn.Sigmoid()()`
- 在实现`vjp`时，使用[`paddle.fluid.dygraph.grad(outputs,inputs,grad_outputs)`](https://github.com/PaddlePaddle/Paddle/blob/d9a41fc479009f75aa976ea18bd759504497796b/python/paddle/fluid/dygraph/base.py#L428)，例如：
```julia
function vjp(stateless_module::PaddleStatelessModule, pyparams, pyargs...; kwargs...)
    res = stateless_module(pyparams, pyargs...; kwargs...)
    function vjp_func(Δ)
        grad = paddle.fluid.dygraph.grad([res], [pyparams...], Δ, retain_graph=true)
        return grad
    end
    return res, vjp_func
end
```

# 六、测试和验收的考量

最后提供的代码为一个Julia的Package，主要内容为对Paddle网络的封装和对`Zygote.gradient`的支持，在测试代码中考虑的case如下：
- 构造全连接网络，前向传播的结果与paddle的api计算结果一致
- 梯度运算结果与paddle的api计算的结果一致
- 在GPU和CPU上的前向和反向传播

同时提供一些benchmark，方便性能上的研究和进一步优化
- 和paddle的原生api相比，前向传播的计算效率，以及梯度运算的计算效率
- 和NeuralPDE.jl结合使用的示例代码，将[示例](https://github.com/SciML/NeuralPDE.jl#example-solving-2d-poisson-equation-via-physics-informed-neural-networks)中的神经网络模块替换成封装后的paddle模块，和使用DiffEqFlux.jl或PyCallChainRules.jl之间的性能差别


# 七、可行性分析和排期规划

方案参考[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)，实现后需要进一步分析性能消耗，考虑优化方案，总体可以在活动时间内完成。

# 八、影响面

在Julia中对paddle进行一定程度的封装，对其他模块无影响


