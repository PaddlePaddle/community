
# 任务名

> RFC 文档相关记录信息

|              |                             |
| ------------ | --------------------------- |
| 提交作者     | lixiaoming233               |
| 提交时间     | 2024-11-05                  |
| RFC 版本号   | v1.0                        |
| 依赖飞桨版本 | develop                     |
| 文件名       | 20241105_torch_harmonics.md |

## 1. 概述

### 1.1 相关背景

> [飞桨科学计算工具组件开发大赛] (https://github.com/PaddlePaddle/PaddleScience/issues/1000)

'torch_harmonics' 是一个基于 PyTorch 的库，专注于处理和分析球谐函数（Spherical Harmonics，简称 SH）。球谐函数在图形学、物理、天文学等多个领域有着广泛的应用，特别是在表示和处理定义在球面上的函数时。

为 'torch_harmonics' 添加 paddle 适配，用于在 paddle 框架下处理和分析球谐函数。

### 1.2 功能目标

> 飞桨适配 'torch-harmonics' 
> [组件地址] (https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5)

1. 整理 'torch-harmonics' 的所有公开 API

2. 使用 paddle 的 python API 等价组合实现上述公开 API 的功能

3. 参考 pytorch 后端已有代码，撰写飞桨后端的单测文件，并自测通过

4. 整理代码并提交PR至 PaddleScience 官方仓库

### 1.3 意义

为 'torch_harmonics' 添加 paddle 适配，用于在 paddle 框架下处理和分析球谐函数。在图形学、物理、天文学等多个领域广泛使用。

## 2. PaddleScience 现状

> PaddleScience 暂无与 'torch_harmonics' 的相关的API适配。

## 3. 目标调研

> 参考 'torch_harmonics' 文档与源码，包含以下API：

+   [torch\_harmonics]
    +   [torch\_harmonics.convolution module]
        +   [`DiscreteContinuousConv`]
        +   [`DiscreteContinuousConvS2`]
        +   [`DiscreteContinuousConvTransposeS2`]
    +   [torch\_harmonics.legendre module]
        +   [`clm()`]
        +   [`legpoly()`]
    +   [torch\_harmonics.quadrature module]
        +   [`clenshaw_curtiss_weights()`]
        +   [`fejer2_weights()`]
        +   [`legendre_gauss_weights()`]
        +   [`lobatto_weights()`]
        +   [`trapezoidal_weights()`]
    +   [torch\_harmonics.random\_fields module]
        +   [`GaussianRandomFieldS2`]
    +   [torch\_harmonics.sht module]
        +   [`InverseRealSHT`]
        +   [`InverseRealVectorSHT`]
        +   [`RealSHT`]
        +   [`RealVectorSHT`]
    +   [torch\_harmonics.distributed package]
        +   [torch\_harmonics.distributed.distributed\_sht module]
            +   [`torch_harmonics.distributed.distributed_sht.DistributedInverseRealSHT`]
            +   [`torch_harmonics.distributed.distributed_sht.DistributedInverseRealVectorSHT`]
            +   [`torch_harmonics.distributed.distributed_sht.DistributedRealSHT`]
            +   [`torch_harmonics.distributed.distributed_sht.DistributedRealVectorSHT`]
        +   [torch\_harmonics.distributed.primitives module]
            +   [`torch_harmonics.distributed.primitives.compute_split_shapes()`]
            +   [`torch_harmonics.distributed.primitives.distributed_transpose_azimuth`]
            +   [`torch_harmonics.distributed.primitives.distributed_transpose_polar`]
            +   [`torch_harmonics.distributed.primitives.get_memory_format`]
            +   [`torch_harmonics.distributed.primitives.split_tensor_along_dim`]
        +   [torch\_harmonics.distributed.utils module]
            +   [`torch_harmonics.distributed.utils.azimuth_group()`]
            +   [`torch_harmonics.distributed.utils.azimuth_group_rank()`]
            +   [`torch_harmonics.distributed.utils.azimuth_group_size()`]
            +   [`torch_harmonics.distributed.utils.init()`]
            +   [`torch_harmonics.distributed.utils.is_distributed_azimuth()`]
            +   [`torch_harmonics.distributed.utils.is_distributed_polar()`]
            +   [`torch_harmonics.distributed.utils.is_initialized()`]
            +   [`torch_harmonics.distributed.utils.polar_group()`]
            +   [`torch_harmonics.distributed.utils.polar_group_rank()`]
            +   [`torch_harmonics.distributed.utils.polar_group_size()`]
    +   [torch\_harmonics.examples package]
        +   [torch\_harmonics.examples.pde\_sphere module]
            +   [`torch_harmonics.examples.pde_sphere.SphereSolver`]
        +   [torch\_harmonics.examples.shallow\_water\_equations module]
            +   [`torch_harmonics.examples.shallow_water_equations.ShallowWaterSolver`]
        +   [torch_harmonics.examples.sfno package]
            +   [torch_harmonics.examples.sfno.model package]
                +   [torch\_harmonics.examples.sfno.models.activations module]
                    +   [`torch_harmonics.examples.sfno.models.activations.ComplexCardioid`]
                +   [torch\_harmonics.examples.sfno.models.contractions module]
                +   [torch\_harmonics.examples.sfno.models.factorizations module]
                    +   [`torch_harmonics.examples.sfno.models.factorizations.get_contract_fun()`]
                +   [torch\_harmonics.examples.sfno.models.layers module]
                    +   [`torch_harmonics.examples.sfno.models.layers._no_grad_trunc_normal_()`]
                    +   [`torch_harmonics.examples.sfno.models.layers.trunc_normal_()`]
                    +   [`torch_harmonics.examples.sfno.models.layers.drop_path()`]
                    +   [`torch_harmonics.examples.sfno.models.layers.DropPath`]
                    +   [`torch_harmonics.examples.sfno.models.layers.MLP`]
                    +   [`torch_harmonics.examples.sfno.models.layers.RealFFT2`]
                    +   [`torch_harmonics.examples.sfno.models.layers.InverseRealFFT2`]
                    +   [`torch_harmonics.examples.sfno.models.layers.SpectralConvS2`]
                    +   [`torch_harmonics.examples.sfno.models.layers.FactorizedSpectralConvS2`]
                +   [torch\_harmonics.examples.sfno.models.sfno module]
                    +   [`torch_harmonics.examples.sfno.models.sfno.SpectralFilterLayer`]
                    +   [`torch_harmonics.examples.sfno.models.sfno.SphericalFourierNeuralOperatorBlock`]
                    +   [`torch_harmonics.examples.sfno.models.sfno.SphericalFourierNeuralOperatorNet`]
            +   [torch_harmonics.examples.sfno.utils package]
                +   [torch\_harmonics.examples.sfno.utils.pde\_dataset module]
                    +   [`torch_harmonics.examples.sfno.utils.pde_dataset.PdeDataset`]



其中基于 pytorch 的API在 Paddle 中绝大多数都有对应或通过组合替代的方式实现。

但是在 'torch_harmonics.examples.sfno' 这个包中所导入的 'tensorly' 库与 'tltorch' 只支持 pytorch 框架下使用，暂未实现 paddle 的适配工作，因此暂时不对这个 example 进行适配。

> 此外，也将对 notebook 部分的可视化示例进行适配。

对于分布式 sht 的实现方式，尝试在 paddle 中寻找更高层的 API 进行适配。

## 4. 设计思路与实现方案

保持所有文件组织结构与原有代码一致，保持注释、换行、空格、开源协议等内容一致。

> 参考 pytorch 和 paddle 的 API 映射文档，直接替换或组合替换原有API，实现飞桨适配 'torch-harmonics'


## 5. 测试和验收的考量

参考 pytorch 后端已有代码，撰写飞桨后端的单测文件，并自测通过。

参考 'torch_harmonics' 所提供的test文件，对 paddle 适配的效果进行测试。

## 6. 可行性分析和排期规划

2024.11.5 RFC文档

2024.11.15 提交paddle适配与单测代码

## 7. 影响面

> 为 'torch_harmonics' 添加 paddle 适配
 
用于在 paddle 框架下处理和分析球谐函数，增加 paddle 框架的使用场景。
