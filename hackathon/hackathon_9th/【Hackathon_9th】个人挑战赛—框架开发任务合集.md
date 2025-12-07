此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛框架开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-API 开发】任务详情

**NO.1 - NO.19 API正确性**

**详细描述：**

Paddle目前正在对全量API的边界正确性做系统性排查，我们开发了 PaddleAPITest 用于测试存在正确性问题的API。通过与Torch执行相同的API进行精度对比，我们发现一些API与Torch的API存在精度diff。经初步少量API确认，我们发现Paddle API确实存在一些正确性问题（过程中也发现了少量Torch API的正确性问题，如torch.tril、torch.triu）。现将这些问题Paddle API公开，邀请社区同学共同解决问题。参与本项活动，你将学习到Paddle算子库框架的设计，并对Paddle CPU、GPU Kernel的实现风格有详细的了解，对算子精度问题的调试技能积累一定经验。

**验收说明：**

- 验收说明：PaddleAPITest 回测 case 全部 Pass

**技术要求：**

- 熟悉深度学习框架API、熟悉CUDA编程

**参考资料：**

- [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest)、[#72637](https://github.com/PaddlePaddle/Paddle/issues/72637)（0-size问题修复）、[#72667](https://github.com/PaddlePaddle/Paddle/issues/72667)（精度问题修复）

**题目内容：**

### NO.1 完成 paddle.nn.functional.batch_norm 0-Size 问题修复
### NO.2 完成 paddle.expand 0-Size 问题修复
### NO.3 完成 paddle.incubate.nn.functional.fused_layer_norm 0-Size 问题修复
### NO.4 完成 paddle.index_add 0-Size 问题修复
### NO.5 完成 paddle.index_sample 0-Size 问题修复
### NO.6 完成 paddle.incubate.nn.functional.fused_multi_head_attention 0-Size 问题修复
### NO.7 完成 paddle.incubate.nn.functional.variable_length_memory_efficient_attention 0-Size 问题修复
### NO.8 完成 paddle.as_stride 0-Size 问题修复
### NO.9 完成 paddle.copysign 精度问题修复
### NO.10 完成 paddle.linalg.eigvals 精度问题修复
### NO.11 完成 paddle.linalg.eigvalsh 精度问题修复
### NO.12 完成 paddle.Tensor.cholesky_solve 精度问题修复
### NO.13 完成 paddle.unique 精度问题修复
### NO.14 完成 paddle.incubate.nn.functional.fused_multi_head_attention 精度问题修复
### NO.15 完成 paddle.incubate.nn.functional.variable_length_memory_efficient_attention 精度问题修复
### NO.16 完成 paddle.index_put 精度问题修复
### NO.17 完成 paddle.nn.functional.conv2d 精度问题修复
### NO.18 完成 paddle.nn.functional.conv2d_transpose 精度问题修复
### NO.19 完成 paddle.put_along_axis 精度问题修复

**NO.109 自定义算子**

### NO.109 基于 Setuptools 80+ 版本自定义算子机制适配

**详细描述：**

使用 C++ 实现自定义算子是深度学习框架中一种非常常见的需求，这可以使得框架自身足够整洁的情况下，灵活接入第三方生态开发的算子。
PaddlePaddle 目前提供了[两种自定义算子接入机制](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/index_cn.html)，分别为[自定义 C++ 算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html)、和[自定义 C++ 扩展](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/cpp_extension_cn.html)，自定义 C++ 扩展即将 C++ 函数通过 pybind11 暴露到 Python 端，而自定义 C++ 算子则是需要将 C++ 函数接入到算子库中。
PaddlePaddle 目前对于自定义 C++ 算子的实现是基于 setuptools 做了一些 patch，在 `bdist_egg` 阶段通过 patch `write_stub` 实现的，然而在 setuptools 80+，被 patch 的逻辑在 `install` command 不会被走到（于 [pypa/setuptools#2908](https://github.com/pypa/setuptools/pull/2908) 移除），因此我们希望基于 setuptools 80+ 对自定义 C++ 算子进行适配，确保自定义 C++ 算子在 setuptools 80+ 是可用的。

**验收说明：**

- 基于 setuptools 80+ 实现自定义 C++ 算子机制适配（需要能够通过 `python setup.py install` 或者 `pip install . --no-build-isolation` 安装并成功调用）
- 确保框架现存自定义算子单测在 setuptools 80 验证通过
- 确保低版本 setuptools 编译好的自定义算子在 setuptools 80 仍然能够正常加载

**技术要求：**

- 熟悉深度学习框架自定义算子机制
- 了解 setuptools 内部实现机制

**参考资料：**

- [PaddlePaddle 自定义算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/index_cn.html)
- [setuptools 源码](https://github.com/pypa/setuptools)

**NO.128 SOT 语法支持**

### NO.128 Paddle SOT 支持 import 语句

**详细描述：**

当前 Paddle SOT 在遇到函数内部的 import 语句时会直接 fallback 回动态图执行，无法充分利用静态图的性能优势。本任务需要修复此问题，使 SOT 能够正确处理 import 语句，保持静态图执行模式。

**验收说明：**

- 支持各类 import 语句在 SOT 转静过程中正确执行，包括：基础 import、from...import、import as、函数内 import、from .xxx import 等场景
- 编写全面的单元测试，覆盖 Python 3.9-Python 3.14 版本的所有 import 使用场景
- 确保修复后 import 语句不再触发 fallback 机制/子图打断机制，保持静态图执行
- 验证/修复下游模型（包括 FastDeploy、PaddleX）正确性，不引入新的错误或性能下降

**技术要求：**

- 熟悉PaddlePaddle框架架构和动转静机制
- 深入理解Paddle SOT的字节码翻译和符号执行原理
- 掌握Python import机制及字节码层面的实现细节
- 能够分析import语句在SOT中的处理流程，定位fallback的根本原因并提供解决方案
