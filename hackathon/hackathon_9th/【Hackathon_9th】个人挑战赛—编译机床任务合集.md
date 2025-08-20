此文档展示 \*_PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛编译机床方向任务_- 详细介绍

## 【开源贡献个人挑战赛-编译机床】任务详情

本期任务贡献于 AI Infra 编译机床 计划的子项目 GraphNet（[https://github.com/PaddlePaddle/GraphNet](https://github.com/PaddlePaddle/GraphNet)），目标计算图数据集收集与通用评价指标建设。

1. 该项目要求 **8月27日中期检查，9月3日收尾**，无法保证完成时间的开发者不建议领取
2. 开发说明：

GraphNet设计了test_compiler接口来评测编译器在GraphNet数据集上的指标，实现代码在graph_net/torch/test_compiler.py。用法如下：

```
python -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /path/to/custom/compiler
```

参数功能包括：- --model-path，字符串类型，指定模型文件的路径，可以支持单个模型或多个模型批量测试。- --compiler，字符串类型，指定要评测的编译器后端，默认值为"default"，即torch.compile并且设置backend="inductor"。

给定一份计算图，当前已支持统计一些原始的指标，包括：1. 不同精度下的正确率；2. 编译前后的运行时间对比。

### NO.97 适配 tvm 编译器

**详细描述：**

- 功能需求：graph_net.torch.test_compiler支持后端使用 tvm 编译器，即支持配置 --compiler "tvm"，读取GraphNet/samples目录下的子图，可成功执行并获得正确的评测结果。
- 可选方案（可选择列举的一种或多种方案实现，或者自行调研更优的方案，若能对多种方案进行对比为最佳）
  1. 使用torch.compile(m, backend="tvm")，可参考 [https://docs.pytorch.org/docs/stable/torch.compiler.html](https://docs.pytorch.org/docs/stable/torch.compiler.html)。
  2. 在 PyTorch 中用 torch.jit.trace 或 torch.jit.script 把模型转换为 TorchScript，用 tvm.frontend.from_pytorch（或 relay 接口）加载。

- 测试要求：
  - GraphNet/samples 目录下，每种类型至少需要验证一个模型，需通过 Profiler 或者日志确认子图是否真的用到了 tvm 编译器后端。若遇到整个类型无法支持的问题，需通过 issue 或其他方式找官方确认。
  - 选取 GraphNet/samples 目录下一个子类型进行批量测试。

**提交内容**：

1. 撰写设计文档，提交 PR 添加至 GraphNet/docs。
2. 提交 PR 增强 GraphNet/graph_net/torch/test_compiler.py 功能，并在 PR 描述中记录单个模型测试和批量模型测试的结果。

**技术要求：**

- 熟练掌握 Python
- 对 torch、tvm 有一定的了解

**中期检查：**

- 确定实现方案，提交设计文档。

### NO.98 适配 xla 编译器

**详细描述：**

- 功能需求：graph_net.torch.test_compiler支持后端使用 xla 编译器，即支持配置 --compiler "xla"，读取GraphNet/samples目录下的子图，可成功执行并获得正确的评测结果。
- 可选方案（可选择列举的一种或多种方案实现，或者自行调研更优的方案，若能对多种方案进行对比为最佳）
  1. 使用torch_xla包，参考 [https://github.com/pytorch/xla](https://github.com/pytorch/xla)，然后使用device='xla'
  2. 使用torch_xla包，然后使用torch.compile(..., backend='openxla')

- 测试要求：
  - GraphNet/samples 目录下，每种类型至少需要验证一个模型，需通过 Profiler 或者日志确认子图是否真的用到了 tvm 编译器后端。若遇到整个类型无法支持的问题，需通过 issue 或其他方式找官方确认。
  - 选取 GraphNet/samples 目录下一个子类型进行批量测试。

**提交内容**：

1. 撰写设计文档，提交 PR 添加至 GraphNet/docs。
2. 提交 PR 增强 GraphNet/graph_net/torch/test_compiler.py 功能，并在 PR 描述中记录单个模型测试和批量模型测试的结果。

**技术要求：**

- 熟练掌握 Python
- 对 torch、xla 有一定的了解

**中期检查：**

- 确定实现方案，提交设计文档。

### NO.99 适配 TensorRT 编译器

**详细描述：**

- 功能需求：graph_net.torch.test_compiler支持后端使用 xla 编译器，即支持配置 --compiler "tensorrt"，读取GraphNet/samples目录下的子图，可成功执行并获得正确的评测结果。
- 可选方案（可选择列举的一种或多种方案实现，或者自行调研更优的方案，若能对多种方案进行对比为最佳）
  1. 使用torch.compile(m, backend="tensorrt")，可参考 [https://docs.pytorch.org/docs/stable/torch.compiler.html](https://docs.pytorch.org/docs/stable/torch.compiler.html)
  2. 使用AOT编译，即torch_tensorrt.compile(m, ir, inputs, ...)，可参考 [https://docs.pytorch.org/TensorRT/py_api/torch_tensorrt.html?highlight=torch+compile#torch_tensorrt.compile](https://docs.pytorch.org/TensorRT/py_api/torch_tensorrt.html?highlight=torch+compile#torch_tensorrt.compile)
  3. 使用TorchScript，将模型转换为TorchScript后调用torch.\_C.\_jit_to_backend("tensorrt", ...)

- 测试要求：
  - GraphNet/samples 目录下，每种类型至少需要验证一个模型，需通过 Profiler 或者日志确认子图是否真的用到了 tvm 编译器后端。若遇到整个类型无法支持的问题，需通过 issue 或其他方式找官方确认。
  - 选取 GraphNet/samples 目录下一个子类型进行批量测试。

**提交内容**：

1. 撰写设计文档，提交 PR 添加至 GraphNet/docs。
2. 提交 PR 增强 GraphNet/graph_net/torch/test_compiler.py 功能，并在 PR 描述中记录单个模型测试和批量模型测试的结果。

**技术要求：**

- 熟练掌握 Python
- 对 torch、TensorRT 有一定的了解

**中期检查：**

- 确定实现方案，提交设计文档。

### NO.100 适配 BladeDISC 编译器

**详细描述：**

- 功能需求：graph_net.torch.test_compiler支持后端使用 xla 编译器，即支持配置 --compiler "blade-disc"，读取GraphNet/samples目录下的子图，可成功执行并获得正确的评测结果。
- 可选方案（可选择列举的一种或多种方案实现，或者自行调研更优的方案，若能对多种方案进行对比为最佳）
  1. TorchScript 模式，即先用 torch.jit.trace 或 torch.jit.script 把 PyTorch 模型转成 TorchScript，再用 BladeDISC 的 torch_blade.optimize 进行编译优化，可参考 [https://github.com/alibaba/BladeDISC/blob/main/docs/developers/bladedisc_torch_overview.md](https://github.com/alibaba/BladeDISC/blob/main/docs/developers/bladedisc_torch_overview.md)。
  2. FX Graph 模式，即基于 PyTorch FX 捕获计算图，直接交给 BladeDISC 编译，支持动态图和部分复杂控制流

- 测试要求：
  - GraphNet/samples 目录下，每种类型至少需要验证一个模型，需通过 Profiler 或者日志确认子图是否真的用到了 tvm 编译器后端。若遇到整个类型无法支持的问题，需通过 issue 或其他方式找官方确认。
  - 选取 GraphNet/samples 目录下一个子类型进行批量测试。

**提交内容**：

1. 撰写设计文档，提交 PR 添加至 GraphNet/docs。
2. 提交 PR 增强 GraphNet/graph_net/torch/test_compiler.py 功能，并在 PR 描述中记录单个模型测试和批量模型测试的结果。

**技术要求：**

- 熟练掌握 Python
- 对 torch、BladeDISC 有一定的了解

**中期检查：**

- 确定实现方案，提交设计文档。

### NO.101 多图抽取问题修复

**详细描述：**

- 问题描述：当TorchDynamo遇到子图打断时，会回退到Python解释器来运行，运行完被打断的地方后，继续用dynamo来跟踪。当前GraphNet的extractor抽取方法仍不完善，当被打断时只默认保存最后一个子图，不停的写入到文件里。
- 该任务大致对应 [https://github.com/PaddlePaddle/GraphNet/issues/165](https://github.com/PaddlePaddle/GraphNet/issues/165) issue 中提到的情况一，[https://github.com/PaddlePaddle/GraphNet/pull/133](https://github.com/PaddlePaddle/GraphNet/pull/133) PR通过设置 config 绕过了这个问题，目前仍需对extractor实现优化，最终支持图打断，无需设置config方可实现完整功能。
- **该任务与前面几项不同**，为extractor抽取方法的功能改进，与test_compiler无关。

**提交内容**：

1. 撰写设计文档（简要描述即可），提交 PR 添加至 GraphNet/docs。
2. 提交 PR 增强 GraphNet/graph_net/torch/extractor.py 及 GraphNet/graph_net/torch/utils.py 功能，在 PR 描述中引用问题 issue，并且记录模型测试的结果。

**技术要求：**

- 熟练掌握 Python
- 对 PyTorch 和 GraphNet 的运作机制有一定的了解

**中期检查：**

- 确定实现方案，提交设计文档。

### NO.102 vmap抽取问题修复

**详细描述：**

- 问题描述：该任务对应 [https://github.com/PaddlePaddle/GraphNet/issues/130](https://github.com/PaddlePaddle/GraphNet/issues/130) 中提到的问题：捕捉GraphMoudle时，遇到torch.\_C.\_functorch.PyCapsule.\_vmap_increment_nesting这类函数，dynamo的捕捉过程会被打断，这时候只会有前面捕捉的第一个子图。
- **该任务与前面几项不同**，与test_compiler无关，具体修改内容需要探索。
- 可能的解决方案，目前大概有三个思路：
  1. 让 dynamo感知\_vmap_increase_nesting / \_vmap_decrease_nesting；
  2. 在 validate 的时候不再使用 torch.compile 来抽取计算图，而是直接用 symbolic_trace；
  3. 编写 fx.Graph 的 revert pass，把\_vmap_increase_nesting / \_vmap_decrease_nesting 复原到高阶函数 vmap 调用。

**提交内容**：

1. 撰写设计文档（简要描述即可），提交 PR 添加至 GraphNet/docs。
2. 提交 PR 增强 GraphNet/graph_net/torch/extractor.py 及 GraphNet/graph_net/torch/utils.py 功能，在 PR 描述中引用问题 issue，并且记录模型测试的结果。

**技术要求：**

- 熟练掌握 Python
- 对 PyTorch 和 GraphNet 的运作机制有一定的了解

**中期检查：**

- 确定实现方案，提交设计文档。
