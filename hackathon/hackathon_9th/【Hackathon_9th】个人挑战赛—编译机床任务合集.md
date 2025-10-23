此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛编译机床方向任务** 详细介绍

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

参数功能包括：
- --model-path，字符串类型，指定模型文件的路径，可以支持单个模型或多个模型批量测试。
- --compiler，字符串类型，指定要评测的编译器后端，默认值为"default"，即torch.compile并且设置backend="inductor"。

给定一份计算图，当前已支持统计一些原始的指标，包括：1. 不同精度下的正确率；2. 编译前后的运行时间对比。

> 交流微信群
> 
> <img src="https://github.com/user-attachments/assets/f0b57cda-63a2-44b0-ab14-87e04fd59ead" width="30%" height="60%">


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

### NO.110 AI4C**计算图分解验证器**

**任务背景**:

编译机床建设的第一个任务是对 GraphNet 的计算图数据集（及其元信息）进行有效切分拆解，以便 Agentic AI 学习样本，从而进行IR迁移、相似模式优化等任务。

我们设想中的 AI4C 子图分解功能包含以下模块：

1. 计算图区间分解器，负责分解操作执行，需要包含分解区间配置（本次任务不涉及）
2. 计算图分解方案验证器，对拆分后的子图做有效性验证（在本次任务中，使用一个极简分解器或分解模型测试用例来评估验证器的效果）

基于以上设计，我们计划先完成【计算图区间分解方案验证器】，复用现有的test_compiler及后续的评估方法，只新增一个RangeDecomposerValidatorBackend，作为验证器核心。

**任务描述**:

该任务的目标是在todo_works.range_decomposer_validator下，实现一个完善的range_decomposer_validator：

1. 该validator作为backend导入graph_net.torch.test_compiler，相应的配置已写入test_compiler代码；
2. 该validator接收一个【原模型】的torch.nn.Module，通过文件路径解析【分解后模型】的多个subgraph，重新输出一个组合后的torch.nn.Module；
3. 在解析过程中，默认【分解后模型】路径为【原模型】路径加上_decomposed，下有多个subgraph单独目录，例如/test/simple_CNN/的分解后模型包括/test/simple_CNN_decomposed/subgraph_0/.../test/simple_CNN_decomposed/subgraph_n/，每个subgraph的文件组成等同一份标准的GraphNet样本；
4. 在组合过程中，组合模型的forward是每个分解模型依次连接、嵌套而成，前一个模型的输出作为下一个模型的输入；
5. 该validator应当能够检测【原模型】与【分解后模型】的算子数量，去掉placeholder后进行比对，以验证其完整性；
6. 该validator应当根据【区间分解模式】配置文件（eg., 存放于/test/simple_CNN_decomposed/路径下，记录模型信息、子图数量、子图规模等，可参照graph_net.json设计），能够检测【分解后模型】是否按照预期的分解模式，以充分验证其完整性、分解验证过程前后一致性。

**预期效果**:

根据 [https://github.com/PaddlePaddle/GraphNet?tab=readme-ov-file#%EF%B8%8F-compiler-evaluation](https://github.com/PaddlePaddle/GraphNet?tab=readme-ov-file#%EF%B8%8F-compiler-evaluation) 提供的评估方法，对于【单个模型计算图】存在以下测试步骤：

1. graph_net.torch.test_compiler，记录下原始log
2. graph_net.log2json，将log转化为JSON
3. graph_net.plot_ESt，生成ESt图象，并输出其各项参数

随后，根据ESt图象在t>0时的阶梯表现，我们可以分析出该样本的正确性。在当前GraphNet repo中，默认t=1的抬升代表输出精度错误，t=3的抬升代表编译运行等其它类别错误。

为了方便测试，我们在GraphNet/todo_works/range_decomposer_validator/test/下提供了简单的测试用例：

* simple_CNN为分解前的原模型样本；
* simple_CNN_decomposed下面的subgraph_0到subgraph_2为分解后的模型样本，其model.py中的forward和weight_meta.py来自对simple_CNN模型的区间拆分。

开发者需要构造出更多测试用例，包含分解错误以及有placeholder的情形，以说明验证器达成了功能。

由于是单个样本测试，无需考虑性能提升，故预期使用所需的RangeDecomposerValidatorBackend后，对于正确拆分样本，ES图象应当是y=1的【一条直线】；对于错误或不完整的拆分样本，应当打印【错误报告】，或ES图象在t>0区域存在【阶梯状抬升】。

**技术要求：**

- 熟练掌握 Python
- 对 PyTorch 和 GraphNet 的运作机制有一定的了解



### NO.111 （GraphNet样本修复）27个因 torch 版本带来的样本问题
**任务背景**
通过指定graph_net.torch.test_compiler的--compiler参数为nope，一共有27个模型样本由于经过PyTorch 2.2.x或之前的版本抽取，其model.py中例如`l_self_modules_... = L_self_modules_... `此类的 predispatch 调用，在旧版 PyTorch 上被自动替换为`torch._functorch.predispatch(...)`；而PyTorch 2.3 以后把 torch._functorch 内部的实现做了重构，不包含 predispatch 属性，于是抛出错误 `AttributeError: module 'torch._functorch' has no attribute 'predispatch'`。涉及的样本如下：

```
samples/transformers-auto-model/42dot_LLM-SFT-1.3B
samples/transformers-auto-model/dansk-gpt-wiki
samples/transformers-auto-model/distilgpt2
samples/transformers-auto-model/EleutherAI_pythia-70m
samples/transformers-auto-model/GePpeTto
samples/transformers-auto-model/gpt2
samples/transformers-auto-model/gpt-sw3-356m
samples/transformers-auto-model/japanese-gpt-1b
samples/transformers-auto-model/joancipria_gpt2-base-bne-FineTunedEmoEvent
samples/transformers-auto-model/kogpt
samples/transformers-auto-model/LFM2-350M
samples/transformers-auto-model/LLaMmlein_120M
samples/transformers-auto-model/LYTinn_gpt2-finetuning-sentiment-model-3000-samples
samples/transformers-auto-model/nie3e_sentiment-polish-gpt2-large
samples/transformers-auto-model/orca_mini_3b
samples/transformers-auto-model/PolyCoder-160M
samples/transformers-auto-model/polyglot-ko-1.3b
samples/transformers-auto-model/Qwen1.5-0.5B
samples/transformers-auto-model/Qwen2.5-0.5B
samples/transformers-auto-model/Qwen3-Embedding-0.6B
samples/transformers-auto-model/sarvam-0.5
samples/transformers-auto-model/super-fast-llm
samples/transformers-auto-model/TinyLlama-1.1B-step-50K-105b
samples/transformers-auto-model/tiny_starcoder_py
samples/transformers-auto-model/Tucano-2b4
samples/transformers-auto-model/w11wo_javanese-gpt2-small-imdb-classifier
samples/transformers-auto-model/w11wo_sundanese-gpt2-base-emotion-classifier
```

**任务描述**
1. 对于上述模型，使用torch2.8重新抽取计算图并更新样本。
2. 自行测试，确保新样本运行符合预期效果。
3. 提交PR。



**预期效果**
1. 验证`test_compiler`评测结果，上述模型设置 nope 编译器后端测试不出现错误。




### NO.112 （GraphNet样本修复）batch_norm算子添加weight_meta约束
**任务背景**
GraphNet支持对深度学习模型的推理样本进行统一评测。batch_norm算子在CV模型中被普遍应用，推理模式下计算公式为：

$batch\_norm = weight * (x - running\_mean) / sqrt(running\_var + eps) + bias$

计算中存在除法和`sqrt`计算，为了得到正常的计算结果，需满足约束`running_var + eps > 0`，否则会产生异常的结果（如`inf`或`nan`）。在GraphNet评测任务中，算子的权重和输入都统一当做样本模型的输入来对待，由评测任务按照样本中保存的 meta 信息进行随机初始化。当前 Torch 样本浮点输入 meta 信息中保存了`mean` 和 `std`。依据 `mean`和`std`随机初始化的数据，难以保证一定满足该约束。Torch 样本在批量评测时，发现 150 个样本因 batch_norm 算子weight_meta 不满足上述要求，导致计算结果出现 `nan`，[https://github.com/PaddlePaddle/GraphNet/pull/301](https://github.com/PaddlePaddle/GraphNet/pull/301) 中采用一种临时方式规避了该问题。最终修复方法，需要为所有样本中 batch_norm 算子的 `running_var`参数添加`min_val = 0`约束。



**任务描述**
通过修改样本，为所有样本中 batch_norm 算子的 `running_var`参数添加`min_val = 0`约束。具体步骤如下：

1. 移除 [https://github.com/PaddlePaddle/GraphNet/pull/301](https://github.com/PaddlePaddle/GraphNet/pull/301) 添加的临时修复代码 [https://github.com/PaddlePaddle/GraphNet/blob/fd025b0c0c0e480577fa527e3d72aa1781484846/graph_net/torch/utils.py#L279](https://github.com/PaddlePaddle/GraphNet/blob/fd025b0c0c0e480577fa527e3d72aa1781484846/graph_net/torch/utils.py#L279) - L281，执行如下命令复现`nan`问题。

```
$ python -m graph_net.torch.test_compiler --model-path samples/mmseg/MAE --compiler "nope" --warmup 0 --trials 1
graph-net-test-compiler-log [Processing] /work/GraphNet/samples/torchvision/mnasnet1_3
graph-net-test-compiler-log [Config] model: mnasnet1_3
graph-net-test-compiler-log [Config] device: cuda
graph-net-test-compiler-log [Config] hardware: NVIDIA H20-3e
graph-net-test-compiler-log [Config] compiler: nope
graph-net-test-compiler-log [Config] warmup: 0
graph-net-test-compiler-log [Config] trials: 1
graph-net-test-compiler-log [Config] compile_framework_version: unknown
graph-net-test-compiler-log [Profiling] Using device: cuda NVIDIA H20-3e, warm up 0, trials 1
Trial 1: e2e=217.01026 ms, gpu=216.88783 ms
graph-net-test-compiler-log [Performance][eager]: {"e2e": {"mean": 217.0, "std": 0.0, "min": 217.0, "max": 217.0}, "gpu": {"mean": 217.0, "std": 0.0, "min": 217.0, "max": 217.0}}
graph-net-test-compiler-log [Datatype][eager]: float32
graph-net-test-compiler-log [Profiling] Using device: cuda NVIDIA H20-3e, warm up 0, trials 1
Trial 1: e2e=1.78576 ms, gpu=1.72432 ms
graph-net-test-compiler-log [Performance][compiled]: {"e2e": {"mean": 1.79, "std": 0.0, "min": 1.79, "max": 1.79}, "gpu": {"mean": 1.72, "std": 0.0, "min": 1.72, "max": 1.72}}
graph-net-test-compiler-log [Datatype][compiled]: float32
graph-net-test-compiler-log [DataType] eager:['float32'] compiled:['float32'] match:True
graph-net-test-compiler-log [Correctness][equal]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-10_rtol_1.00E-06]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-10_rtol_2.56E-04]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-10_rtol_1.69E-12]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-14_rtol_1.00E-14]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-09_rtol_3.98E-06]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-09_rtol_5.85E-04]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-09_rtol_2.54E-11]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_2.51E-13_rtol_2.51E-13]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-08_rtol_1.58E-05]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-08_rtol_1.34E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-08_rtol_3.82E-10]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_6.31E-12_rtol_6.31E-12]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-07_rtol_6.31E-05]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-07_rtol_3.06E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-07_rtol_5.75E-09]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.58E-10_rtol_1.58E-10]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-06_rtol_2.51E-04]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-06_rtol_7.00E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-06_rtol_8.65E-08]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_3.98E-09_rtol_3.98E-09]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-05_rtol_1.00E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-05_rtol_1.60E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-05_rtol_1.30E-06]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-07_rtol_1.00E-07]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-04_rtol_3.98E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-04_rtol_3.66E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-04_rtol_1.96E-05]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_2.51E-06_rtol_2.51E-06]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-03_rtol_1.58E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-03_rtol_8.36E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-03_rtol_2.94E-04]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_6.31E-05_rtol_6.31E-05]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-02_rtol_6.31E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-02_rtol_1.91E-01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-02_rtol_4.42E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.58E-03_rtol_1.58E-03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-01_rtol_2.51E-01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-01_rtol_4.37E-01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E-01_rtol_6.65E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_3.98E-02_rtol_3.98E-02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+00_rtol_1.00E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+00_rtol_1.00E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+00_rtol_1.00E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+00_rtol_1.00E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+01_rtol_3.98E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+01_rtol_2.29E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+01_rtol_1.50E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_2.51E+01_rtol_2.51E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+02_rtol_1.58E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+02_rtol_5.23E+00]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+02_rtol_2.26E+02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_6.31E+02_rtol_6.31E+02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+03_rtol_6.31E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+03_rtol_1.20E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+03_rtol_3.40E+03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.58E+04_rtol_1.58E+04]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+04_rtol_2.51E+02]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+04_rtol_2.73E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+04_rtol_5.11E+04]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_3.98E+05_rtol_3.98E+05]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+05_rtol_1.00E+03]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+05_rtol_6.25E+01]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+05_rtol_7.69E+05]: 0
graph-net-test-compiler-log [Correctness][all_close_atol_1.00E+07_rtol_1.00E+07]: 0
graph-net-test-compiler-log [Correctness][max_diff]: nan
graph-net-test-compiler-log [Correctness][mean_diff]: nan
graph-net-test-compiler-log [Result] status: success
graph-net-test-compiler-log [Speedup][e2e]: 121.2291
graph-net-test-compiler-log [Speedup][gpu]: 126.1628
```


2. 编写脚本，批量修改`GraphNet/samples`目录下面的样本的weight_meta.py，为batch_norm算子的`running_var`添加`min_val = 0`meta信息。
3. 修改`GraphNet/graph_net/torch/utils.py`，`replay_tensor`中使用`min_val`约束随机Tensor值的下界。
4. 提交PR。



**预期效果**
1. 验证`test_compiler`评测结果，`max_diff`和`mean_diff`中不再出现`nan`，设置 nope 编译器后端测试所有容忍度下`allclose`检查结果均通过。



### NO.113 （GraphNet样本修复）非法Torch样本修复
**任务背景**
一些样本在使用`test_compiler`评测时，计算结果中会产生`inf`或`nan`，最终导致`max_diff`为`nan`，精度检测不通过。样本列表如下：

```
samples/transformers-auto-model/IDEA-Research_grounding-dino-base
samples/transformers-auto-model/canary-1b-v2
samples/transformers-auto-model/fushh7_llmdet_swin_tiny_hf
samples/nemo/stt_en_squeezeformer_ctc_small_ls
samples/nemo/stt_en_conformer_ctc_large
samples/nemo/stt_en_squeezeformer_ctc_small_medium_ls
samples/nemo/parakeet-ctc-1.1b
samples/nemo/stt_en_squeezeformer_ctc_medium_large_ls
samples/nemo/parakeet-tdt-1.1b
samples/nemo/stt_en_fastconformer_hybrid_large_pc
samples/nemo/parakeet-ctc-0.6b
samples/nemo/stt_en_conformer_ctc_medium
samples/nemo/parakeet-tdt-0.6b-v3
samples/nemo/stt_en_squeezeformer_ctc_xsmall_ls
samples/nemo/stt_en_squeezeformer_ctc_medium_ls
samples/nemo/stt_en_squeezeformer_ctc_large_ls
samples/nemo/stt_en_conformer_ctc_small
```


**任务描述**
修复上述17个样本，使得`nope`和`inductor`评测结果均不再出现`nan`。具体步骤如下：

1. 执行`python -m graph_net.torch.test_compiler --model-path samples/xxx/xxx --compiler "inductor" --warmup 0 --trials 1`复现`nan`问题。
2. 通过修改样本、数据初始化或评测执行流程解决问题。
3. 提交PR。



**预期效果**
1. 给定17个样本`test_compiler`在使用`nope`和`inductor`后端时评测结果中均不出现`nan`。


### NO.114 torch._C._fft.fft_irfft API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`fft_irfft_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._fft.fft_irfft/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._fft.fft_irfft`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.115 torch._C._linalg.linalg_vector_norm API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`linalg_vector_norm_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._linalg.linalg_vector_norm/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._linalg.linalg_vector_norm`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.116 torch._C._nn.pad API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`pad_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.pad/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.pad`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.117 torch._C._log_api_usage_once API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`log_api_usage_once_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._log_api_usage_once/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._log_api_usage_once`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.118 torch._C._fft.fft_rfft API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`fft_rfft_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._fft.fft_rfft/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._fft.fft_rfft`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.119 torch._C._nn.linear API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`linear_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.linear/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.linear`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.120 torch._C._nn.scaled_dot_product_attention API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`scaled_dot_product_attention_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.scaled_dot_product_attention/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.scaled_dot_product_attention`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.121 torch._C._nn.avg_pool2d API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`avg_pool2d_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.avg_pool2d/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.avg_pool2d`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.122 torch._C._nn.gelu API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`gelu_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.gelu/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.gelu`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.123 torch._C._nn.softplus API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`softplus_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.softplus/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.softplus`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.124 torch._C._set_grad_enabled API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`set_grad_enabled_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._set_grad_enabled/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._set_grad_enabled`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.125 torch._C._fft.fft_fftn API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`fft_fftn_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._fft.fft_fftn/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._fft.fft_fftn`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.126 torch._C._linalg.linalg_norm API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`linalg_norm_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._linalg.linalg_norm/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._linalg.linalg_norm`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.127 torch._C._nn.one_hot API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`one_hot_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._nn.one_hot/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._nn.one_hot`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。

### NO.128 torch._C._special.special_logit API转换

**任务背景**

GraphNet中很多计算图由于用到了torch中的unstable\_api，因此无法直接通过PaConvertor转换为Paddle计算图。

本任务需要新增对应的api转换函数，将对应unstable api转换为pytorch stable api。

**任务描述**

为`graph_net/torch/backend/unstable_to_stable_backend.py`新增对应的`unstable_to_stable`函数。

并绘制出y=1的水平直线，作为可供验收的预期效果。具体细则如下：
1. 在pytorch stable api列表：[https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)中找到可供转换的 `<stable_api>`。
2. 在`unstable_to_stable_backend.py`中找到有"# TODO"标记的`unstable_to_stable`函数，将其函数名改为`special_logit_to_<stable_api>`，并同步修改`UnstableToStableBackend`的`__call__`函数。
3. 完善转换api的逻辑，返回一个修改好的gm对象。

**预期效果**

绘制出y=1的ES(t)曲线。具体细则如下：
1. 在GraphNet目录下，运行`todo_works/unstable_api_to_stable_api/torch._C._special.special_logit/test.sh`，进行检查。
2. 查看Es曲线结果，结果位于`todo_works/unstable_api_to_stable_api/torch._C._special.special_logit`。
3. 为了便于开发者debug，可以参考运行过程中的`log.log`，观察print等信息，以及代码运行报错信息，路径与Es曲线相同。
