此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛冲刺赛任务** 详细介绍

## 【开源贡献个人挑战赛冲刺赛-框架】任务详情

### **NO.1 - NO.6 API兼容性**

**开发流程规范**

**详细描述**

为了降低新模型（特别是新的大模型）使用飞桨开发或迁移到飞桨的成本，飞桨在3.2版本开展了 API兼容性适配 工作，提升了API针对不同框架写法的自适应能力，当前已完成了378个API与Pytorch API的无缝兼容工作（包括：API名称、参数名称、参数个数、参数语义）。针对Pytorch项目，仅需修改接口代码前缀torch为paddle，即可无缝迁移到Paddle。

但仍有很多Paddle API尚未完成兼容性适配 工作，本次活动旨在对Paddle API进行整体兼容性推全。

**示例修改**

* **要求**：尽可能采用C++下沉方式支持参数别名，只有无法C++下沉的情况下才考虑使用装饰器修改
* **C++下沉参考**：
  * [https://github.com/PaddlePaddle/Paddle/pull/75026](https://github.com/PaddlePaddle/Paddle/pull/75026)
  * [https://github.com/PaddlePaddle/Paddle/pull/76255](https://github.com/PaddlePaddle/Paddle/pull/76255)
* **装饰器参考**：
  * [https://github.com/PaddlePaddle/Paddle/pull/74420](https://github.com/PaddlePaddle/Paddle/pull/74420)
* **新增API参考**：官网相关开发文档[新增API流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)

**提交方式**

每一个任务包含以下5项修改内容，**请逐一核对**：

1. Paddle API自身修改（代码提交到 [Paddle Repo](https://github.com/PaddlePaddle/Paddle)）
按照『与Pytorch完全对齐一致』的标准来修改API，在任务表中已标注参考方案，但注意方案**仅供参考**，最终验收标准还是与『与Pytorch完全对齐一致』。

2. Paddle 单测测试（代码提交到 [Paddle Repo](https://github.com/PaddlePaddle/Paddle)）
根据修改点，增强对应的OP单测或API单测，注意自测充分。

3. Pytorch单测测试（代码提交到 [PaConvert Repo](https://github.com/PaddlePaddle/PaConvert)）
与Pytorch进行对比测试，运行Pytorch单测很容易发现两者对不齐地方，能确认是否实现了『完全对齐一致』的效果。

**测试方式**：`git clone https://github.com/PaddlePaddle/PaConvert`，如下步骤修改。

* **Step1：标记对齐的Pytorch API**，对齐完一个Paddle API后，需要将对应的Pytorch API加到`NO_NEED_CONVERT_LIST`里，此名单API会通过 _替换torch前缀为paddle_ 的形态来运行，并且比对两者运行结果，只有数值和属性都一致才可通过。

  <img width="336" height="150" alt="Image" src="https://github.com/user-attachments/assets/f5569119-9a22-4223-8dbe-9ee13f5d0a24" />

* **Step2：运行Pytorch单测**，首先检查你认领的Pytorch API是否有对应单测：
    * **已有单测**：[PaConvert Repo](https://github.com/PaddlePaddle/PaConvert) 的tests目录有绝大部分Pytorch API单测，`pytest tests/test_xxx.py` 运行即可，需要保证运行通过。
    * **没有单测**：对于 _新增参数、新增API_ 任务则可能没有单测，还需要补充torch单测，要求如下：
        * **精度与输入要求**：单测的输入Tensor 不能全为0值等无效输入，默认对比Pytorch、Paddle的输出精度以及shape等属性。
        * **单测覆盖范围要求**：Pytorch单测本质为模仿用户Pytorch代码写法，因此需要考虑该torch api的所有可能用法case。涉及到多个API形参的，应包含各种参数用法（全部指定关键字、全部不指定关键字、改变关键字顺序、默认参数均不指定、参数取值以变量形式传入），不能只考虑最简单常见的用法。例如新增了2个参数，则需至少补充4种以上排列组合用法（越多越好）。
        * **详细规范可参考**：[Pytorch单元测试规范](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md#%E6%AD%A5%E9%AA%A45%E7%BC%96%E5%86%99%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95) 。
        * **参考PR**：[https://github.com/PaddlePaddle/PaConvert/pull/613](https://github.com/PaddlePaddle/PaConvert/pull/613) 。

  例如以下为torch.tensor的测试，需保证这些测试都能通过（_注意PaConvert的CI会抓取nightly最新编译的paddle whl包 + torch最新release包来测试，因此**需要先合入Paddle，隔天才可通过PaConvert的CI**_）：

  <img width="559" height="651" alt="Image" src="https://github.com/user-attachments/assets/0b944ff8-1523-486f-bbdd-cf87792df500" />

4. Paddle API 英文文档（代码提交到 [https://github.com/PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 目录）
在英文文档中描述对应的修改点。

5. Paddle API 中文文档（代码提交到 [https://github.com/PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 目录）
在中文文档中描述对应的修改点。

**题目内容：**
### NO.1 paddle.unique兼容性增强

**详细描述：**

新增参数sorted；c++下沉或装饰器 {'input': 'x', 'sorted': '', 'dim': 'axis'}； 

### NO.2 paddle.autograd.function.*一系列API兼容性增强

**详细描述：**

torch.autograd.function.Function/FunctionCtx整体兼容，paddle目前对应API为paddle.autograd.PyLayer/PyLayerContext，需要新增paddle.autograd.function.*兼容这一系列API。涉及到自定义Function的API都看下，当前paddle若支持的API都需要兼容。

可参考 [官网映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) 来找到对应映射。

### NO.3 paddle.utils.data.*一系列API兼容性增

**详细描述：**

torch.utils.data.*整体兼容，paddle目前对应路径为paddle.io.*，需要新增paddle.utils.data.*兼容这一系列API。涉及到torch.utils.data.*路径下都看下，当前paddle若支持的API都需要兼容。

可参考 [官网映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) 来找到对应映射，例如 paddle.io.ConcatDataset应对应到paddle.utils.data.ConcatDataset。

### NO.4 paddle.nn.*一系列API兼容性增强

**详细描述：**

详细描述：torch.nn.*下面有很多网络层、激活层都含有inplace参数，需要给paddle.nn.*对应API都支持inplace功能。

参考PR：[#74788](https://github.com/PaddlePaddle/Paddle/pull/74788)

### NO.5 paddle.nn.functional.interpolate兼容性增强

**详细描述：**

新增antialias抗锯齿功能，需要新增Kernel，不同插值方式下2+套kernel。（包括CPU/GPU/XPU）

### NO.6 paddle.nn.MaxPool2D兼容性增强

**详细描述：**

新增dilation参数，支持空洞池化，新增新增Kernel。（包括CPU/GPU/XPU）


## 【开源贡献个人挑战赛-编译机床】任务详情

### **NO.7 - NO.10 编译机床**

### NO.7 PyTorch to Paddle 计算图转换

**详细描述：**

1. 计算图转换：
    1. 对于 GraphNet/samples 中所有PyTorch样本，应用 PaConvert 代码转换工具（[https://github.com/PaddlePaddle/PaConvert](https://github.com/PaddlePaddle/PaConvert)），实现 GraphNet 中 torch 样本到 paddle 的迁移；
    2. 转换后样本位置GraphNet/torch_to_paddle_samples；
    3. 记录log和转换失败案例，简单分析错误原因。

2. 计算图测试：
    1. 将 GraphNet/torch_to_paddle_samples 目录下转换后样本进行批量测试；
    2. 记录log和测试失败案例。

3. 在 graph_net/config 中分别新增两组模型列表，格式参照 [torch_samples_list.txt](https://github.com/JewelRoam/GraphNet/blob/dcu/graph_net/config/torch_samples_list.txt)：
    1. torch_to_paddle_samples_list_full.txt: 在全量样本中，仅剔除转换过程中失败样本；
    2. torch_to_paddle_samples_list.txt: 在全量样本中，同时剔除转换过程中和测试过程中的失败样本。


**提交内容**：

1. 撰写设计文档，提交 PR 添加至 GraphNet/docs。
2. 在新增样本的 PR 描述中记录模型样本转换、运行测试的结果，及必要的log片段。

### NO.8 ai4c计算图粗分解器设计与实现

**任务背景**

 AI4C 子图分解功能包含以下模块：

1. 计算图区间分解器，负责分解操作执行，需要包含分解区间配置
2. 计算图分解方案验证器，以RangeDecomposerValidatorBackend为核心，对拆分后的子图做有效性验证

当前任务聚焦【计算图区间分解器】，采用一种可能的粗分解方案实现，与验证器交叉验证。

**任务描述**

该任务的目标是实现一个 range_decomposer 基类，和粗分解方案的一种实现。拥有如下特性：

1. 作为 backend 导入 graph_net.torch.test_compiler，相应的配置已写入 test_compiler 代码；
2. 接收一个【原模型】的 torch.nn.Module，输出【分解后模型】的多个subgraph；
3. 在分解过程中，默认【分解后模型】路径为【原模型】路径加上_decomposed，下有多个subgraph单独目录，例如 /test/simple_CNN/ 的分解后模型包括 /test/simple_CNN_decomposed/subgraph_0/.../test/simple_CNN_decomposed/subgraph_n/，每个subgraph的文件组成等同一份标准的GraphNet样本；
4. 在组合过程中，组合模型的forward是每个分解模型依次连接、嵌套而成，前一个模型的输出作为下一个模型的输入；
5. 粗分解方案可参照 [https://github.com/PaddlePaddle/GraphNet/blob/develop/graph_net/test/rp_expr_parser_test.py](https://github.com/PaddlePaddle/GraphNet/blob/develop/graph_net/test/rp_expr_parser_test.py)，实现经典子图（即高频子模式）提取；
6. 参照 [朴素子图切分脚本](https://github.com/PaddlePaddle/GraphNet/blob/develop/graph_net/test/naive_graph_decomposer_test.sh) 和 [朴素子图链式切分脚本](https://github.com/PaddlePaddle/GraphNet/blob/develop/graph_net/test/chain_naive_graph_decomposer_test.sh)，在此基础上开发图分解功能。

**预期效果**

分解正确性验证：以通过 range_decomposer_validator的compose 操作后ESt图象的表现为标准：

1. t=1 的抬升代表输出精度错误，t=3的抬升代表编译运行等其它类别错误。
2. 由于是单个样本测试，无需考虑性能提升，故预期对于正确拆分样本，ES图象应当是y=1的【一条直线】；
3. 对于错误或不完整的拆分样本，应当打印【错误报告】，或ES图象在 t>0 区域存在【阶梯状抬升】。

### NO.9 GraphNet Analysis功能及ESt绘图优化

**Analysis 读取 log 功能优化**

原 GraphNet 的 benchmark 功能有三个步骤：

1. 使用 test compiler（以及刚做好 test device 的最终步骤）批量测试并记录下合并记录的一份log
2. 使用 graph_net.log2json 读取这份 log，在另一个目录下生成每个模型

之前这么做的原因是 json 方便 graph_net.analysis_uti l读取，可读性高；而 test_compiler 中如果遇到底层的 C++ runtime 报错等无法被 catch 住，可能无法直接记录下 json。但实际操作过程中debug 看 log 已经足够，log2json 的中间过程显得粗糙，同时增加了使用者的学习成本。

于是，本任务需求为去除 log2json 中间步骤，修改 graph_net.analysis_util（在 plot_ESt 和 plot_St 过程中调用），使其直接读取 log 来解析。

解析过程仍可以参考 log2json 的方式，需要注意的是 paddle 样本带有 subgraph 序号而 torch 样本没有，这个特性 log2json 的处理在 [https://github.com/PaddlePaddle/GraphNet/blob/e7c6e0383aec1c9f6fef775463e8fe68db050389/graph_net/log2json.py#L138](https://github.com/PaddlePaddle/GraphNet/blob/e7c6e0383aec1c9f6fef775463e8fe68db050389/graph_net/log2json.py#L138)，比较粗糙，可以优化兼容解析方式。

**ESt 绘图中参数计算优化**

原 graph_net.plot_St 和 graph_net.plot_ESt 脚本调用 graph_net.analysis_util，实现对技术报告 [https://arxiv.org/abs/2510.24035](https://arxiv.org/abs/2510.24035) 中 3.2 Evaluation Metrics 的图象绘制，公式推导、tolerance 配置、各项参数参见附录。

graph_net.analysis_util 以技术报告中 ESt 公式为基础，通过两种计算方式交叉验证：

* 微观计算 rectified_speedup 之后做几何平均
* 通过宏观统计参数计算

由于计算过程比较复杂，需要验证计算的有效性。本任务拆分出独立脚本计算每个宏观参数，打印出结果，从而验证 graph_net.plot_ESt 得出的结果。在graph_net.plot_ESt中，改为必须宏观/微观计算结果相匹配情况下才能采用。

**提交内容**：

1. 对于上面两个功能，可以遵循软件工程的更好设计，重构 graph_net.analysis_util 的处理逻辑，例如把宏观统计量的计算单独拆开作为一个脚本、每个参数独立一个函数，提高可维护度。
2. 提交PR，在 graph_net/相应位置修改代码，修改 [readme中的相关描述](https://github.com/PaddlePaddle/GraphNet?tab=readme-ov-file#%EF%B8%8F-compiler-evaluation)。

### NO.10 GraphNet自动样本抽取Agent（Huggingface）

**详细描述：**

实现一个自动从hf上下载模型，使用GraphNet组件端到端抽取样本的Agent，自动完成运行拉取、撰写代码、抽图、验证、提交的流程；

操作过程中应充分使用GraphNet的开放接口，并在架构设计上保留可拓展性（例如方便后续增加面向其它源抽取的Agent组件）

要求结构尽可能稳定、易于理解，功能稳定、方便部署，但Agent的技术选型没有限制。

**提交内容**：

1. 提交代码到graph_net/agent
2. 撰写设计文档，提交 PR 添加至 GraphNet/docs。

## 【开源贡献个人挑战赛冲刺赛-套件开发】任务详情

### FastDeploy套件开发

### **NO.11 - NO.55 单测补充**

**详细描述：**

当前FastDeploy下一些文件缺少单测监控，需要添加单测代码，来提高文件中代码的单测覆盖率。
本任务中，通过添加单测后提高的代码覆盖行数来确定PR的贡献度，每提高100行（四舍五入，比如150等同200行，140行等同100行）代码覆盖，贡献度累计0.1⭐️。

开发者可通过链接来查看最新的代码覆盖情况：https://paddle-github-action.bj.bcebos.com/BRANCH/FastDeploy/develop/{完整的commit-id}/SM/CoverageData/full_coverage_report.csv，
在这个链接里，通过指定commit-id来查看对应commit-id下代码的覆盖情况（当前仅支持查看某一天最后一个commit的覆盖率）：

<img width="984" height="40" alt="Image" src="https://github.com/user-attachments/assets/5d6d1dd5-a455-40d7-a430-024cbf29eca3" />

比如打开覆盖率表格可以看到如上内容，通过Miss列可以看到总的未覆盖代码行号，比如上边的audio.py里有25行有效代码没有单测覆盖；通过Missing列可看到具体未覆盖代码的行号，比如这里表示行号17-127行未被覆盖（这里Missing列会把注释等无效代码算进去，所以数字会比Miss列要大）。

PR验收的标准是看文件代码的覆盖率(Cover)是否达到了80%，这个覆盖率在Coverage CI的日志里有显示，在达到80%的基础上，贡献单测越多，获得的⭐️越高

提交内容：
* Python 单测代码
* PR中评论：当前develop分支的单测覆盖率情况，增加该PR后的单测覆盖率情况，本PR代码覆盖行数

技术要求：
* 熟悉python及unittest、pytest单测工具，会基于ai工具的单测开发

**题目内容**：

### NO.11  功能模块 fastdeploy/worker/gpu_model_runner.py 单测补充
### NO.12  功能模块 fastdeploy/spec_decode/mtp.py 单测补充
### NO.13  功能模块 fastdeploy/model_executor/ops/triton_ops/triton_utils.py 单测补充
### NO.14 功能模块 fastdeploy/rl/rollout_model.py 单测补充
### NO.15 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py 单测补充
### NO.16 功能模块 fastdeploy/input/ernie4_5_vl_processor/process.py 单测补充
### NO.17 功能模块 fastdeploy/input/text_processor.py 单测补充
### NO.18 功能模块 fastdeploy/model_executor/layers/moe/ep.py 单测补充
### NO.19 功能模块 fastdeploy/model_executor/layers/pooler.py 单测补充
### NO.20 功能模块 fastdeploy/model_executor/layers/sample/sampler.py 单测补充
### NO.21 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_triton_backend.py 单测补充
### NO.22 功能模块 fastdeploy/input/ernie4_5_vl_processor/ernie4_5_vl_processor.py 单测补充
### NO.23 功能模块 fastdeploy/config.py 单测补充
### NO.24 功能模块 fastdeploy/model_executor/models/tp_utils.py 单测补充
### NO.25 功能模块 fastdeploy/input/ernie4_5_vl_processor/image_preprocessor/image_preprocessor_adaptive.py 单测补充
### NO.26 功能模块 fastdeploy/model_executor/load_weight_utils.py 单测补充
### NO.27 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_wint2_backend.py 单测补充
### NO.28 功能模块 fastdeploy/model_executor/ops/triton_ops/triton_utils_v2.py 单测补充
### NO.29 功能模块 fastdeploy/model_executor/models/ernie4_5_mtp.py 单测补充
### NO.30 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py 单测补充
### NO.31 功能模块 fastdeploy/input/ernie4_5_processor.py 单测补充
### NO.32 功能模块 fastdeploy/input/ernie4_5_vl_processor/process_video.py 单测补充
### NO.33 功能模块 fastdeploy/cache_manager/cache_messager.py 单测补充
### NO.34 功能模块 fastdeploy/scheduler/splitwise_scheduler.py 单测补充
### NO.35 功能模块 fastdeploy/engine/common_engine.py 单测补充
### NO.36 功能模块 fastdeploy/cache_manager/prefix_cache_manager.py 单测补充
### NO.37 功能模块 fastdeploy/output/token_processor.py 单测补充
### NO.38 功能模块 fastdeploy/scheduler/global_scheduler.py 单测补充
### NO.39 功能模块 fastdeploy/engine/sched/resource_manager_v1.py 单测补充
### NO.40 功能模块 fastdeploy/entrypoints/openai/api_server.py 单测补充
### NO.41 功能模块 fastdeploy/splitwise/splitwise_connector.py 单测补充
### NO.42 功能模块 fastdeploy/entrypoints/openai/serving_completion.py 单测补充
### NO.43 功能模块 fastdeploy/utils.py 单测补充
### NO.44 功能模块 fastdeploy/engine/engine.py 单测补充
### NO.45 功能模块 fastdeploy/cache_manager/cache_transfer_manager.py 单测补充
### NO.46 功能模块 fastdeploy/model_executor/guided_decoding/xgrammar_backend.py 单测补充
### NO.47 功能模块 fastdeploy/inter_communicator/zmq_server.py 单测补充
### NO.48 功能模块 fastdeploy/engine/resource_manager.py 单测补充
### NO.49 功能模块 fastdeploy/entrypoints/openai/serving_chat.py 单测补充
### NO.50 功能模块 fastdeploy/entrypoints/engine_client.py 单测补充
### NO.51 功能模块 fastdeploy/scheduler/dp_scheduler.py 单测补充
### NO.52 功能模块 fastdeploy/model_executor/guided_decoding/ernie_tokenizer.py 单测补充
### NO.53 功能模块 fastdeploy/scheduler/workers.py 单测补充
### NO.54 功能模块 fastdeploy/inter_communicator/engine_worker_queue.py 单测补充
### NO.55 功能模块 fastdeploy/scheduler/local_scheduler.py 单测补充
### NO.56 功能模块 fastdeploy/multimodal/utils.py 单测补充

### **NO.57 - NO.58 编译支持**

### NO.57 FastDeploy 支持在 T4/V100 硬件的编译

**详细描述：**

FastDeploy支持在T4、V100硬件编译

**提交内容：**

编代码提交到FastDeploy仓库

**技术要求：**

- 熟悉C++/CUDA开发编译，有多硬件开发经验更佳
- 熟悉 shell 以及setuptools 等编译工具

### NO.58 FastDeploy 支持在 windows 平台的编译

**详细描述：**

FastDeploy支持在Windows平台编译

**提交内容：**

编代码提交到FastDeploy仓库

**技术要求：**

- 熟悉C++/CUDA开发编译，有多硬件开发经验更佳
- 熟悉 shell 以及setuptools 等编译工具

### **NO.59 - NO.70 功能开发** 

### NO.59 FastDeploy Deterministic Inference 模式开发

**任务目标：**

参与 FastDeploy Deterministic Inference 项目

**详细描述：**

Project 地址： https://github.com/orgs/PaddlePaddle/projects/18

总Issue 地址： https://github.com/PaddlePaddle/FastDeploy/issues/4651

背景：大模型推理的不确定是由于算子不具备批处理不变性引起的，需要为算子实现批处理不变性来保证大模型的确定性推理。https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

MileStone 1（🌟）: 支持 Dense 模型 (Qwen3-8B & ERNIE-4.5-0.3B-Paddle) 的确定性推理 
1. [Doing]参考 https://github.com/thinking-machines-lab/batch_invariant_ops ，实现 Paddle 版本 batch_invariant_ops，推荐使用原版Triton算子
2. [Doing]构建请求级别与算子级别的批处理不变性测试脚本
3. 为 Append Attention、FA3 支持批处理不变性 （0.5🌟）
4. 整体端到端跑通 Dense 模型，可以通过请求级别测试  （0.5🌟）

MileStone 2（🌟🌟）: 支持 Moe、CUDAGraph、Chunk Prefill、PrefixCache（三个默认开启的加速方法）
1. 支持 Moe 模型，（Qwen3-30B-A3B & ERNIE-4.5-21B-A3B-Paddle）
2. 适配 CUDAGraph 
3. 适配 Chunked Prefill 
4. 适配 Prefix Cache

MileStone 3（🌟🌟）: 支持混合并行（TP、DP、EP），MTP，量化 
1. 适配 MTP
2. 适配混合并行（TP=4，TP=8）
3. 适配量化（blockwise fp8）

**路线参考：**

sglang：https://github.com/sgl-project/sglang/issues/10278
vllm：https://github.com/orgs/vllm-project/projects/29

**技术要求：**

熟悉大模型推理框架、了解大模型常见算子的实现，可一对一指导

**机器要求：**

A800 GPU（星河社区可提供） 或 Hopper GPU

### NO.60 为 FastDeploy 新增支持 DeepSeek 模型的 Reasoning Parser & Tool Parser

**任务目标：**

针对支持思考或工具调用的模型，根据模型输出协议，流式、非流式地解析其中reasoning_content（思考内容）、content（回复内容）、tool_calls（工具调用） 内容。

**详细描述：**

* reasoning_parser🌟： 
    * 明确模型的输入拼接方案：根据 prompt， 判断模型生成对应的阶段（思考开始、回复开始等），思考长度截断、流式后解析控制均有所依赖。
    * 明确模型输出协议， 通过 special_token或其他规则，解析出reasoning_content及content 内容。对于属于tool_calls的部分，在 tool_parser 中解析，与content内容互斥。
    * 流式与非流式对结果解析保持结果一致性，非流式场景重点关注是否存在关思考情况。
    * 根据解析逻辑， 说明在模型生成格式不合法情况下，解析后结果呈现。

* tool_parser🌟：
    * 支持流式、非流式工具解析； 流式场景，返回结果需区分攒包、非工具内容。
    * 根据模型属性，确认是否有parallel_tool_calls、tool_choice等后处理解析逻辑。

* 加分项🌟：
    *  对于多数模型统一使用的模型协议（较通用的）， 归一成一套解析方案


**提交内容：**

* reasoning_parser代码实现，提交到FastDeploy/fastdeploy/reasoning目录下
* tool_parser代码实现，提交到FastDeploy/fastdeploy/entrypoints/openai/tool_parsers目录下
* 单测提交到  tests/reasoning/ 及 tests/entrypoints/openai/tool_parsers/ 目录下
* 补充Docs文档说明：
    * reasoning_parser：
        * docs/features/reasoning_output.md
        * docs/zh/features/reasoning_output.md

    * tool_call_parser：
        * docs/features/tool_calling.md
        * docs/zh/features/tool_calling.md

**模型范围：**

已支持的 DeepSeek 系列模型
[https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/supported_models.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/supported_models.md)

**技术要求：**

熟悉 python， 了解FastDeploy 数据处理流程

### NO.61 为 FastDeploy 新增支持 Qwen 模型的 Reasoning Parser & Tool Parser

**任务目标：**

针对支持思考或工具调用的模型，根据模型输出协议，流式、非流式地解析其中reasoning_content（思考内容）、content（回复内容）、tool_calls（工具调用） 内容。

**详细描述：**

* reasoning_parser🌟： 
    * 明确模型的输入拼接方案：根据 prompt， 判断模型生成对应的阶段（思考开始、回复开始等），思考长度截断、流式后解析控制均有所依赖。
    * 明确模型输出协议， 通过 special_token或其他规则，解析出reasoning_content及content 内容。对于属于tool_calls的部分，在 tool_parser 中解析，与content内容互斥。
    * 流式与非流式对结果解析保持结果一致性，非流式场景重点关注是否存在关思考情况。
    * 根据解析逻辑， 说明在模型生成格式不合法情况下，解析后结果呈现。

* tool_parser🌟：
    * 支持流式、非流式工具解析； 流式场景，返回结果需区分攒包、非工具内容。
    * 根据模型属性，确认是否有parallel_tool_calls、tool_choice等后处理解析逻辑。

**提交内容：**

* reasoning_parser代码实现，提交到FastDeploy/fastdeploy/reasoning目录下
* tool_parser代码实现，提交到FastDeploy/fastdeploy/entrypoints/openai/tool_parsers目录下
* 单测提交到  tests/reasoning/ 及 tests/entrypoints/openai/tool_parsers/ 目录下
* 补充Docs文档说明：
    * reasoning_parser：
        * docs/features/reasoning_output.md
        * docs/zh/features/reasoning_output.md

    * tool_call_parser：
        * docs/features/tool_calling.md
        * docs/zh/features/tool_calling.md

**模型范围：**

已支持的 Qwen 系列模型
[https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/supported_models.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/supported_models.md)

**技术要求：**

熟悉 python， 了解FastDeploy 数据处理流程

### NO.62 DeepseekV3 编译优化+CudaGraph 子图机制支持

**详细描述：**

这个问题分为三步，消除打断、跑通CUDAGraph、模型性能分析与优化
* 消除打断，在DeepseekV3的DeepseekV3MLAAttention中，参考这个PR的方法

https://github.com/PaddlePaddle/Paddle/pull/75548，将控制流移出形成新函数，在外部新建一个empty Tensor传入固定输出地址，可参考：https://github.com/PaddlePaddle/FastDeploy/pull/3302
* 跑通CUDAGraph，可先阅读这两个PR——SOT支持CUDAGraph：

https://github.com/PaddlePaddle/Paddle/pull/73393
https://github.com/PaddlePaddle/FastDeploy/pull/3478，开启CUDAGraph，跑通整个推理流程
* 模型性能分析与优化，消除子图打断后，分析性能瓶颈，性能不低于动态图+CUDAGraph

**提交内容：**

* 代码实现，提交至FastDeploy形成PR；
* 单测；
* 相关设计文档，流水线示意图等；
* kernel 性能分析文件，性能不低于动态图+CUDAGraph；

**技术要求：**

了解Paddle动转静机制、熟悉 FastDeploy、会修改常见模型、性能分析能力

**机器要求：**

需要 Hopper GPU

### NO.63 CINN 编译 Kernel 缓存机制

**详细描述：**

**任务背景**

CINN神经网络编译器背景：[https://www.paddlepaddle.org.cn/documentation/guides/paddle_v3_features/cinn_cn.html](https://www.paddlepaddle.org.cn/documentation/guides/paddle_v3_features/cinn_cn.html)

CINN在编译生成Kernel的过程中，存在编译耗时的开销。特别是在大模型编译优化过程中，这种时间开销会直接影响到模型的服务部署，大大影响开发人员的工作效率。因此我们需要优化编译流程，降低线上业务的部署成本。
目前我们打算通过编译Kernel缓存机制，来避免重复编译，并在多卡上实现一处编译多处复用。现计划分两阶段实现：
* STEP1：单卡动态链接库缓存（已完成技术验证，预计于11月中旬发布版本，联系 [zyfncg](https://github.com/zyfncg)，[YuhanXu](https://github.com/YuhanXu)）
* STEP2：多卡拓展（当前任务）

**任务目标**

将单卡 Kernel 缓存机制扩展至多卡环境，支持节点内的多卡编译 Kernel 复用，节省编译耗时。

**提交内容**

* 代码实现，提交至Paddle develop 分支形成PR；
* 单测；
* 相关设计文档，流水线示意图等；
* 提供编译耗时对比报告（vs 无缓存方案）；

**技术要求：**

熟悉C++, 了解相关原理

**机器要求：**

A100

### NO.64 新版模型加载Loader 适配 Marlin MoE Backend

**详细描述：**

为 FastDeploy的 marlin backend 适配 v1loader 加载，并验证单卡/多卡精度

**提交内容**

* 1.FD 组网代码提交到 fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py
需保证marlin backend 可以正常跑通 v1loader 下的FD加载并验证精度
* 2.baseline为v0 laoder开启方法为  --load-choices 'default'/load_choices='default' 

开启v1 loader方法

开启 --load-choices 'default_v1'/load_choices='default_v1' 

baseline:

 --load-choices 'default'/load_choices='default'

**验证模型：**

[https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Paddle](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Paddle)

**开启marlin方法:**

export FD_MOE_BACKEND=marlin

**技术要求：**

* 熟悉python
* 熟悉VLLM/SGLANG加载流程。

### NO.65 新版模型加载Loader适配 Wint2 MoE Backend

**详细描述：**

为FastDeploy的 wint2 backend 适配 v1loader 加载，并验证单卡/多卡精度

**提交内容**

* 1.FD 组网代码提交到 fastdeploy/model_executor/layers/quantization/wint2.py
需保证wint2 backend 可以正常跑通 v1loader 下的FD加载并验证精度
* 2.需验证单卡/多卡精度，baseline为v0loader 开启方法为  --load-choices 'default'/load_choices='default' 
* 3.给v1loader支持预切模型加载能力，提交至fastdeploy，代码较为分散，需要修改所有def weight_loader函数才能适配预切加载 
开启v1 loader方法

开启 --load-choices 'default_v1'/load_choices='default_v1' 

baseline:

 --load-choices 'default'/load_choices='default'

**验证模型：**

[https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-2Bits-TP4-Paddle](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-2Bits-TP4-Paddle)

**开启marlin方法:**

export FD_MOE_BACKEND=marlin

**技术要求：**

* 熟悉python
* 熟悉VLLM/SGLANG加载流程。

### NO.66 为 FastDeploy 推全 Pooling 的 classify 任务

**详细描述：**

为 FastDeploy 支持 runner 为 pooling、convert 为 classify，并支持相应 openai 的请求接口，和 vlm 验证精度。

**提交内容**

1. 支持服务传递 convert 为 classify，并认证 convert 为 classify 为 pooling 模型
2. 支持 classify 的 pooling 任务
3. 支持请求接口，命名 serving_classification.py
4. 添加 create_classify 接口，支持 classificationRequest、ClassificationData、classificationResponse 接口

可参考 runner 为 pooling、convert 为 classify 的相关pr：[pr#3827](https://github.com/PaddlePaddle/FastDeploy/pull/3827)，[pr#4344](https://github.com/PaddlePaddle/FastDeploy/pull/4344)，[pr#4345](https://github.com/PaddlePaddle/FastDeploy/pull/4345),[pr#4590](https://github.com/PaddlePaddle/FastDeploy/pull/4590)

### NO.67 为 FastDeploy 推全 Pooling 的 score 任务

**详细描述：**

为 FastDeploy 支持 runner 为 pooling、convert 为 score，并支持相应 openai 的请求接口，和 vlm 验证精度。

**提交内容**

1. 支持服务传递 convert 为 score，并认证 convert 为 score 为 pooling 模型
2. 支持 score 的 pooling 任务
3. 支持请求接口，命名 serving_score.py
4. 添加 create_score 接口，支持 ScoreRequest、ScoreResponseData、ScoreResponse接口

可参考 runner 为 pooling、convert 为 classify 的相关pr：[pr#3827](https://github.com/PaddlePaddle/FastDeploy/pull/3827)，[pr#4344](https://github.com/PaddlePaddle/FastDeploy/pull/4344)，[pr#4345](https://github.com/PaddlePaddle/FastDeploy/pull/4345),[pr#4590](https://github.com/PaddlePaddle/FastDeploy/pull/4590)

### NO.68 为 FastDeploy 支持 Pooling 离线推理

**详细描述：**

为 FastDeploy 支持 runner 为 pooling、convert 为 embed 的离线方式
离线方式支持pooling推理，路线参考: [https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py)

**提交内容**

添加embed接口

### NO.69 为 FastDeploy 支持投机解码功能

**详细描述：**

* 背景：

    1. 投机解码有多种方法，目前 FastDeploy 中 ngram_match / hybrid_mtp_ngram 两种方法都用到了字符串匹配方法。
    2. 但目前两个方法的核心匹配算子实现是 CPU 版本，需要做同步的 Device->CPU 的拷贝操作，对性能影响较大

* 验证模型：[https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Paddle](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Paddle)
* 其他：该任务涉及到诸多细节，可1对1交流、沟通

**提交内容**

* 将两个 Kernel 优化为 GPU 版本，且性能不低于CPU版本，Kernel 分别是

    1. FastDeploy/custom_ops/gpu_ops/speculate_decoding/ngram_match.cc
    2. FastDeploy/custom_ops/gpu_ops/speculate_decoding/draft_model/ngram_match_mixed.cu

* 两个Kernel逻辑有较为相似部分，Kernel 形式为提取共用的匹配逻辑，外加业务逻辑

**验收要求：**

* 在较长的匹配下，Kernel 性能优于或基本不劣于目前的 CPU kernel

### NO.70 DeepSeek-v3.1-Terminus 模型支持 MTP

**详细描述：**

基于FastDeploy现有的 MTP 相关功能模块，进一步开发 DeepSeek 系列模型 MTP功能模块；支持DeepSeek系列模型+MTP推理部署。

**提交内容**

* DeepSeek系列模型 MTP 模块设计文档；
* DeepSeek系列模型 MTP 所需模块代码需提交至 FastDeploy；

**技术要求：**

* 要求支持 Deepseek-v3、R1、V3.1系列模型 MTP；
* DeepSeek-v3.1-Terminus MTP 性能不差于sglang/vllm；

**参考：**

* 可以参考文心系列模型 MTP 进行适配和支持。
* FastDeploy 投机解码相关文档：[https://paddlepaddle.github.io/FastDeploy/zh/features/speculative_decoding/](https://paddlepaddle.github.io/FastDeploy/zh/features/speculative_decoding/)
* FastDeploy现有模块：
    * [https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/spec_decode](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/spec_decode)
    * [https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/spec_decode/mtp.py](https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/spec_decode/mtp.py)
    * [https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/model_executor/layers/sample/sampler.py](https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/model_executor/layers/sample/sampler.py)


### **NO.71 - NO.80 模型新增** 
   
### NO.71 为 FastDeploy 新增 Qwen3-Next-80B-A3B-Thinking 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 [Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) 模型的能力. 

**提交内容**

* Qwen3-Next-80B-A3B-Thinking 模型组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 Qwen3-Next-80B-A3B-Thinking 适配 FastDeploy 现有的各种低 bit 量化推理的能力.

**技术要求：**

* 熟悉常见的LLM模型结构和计算流程. 了解 Qwen3-Next-80B-A3B-Thinking模型结构.
* 熟悉python, 熟悉cuda

### NO.72 为 FastDeploy 新增 Qwen3-Omni-30B-A3B-Thinking 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) （🌟🌟）以及 [Qwen3-Omni-30B-A3B-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) 模型（🌟）的能力. 

**提交内容**

* Qwen3-Omni-30B-A3B-Thinking 模型以及 Qwen3-Omni-30B-A3B-Captioner 模型组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 Qwen3-Omni-30B-A3B-Thinking 模型以及 Qwen3-Omni-30B-A3B-Captioner 模型适配 FastDeploy 现有的各种低 bit 量化推理的能力.

**技术要求：**

* 熟悉常见的LLM模型结构和计算流程. 了解 Qwen3-Next-80B-A3B-Thinking 模型以及Qwen3-Omni-30B-A3B-Captioner 模型结构.
* 熟悉 python, 熟悉 cuda

### NO.73 为 FastDeploy 新增 Qwen3-VL-30B-A3B-Thinking 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 Qwen3-VL 系列模型的能力。包括：[Qwen/Qwen3-VL-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking)（🌟🌟）、[Qwen/Qwen3-VL-30B-A3B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8)（0.3🌟）、[Qwen/Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)（0.3🌟）、[Qwen/Qwen3-VL-4B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8)（0.4🌟）

**提交内容**

* Qwen3-VL 相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 Qwen3-VL 系列模型适配FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

* 熟悉常见的 LLM 模型结构和计算流程. 了解 Qwen3-VL 类模型结构.
* 熟悉 python, 熟悉 cuda

### NO.74 为 FastDeploy 新增 MiniCPM4.1-8B 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 [openbmb/MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B) 系列模型的能力. 

**提交内容**

* MiniCPM4.1-8B相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 MiniCPM4.1-8B系列模型适配FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

* 熟悉常见的 LLM 模型结构和计算流程. 了解 MiniCPM4.1-8B 类模型结构.
* 熟悉 python, 熟悉 cuda

### NO.75 为 FastDeploy 新增 Llama-4-Scout-17B-16E-Instruct 模型

**详细描述：**

为FastDeploy 提供部署高性能的 [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) 系列模型的能力. 

**提交内容**

* Llama-4 相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 Llama-4 系列模型适配 FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

* 熟悉常见的 LLM 模型结构和计算流程. 了解 Llama-4类模型结构.
* 熟悉 python, 熟悉 cuda

### NO.76 为 FastDeploy 新增 LongCat-Flash-Chat 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 [meituan-longcat/LongCat-Flash-Chat](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat) 系列模型的能力. 

**提交内容**

* LongCat 相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 LongCat 系列模型适配FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

* 熟悉常见的LLM模型结构和计算流程. 了解 LongCat 类模型结构.
* 熟悉 python, 熟悉 cuda

### NO.77 为 FastDeploy 新增 Kimi-VL-A3B-Thinking-2506 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 [moonshotai/Kimi-VL-A3B-Thinking-2506](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506) 系列模型的能力. 

**提交内容**

* Kimi-VL相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 Kimi-VL 系列模型适配 FastDeploy 现有的各种低bit量化推理的能力.

**技术要求：**

* 熟悉常见的 LLM 模型结构和计算流程. 了解Kimi-VL类模型结构.
* 熟悉 python, 熟悉 cuda

### NO.78 为 FastDeploy 新增 DeepSeek-OCR 模型

**详细描述：**

为 FastDeploy 提供部署高性能的 [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) 系列模型的能力. 

**提交内容**

* DeepSeek-OCR相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档. 
* 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
* 为 DeepSeek-OCR系列模型适配 FastDeploy 现有的各种低bit量化推理的能力.

**技术要求：**

* 熟悉常见的 LLM 模型结构和计算流程. 了解 DeepSeek-OCR 类模型结构.
* 熟悉 python, 熟悉 cuda

### NO.79 适配 HF Safetenosrs  Qwen3-4B-AWQ 量化模型并支持 AWQ量化kernel

**详细描述：**

当前 Paddle 的 weight_quant kernel 对 AWQ 量化未支持/未验证，需要开发者实现一版可用的 AWQ 量化方案，加载 Qwen/Qwen3-4B-AWQ 模型，跑通并验证精度

**提交内容**

1. AWQ kernel 提交到 Paddle 仓库中 [https://github.com/PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
    1. 可参考 gpu实现 paddle/phi/kernels/gpu/weight_quantize_kernel.cu
    2. 可参考 cpu实现 paddle/phi/kernels/cpu/weight_quantize_kernel.cc
    3. 可参考前项实现 paddle/phi/kernels/gpu/weight_only_linear_kernel.cu
    4. gpu实现提交到：paddle/phi/kernels/gpu
    5.  cpu实现提交到：paddle/phi/kernels/cpu
    6. 前项实现提交到 paddle/phi/kernels/gpu

2. FD 代码提交到 
    1. dense部分组网实现提交到:fastdeploy/model_executor/layers/quantization/weight_only.py 中，新增class AWQMethod(QuantMethodBase)
    2. moe部分awq kernel实现提交到 custom_ops/gpu_ops/moe/moe_ffn.cu
    3. moe部分组网实现提交到 fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py 新增 class AWQMoEMethod(CutlassMoEMethod):

3. 确保精度正常，以 vllm/sgalng/transform 作为 baseline

**模型链接：**

[https://huggingface.co/Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)

**技术要求：**

* 熟悉python, 熟悉cuda
* 熟悉VLLM/SGLANG加载量化流程。

### NO.80 适配 HF Safetenosrs  Qwen3-1.7B-GPTQ-Int8/Qwen3-30B-A3B-GPTQ-Int4 量化模型 并支持 GPTQ量化kernel

**详细描述：**

* 当前 Paddle 的 weight_quant kernel 对 GPTQ（如 Qwen3-1.7B-GPTQ-Int8）量化未支持或未验证。
* 目标是开发一版可用的 GPTQ 量化实现，确保能：

    1. 正确加载 GPTQ 量化模型。
    2. 跑通推理。
    3. 验证量化精度。
    4. 为FD 适配 Qwen/Qwen3-1.7B-GPTQ-Int8
    5. 为FD 适配 Qwen/Qwen3-30B-A3B-GPTQ-Int4

**提交内容**

1. dense部分 GPTQ kernel 提交到 Paddle 仓库中 [https://github.com/PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
    1. 可参考 gpu实现 paddle/phi/kernels/gpu/weight_quantize_kernel.cu
    2. 可参考 cpu实现 paddle/phi/kernels/cpu/weight_quantize_kernel.cc
    3. 可参考前项实现 paddle/phi/kernels/gpu/weight_only_linear_kernel.cu
    4. gpu实现提交到：paddle/phi/kernels/gpu
    5.  cpu实现提交到：paddle/phi/kernels/cpu
    6. 前项实现提交到 paddle/phi/kernels/gpu

2. FD 代码提交到 
    1. dense部分组网实现提交到:fastdeploy/model_executor/layers/quantization/weight_only.py 中，新增class GPTQMethod(QuantMethodBase)
    2. moe部分awq kernel实现提交到 custom_ops/gpu_ops/moe/moe_ffn.cu
    3. moe部分组网实现提交到 fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py 新增 class GPTQMoEMethod(CutlassMoEMethod):

3. 确保精度正常，以 vllm/sgalng/transform 作为 baseline

**模型链接：**

[https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8](https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8)
[https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4)

**技术要求：**

* 熟悉 python, 熟悉 cuda
* 熟悉 VLLM/SGLANG加载量化流程。

### PaddleScience套件开发

### **NO.81 - NO.82 算子学习和第三方库新增**

### NO.81 基于PaddleScience复现GAOT模型，精度对齐论文

**论文链接：**

[https://camlab-ethz.github.io/GAOT/static/pdfs/gaot.pdf](https://camlab-ethz.github.io/GAOT/static/pdfs/gaot.pdf)

**复现要求：**

基于数据集Elasticity，能够复现论文声称的1.3% Median relative L¹ error

**参考代码链接：**

[https://github.com/camlab-ethz/GAOT](https://github.com/camlab-ethz/GAOT)

### NO.82 基于Paddle实现Pytorch Geometric库的conv模块

**详细描述：**

实现Pytorch Geometric库（2.6.1版本）的conv模块，并将实现结果合入到paddle_geometric仓库
* 参考代码链接：[https://github.com/pyg-team/pytorch_geometric/tree/2.6.1/torch_geometric/nn/conv](https://github.com/pyg-team/pytorch_geometric/tree/2.6.1/torch_geometric/nn/conv)
* 相关实现：[https://github.com/PFCCLab/paddle_geometric](https://github.com/PFCCLab/paddle_geometric)
* conv模块下的依赖该库的其他模块已经基本实现完成，如遇问题可以在paddle_geometric仓库下提交Issue，我们会尽快解决。

**验收标准：**

* 实现conv模块的全部API，cugraph可不实现，pyg_lib相关可不实现。
* 根据参考代码，完成对应的单元测试。

**技术要求：**

* 熟练掌握 Python 语言
* 熟悉 Paddle、PyTorch等框架

### PaddleOCR套件开发

### NO.83 总结PaddleOCR/PaddleX issue区、用户群的核心高频问题，定位并解决

**任务描述：**

1. 跟进PaddleOCR/PaddleX的ISSUE区的用户问题（每天约几十条），在小于24小时内给予回复，关键问题配合定位和解决。
2. 跟进PaddleOCR/PaddleX的用户群的用户问题（每天约几十条），配合回复问题，关键问题配合定位和解决。

**注意事项：**

1. 每周一、周三、周五、周日，需要总结并上交日报开日会，日报详细描述跟进的问题和解决思路。每周开发时间至少25小时。
2. 每周选定一天上交本周周报，周报详细描述本周跟进和解决的问题，并反馈PaddleOCR/PaddleX中暴露的问题。
3. 用户遇到的基本的bug的问题需要配合定位：
    1. 如是确实是bug，需要配合修复
    2. 如果不会修复，需要在 mentor 的指导下完成修复
    3. 如果涉及的问题确实比较难修复，可以记录问题，由mentor等研发同学来修复

### PaddleSpeech套件开发

### NO.84 支持 PaddleSpeech 中Whisper large/turbo 模型的推理加速

**任务描述：**

根据 PaddleSpeech 中已经实现的 Whisper 推理，在此基础上完成对large/turbo 模型的推理加速，优化推理速度与[faster-whisper](https://github.com/SYSTRAN/faster-whisper)持平。

**注意事项：**

1. 过程中遇到的问题及时反馈，每周参与一次周会。
2. 模型推理加速成功后，需要在 repo 中书写并提交教程 readme，帮助其他开发者使用。
