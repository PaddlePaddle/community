此文档展示 **PaddlePaddle Hackathon 第十期活动——开源贡献个人挑战赛春节特别季任务** 详细介绍

## 【开源贡献个人挑战赛春节特别季-框架】任务详情

### NO.1 paddle.nn.MaxPool1D / MaxPool2D / MaxPool3D兼容性增强

**详细描述：**

详细描述：新增dilation参数，支持空洞池化，需要新增Kernel。（包括CPU/GPU/XPU）
开发流程规范参考：[issue#76301](https://github.com/PaddlePaddle/Paddle/issues/76301#issuecomment-3502554429)
注：需同时完成3个API的功能增强，新增kernel逻辑较复杂，需要同时新增1D、2D、3D的kernel逻辑，整体考虑和开展。

**提交内容**：

1. 撰写设计文档，提交 PR 添加至 GraphNet/docs。
2. 在新增样本的 PR 描述中记录模型样本转换、运行测试的结果，及必要的log片段。

### NO.2 跨生态自定义算子注册层兼容能力增强

**任务说明**：

为了能够低成本接入其他框架生态中丰富的算子库，我们目前引入了一个兼容方案，通过自底向上「C++ API 兼容层」、「算子注册兼容层」、「Python 接口兼容层」、「Python API 代理层」等多层兼容机制，实现对其他框架生态算子的支持。目前生态兼容方案架构已经逐渐成型，并在基于 PyTorch、TVM FFI 生态的算子库中完成了验证，但在实际使用过程中暴露了一些机制不完善的问题，需要进一步完善兼容机制，提升跨生态自定义算子注册的兼容能力和易用性。

本任务专注于「算子注册兼容层」，需要完成以下工作内容：

- 为 `TORCH_LIBRARY` 注册机制添加 schema 支持，为类型推导和参数绑定提供支持；
- 基于 schema 机制，完成复杂传参场景的支持，包括「默认参数」、「keyword 参数」功能；
- 注册兼容层中实现 backend 选择逻辑，支持多 backend 场景下的算子注册和调用。

**验收说明**：

- 完成关于本任务的 RFC（合入 community 仓库）；
- 完成 `TORCH_LIBRARY` 注册机制的 schema 支持，能够正确处理算子参数的类型，以及默认参数、keyword 参数传递功能，单测添加到 `test/cpp/compat/torch_library_test.cc`（合入 Paddle repo）；
- 完成多 backend 支持的注册兼容层实现，并新增相关单测（合入 Paddle repo）；
- 基于上述功能，减少适配 [paddlecodec](https://github.com/meta-pytorch/torchcodec/compare/main...PFCCLab:paddlecodec:paddle) 过程中进行的改动，并在相关改动恢复后仍能保证适配正确性（合入 [PFCCLab/paddlecodec](https://github.com/PFCCLab/paddlecodec) repo）。

> 注意，利用 schema 的检查功能是可选的，我们不一定需要实现在 Paddle 兼容层做相关检查，因为我们默认这些库在其原生框架中已经经过了充分的测试和验证，我们需要保证的是在原生框架中能够通过的算子调用，在 Paddle 兼容层中也能正确通过即可。

**技术要求**：

- 熟悉 PaddlePaddle 框架代码，熟悉 C++ 编译和调试流程
- 了解 PyTorch 框架的自定义算子注册机制和 schema 定义方式

**参考资料：**

- [跨生态自定义算子接入 - 原理和迁移方式](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/custom_op/cross_ecosystem_custom_op/design_and_migration_cn.html)
- [兼容性生态库 paddlecodec](https://github.com/PFCCLab/paddlecodec)

## 【开源贡献个人挑战赛春节特别季-套件开发】任务详情

### PaddleScience套件开发

### NO.3 基于Paddle实现Pytorch Sparse库的spmm等模块

**详细描述**：

基于Paddle实现Pytorch Sparse库的spmm等模块，并将实现结果合入到paddle_sparse仓库
- 相关实现：[https://github.com/PFCCLab/paddle_sparse](https://github.com/PFCCLab/paddle_sparse)
- 具体需要实现的模块包括：spmm, spspmm, matmul, __matmul__, random_walk, partition, reverse_cuthill_mckee, saint_subgraph, remove_diat, set_diag, fill_diag, get_diag

**验收标准**：

- 实现相关模块，完成对应的单元测试。

**技术要求**：

- 熟练掌握 Python 语言
- 熟悉 Paddle、PyTorch等框架

### NO.4 基于Paddle实现Pytorch Geometric库的data模块

**详细描述**：

实现Pytorch Geometric库（2.6.1版本）的data模块，并将实现结果合入到paddle_geometric仓库
- 参考代码链接：[https://github.com/pyg-team/pytorch_geometric/tree/2.6.1/torch_geometric](https://github.com/pyg-team/pytorch_geometric/tree/2.6.1/torch_geometric)
- 相关实现：[https://github.com/PFCCLab/paddle_geometric](https://github.com/PFCCLab/paddle_geometric)
- data 模块下面已有部分完成实现，本次需要实现的模块有：collate, databse, datapipes, download, extract, hypergraph_data, on_dist_dataset, remote_backend_utils, temporal

**验收标准**：

- 实现要求模块的全部API。
- 根据参考代码，完成对应的单元测试。

**技术要求**：

- 熟练掌握 Python 语言
- 熟悉 Paddle、PyTorch等框架

### NO.5 基于Paddle实现Pytorch Geometric库的conv模块

**详细描述**：

实现Pytorch Geometric库（2.6.1版本）的conv模块，并将实现结果合入到paddle_geometric仓库
- 参考代码链接：https://github.com/pyg-team/pytorch_geometric/tree/2.6.1/torch_geometric/nn/conv
- 相关实现：https://github.com/PFCCLab/paddle_geometric

**验收标准**：

- 实现conv模块的全部API，cugraph可不实现，pyg_lib相关可不实现。
- 根据参考代码，完成对应的单元测试。

**技术要求**：

- 熟练掌握 Python 语言
- 熟悉 Paddle、PyTorch等框架

### **NO.6 - NO.19 PaddleMateirals模型复现**

**详细描述**：

依托PaddleMaterials实现模型复现计划中的模型，参考列表里的非MIIT Program的模型进行复现[https://github.com/PaddlePaddle/PaddleMaterials/issues/194](https://github.com/PaddlePaddle/PaddleMaterials/issues/194)

**验收标准**：

1. 单卡前向精度对齐：前向logits diff 1e-4 量级（生成式1e-6）。
2. 反向对齐：训练2轮以上，loss一致。
3. 训练精度对齐：ImageNet数据集精度 diff 0.2%以内。
4. 监督类任务：metirc误差控制在1%以内
5. 生成式模型：采样指标保持误差5%以内

**备注说明**：

1. 模型文件放到ppmat/model下，所有模型需采用统一的trainer/predictor/sampler
2. 数据集/预训练模型文件/log原始文件通过百度网盘链接通过
pr/微信等方式给到百度工程师，工程师会给到百度云链接，需把相应的链接放到相应的dataset/model/readme等文件中
3. 扩散模型相关组件在ppmat/scheduler里
4. 对于套件里暂时还没有的任务，新模型对应的新任务，需新建相应的任务readme文件
5. 数据处理模块，需使用已有的build_structure/build molecule等工厂函数
6. 每个复现的模型PR需保证囊括了原论文的所有数据集，模型训练精度和推理/采样metric指标均可以对应原始论文
7. 对应套件内还没有的任务类型，需要添加新增的任务类型说明readme文档

**技术要求**：

- 熟练掌握 Python 语言
- 熟悉 Paddle、PyTorch等框架

**题目内容**：

### NO.6  CrystalLLM模型复现
### NO.7  wD-MPNN模型复现
### NO.8  Crystalformer模型复现
### NO.9  NewtonNet模型复现
### NO.10 ECDFormer模型复现
### NO.11 TrinityLLM模型复现
### NO.12 AlloyGAN模型复现
### NO.13 GDI-NN模型复现
### NO.14 SFIN模型复现
### NO.15 DM2模型复现
### NO.16 DiffSyn模型复现
### NO.17 MOFDiff模型复现
### NO.18 SchNet模型复现
### NO.19  SphereNet模型复现

### FastDeploy套件开发

### **NO.20 - NO.44 单测补充**

**详细描述：**

当前FastDeploy下一些文件缺少单测监控，需要添加单测代码，来提高文件中代码的单测覆盖率。
本任务中，通过添加单测后提高的代码覆盖行数来确定PR的贡献度，每提高100行（四舍五入，比如150等同200行，140行等同100行）代码覆盖，贡献度累计0.1⭐️。

开发者可通过链接来查看最新的代码覆盖情况：https://paddle-github-action.bj.bcebos.com/BRANCH/FastDeploy/develop/{完整的commit-id}/SM/CoverageData/full_coverage_report.csv，
在这个链接里，通过指定commit-id来查看对应commit-id下代码的覆盖情况（当前仅支持查看某一天最后一个commit的覆盖率）：

<img width="984" height="40" alt="Image" src="https://github.com/user-attachments/assets/5d6d1dd5-a455-40d7-a430-024cbf29eca3" />

比如打开覆盖率表格可以看到如上内容，通过Miss列可以看到总的未覆盖代码行号，比如上边的audio.py里有25行有效代码没有单测覆盖；通过Missing列可看到具体未覆盖代码的行号，比如这里表示行号17-127行未被覆盖（这里Missing列会把注释等无效代码算进去，所以数字会比Miss列要大）。

PR验收的标准是看文件代码的覆盖率(Cover)是否达到了80%，这个覆盖率在Coverage CI的日志里（Run FastDeploy Unit Tests and Coverage / run_tests_with_coverage 中的 Run FastDeploy Unit Tests and Coverage）有显示，在达到80%的基础上，贡献单测越多，获得的⭐️越高。

提交内容：
* Python 单测代码, **可以适当使用AI，但不能过度依赖完全生成**
* PR中评论：当前develop分支的单测覆盖率情况，增加该PR后的单测覆盖率情况，本PR代码覆盖行数。可参考 [https://github.com/PaddlePaddle/FastDeploy/pull/5007](https://github.com/PaddlePaddle/FastDeploy/pull/5007) 

技术要求：
* 熟悉python及unittest、pytest单测工具

**题目内容**：

### NO.20 功能模块 fastdeploy/engine/common_engine.py 单测补充
### NO.21 功能模块 fastdeploy/engine/sched/resource_manager_v1.py 单测补充
### NO.22 功能模块 fastdeploy/cache_manager/cache_transfer_manager.py 单测补充
### NO.23 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py 单测补充
### NO.24 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_triton_backend.py 单测补充
### NO.25 功能模块 fastdeploy/inter_communicator/zmq_server.py 单测补充
### NO.26 功能模块 fastdeploy/utils.py 单测补充
### NO.27 功能模块 fastdeploy/cache_manager/prefix_cache_manager.py 单测补充
### NO.28 功能模块 fastdeploy/entrypoints/engine_client.py 单测补充
### NO.29 功能模块 fastdeploy/engine/engine.py 单测补充
### NO.30 功能模块 fastdeploy/inter_communicator/engine_worker_queue.py 单测补充
### NO.31 功能模块 fastdeploy/model_executor/layers/sample/sampler.py 单测补充
### NO.32 功能模块 fastdeploy/model_executor/load_weight_utils.py 单测补充
### NO.33 功能模块 fastdeploy/config.py 单测补充
### NO.34 功能模块 fastdeploy/eplb/async_expert_loader.py 单测补充
### NO.35 功能模块 fastdeploy/engine/resource_manager.py 单测补充
### NO.36 功能模块 fastdeploy/worker/worker_process.py 单测补充
### NO.37 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_wint2_backend.py 单测补充
### NO.38 功能模块 fastdeploy/entrypoints/openai/serving_completion.py 单测补充
### NO.39 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py 单测补充
### NO.40 功能模块 fastdeploy/model_executor/layers/linear.py 单测补充
### NO.41 功能模块 fastdeploy/entrypoints/llm.py 单测补充
### NO.42 功能模块 fastdeploy/entrypoints/openai/serving_chat.py 单测补充
### NO.43 功能模块 fastdeploy/model_executor/models/ernie4_5_mtp.py 单测补充
### NO.44 功能模块 fastdeploy/model_executor/layers/moe/fused_moe_deepgemm_backend.py 单测补充

### **NO.45 - NO.46 编译支持**

### NO.45 FastDeploy 支持在 T4/V100 硬件的编译

**详细描述：**

FastDeploy支持在T4、V100硬件编译

**提交内容：**

编代码提交到FastDeploy仓库

**技术要求：**

- 熟悉C++/CUDA开发编译，有多硬件开发经验更佳
- 熟悉 shell 以及setuptools 等编译工具

### NO.46 FastDeploy 支持在 windows 平台的编译

**详细描述：**

FastDeploy支持在Windows平台编译

**提交内容：**

编代码提交到FastDeploy仓库

**技术要求：**

- 熟悉C++/CUDA开发编译，有多硬件开发经验更佳
- 熟悉 shell 以及setuptools 等编译工具

### NO.47 为 FastDeploy 新增 MiniMax-M1模型

**详细描述：**

- 为FastDeploy 提供部署高性能的MiniMax-M1模型的能力.

**提交内容**：

1. MiniMax-M1 模型组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档.
2. 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
3. 为MiniMax-M1适配FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

- 熟悉常见的LLM模型结构和计算流程. 了解 MiniMax-M1 模型结构.
- 熟悉python, 熟悉cuda

### NO.48 为 FastDeploy 新增 SD、Flux扩散模型

**详细描述：**

- 为FastDeploy 提供部署高性能的Stable-diffusion、Flux模型的能力.

**提交内容**：

1. SD3、Flux扩散模型组网代码, 提交至 FastDeploy/fastdeploy/model_executor/diffusion_models/ 目录下. 同时提交模型使用说明文档.
2. 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
3. 为扩散模型适配FastDeploy现有的各种并行、低bit量化推理的能力.

**技术要求：**

- 熟悉常见的扩散模型结构和计算流程. 了解SD、Flux模型结构.
- 熟悉python, 熟悉cuda
