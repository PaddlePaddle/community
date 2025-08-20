此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛套件开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-套件开发】任务详情

### NO.20 - NO.85 为 FastDeploy 各个模块及自定义算子补充单测

**详细描述：**

- 自定义单测补充（20题-69题）：定义算子由C++代码实现，是FD底层执行的核心组件，其单测主要存放在FastDeploy的test/operators目录下，自定义算子的单测要求能够对算子执行的正确性进行验证。
- 功能模块单测补充（70题-85题）：这一部分的单测主要聚焦FD各个基础功能模块的测试，各个功能模块都是由Python实现的，需要进行单测补充。

**题目内容**：

20. 自定义算子 masked_per_token_quant 单测补充
21. 自定义算子 moe_fused_hadamard_quant_fp8 单测补充
22. 自定义算子 share_external_data 单测补充
23. 自定义算子 fused_hadamard_quant_fp8 单测补充
24. 自定义算子 rebuild_padding 单测补充
25. 自定义算子 fused_get_rotary_embedding 单测补充
26. 自定义算子 set_value_by_flags_and_idx 单测补充
27. 自定义算子 get_padding_offset 单测补充
28. 自定义算子 cutlass_fp8_fp8_fp8_dual_gemm_fused 单测补充
29. 自定义算子 cutlass_fp8_fp8_half_block_gemm_fused 单测补充
30. 自定义算子 tritonmoe_preprocess 单测补充
31. 自定义算子 gptq_marlin_repack 单测补充
32. 自定义算子 group_swiglu_with_masked 单测补充
33. 自定义算子 moe_wna16_marlin_gemm 单测补充
34. 自定义算子 get_position_ids_and_mask_encoder_batch 单测补充
35. 自定义算子 moe_redundant_topk_select 单测补充
36. 自定义算子 extract_text_token_output 单测补充
37. 自定义算子 top_k_renorm_probs 单测补充
38. 自定义算子 winx_unzip 单测补充
39. 自定义算子 moe_expert_ffn_wint2 单测补充
40. 自定义算子 top_p_candidates 单测补充
41. 自定义算子 speculate_update_v2 单测补充
42. 自定义算子 speculate_get_output_padding_offset 单测补充
43. 自定义算子 speculate_get_seq_lens_output 单测补充
44. 自定义算子 speculate_get_token_penalty_multi_scores 单测补充
45. 自定义算子 speculate_get_padding_offset 单测补充
46. 自定义算子 fused_rotary_position_encoding 单测补充
47. 自定义算子 append_attention 单测补充
48. 自定义算子 ep_moe_expert_dispatch_fp8 单测补充
49. 自定义算子 pre_cache_len_concat 单测补充
50. 自定义算子 ep_moe_expert_dispatch 单测补充
51. 自定义算子 gqa_rope_write_cache 单测补充
52. 自定义算子 dynamic_per_token_scaled_fp8_quant 单测补充
53. 自定义算子 multi_head_latent_attention 单测补充
54. 自定义算子 per_token_quant 单测补充
55. 自定义算子 update_inputs_v1 单测补充
56. 自定义算子 get_data_ptr_ipc 单测补充
57. 自定义算子 per_token_quant_padding 单测补充
58. 自定义算子 speculate_rebuild_append_padding 单测补充
59. 自定义算子 speculate_set_value_by_flags_and_idx 单测补充
60. 自定义算子 eagle_get_self_hidden_states 单测补充
61. 自定义算子 speculate_update_v3 单测补充
62. 自定义算子 eagle_get_hidden_states 单测补充
63. 自定义算子 draft_model_postprocess 单测补充
64. 自定义算子 draft_model_set_value_by_flags 单测补充
65. 自定义算子 draft_model_update 单测补充
66. 自定义算子 speculate_set_stop_value_multi_seqs 单测补充
67. 自定义算子 speculate_verify 单测补充
68. 自定义算子 ngram_match 单测补充
69. 自定义算子 draft_model_preprocess 单测补充
70. 功能模块 CUDAPlatform、CPUPlatform 单测补充

- 详细描述：本任务中需要补充功能模块 CUDAPlatform、CPUPlatform 的单测
- 测试内容：类中各个接口正常可用，功能包括正确判断所在硬件类型，硬件是否可用，正确返回 attention_backend
- 单测名称：tests/platforms/test_platforms.py

71. 功能模块 WeightOnlyLinearMethod 单测补充

- 详细描述：本任务中需要补充功能模块 WeightOnlyLinearMethod 的单测
- 测试内容：创建的 Parameter，apply 计算结果是否符合预期
- 单测名称：tests/quantization/test_weight_only.py

72. 功能模块 Worker/ModelRunner 单测补充

- 详细描述：本任务中需要补充功能模块 Worker/ModelRunner 的单测
- 测试内容：构造一个可以随意指定BatchSize（无动态插入）、Prompt 的 token 数、Decode 的 token 数的只跑假数据的 Worker/ModelRunner
- 单测名称：tests/worker/model_runner.py

73. 功能模块 graph_optimization 单测补充

- 详细描述：本任务中需要补充功能模块 graph_optimization 的单测
- 测试内容：添加一个单测，测试Numpy实现(BaseLine)、动态图、静态图、CINN、动态图+CudaGraph、静态图+CudaGraph、CINN+CudaGraph 七种情况下精度正常且能对齐
- 单测名称：test/graph_optimization/graph_opt_backend.py

74. 功能模块 fastdeploy/cache_manager/RDMACommManager 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/cache_manager/RDMACommManager 的单测
- 测试内容：测试任意两个实例能否进行 kvcache 传输, 并验证传输内容是否完全一致，包括机内与机间
- 单测名称：test/cache_manager/rdma_connect.py

75. 功能模块 fastdeploy/cache_manager/IPCCommManager 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/cache_manager/IPCCommManager 的单测
- 测试内容：测试任意两个实例单机内能否进行 kvcache 传输，且结果一致
- 单测名称：test/cache_manager/ipc_connect.py

76. 功能模块 fastdeploy/model_executor/guided_decoding/XGrammarChecker 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/model_executor/guided_decoding/XGrammarChecker 的单测
- 测试内容：测试能否正确识别 guided_json、guided_grammar、guided_json_object、guided_choice、structural_tag、regex 语法是否合法
- 单测名称：test/model_executor/guided_decoding/test_xgrammar.py

77. 功能模块 fastdeploy/metrics/metrics/get_filtered_metrics 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/metrics/metrics/get_filtered_metrics 的单测
- 测试内容：测试过滤制指定指标，保留其他指标功能，extra_register_func 的指标是否生效
- 单测名称：test/metrics/test_metrics.py

78. 功能模块 fastdeploy/entrypoints 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/entrypoints 的单测
- 测试内容：chat/generation 接口测试，涵盖不同输入格式
- 单测名称：tests/entrypoints/test_generation.py、tests/entrypoints/test_chat.py

79. 功能模块 fastdeploy/entrypoints/openai 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/entrypoints/openai 的单测
- 测试内容：chat/completion 接口测试，流式非流式，异常报错抛出
- 单测名称：tests/entrypoints/openai

80. 功能模块 fastdeploy/splitwise 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/splitwise 的单测
- 测试内容：增加 e2e 单机pd 分离单测，可以正常推理
- 单测名称：tests/splitwise

81. 功能模块 fastdeploy/output 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/output 的单测
- 测试内容：增加 e2e 单机pd 分离单测，可以正常推理
- 单测名称：tests/output

82. 功能模块 fastdeploy/cache_manager 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/cache_manager 的单测
- 测试内容：增加 e2e prompt cache 单测，验证命中率，cache 驱逐，cache swap 是否正常
- 单测名称：fastdeploy/cache_manager/test_prefix_cache.py

83. 功能模块 fastdeploy/model_executor/models 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/model_executor/models 的单测
- 测试内容：每个模型创造一个少量层、fake parameters 的小模型(涵盖 dense layer 和 moe layer)，完成正常推理不报错
- 单测名称：tests/models

84. 功能模块 fastdeploy/reasoning/ 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/reasoning/ 的单测
- 测试内容：测试基类的注册、获取函数功能是否正常
- 单测名称：test/reasoning/test_reasoning_parser.py

85. 功能模块 fastdeploy/inputs/ 单测补充

- 详细描述：本任务中需要补充功能模块 fastdeploy/inputs/ 的单测
- 测试内容：input 为数据处理模块，测试这个目录下四个 processor 类的process_request_dict、process_response、process_response_dict 类能否返回正确值
- 单测名称：test/inputs

**提交内容**：

1. python单测文件代码，提交至FastDeploy/test目录下

**参考文档**：

- 单测开发规范：[Fastdeploy单测规范](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/fastdeploy_unit_test_guide.md)
- 单测写法示例可参考FastDeploy/test目录下已有的单测

**技术要求：**

- 熟悉 python 及 C++ ，会使用 unittest 开发单测
- 能够快速熟悉一个局部模块代码逻辑并制定可行的测试方法
- 熟悉Paddle自定义算子的使用
- 熟悉大模型推理各个模块的功能更佳

### NO.86 FastDeploy编译加速

**详细描述：**

- 优化FastDeploy编译效率，提升编译源码速度

**提交内容**：

- FastDeploy编译加速设计方案，包括调研vllm/sglang等开源库编译方式与流程，分析FastDeploy当前编译耗时情况，找出头部编译耗时单元
- 编译优化代码提交到FastDeploy仓库

**参考文档**：

- vllm官方代码库：[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

**技术要求：**

- 熟悉 cmake、ninja、setuptools 等编译工具
- 熟悉编译有关原理，有调优经验者更佳

### NO.87 为FastDeploy增加Profiler模块

**详细描述：**

- 为 FastDeploy 框架设计、实现类似 vLLM 框架的 profiling（性能分析与 memory 占用）能力。可参考下方 vLLM profiling 说明文档。

**提交内容**：

1. 调研 FastDeploy 中模块层级（数据加载、模型加载、forward、batching、kernel 等），并定义 profiling 划分颗粒度。
2. 产出设计文档（模块划分、 profiling 接口、trace schema、输出格式、可配置 profile 模式等）。
3. 功能对齐 vLLM 的 profiling 能力实现 PR、详细说明文档（使用方式、测试用例、Demo 示例代码等）。
4. 加分项：若发现 paddle.profiler 模块相比 torch.profiler 模块有功能差异，能够在 paddle 仓库中开发补齐完善 paddle.profiler 模块。

**参考文档**：

- vllm profiling: [https://docs.vllm.ai/en/latest/contributing/profiling.html](https://docs.vllm.ai/en/latest/contributing/profiling.html)
- paddle profiler: [https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Profiler_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Profiler_cn.html)

**技术要求：**

- 熟悉 python，了解 paddle
- 熟悉 LLM 推理框架计算流程
- 熟悉 vLLM 更佳

### NO.88 为FastDeploy重构log日志打印范式

**详细描述：**

- 为 FastDeploy 框架设计、实现清晰、易读、有条理、合理且正确的log打印范式。针对大模型推理从启动到输出结果的全执行流程进行分析、梳理、划分，使得在整个执行过程中，终端或日志文件能够打印出关键信息、必要的debug信息等内容。

**提交内容**：

1. 调研 FastDeploy 中模型推理的全环节各模块实现位置，简要梳理仓库源码，产出log打印现状分析文档；
2. 产出全新的log范式划分与设计文档；
3. 开发并为FastDeploy提交一份PR。

**技术要求：**

- 熟悉 python
- 熟悉 LLM 推理框架计算流程

### NO.89 为FastDeploy集成 SageAttn v2/2++

**详细描述：**

- 为FastDeploy 集成SageAttn v2/2++ 高性能量化 Attention能力。

**提交内容**：

1. 调研 SageAttn 适合场景，性能提升幅度，精度损失等。
2. 产出设计文档（接入方式「自定义算子」/「其它」，多Batch、动态插入方案设计，接口设计，功能支持「激活/权重量化等多个场景」，开启方式）
3. 将相关算子集成到FastDeploy, (如果为自定义算子)提交至 FastDeploy/custom_ops/gpu_ops/ 目录下，添加算子单测、对齐算子精度和性能。
4. 产出文档，将相关参数接口说明，算子使用说明，测试用例，相关能力等以文档形式阐述清晰。
5. 将上述算子集成到EB、DeepSeek、Qwen等模型中，测试端到端精度和性能；
6. 加分项：基于开源代码进行量化算子调优，性能优于开源版本。

**技术要求：**

- 熟悉Attention算子，熟悉常见的LLM模型结构和计算流程.
- 熟悉python, 有GEMM/Attention算子等开发经验优先

### NO.90 为FastDeploy集成 SpargeAttn

**详细描述：**

- 为FastDeploy 集成SpargeAttn(基于sageattn-v2++) 的高性能稀疏 Attention能力。

**提交内容**：

1. 调研 SageAttn 适合场景，性能提升幅度，精度损失等。
2. 产出设计文档（接入方式「自定义算子」/「其它」，多Batch、动态插入方案设计，接口设计，功能支持「多Attnetion后段」，开启方式）
3. 将相关算子集成到FastDeploy, (如果为自定义算子)提交至 FastDeploy/custom_ops/gpu_ops/ 目录下，添加算子单测、对齐算子精度和性能。
4. 产出文档，将相关参数接口说明，算子使用说明，测试用例，相关能力等以文档形式阐述清晰。
5. 将上述算子集成到EB、DeepSeek、Qwen等模型中，测试端到端精度和性能；
6. 加分项：基于开源代码进行量化算子调优，性能优于开源版本。

**技术要求：**

- 熟悉Attention算子，熟悉常见的LLM模型结构和计算流程.
- 熟悉python, 有GEMM/Attention算子等开发经验优先

### NO.91 FastDeploy中的MoE GroupGEMM支持INT8\*INT8实现

**详细描述：**

- 为FastDeploy 开发高性能 MoE算子(INT8\*INT8)

**提交内容**：

1. 调研 开源的MoE GroupGEMM实现，分析不同shape下性能优劣。
2. 产出设计文档（激活/权重量化粒度，算子接入方式「离线编译」或「JIT」）
3. 将相关算子集成到FastDeploy，提交至 FastDeploy/custom_ops/gpu_ops/moe 目录下，添加算子单测、对齐算子精度和性能。
4. 产出文档，将相关参数接口说明，算子使用说明，测试用例，相关能力等以文档形式阐述清晰。
5. 将上述算子集成到EB、Qwen等开源模型中，测试端到端精度和性能；
6. 加分项：基于常用开源模型MoE shape 性能优于开源实现。

**技术要求：**

- 熟悉python, 有cuda、cutlass、triton等算子开发经验优先

### NO.92 为 FastDeploy 新增 K2模型

**详细描述：**

- 为FastDeploy 提供部署高性能的Kimi K2 模型的能力.

**提交内容**：

1. Kimi K2 模型代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档.
2. 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
3. 为 Kimi K2 适配FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

- 熟悉常见的LLM模型结构和计算流程. 了解 Kimi K2 模型结构.
- 熟悉python, 熟悉cuda

### NO.93 为 FastDeploy 新增 MiniMax-M1模型

**详细描述：**

- 为FastDeploy 提供部署高性能的MiniMax-M1模型的能力.

**提交内容**：

1. MiniMax-M1 模型组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档.
2. 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
3. 为MiniMax-M1适配FastDeploy现有的各种低bit量化推理的能力.

**技术要求：**

- 熟悉常见的LLM模型结构和计算流程. 了解 MiniMax-M1 模型结构.
- 熟悉python, 熟悉cuda

### NO.94 为 FastDeploy 新增 SD、Flux扩散模型

**详细描述：**

- 为FastDeploy 提供部署高性能的Stable-diffusion、Flux模型的能力.

**提交内容**：

1. SD3、Flux扩散模型组网代码, 提交至 FastDeploy/fastdeploy/model_executor/diffusion_models/ 目录下. 同时提交模型使用说明文档.
2. 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
3. 为扩散模型适配FastDeploy现有的各种并行、低bit量化推理的能力.

**技术要求：**

- 熟悉常见的扩散模型结构和计算流程. 了解SD、Flux模型结构.
- 熟悉python, 熟悉cuda

### NO.95 为 FastDeploy 新增 MTP 的 Multi-layer功能

**详细描述：**

- 目前 FastDeploy 支持 MTP/Eagle 的多次自回归，但不支持多个独立的MTP Layer 推理，需支持此功能；同时兼容已有的 PromptCache/PD分离/ChunkPrefill 等功能

**提交内容**：

1. 设计文档，提 PR 至 FastDeploy/docs/features/speculative_decoding.md
2. 代码文件，提交至 FastDeploy/fastdeploy/spec_decode
3. 如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.

**技术要求：**

- 熟悉投机解码原理，熟悉 Multi-Token-Prediction(MTP)/Eagle
- 熟练掌握 Python，熟悉 CUDA

### NO.96 为FastDeploy新增MLA的FP8版本实现

**详细描述：**

- 目前FastDeploy中已经支持了MLA的16 bit版本实现，在decode阶段，访存带宽可达到硬件理论带宽的80%以上；此外，各开源实现也基本没有MLA相关的FP8版本实现；因此可在FastDeploy中储备QuantMLA的能力。

**提交内容**：

1. 代码实现，提交至FastDeploy/custom_ops/gpu_ops/mla_attn；
2. 单测，提交至FastDeploy/test/operators；
3. 相关设计文档，流水线示意图等；
4. kernel性能分析文件，要求最终访存带宽达到80%以上；

**技术要求：**

- 熟悉CUDA编程，熟悉NVIDIA GPU架构；
- 熟悉MLA算法，熟悉Attention的常规GPU实现；
