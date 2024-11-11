此文档展示 **PaddlePaddle Hackathon 第七期活动——Fundable Projects** 任务详细介绍。Fundable Projects 赛道定位硬核任务，要求高水平的开发者独立进行任务拆解和完成。

## 产出要求

- 任务拆解 tracking issue
- 答辩 PPT
- 书面的技术报告
- 代码运行无误，通过社区 maintainers 的评审并合入代码仓库。

## 任务详情

### 一、Paddle Operator 算子库非通信算子退场/迁移

**任务背景：**

飞桨 Paddle 自 2.0 版本以来，进行了多个重大机制改造。包括：高可复用算子库 PHI、全新的动态图体系。随着新机制的发布使用，旧机制和功能代码需要进行退场和移除，保持架构清晰和代码库的条理性，为内外部开发者提供更好的二次开发环境。这就包括了 Operators 算子库的清理。

**详细描述：**

- 梳理 105 个 Operators 算子库算子，明确迁移/删除
- 105 个算子清单如下：soft_relu,class_center_sample,feed_sparse_coo_tensor,feed_dense_tensor,feed_strings,fetch_v2,crop,crop_grad,data_norm,data_norm_grad,data_norm,data_norm_grad,anchor_generator,anchor_generator,collect_fpn_proposals,collect_fpn_proposals,generate_proposals,generate_proposals,moving_average_abs_max_scale,straight_through_estimator_grad,moving_average_abs_max_scale,straight_through_estimator_grad,resnet_basic_block,resnet_basic_block_grad,resnet_unit,resnet_unit_grad,resnet_unit,resnet_unit_grad,get_tensor_from_selected_rows,bilinear_interp,bilinear_interp_grad,nearest_interp,nearest_interp_grad,trilinear_interp,trilinear_interp_grad,linear_interp,linear_interp_grad,bicubic_interp,bicubic_interp_grad,bilinear_interp,bilinear_interp_grad,nearest_interp,nearest_interp_grad,trilinear_interp,trilinear_interp_grad,linear_interp,linear_interp_grad,bicubic_interp,bicubic_interp_grad,load_combine,load_combine,load_combine,load,load_sr,lod_reset,lod_reset_grad,lod_reset,lod_reset_grad,lookup_table,lookup_table_grad,lookup_table,lookup_table_grad,margin_cross_entropy,margin_cross_entropy_grad,matmul,matmul_grad,matmul_grad_grad,matmul,matmul_grad,memcpy_d2h,memcpy_h2d,nce,nce_grad,nearest_interp,bilinear_interp,matmul,matmul_grad,pull_box_sparse,push_box_sparse,pull_gpups_sparse,push_gpups_sparse,pull_gpups_sparse,push_gpups_sparse,push_dense,reshape,reshape_grad,row_conv,row_conv_grad,row_conv,row_conv_grad,save_combine_tensor,save_combine_vocab,save_combine_tensor,save_combine_vocab,save,save_sr,seed,seed,soft_relu,soft_relu_grad,faster_tokenizer,sync_batch_norm,sync_batch_norm_grad,sync_batch_norm_coo,sync_batch_norm_coo_grad
- 根据梳理结论，迁移/删除算子

**验收说明：**

- 完成 105 个算子的迁移/删除

**技术要求：**

- 熟练掌握 Python 语言
- PIH 算子库的基本原理
- 了解 Operator 算子库的基本原理

**参考资料：**

- [飞桨 PHI 算子库介绍](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2994)
- [PHI 算子库 kernel 注册全流程](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/PHI_kernel_registration/PHI_kernel_registration.md)
- [Kernel 选择分发体系梳理与优化](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/kernel_selection/20221130_kernel_selection.md)

### 二、Paddle Operator 算子库通信算子退场/迁移

**任务背景：**

飞桨 Paddle 自 2.0 版本以来，进行了多个重大机制改造。包括：高可复用算子库 PHI、全新的动态图体系。随着新机制的发布使用，旧机制和功能代码需要进行退场和移除，保持架构清晰和代码库的条理性，为内外部开发者提供更好的二次开发环境。这就包括了 Operators 算子库的清理。

**详细描述：**

- 梳理 115 个 Operators 算子库算子，明确迁移/删除
- 115 个算子清单如下：alltoall,alltoall,barrier,barrier,c_allgather,c_allgather,c_allgather,c_allreduce_avg,c_allreduce_max,c_allreduce_max,c_allreduce_max,c_allreduce_min,c_allreduce_min,c_allreduce_min,c_allreduce_prod,c_allreduce_prod,c_allreduce_prod,c_allreduce_sum,c_allreduce_sum,c_allreduce_sum,c_broadcast,c_broadcast,c_broadcast,c_comm_init_all,c_comm_init_all,c_concat,c_concat,c_concat,c_reduce_avg,c_reduce_max,c_reduce_max,c_reduce_max,c_reduce_min,c_reduce_min,c_reduce_min,c_reduce_prod,c_reduce_prod,c_reduce_prod,c_reduce_sum,c_reduce_sum,c_reduce_sum,c_reducescatter,c_reducescatter,c_scatter,c_scatter,c_softmax_with_cross_entropy,c_softmax_with_cross_entropy,c_softmax_with_cross_entropy_grad,c_softmax_with_cross_entropy,c_softmax_with_cross_entropy_grad,c_sync_calc_stream,c_sync_calc_stream,c_sync_comm_stream,c_sync_comm_stream,global_gather,global_gather,global_scatter,global_scatter,mp_allreduce_sum,mp_allreduce_sum,mp_allreduce_sum,partial_allgather,partial_allgather,partial_recv,partial_recv,partial_send,partial_send,recv_v2,recv_v2,send_v2,send_v2,save_combine,load_combine,c_concat,c_split,c_embedding,c_embedding_grad,c_softmax_with_cross_entropy,c_softmax_with_cross_entropy_grad,c_identity,c_sync_calc_stream,c_allreduce_sum,mp_allreduce_sum,c_allreduce_min,c_allreduce_max,c_allreduce_prod,c_broadcast,barrier,number_count,limit_by_capacity,prune_gate_by_capacity,random_routing,assign_pos,global_scatter,global_gather,fused_attention,fused_attention_grad,fused_feedforward,fused_feedforward_grad,fused_multi_transformer_int8,fused_multi_transformer,ncclAllReduce,ncclBcast,ncclReduce,distributed_fused_lamb,distributed_fused_lamb,distributed_lookup_table,distributed_lookup_table,distributed_push_sparse,distributed_push_sparse,send_and_recv,pull_sparse,push_sparse,pull_sparse_v2,push_sparse_v2
- 根据梳理结论，迁移/删除算子

**验收说明：**

- 完成 115 个算子的迁移/删除

**技术要求：**

- 熟练掌握 Python 语言
- PIH 算子库的基本原理
- 了解 Operator 算子库的基本原理
- 了解通信算子的基本原理

**参考资料：**

- [飞桨 PHI 算子库介绍](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2994)
- [PHI 算子库 kernel 注册全流程](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/PHI_kernel_registration/PHI_kernel_registration.md)
- [Kernel 选择分发体系梳理与优化](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/kernel_selection/20221130_kernel_selection.md)

### 三、x2paddle 套件能力建设

**任务背景**：

X2Paddle 是飞桨生态下的模型转换工具，致力于帮助其它深度学习框架（Caffe/TensorFlow/ONNX/PyTorch）用户快速迁移至飞桨框架。由于近期 Paddle 新版本的升级存在不兼容部分（如 `paddle.fluid` API 全面退场，0-d tensor，view 行为修改等），需要重新对 X2Paddle 进行适配开发与回归测试，保证模型转换工具的正常运转，转换后模型的功能与精度不受损失。同时，近期大模型领域有新的热门模型发布，需要添加此类模型的转换策略，保证模型转换工具的丰富度和领先性。

**详细描述：**

1. 基于 Paddle 3.0.0-beta 版本对 X2Paddle 进行适配升级，梳理已有堵点并解决。保证 [test_benchmark](https://github.com/PaddlePaddle/X2Paddle/tree/develop/test_benchmark) 目录下已适配的 100+ 模型在 新 Paddle 版本 & 新其他深度学习框架版本下的正常运转。（当时适配版本见[这里](https://github.com/PaddlePaddle/X2Paddle?tab=readme-ov-file#环境依赖)）
2. 基于 Paddle 3.0.0-beta 版本，新增 1-2 个大语言模型的转换策略。

**验收说明：**

1. X2Paddle 基于 Paddle 3.0.0-beta 版本，完成 100+ 原有模型的适配。
2. X2Paddle 基于 Paddle 3.0.0-beta 版本，新增 1-2 个大语言模型的转换策略。

**技术要求：**

- 熟悉 Python，工程能力强
- 对深度学习有一定了解，有深度学习框架应用和研发经验（加分项）
- 有大模型相关经验（加分项）

**参考资料：** https://github.com/PaddlePaddle/X2Paddle

### 四、前沿扩散模型飞桨复现

**任务背景：**

FLUX.1 是由 Black Forest Labs 推出的 AI 图像生成模型，拥有 12B 参数，是迄今为止最大的开源文生图模型，目前该团队推出了三款模型变体，分别是快速版-FLUX.1[schnell]、开发版-FLUX.1[dev]和专业版 FLUX.1[pro]。PPDiffusers 是 PaddleMIX 下一款支持多种模态（如文本图像跨模态、图像、语音）扩散模型（Diffusion Model）训练和推理的国产化工具箱，依托于 PaddlePaddle 框架和 PaddleNLP 自然语言处理开发库，目前我们已经完成了套件的训练、推理、应用等基础能力建设，但跨模态文图领域发展迅速，需要从模型、训练、推理等方面不断地跟进与丰富。

**详细描述：**

参考 Diffusers 及 flux 原库完成模型复现、推理 pipeline 复现、模型转换等、lora 训练等，具体包括

- flux.1 基础模型复现，包含其依赖的相关基础组件
- fulx.1 推理 pipeline 构建，包含 FluxPipeline 及 FluxControlNetPipeline
- 提供相关的 paddle 模型权重
- 支持并对齐 flux 的 dreambooth lora 训练

**验收说明：**

- 相关 PR 合入 PaddleMIX/PPDiffusers
- 产出精度对齐数据
- 提供单侧代码，在本地及 CI 测试通过

**技术要求：**

- 熟练掌握 Python 语言
- 熟练使用 PyTorch/Paddle 框架 API
- 熟悉扩散模型原理

**参考资料：**

- https://github.com/black-forest-labs/flux

- https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux

### 五、前沿文档多模态大模型飞桨复现

**任务背景：**

GOT-OCR2.0 是由 StepFun 和中国科学院大学推出的专用于通用 OCR 任务的多模态大模型，参数量 0.6B，采用 vision encoder+input embedding layer+decoder 的 pipeline。PaddleMIX 是基于飞桨的多模态大模型开发套件，聚合图像、文本、视频等多种模态，覆盖视觉语言预训练，文生图，文生视频等丰富的多模态任务。依托于 PaddlePaddle 框架和 PaddleNLP 自然语言处理开发库，目前我们已经完成了套件的训练、推理、应用等基础能力建设，但跨模态文图领域发展迅速，需要从模型、训练、推理等方面不断地跟进与丰富。

**详细描述：**

参考 GOT-OCR2.0 原库完成模型复现、推理 pipeline 复现、模型转换等，具体包括：

- GOT-OCR2.0 基础模型复现，包含其依赖的相关基础组件
- GOT-OCR2.0 推理 pipeline 构建
- 提供相关的 paddle 模型权重
- 支持并对齐 GOT-OCR2.0 的 post-training 训练

**验收说明：**

- 相关 PR 合入 PaddleMIX/paddlemix
- 产出精度对齐数据
- 提供单侧代码，在本地及 CI 测试通过

**技术要求：**

- 熟练掌握 Python 语言
- 熟练使用 PyTorch/Paddle 框架 API
- 熟悉多模态大模型原理

**参考资料：**

https://github.com/Ucas-HaoranWei/GOT-OCR2.0/
https://huggingface.co/stepfun-ai/GOT-OCR2_0

### 七、PaddleSpeech 套件能力建设

**任务背景**：

PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源套件，囊括语音识别、语音合成、语音唤醒、声纹识别等多种语音常用功能的支持。由于近期 Paddle 新版本的升级存在不兼容部分（如 `paddle.fluid` API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等），需要重新对 PaddleSpeech 中的模型进行适配开发与回归测试，保证套件正常运转，模型功能与精度不受损失。外部开发者需要做的事情包括：

**详细描述：**

1. 基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 进行适配升级，梳理已有堵点并解决。保证 [demo](https://github.com/PaddlePaddle/PaddleSpeech/tree/doc/demos) 和 [example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples) 目录下已适配的模型在 新 Paddle 版本 & 新其他深度学习框架版本下的正常运转。目前适配版本为 Paddle 2.5.1。
2. 基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 中支持转静的模型重新按照 PIR + predictor 的方式导出，并成功推理。

**验收说明：**

1. PaddleSpeech 基于 Paddle 3.0.0-beta 版本，完成 80+ 原有模型的适配。
2. PaddleSpeech 基于 Paddle 3.0.0-beta 版本，完成 20+ 原有静态图模型的重新导出和上传。

**技术要求：**

- 熟悉 Python，工程能力强
- 对语音识别或合成有一定了解，有训练或者研发经验（加分项）
- 对 PaddleSpeech 套件比较熟悉（加分项）

**参考资料：** https://github.com/PaddlePaddle/PaddleSpeech

### 八、Netron 原生支持 Paddle PIR 可视化

**任务背景**：

Netron 是一个开源的神经网络模型可视化工具，它支持多种深度学习框架的模型格式。通过 Netron，用户可以直观地查看神经网络模型的结构、层次关系、参数信息和数据流，帮助开发者调试和优化模型。早在 2018 年 Netron 已支持 Paddle 模型文件的解析和展示。但 2024 年，Paddle 对静态图 IR 进行了全面的升级换代（PIR），Save 的模型文件格式也从 protobuf 格式变为了 Json 格式。因此，Paddle 需要重新适配 Netron。主要工作包括：

**详细描述：**

1. 编写 JavaScript，解析 Json 格式的模型文件。关于 Json 格式的细节将会有 Paddle 相关导师答疑。
2. 适配 Netron，将解析的模型与 Netron 前端逻辑适配。

**验收说明：**
打开 Netron 网站，提交多种 case 的模型文件，能够正确展示模型结构。

**技术要求：**

- 熟练掌握 JavaScript
- 了解 Json
- 了解深度学习基本知识

**参考资料：** 
https://github.com/lutzroeder/netron
https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/IR_Dialect/pir_save_load.md

### 九、Paddle LoD 退场清理

**任务背景**：

早年Paddle设计之初，Paddle设计LodTensor用来表示嵌套的、每条数据长度不一的一组数据。（例：一个 batch 中包含了长度为 3，10，7，50 的四个句子）。自Paddle2.0开始，Paddle废弃了LodTensor的概念。但由于LoD是Paddle设计之初就存在的最为基础的数据结构，虽然Paddle废弃了LodTensor，但是考虑兼容性问题，LoD并没有被及时清理，且目前“弥漫”在Paddle代码的各个角落。目前整个Paddle源码中，以“lod”字符串检索，可命中1105个文件的10367行代码。

本题目是LoD退场清理的首期工作，剩余清理工作将以二期、三期形式开展。

**详细描述：**

清理的原则如下：

1. 对于存在lod字样的代码，如果已经没有任何地方调用/使用，直接删除即可。参考PR：https://github.com/PaddlePaddle/Paddle/pull/69176
2. 有一些lod的使用逻辑，虽然有地方在使用，但仍然是可以删除的。需要同步删除调用的地方，甚至修改一些代码上下文确保原来代码的正确性。典型的如：paddle.bass.core.LoDTensor的set_lod、lod、set_recursive_sequence_lengths、recursive_sequence_lengths、has_valid_recursive_sequence_lengths相关接口。
3. 所有以“LodTensor”、“lod_tensor”、“lodtensor”命名的函数名/变量名/文件名，原则上都可以替换为“DenseTensor”、“dense_tensor”、“densetensor”。但替换时仍有必要认真Check是否重命名存在不恰当的情况，比如这里的代码应该删除等。
4. 能删除的，绝不要修改名称了之。
5. 所有涉及API名字修改、或者API直接删除的，必须对Paddle下所有repo，做一遍检索，修改所有模型库调用该API的地方。
6. 有些不得不保留的lod相关逻辑，将lod命名改为deprecated_lod，以标记这是废弃的逻辑，只是暂时未删除。


清理步骤可以如下：

步骤1，paddle.bass.core.LoDTensor的专项清理：

1. 它在Python的声明在：https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/base/core.py#L278
2. 他从C++暴露到Python的逻辑在：https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/tensor.cc#L202
3. 清理后的目标是：
  - pybind直接暴露的名字为DenseTensor，而不是Tensor或者LoDTensor。所有相关API文档里代码样例也要同时伴随修改。重命名为DenseTensor后，必须对Paddle下所有repo，做一遍检索，修改所有模型库调用paddle.bass.core.LoDTensor或者paddle.bass.core.Tensor的地方。
  - paddle.bass.core.LoDTensor.set的文档写的有问题，“lod (numpy.ndarray): The data to set.”应改成"array (numpy.ndarray): The shape where the DenseTensor is to be set."
  - py::init构造函数，去掉传参const std::vector<std::vector<size_t>> &recursive_sequence_lengths的重载
  - 删除set_lod、lod、set_recursive_sequence_lengths、recursive_sequence_lengths、has_valid_recursive_sequence_lengths相关接口。同样需要检索Paddle下所有repo，处理有相应API调用的逻辑。
  - 删除其余接口代码及代码示例中对lod的使用（如果CI能过的话）

步骤2，一些重点名称的重命名：

1. paddle.bass.core.LoDTensorArray 重命名为 paddle.bass.core.DenseTensorArray。
2. paddle.bass.core.VarDesc.VarType.LOD_TENSOR 重命名为 paddle.bass.core.VarDesc.VarType.DENSETENSOR_TENSOR。
3. paddle.bass.core.VarDesc.VarType.LOD_TENSOR_ARRAY 重命名为 paddle.bass.core.VarDesc.VarType.DENSETENSOR_TENSOR_ARRAY。
4. framework::proto::VarType::LOD_TENSOR 重命名为 framework::proto::VarType::DENSETENSOR_TENSOR
5. framework::proto::VarType::LOD_TENSOR_ARRAY 重命名为 framework::proto::VarType::DENSETENSOR_TENSOR_ARRAY
6. 同样需要检索、处理Paddle下所有repo的老名字使用。

步骤3，paddle/fluid/pybind/目录下其余lod字样代码的清理

1. 参考步骤1的原则，清理pybind下其余C++对Python暴露的类、接口，如DistModelTensor
2. 能删除尽可能删除，同时一并删除Python端对类、接口的使用
3. 需要保留的，往往是需要将“lod”重命名为“dense”的，可以重命名处理
4. 同样需要检索、处理Paddle下所有repo，防止不兼容升级造成生代码仍在调用老的API


**验收说明：**
完成详细描述中三个步骤的清理工作。

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉Paddle框架底层机制

**参考资料：** 
本工作，参考详细描述即可。
