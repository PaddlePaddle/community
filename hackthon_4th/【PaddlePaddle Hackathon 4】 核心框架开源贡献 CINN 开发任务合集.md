# 【PaddlePaddle Hackathon 4】核心框架开源贡献 CINN 开发任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/50629)）

注：为神经网络编译器 CINN 增加一系列基础算子，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下：

### No.81：为神经网络编译器 CINN 增加 bitcast_convert 算子 <a name='task81'></a>

- 任务难度：基础
- 详细描述：
  - 现状：在对接上层框架时，编译器会将上层的框架复杂算子进一步拆分为若干基础算子便于编译器执行算子融合，同时减少开发成本。
  - 目标：通过 extern call 的方式，实现 bitcast_convert 算子。在不改变底层存储的情况下，强制转换数据类型。若转换前后数据类型的字节大小不相同，则形状会改变。比如一个 shape=[10] 的 float32 类型数据被强制转换为 float16 类型后，其 shape 应为[10, 2]。
- 任务提交：
  - 参考示例 [PR1018](https://github.com/PaddlePaddle/CINN/pull/1018)。
  - 算子接口应有一个输入，一个 string 类型属性指定要转换的类型，以及一个输出。
- 技术要求：
  - 了解 AI 编译器 Compute 和 Schedule 含义
  - 熟练使用 C++ 开发以及 cmake 编译

### No.82：为神经网络编译器 CINN 增加 triangular_solve 算子 <a name='task82'></a>

- 任务难道：基础
- 详细描述：
  - 现状：在对接上层框架时，编译器会将上层的框架复杂算子进一步拆分为若干基础算子便于编译器执行算子融合，同时减少开发成本。
  - 目标：通过 custom call 的方式，实现 triangular_solve 算子。triangular_solve 算子用于计算具有唯一解的线性方程组，其中系数方阵A为上（下）三角系数矩阵。若系数方阵A不可逆，则线性方程不可解。
- 任务提交：
  - 参考示例 [#1133](https://github.com/PaddlePaddle/CINN/pull/1133)。当前仅需考虑CUDA环境，可调用 cuslover 库。
  - 算子接口应有两个输入，分别指定矩阵A（形状[*, M, M]）和B（形状[*, M, K]），两矩阵batch维度大小相同。四个 bool 类型的属性，left_side 用于指定系数方阵是位于求解矩阵的左侧还是右侧，upper 指定对系数方阵取上三角还是下三角，transpose_a 指定是否对系数方阵进行转置，unit_diagonal 指定是否假定系数方阵对角线上的元素都为1。输出只有一个，且形状为[*, M, K]。
- 技术要求：
  - 了解 cuda
  - 熟练使用 C++ 开发以及 cmake 编译

### No.83：为神经网络编译器 CINN 增加 resize 算子 <a name='task83'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：在对接上层框架时，编译器会将上层的框架复杂算子进一步拆分为若干基础算子便于编译器执行算子融合，同时减少开发成本。
  - 目标：resize 算子是图像处理领域中的常见算子，resize 算子将输入图片通过指定插值方法调整为指定大小。可通过 CINN IR + extern call 或者 custom call 的方式实现该算子。
- 任务提交：
  - 参考示例 [PR1018](https://github.com/PaddlePaddle/CINN/pull/1018)，以及 [PR1133](https://github.com/PaddlePaddle/CINN/pull/1133)。当前仅需考虑CUDA环境。
  - 算子接口应有一个输入，该输入应为 4-D 张量，形状为[N, C, H, W]。有两属性 out_shape 和 mode，其中 out_shape 指定调整后的张量大小，且形状为[out_H, out_W]；mode 指定插值方法，当前只需实现 bilinear、bicubic、nearest 方法。输出只有一个，且形状为[N, C, out_H, out_W]。
- 技术要求：
  - 有 gpu 算子或 AI 编译器算子开发经验
  - 了解 AI 编译器 Compute 和 Schedule 含义
  - 熟练使用 C++ 开发以及 cmake 编译

### No.84：为神经网络编译器 CINN 增加 ReverseComputeInline 原语 <a name='task84'></a>

- 任务难度：基础
- 详细描述：
  - 现状：Schedule 原语是 CINN 编译器优化算子计算实现的接口，目前已经实现了Split、Fuse、Reorder等常用原语，其中 ComputeInline 原语操作是将一个 tensor 的计算过程内联到其消费者中完成，简化计算过程。
  - 目标：参考已有的 ComputeInline 操作和 [CINN 调度原语开发说明文档](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/CINN/CINN_ir_schedule.md)，添加 ReverseComputeInline 原语，实现将一个 tensor 的计算内联到其生产者中。
- 任务提交：
  - ComputeInline 原语：分别添加接口及实现至[ cinn/ir/ir_schedule.h](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/ir_schedule.h)、[cinn/ir/ir_schedule.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/ir_schedule.cc)，单测添加至 [cinn/backends/ir_schedule_test.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/backends/ir_schedule_test.cc)
  - 支持新增原语 Trace 重放：在 [cinn/ir/schedule_desc.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/schedule_desc.cc) 中使用CINN_BUILD_STEP_KIND 注册 ComputeInline 原语的重放函数，单测添加至 [cinn/ir/schedule_desc_test.cc](https://github.com/PaddlePaddle/CINN/blob/f3dfd275d6ed189e6d18a885178632a4b02afddd/cinn/ir/schedule_desc_test.cc)
- 技术要求：
  - 了解 AI 编译器 Schedule 作用及含义，熟悉 CINN IR 结构
  - 熟练掌握 C++



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 详细描述
使用神经网络编译器 CINN IR 编写算子，大致步骤为在 CINN 底层使用 CINN IR 实现相关计算的描述，如果有优化需求再实现相关 strategy，之后打通上层的 net_builder 和 load_paddle_model，并进行相关单测。

### 提交内容
1. 包含算子的设计文档，以及必要的代码讲解和背景知识。 提交至 PaddlePaddle/community repo 的 [rfcs/CINN/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/CINN/APIs) 目录；
2. Github fork [PaddlePaddle/CINN](https://github.com/PaddlePaddle/CINN) repo 为自己 repo 后，提交 PR 到 PaddlePaddle/CINN 公开 repo，提交 PR 的链接：
 a. PR 代码中，使用 CINN IR 编写的算子，相应的 strategy，根据具体算子在 CINN 情况提交至相关路径；
  b. 上层 net_builder 调用提交到 cinn/frontend/net_builder.h 和 .cc 文件下；
  c. 上层 load_paddle_model 调用提交到 cinn/frontend/paddle_model_to_program.h 和 .cc 文件下。

### 技术要求

- 熟悉神经网络编译器，如 TVM 或 CINN；
- 熟悉神经网络框架，如 TensorFlow 或 Pytorch 或者 PaddlePaddle；
- 熟练掌握 C++ 、Python。

### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流；
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请大家关注官网&微信群的通知，及时参与。