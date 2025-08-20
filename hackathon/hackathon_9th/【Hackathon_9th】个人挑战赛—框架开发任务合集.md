此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛框架开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-API 开发】任务详情

### NO.1 - NO.19 API正确性

**详细描述：**

Paddle目前正在对全量API的边界正确性做系统性排查，我们开发了 PaddleAPITest 用于测试存在正确性问题的API。通过与Torch执行相同的API进行精度对比，我们发现一些API与Torch的API存在精度diff。经初步少量API确认，我们发现Paddle API确实存在一些正确性问题（过程中也发现了少量Torch API的正确性问题，如torch.tril、torch.triu）。现将这些问题Paddle API公开，邀请社区同学共同解决问题。参与本项活动，你将学习到Paddle算子库框架的设计，并对Paddle CPU、GPU Kernel的实现风格有详细的了解，对算子精度问题的调试技能积累一定经验。

**验收说明：**

- 验收说明：PaddleAPITest 回测 case 全部 Pass

**技术要求：**

- 熟悉深度学习框架API、熟悉CUDA编程

**参考资料：**

- [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest)、[#72637](https://github.com/PaddlePaddle/Paddle/issues/72637)（0-size问题修复）、[#72667](https://github.com/PaddlePaddle/Paddle/issues/72667)（精度问题修复）

**题目内容：**

##### NO.1 完成 paddle.nn.functional.batch_norm 0-Size 问题修复
##### NO.2 完成 paddle.expand 0-Size 问题修复
##### NO.3 完成 paddle.incubate.nn.functional.fused_layer_norm 0-Size 问题修复
##### NO.4 完成 paddle.index_add 0-Size 问题修复
##### NO.5 完成 paddle.index_sample 0-Size 问题修复
##### NO.6 完成 paddle.incubate.nn.functional.fused_multi_head_attention 0-Size 问题修复
##### NO.7 完成 paddle.incubate.nn.functional.variable_length_memory_efficient_attention 0-Size 问题修复
##### NO.8 完成 paddle.as_stride 0-Size 问题修复
##### NO.9 完成 paddle.copysign 精度问题修复
##### NO.10 完成 paddle.linalg.eigvals 精度问题修复
##### NO.11 完成 paddle.linalg.eigvalsh 精度问题修复
##### NO.12 完成 paddle.Tensor.cholesky_solve 精度问题修复
##### NO.13 完成 paddle.unique 精度问题修复
##### NO.14 完成 paddle.incubate.nn.functional.fused_multi_head_attention 精度问题修复
##### NO.15 完成 paddle.incubate.nn.functional.variable_length_memory_efficient_attention 精度问题修复
##### NO.16 完成 paddle.index_put 精度问题修复
##### NO.17 完成 paddle.nn.functional.conv2d 精度问题修复
##### NO.18 完成 paddle.nn.functional.conv2d_transpose 精度问题修复
##### NO.19 完成 paddle.put_along_axis精度问题修复


