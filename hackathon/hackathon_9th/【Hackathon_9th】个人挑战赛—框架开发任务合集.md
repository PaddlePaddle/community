此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛框架开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-API 开发】任务详情

注：为飞桨框架新增一系列 API，提交流程请参考 [新增 API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项：

- 合入标准
  - 按 [API 设计规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) 完成 API 设计文档；需要在设计文档中说明支持哪些数据类型（默认都要支持 fp16/bf16/complex64/complex128），对不支持的要给出理由
  - 按 [API 验收标准](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) 完成 API 功能实现、单测、API 文档；
  - 按 [API 映射关系-格式规范](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 完成 **_API 映射关系文档_** 的编写，文件统一提交到 [convert_from_pytorch/api_difference](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/) 中相应目录，需详细描述与竞品（Pytorch）的差异之处，对差异之处需要在 **_API 设计文档_** 中阐述理由；
- 参考内容
  - [新增 API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)
  - [新增 API 设计模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)
  - [飞桨 API Python 端开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html)
  - [C++ 算子开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
  - [飞桨 API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
  - [API 映射关系-格式规范](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md)
  - [API 单测开发及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)
  - 复数数据类型相关资料：
    - [On the Computation of Complex-valued Gradients with Application to Statistically Optimum Beamforming](https://arxiv.org/abs/1701.00392)
    - [复数梯度推导计算](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/complex_autograd)
    - [paddlepaddle 支持复数任务](https://github.com/PaddlePaddle/Paddle/issues/61975)

### NO.1 - NO.19 API正确性

**详细描述：**

Paddle目前正在对全量API的边界正确性做系统性排查，我们开发了 PaddleAPITest 用于测试存在正确性问题的API。通过与Torch执行相同的API进行精度对比，我们发现一些API与Torch的API存在精度diff。经初步少量API确认，我们发现Paddle API确实存在一些正确性问题（过程中也发现了少量Torch API的正确性问题，如torch.tril、torch.triu）。现将这些问题Paddle API公开，邀请社区同学共同解决问题。参与本项活动，你将学习到Paddle算子库框架的设计，并对Paddle CPU、GPU Kernel的实现风格有详细的了解，对算子精度问题的调试技能积累一定经验。

**题目内容：**

1. 完成 paddle.nn.functional.batch_norm 0-Size 问题修复
2. 完成 paddle.expand 0-Size 问题修复
3. 完成 paddle.incubate.nn.functional.fused_layer_norm 0-Size 问题修复
4. 完成 paddle.index_add 0-Size 问题修复
5. 完成 paddle.index_sample 0-Size 问题修复
6. 完成 paddle.incubate.nn.functional.fused_multi_head_attention 0-Size 问题修复
7. 完成 paddle.incubate.nn.functional.variable_length_memory_efficient_attention 0-Size 问题修复
8. 完成 paddle.as_stride 0-Size 问题修复
9. 完成 paddle.copysign 精度问题修复
10. 完成 paddle.linalg.eigvals 精度问题修复
11. 完成 paddle.linalg.eigvalsh 精度问题修复
12. 完成 paddle.Tensor.cholesky_solve 精度问题修复
13. 完成 paddle.unique 精度问题修复
14. 完成 paddle.incubate.nn.functional.fused_multi_head_attention 精度问题修复
15. 完成 paddle.incubate.nn.functional.variable_length_memory_efficient_attention 精度问题修复
16. 完成 paddle.index_put 精度问题修复
17. 完成 paddle.nn.functional.conv2d 精度问题修复
18. 完成 paddle.nn.functional.conv2d_transpose 精度问题修复
19. 完成 paddle.put_along_axis精度问题修复

**验收说明：**

- 验收说明：PaddleAPITest 回测 case 全部 Pass

**技术要求：**

- 熟悉深度学习框架API、熟悉CUDA编程

**参考资料：**

- [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest)、[#72637](https://github.com/PaddlePaddle/Paddle/issues/72637)（0-size问题修复）、[#72667](https://github.com/PaddlePaddle/Paddle/issues/72667)（精度问题修复）
