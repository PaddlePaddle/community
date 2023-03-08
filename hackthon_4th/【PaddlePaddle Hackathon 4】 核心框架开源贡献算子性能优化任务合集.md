# 【PaddlePaddle Hackathon 4】核心框架开源贡献算子性能优化任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

注：为飞桨框架优化一系列算子性能，提交流程请参考 [算子性能优化&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/op_optimization/op_optimization_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下：

### No.32：为 Paddle 优化 expand_as 前向&反向 op 在 GPU 上的计算性能 <a name='task32'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 expand_as 前向和反向算子的 GPU 实现采用 Eigen 组合的模式，缺少 GPU Kernel，性能相对不足；
  - 目标：请实现高性能的 GPU 计算 Kernel，为 Paddle 优化 expand_as op 在 GPU 上的计算性能，性能至少提升6倍，对性能极差的 case 提升达700倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/expand_as_grad_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/expand_as_grad_kernel_impl.h)，[paddle/phi/kernels/impl/expand_as_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/expand_as_kernel_impl.h) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.33：为 Paddle 优化 Histogram 在GPU上的性能 <a name='task33'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 Histogram 算子的计算调用了eigen，kernel实现中存在原子操作。GPU kernel的性能有待提升；
  - 目标：请优化计算实现，为 Paddle 优化 Histogram OP 在 GPU 上的计算性能，性能平均提升2x。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf) 目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/histogram_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/histogram_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.34：为 Paddle 优化 Lerp OP在GPU上的性能 <a name='task34'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 lerp 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 lerp op 在 GPU 上的计算性能，性能至少提升20%，针对性能差的case，性能至少提升4+倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/lerp_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/lerp_kernel_impl.h) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)

### No.35：为 Paddle 优化 Prelu OP在GPU上的性能 <a name='task35'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 Prelu 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 Prelu op 在 GPU 上的计算性能，性能至少提升20%，针对性能差的case，性能至少提升4+倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/prelu_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/prelu_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.36：为 Paddle 优化 Tile OP在GPU上的性能 <a name='task36'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：基础
- 详细描述：
  - 现状：目前 Paddle 内 Tile 算子的 GPU 计算和CPU采用相同的逻辑，GPU性能仍有提升空间；
  - 目标：请优化计算实现，为 Paddle 优化 Tile op 在 GPU 上的计算性能，性能平均提升70%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf) 目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/tile_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/tile_kernel_impl.h)目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.37：为 Paddle 优化 matrix_rank op 在 GPU 上的计算性能 <a name='task37'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 matrix_rank 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 matrix_rank op 在 GPU 上的计算性能，性能至少提升3倍，针对性能差的case，性能提升120+倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/matrix_rank_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/matrix_rank_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.38：为 Paddle 优化 p_norm op 在 GPU 上的计算性能 <a name='task38'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 p_norm 算子 Reduce_Any Kernel 的性能较竞品存在差异；
  - 目标：请优化计算实现，为 Paddle 优化 p_norm op 在 GPU 上的计算性能，性能平均提升2.5倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/p_norm_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/p_norm_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.39：为 Paddle 优化 p_norm_grad op 在 GPU 上的计算性能 <a name='task39'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 p_norm_grad 算子 GPU 计算采用了 CUDA Kernel 与 Eigen 混合的模式，用现有的 Reduce OP 等取代 Eigen 可以提升计算性能，减少数据 HtoD 拷贝等开销；
  - 目标：请优化计算实现，为 Paddle 优化 p_norm_grad op 在 GPU 上的计算性能，性能平均提升3倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/p_norm_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/p_norm_grad_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.40：为 Paddle 优化 kthvalue op 在 GPU 上的计算性能 <a name='task40'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 kthvalue 算子 GPU 计算采用了cub库实现，性能仍有不足；
  - 目标：请优化计算实现，为 Paddle 优化 kthvalue op 在 GPU 上的计算性能，性能平均提升2.7倍。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/kthvalue_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/kthvalue_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.41：为 Paddle 优化 cumprod_grad op 在 GPU 上的计算性能 <a name='task41'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内 cumprod_grad 算子 GPU 计算采用了采用GPU Kernel等拼接实现，性能仍有提升空间；
  - 目标：请优化计算实现，为 Paddle 优化 cumprod_grad op 在 GPU 上的计算性能，性能平均提升30%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/cumprod_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/cumprod_grad_kernel.cu) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.42：为 Paddle 优化 FminGrad OP在GPU上的性能 <a name='task42'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内  FminGrad 在 GPU 的计算性能有待提升；
  - 目标：请优化计算实现，为 Paddle 优化  FminGrad OP 在 GPU 上的计算性能，性能平均提升40%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h) 目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.43：为 Paddle 优化 GeluGrad OP在GPU上的性能 <a name='task43'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内  GeluGrad OP 在GPU的计算性能有待提升；
  - 目标：请优化计算实现，为 Paddle 优化  GeluGrad 在 GPU 上的计算性能，性能平均提升60%。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/gpu/gelu_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/gelu_grad_kernel.cu)目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。

### No.44：为 Paddle 优化 logsumexp OP在GPU上的性能 <a name='task44'></a>

- 技术标签：深度学习框架，Python，C++，CUDA
- 任务难度：进阶
- 详细描述：
  - 现状：目前 Paddle 内  logsumexp OP 的 GPU 计算调用了 eigen，性能较差，有较大的提升空间；
  - 目标：请优化计算实现，为 Paddle 优化 logsumexp OP 在 GPU 上的计算，性能平均提升10x。
- 任务提交：
  - 设计文档：提 PR 至 community repo 的 [rfcs/OPs-Perf](https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs-Perf)  目录；
  - C++ 及 GPU kernel 实现代码：提 PR 至 [paddle/phi/kernels/impl/logsumexp_grad_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/logsumexp_grad_kernel_impl.h)目录；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)。



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 合入标准

- 目标：依据各任务描述，达成任务目标（可以超出）;
-  OP Benchmark内优化算子的全部配置 case 性能不出现下降问题，优化算子的计算精度不出现下降问题；
- 要求：测试case需要覆盖全部的计算分支，同时至少覆盖fp32，fp16两种数据类型。 


### 技术要求

- 熟练掌握 Python、C++、CUDA代码编写；
- 掌握 OP Benchmark [使用方法](https://github.com/PaddlePaddle/benchmark/tree/master/api)。

### 参考内容

- 设计文档示例：[op_optimization_example.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/OPs-Perf/op_optimization_example.md)
- 优化方法参考：[算子性能优化方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/op_optimization/op_optimization_method_introduction_cn.html)

### 答疑交流
- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请大家关注官网&微信群的通知，及时参与~
