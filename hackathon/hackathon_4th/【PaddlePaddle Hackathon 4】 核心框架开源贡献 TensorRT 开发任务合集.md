# 【PaddlePaddle Hackathon 4】核心框架开源贡献 TensorRT 开发任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

注：为飞桨框架新增一系列 TensorRT 算子，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项在任务列表后：

### No.71：为 Paddle-TRT 添加 pad3d 算子 <a name='task71'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 pad3d 算子 TRT Layer映射实现，为了让含有 pad3d 的算子以全图形式执行 TensorRT engine，需添加该算子实现。
  - 目标：完成 pad3d 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 pad3d TRT算子映射，并提交 PR
  - 任务要求：
	    - 完成pad3d功能实现代码
	    - 单测python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_pad3d.py 验证通过
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.72：为 Paddle-TRT 添加 flip 算子 <a name='task72'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 flip 算子 TRT Layer 映射实现。
  - 目标：完成 flip 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 flip TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 flip 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.73：为 Paddle-TRT 添加 temporal_shift 算子 <a name='task73'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏temporal_shift 算子 TRT Layer 映射实现。
  - 目标：完成 temporal_shift 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 temporal_shift TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 temporal_shift 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.74：为 Paddle-TRT 添加 grid_sampler 算子 <a name='task74'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 grid_sampler 算子 TRT Layer 映射实现，TRT 8.5 已提供 IGridSampleLayer 实现，基于该 Layer 完成 OP 映射工作。
  - 目标：完成 grid_sampler 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 grid_sampler TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 temporal_shift 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.75：为 Paddle-TRT 添加 expand_as_v2 算子 <a name='task75'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 expand_as_v2 算子 TRT Layer 映射实现。
  - 目标：完成 expand_as_v2 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 expand_as_v2 TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 expand_as_v2 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.76：为 Paddle-TRT 添加elementwise_mod 算子 <a name='task76'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 elementwise_mod 算子 TRT Layer 映射实现。
  - 目标：完成 elementwise_mod 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 elementwise_mod TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 elementwise_mod 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.77：为 Paddle-TRT 添加 bitwise_and 算子 <a name='task77'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：基础
- 详细描述：
  - 背景：Paddle-TRT 缺乏 bitwise_and 算子 TRT 映射实现。
  - 目标：完成 bitwise_and 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 bitwise_and TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 bitwise_and 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.78：为 Paddle-TRT 添加 cumsum 算子 <a name='task78'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：进阶
- 详细描述：
  - 背景：Paddle-TRT 缺乏 cumsum 算子 TRT Layer 映射实现。
  - 目标：完成 cumsum 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 cumsum TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 cumsum 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化

### No.79：为 Paddle-TRT 添加 while 算子 <a name='task79'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：进阶
- 详细描述：
  - 背景：Paddle-TRT 缺乏 while 算子 TRT Layer 映射实现。
  - 目标：完成 while 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 while TRT 算子映射，并提交 PR
  - 任务要求：
	    - 完成 while 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉TensorRT，熟悉推理优化

### No.80：为 Paddle-TRT 添加 conditional_block 算子 <a name='task80'></a>

- 技术标签：深度学习框架，C++，推理优化，GPU
- 任务难度：进阶
- 详细描述：
  - 背景：Paddle-TRT 缺乏 conditional_block 算子TRT Layer 映射实现。
  - 目标：完成 conditional_block 算子 TRT Layer 映射
  - PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)。
- 任务提交：
  - 完成 conditional_block TRT算子映射，并提交 PR
  - 任务要求：
	    - 完成 conditional_block 功能实现代码
	    - 添加单测，并验证通过
  - 单元测试样例：python/paddle/fluid/tests/unittests/ir/inference/test_trt_convert_silu.py
- 技术要求：
  - 熟练掌握 Python、C++ 代码编写
  - 熟悉 TensorRT，熟悉推理优化



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 合入标准

- 完成功能实现代码，添加单测，并验证通过
- PR 参考示例见 [PR47820](https://github.com/PaddlePaddle/Paddle/pull/47820)、[代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，历史开发任务参考 [[tracking issue\] Paddle-TensorRT 算子开发](https://github.com/PaddlePaddle/Paddle/issues/48292)

### 技术要求

- 熟练掌握 Python、C++ 代码编写
- 熟悉 TensorRT，熟悉推理优化

### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&QQ群的通知，及时参与。
