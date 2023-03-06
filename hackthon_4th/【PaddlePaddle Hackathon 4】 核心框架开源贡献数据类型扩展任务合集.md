# 【PaddlePaddle Hackathon 4】核心框架开源贡献数据类型扩展任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/50629)）

注：为飞桨框架一系列算子增加支持的数据类型，提交流程请参考 [算子数据类型扩展&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/op_dtype_extension/op_dtype_extension_contributing_guides_cn.html) & [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)， 任务列表如下：

### No.45：为 Paddle logical 算子实现 float16 数据类型支持 <a name='task45'></a>

- 技术标签：深度学习框架，C++
- 任务难度：基础
- 详细描述：logical 类的算子未支持 float16 类型，因此该功能要求为这类算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。本任务共包含 4 个具体算子：logical_and，logical_or，logical_xor，logical_not
- 任务提交：
  - C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/kps/logical_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/kps/logical_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_logical_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_logical_op.py)；
  - API文档中增加数据类型支持说明，以logical_and为例英文文档在[python/paddle/tensor/logic.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/logic.py)，中文文档在[docs/api/paddle/logical_and_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/logical_and_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.46：为 Paddle gumbel_softmax 算子实现 float16 数据类型支持 <a name='task46'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：gumbel_softmax 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/gumbel_softmax_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/gumbel_softmax_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/gumbel_softmax_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/gumbel_softmax_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_gumbel_softmax_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_gumbel_softmax_op.py)；
  - API文档中增加数据类型支持说明，英文文档在[python/paddle/nn/functional/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/activation.py)，中文文档在[docs/api/paddle/nn/functional/gumbel_softmax_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/functional/gumbel_softmax_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.47：为 Paddle cross 算子实现 float16 数据类型支持 <a name='task47'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：cross 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/cross_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/cross_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/cross_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/cross_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_cross_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_cross_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在 [python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py)，中文文档在 [docs/api/paddle/cross_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/cross_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.48：为 Paddle assign_value、meshgrid、kthvalue、determinant 算子实现 float16 数据类型支持 <a name='task48'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：assign_value 前向，meshgrid、kthvalue、determinant 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/assign_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/assign_kernel.cc)、[paddle/phi/kernels/gpu/meshgrid_kernel.cu.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/meshgrid_kernel.cu.cc)、[paddle/phi/kernels/gpu/kthvalue_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/kthvalue_kernel.cu)、[paddle/phi/kernels/gpu/determinant_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/determinant_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/meshgrid_grad_kernel.cu.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/meshgrid_grad_kernel.cu.cc)、[paddle/phi/kernels/gpu/kthvalue_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/kthvalue_grad_kernel.cu)、[paddle/phi/kernels/gpu/determinant_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/determinant_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_assign_value_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_assign_value_op.py)、[python/paddle/fluid/tests/unittests/test_meshgrid_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_meshgrid_op.py)、[python/paddle/fluid/tests/unittests/test_kthvalue_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_kthvalue_op.py)、[python/paddle/fluid/tests/unittests/test_determinant_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_determinant_op.py)；
  - API文档中增加数据类型支持说明，英文文档在[python/paddle/tensor/creation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py)、[python/paddle/tensor/search.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/search.py)、[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py)，中文文档在[docs/api/paddle/assign_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/assign_cn.rst)、[docs/api/paddle/meshgrid_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/meshgrid_cn.rst)、[docs/api/paddle/kthvalue_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/kthvalue_cn.rst)、[docs/api/paddle/linalg/det_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/linalg/det_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.49：为 Paddle bce_loss 算子实现 float16 数据类型支持 <a name='task49'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：bce_loss 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/bce_loss_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/bce_loss_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/bce_loss_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/bce_loss_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_bce_loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_bce_loss.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/nn/layer/loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/loss.py)，中文文档在[docs/api/paddle/nn/BCELoss_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/BCELoss_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.50：为 Paddle lerp 算子实现 float16 数据类型支持 <a name='task50'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：lerp 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，同时改写算子实现，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/lerp_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/lerp_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/lerp_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/lerp_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_lerp_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_lerp_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在 [python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)，中文文档在 [docs/api/paddle/lerp_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/lerp_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.51：为 Paddle maxout 算子实现 float16 数据类型支持 <a name='task51'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：maxout 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/maxout_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/maxout_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/maxout_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/maxout_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_maxout_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_maxout_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/nn/functional/activation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/activation.py)，中文文档在[docs/api/paddle/nn/functional/maxout_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/functional/maxout_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.52：为 Paddle dist 算子实现 float16 数据类型支持 <a name='task52'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：dist 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/dist_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/dist_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/dist_grad_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/dist_grad_kernel.cc)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_dist_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_dist_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/tensor/linalg.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py)，中文文档在[docs/api/paddle/dist_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/dist_cn.rst)； 
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.53：为 Paddle label_smooth 算子实现 float16 数据类型支持 <a name='task53'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：label_smooth 前反向算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - 前向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/label_smooth_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/label_smooth_kernel.cu)；
  - 反向算子：C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/label_smooth_grad_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/label_smooth_grad_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_label_smooth_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_label_smooth_op.py)；
  - API 文档中增加数据类型支持说明，英文文档在[python/paddle/nn/functional/common.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/python/paddle/nn/functional/common.py)，中文文档在[docs/api/paddle/nn/functional/label_smooth_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/nn/functional/label_smooth_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.54：为 Paddle allclose、isclose 算子实现 float16 数据类型支持 <a name='task54'></a>

- 技术标签：深度学习框架，C++，CUDA
- 任务难度：基础
- 详细描述：allclose、isclose 算子未支持 float16 类型，因此该功能要求为该算子注册 float16 类型，使得 float16 下精度与期望结果的误差不超过 1e-3，并且性能不差于使用 float32 类型计算。
- 任务提交：
  - C++ 及 GPU kernel 实现代码在 [paddle/phi/kernels/gpu/allclose_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/allclose_kernel.cu)、[paddle/phi/kernels/gpu/isclose_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/isclose_kernel.cu)；
  - 单元测试脚本中增加测试样例，在 [python/paddle/fluid/tests/unittests/test_allclose_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_allclose_op.py)、[python/paddle/fluid/tests/unittests/test_isclose_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/python/paddle/fluid/tests/unittests/test_isclose_op.py)；
  - API文档中增加数据类型支持说明，英文文档在[python/paddle/tensor/logic.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/logic.py)，中文文档在[docs/api/paddle/allclose_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/allclose_cn.rst)、[docs/api/paddle/isclose_cn.rst](https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/isclose_cn.rst)；
  - 在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范参考 [示例](https://github.com/PaddlePaddle/Paddle/pull/30380)，需要分别提供 float32 和 float16 下 2 种类型的性能对比。

### No.55：channel_shuffle 等算子FP16/BF16算子及单测完善 <a name='task55'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 argsort、channel_shuffle、pixel_unshuffle、pixel_shuffle、fmax、fmin、erf、erfinv 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在**任务表单**中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.56：set_value 等算子 FP16/BF16算子及单测完善 <a name='task56'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 fill、fill_diagonal_tensor、diag、diagonal、bernoulli、poisson、trunc、searchsorted 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.57：gaussian 等算子 FP16/BF16算子及单测完善 <a name='task57'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 gaussian、cross、dot、conv3d、conv3d_transpose、max_pool2d_with_index、max_pool3d_with_index 、flip算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.58：linear_interp 等算子 FP16/BF16算子及单测完善 <a name='task58'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 logcumsumexp、logsumexp、empty、empty_like、kthvalue、exponential 、atan2、set_value、pad算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.59：addmm 等算子 FP16/BF16算子及单测完善 <a name='task59'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 addmm、bmm、angle、put_along_axis、take_along_axis、index_sample、index_add、hardtanh 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.60：angle 等算子 FP16/BF16算子及单测完善 <a name='task60'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 prelu、multinomial、multi_dot、overlap_add、clip_by_norm、randperm、sign、split、split_with_num 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.61：unfold 等算子 FP16/BF16算子及单测完善 <a name='task61'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 unfold、min、mode、logsigmoid、uniform_inplace、segment_pool、update_loss_scaling、remainder 算子的FP16算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.62：masked_select 等算子 FP16/BF16算子及单测完善 <a name='task62'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 masked_select、dirichlet、lgamma、dropout_nd、digamma、margin_cross_entropy、broadcast_tensors 、pool3d、transfer_layout算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.63：complex 等算子 FP16/BF16算子及单测完善 <a name='task63'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 complex、eye、lerp、frame、embedding、nanmedian、temporal_shift、conj 算子的 FP16 算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.64：trace 等算子 FP16/BF16算子及单测完善 <a name='task64'></a>

- 任务难度：基础
- 详细描述：
  - 参照 FP16 算子开发文档和 FP16 单测开发文档对 trace、elementwise_heaviside、huber_loss、logspace、full_batch_size_like、unbind、einsum、matmul_with_flatten、trace 算子的FP16算子实现和单测的修改。
  - 各算子的对应单测文件及需要完成的任务在任务表单中给出
  - 保证算子正确性和单测正确性，提交代码至 Paddle 代码仓库
- 文档参考：
  - [低精度算子及单测完善任务表单](https://shimo.im/sheets/RKAWVnVNopC1NKk8/Z03pH)
  - [低精度算子开发贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/amp_precision/index_cn.html)

### No.65：为 Paddle matmul_with_flatten/matmul 前向算子实现 int8 数据类型支持 <a name='task65'></a>

- 技术标签：深度学习框架，C++，CUDA, 量化
- 任务难度：基础
- 详细描述：为了更好的支持量化训练和原生量化推理，需要对 matmul 和matmul_with_flatten 算子添加 int8 数据类型的支持。推荐使用框架内接口调用 cublasLt 作为 int8 矩阵乘的计算后端并根据实际情况适配，也可使用其他方式实现。要求精度无误，算子性能在常见 case 下快于 FP16。
- 提交内容：
  - 修改前向算子注册: paddle/phi/kernels/gpu/matmul_kernel.cu
  - 修改前向算子GPU kernel：paddle/phi/kernels/impl/matmul_kernel_impl.h
  - 修改 cublasLt int8 计算后端：paddle/fluid/operators/fused/cublaslt.h
  - 单元测试脚本中增加测试样例：
     - python/paddle/fluid/tests/unittests/test_mul_op.py
     - python/paddle/fluid/tests/unittests/test_matmul_op.py
     - python/paddle/fluid/tests/unittests/test_matmul_v2_op.py 
  - 提交PR，在PR描述中说明做法并附上FP16/32的性能对比
- 技术要求：
  - 熟练掌握 Python、C++、CUDA 代码编写
  - 了解量化原理和量化推理执行流程

### No.66：为Paddle FC 前向算子实现量化计算 <a name='task66'></a>

- 技术标签：深度学习框架，C++，CUDA, 量化
- 任务难度：基础
- 详细描述：为了更好的支持原生量化推理，需要对FC算子添加量化支持。算子的 Input/Bias/Out 类型保持不变，要求W支持int8输入；需要额外添加量化kernel，反量化 kernel，和 int8 类型的矩阵乘运算；要求为OP添加量化开关作为算子属性，同时添加属性 max_bound/min_bound, round_type；算子已有quant_in_scale 属性，可以直接使用该属性作为量化输入的参数(layer_wise)，添加 weight_scale(channel_wise) 作为输入Tensor用于计算输出的反量化参数。INT8矩阵乘推荐使用框架内接口调用 cublasLt 作为int8矩阵乘的计算后端并根据实际情况适配，也可使用其他方式实现。要求精度无误，算子性能在常见 case 下快于 FP16。
- 提交内容：
  - 修改前向算子声明: paddle/fluid/operators/fc_op.cc
  - 修改前向算子GPU kernel：paddle/fluid/operators/fc_op.h
  - 修改cublasLt int8计算后端：paddle/fluid/operators/fused/cublaslt.h
  - 单元测试脚本中增加测试样例：python/paddle/fluid/tests/unittests/test_fc_op.py
  - 提交PR，在PR描述中说明做法并附上 FP16/32 的性能对比
- 技术要求：
  - 熟练掌握 Python、C++、CUDA 代码编写
  - 熟悉量化原理和量化推理执行流程
  - 了解 Paddle 算子体系



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～


### 合入标准

- float16 数据类型支持
  - 精度：依据各任务描述为算子扩展数据类型，通过算子的单元测试；
  - 性能：OP Benchmark 对应 OP 的全部配置 case 性能不出现下降问题，使用 float16 类型的算子性能不差于使用 float32 类型。
- FP16/BF16算子及单测完善
	- 精度：依据单测添加规范，设置或修改FP16单测的精度误差阈值，使之尽可能精确且通过算子的所有单元测试
	- 测例数量：FP16的测例数量不少于FP32的测例
- int8 数据类型支持 
	- 精度：在单元测试中添加模拟量化过程，使算子通过单元测试
	- 性能：在常见case下性能高于FP16

### 技术要求

- 熟练掌握 Python、C++、CUDA 代码编写；
- 掌握 OP Benchmark 使用方法；
- 熟悉量化原理和量化推理执行流程

### 答疑交流
* 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流；
* 对于开发中的共性问题，在活动过程中，会定期组织答疑，请大家关注官网&微信群的通知，及时参与。