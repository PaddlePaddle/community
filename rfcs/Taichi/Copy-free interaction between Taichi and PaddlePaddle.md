# Taichi 和 PaddlePaddle 之间的无拷贝交互设计文档

|Paddle Friends | Taichi |
|---|---|
|提交作者 | 0xzhang |
|提交时间 | 2022-03-30 |
|版本号 | V0.1 |
|飞桨版本 | paddlepaddle-gpu 2.2.2 |
|太极版本 | taichi 0.9.1 |
|文件名 | Copy-free interaction between Taichi and PaddlePaddle.md |

# 一、概述

深度学习框架如 PaddlePaddle 与 Taichi 的结合使用，可以使开发新的 op 更加高效快速。得益于 Paddle Tensor 的内部由一块连续内存实现，Taichi 和 Paddle 之间可以实现同硬件上的无拷贝交互，即 Taichi kernel 直接操作 PaddlePaddle 。Tensor 所在的内存地址，既实现了在同一个 Python 程序中 Taichi 和 PaddlePaddle 的数据交互，又避免了两个框架间的数据拷贝。

# 二、飞桨现状
`Tensor` 是 Paddle 中最为基础的数据结构。

Paddle 中的 `Tensor` 和 Taichi 中的  `Fields` 概念相近，类似于 Numpy 中的 `ndarray` 或 PyTorch 中的 `tensor`。尽管应用场景不一定完全相同，都可以用来表达多维的数组。


# 三、业内方案调研
根据任务说明中提供的参考，Taichi 支持与 Pytorch 中的 `tensor` 交互，而 Paddle 中提供了获取 `Tensor` 地址的接口 `_ptr()`，与 Pytorch 中的 `data_ptr()`等价。

# 四、设计思路与实现方案

以下为目前整理出的 Taichi 中  PyTorch 的相关实现位置。参考 Pytorch 交互的实现，在相应位置添加代码，实现对 Paddle 的无拷贝交互支持。

## 实现

1. **python\taichi\lang\util.py**
   1. to_pytorch_type()
   2. to_taichi_type()
   3. has_pytorch()
2. **python\taichi\lang\kernel_impl.py**
   1. get_torch_callbacks()
   2. get_function_body()
   3. match_ext_arr()
3. python\taichi\lang\matrix.py
   1. to_torch()
4. python\taichi\lang\field.py
   - **ScalarField**:
     1. to_torch()
   - **Field**
     1. to_torch(): **NotImplementedError**
     2. from_torch()
5. python\taichi\lang\mesh.py
   - **MeshElementField**
     1. to_torch()
     2. from_torch()
6. python\taichi\lang\struct.py
   - **StructField**
     1. to_torch()
     2. from_torch()

## 测试

1. **tests\python\test_torch_io.py**
2. tests\python\test_get_external_tensor_shape.py
   - test_get_external_tensor_shape_access_torch()
3. tests\python\test_f16.py
   - test_to_torch()
   - test_from_torch()
4. tests\python\test_api.py
   - user_api[**ti**.**Field**]: 'from_torch',  'to_torch'
   - user_api[**ti**.**MatrixField**]: 'from_torch',  'to_torch'
   - user_api[**ti**.**ScalarField**]: 'from_torch',  'to_torch'
   - user_api[**ti**.**StructField**]: 'from_torch',  'to_torch'

## 文档

1. docs\lang\articles\basic\external.md
   - 支持 Paddle 的 `Tensor`
2. docs\lang\articles\misc\global_settings.md
   - 引入 `TI_ENABLE_PADDLE`

# 五、测试和验收的考量
- 添加交互 API 如 from_paddle/to_paddle 的文档
- 添加对交互 API 的测试

# 六、可行性分析和排期规划

根据现有参考，非常可行。

在 RFC 确认后，尽快开始开发。

# 七、附件及参考资料

1. [Tensor-API文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#tensor)
2. [Taichi 与 torch tensor 交互的实现细节](https://github.com/taichi-dev/taichi/blob/master/python/taichi/lang/kernel_impl.py#L570-L572)
3. [Paddle Paddle 中与 torch Tensor.data_ptr() 等价的_ptr() 样例](https://github.com/PaddlePaddle/Paddle/blob/24b2e8e6c84ec6e75f561c51f170faf76ec70374/python/paddle/fluid/tests/unittests/test_tensor.py#L29)
