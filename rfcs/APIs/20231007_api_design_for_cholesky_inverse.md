# paddle.cholesky_inverse 设计文档

| API名称      | Paddle.cholesky_inverse                     |
| ------------ | ------------------------------------------- |
| 提交作者     | jlx                                        |
| 提交时间     | 2023-10-07                                  |
| 版本号       | V1.0                                        |
| 依赖飞桨版本 | develop                                     |
| 文件名       | 20231007_api_design_for_cholesky_inverse.md |


# 一、概述
## 1、相关背景
丰富Paddle相关数学API，支持更多样的矩阵运算

## 2、功能目标
实现`cholesky_inverse` API,使用 Cholesky 因子U计算对称正定矩阵的逆矩阵，调用方式如下:
 - paddle.cholesky_inverse , 作为独立的函数调用
 - Tensor.cholesky_inverse , 作为 Tensor 的方法使用

## 3、意义
提升飞桨API丰富度，支持矩阵计算类API，对应 PyTorch 的`cholesky_inverse` API 操作

# 二、飞桨现状
飞桨目前有`cholesky`和`cholesky_solve` API分别用于计算cholesk分解和线性方程的求解，但相较于Pytorch仍缺少`cholesky_inverse` API


# 三、业内方案调研

**pytorch**

API为：`torch.cholesky_inverse`(*input*, *upper=False*, *, *out=None*) → Tensor

PyTorch参数的*用于指定后面参数out为keyword-only参数，Paddle中无此参数

# 四、对比分析
  
  - PyTorch是在C++ API基础上实现，使用Python调用C++对应的接口
  - Tensorflow、Numpy中没有对应API的实现

## Pytorch

### 实现解读

在PyTorch中，cholesky_inverse底层是由C++实现的，核心代码为:
```C++
template <typename scalar_t>
static void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(false, "cholesky_inverse: MAGMA library not found in compilation. Please rebuild with MAGMA.");
#else
  // magmaCholeskyInverse (magma_dpotri_gpu) is slow because internally
  // it transfers data several times between GPU and CPU and calls lapack routine on CPU
  // using magmaCholeskySolveBatched is a lot faster
  // note that magmaCholeskySolve is also slow

  // 'input' is modified in-place we need to clone it and replace with a diagonal matrix
  // for apply_cholesky_solve
  auto input_working_copy = cloneBatchedColumnMajor(input);

  // 'input' tensor has to be a batch of diagonal matrix
  input.fill_(0);
  input.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);

  Tensor result_u, input_u;
  if (input.dim() == 2) {
    // unsqueezing here so that the batched version is used
    result_u = input.unsqueeze(0);
    input_u = input_working_copy.unsqueeze(0);
  } else {
    result_u = input;
    input_u = input_working_copy;
  }

  // magma's potrs_batched doesn't take matrix-wise array of ints as an 'info' argument
  // it returns a single 'magma_int_t'
  // if info = 0 the operation is successful, if info = -i, the i-th parameter had an illegal value.
  int64_t info_tmp = 0;
  apply_cholesky_solve<scalar_t>(result_u, input_u, upper, info_tmp);
  infos.fill_(info_tmp);
#endif
}
```
在前序一系列参数检查后，进入`apply_cholesky_inverse`函数，最终通过调用`apply_cholesky_solve`使用MAGMA例程进行求解



# 五、设计思路与实现方案

## 命名与参数设计
`paddle.cholesky_solve`(*x*, *upper=False*, *name=None*) → Tensor

使用Cholesky分解因子u,计算对称正定矩阵A的逆矩阵inv

- Parameters
    - **x** (Tensor) - 是2维矩阵或者2维矩阵以batch形式组成的3维矩阵,每一个矩阵均为对称正定矩阵。
    - **upper** (bool, 可选) - 输入x是否是上三角矩阵，True为上三角矩阵，False为下三角矩阵。默认值False。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

- Returns
    - Tensor, **x** 的逆矩阵

## 底层OP设计
使用 paddle现有 API 进行设计，不涉及底层OP新增与开发。

## API实现方案
函数实现路径：在`python/paddle/tensor/linalg.py`中增加`cholesky_inverse`函数

单元测试路径：在`test/legacy_test`目录下增加`test_cholesky_inverse.py`

初步实现方案如下：

1) 检查输入x, 对输入Tensor x做broadcast shape。
2) 检查是否需要转置(行优先还是列优先)。
3) 调用paddle cholesky api进行计算。
4) 检查输出, 存储Tensor。

# 六、测试和验收的考量
1. 结果正确性:
   - 前向计算: `paddle.cholesky_inverse`计算结果与 `torch.cholesky_inverse` 计算结果一致。
   - 反向计算:由 Python 组合新增 API 无需验证反向计算。
2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

# 七、可行性分析和排期规划
方案实施难度可控，工期上可以满足在当前版本周期内开发完成

# 八、影响面
为已有 API 的增强，对其他模块无影响。

# 名词解释
无

# 附件及参考资料
[cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)

[torch.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch.cholesky_inverse)
