# Paddle.linalg.cholesky_solve 设计文档

| API名称                                                      | Paddle.linalg.cholesky_solve              |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | 致泊                                    |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-29                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                      |
| 文件名                                                       | 20220329_design_for_cholesky_solve.md<br> |

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，支持矩阵计算类API，Paddle需要扩充API`paddle.linalg.cholesky_solve。
该功能实现解一个具有半正定矩阵的线性方程组，其中半正定矩阵通过提供其Cholesky因子矩阵u实现。
## 2、功能目标
增加API`paddle.linalg.cholesky_solve`，实现对cholesky方程求解。

## 3、意义
飞桨支持API`paddle.linalg.cholesky_solve`

# 二、飞桨现状
目前paddle缺少相关功能实现。

无类似功能API或者可组合实现方案。

# 三、业内方案调研
**一、pytorch**

API为：`torch.cholesky_solve`(*input*, *input2*, *upper=False*, *, *out=None*) → Tensor

PyTorch参数的*用于指定后面参数out为keyword-only参数，Paddle中无此参数



**详细介绍**：

`torch.cholesky_solve`(*input*, *input2*, *upper=False*, *\**, *out=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix u*u*.

If `upper` is `False`, u*u* is and lower triangular and c is returned such that:

c = (u u^T)^{{-1}} b*c*=(*u**u**T*)−1*b*

If `upper` is `True` or not provided, u*u* is upper triangular and c is returned such that:

c = (u^T u)^{{-1}} b*c*=(*u**T**u*)−1*b*

torch.cholesky_solve(b, u) can take in 2D inputs b, u or inputs that are batches of 2D matrices. If the inputs are batches, then returns batched outputs c

Supports real-valued and complex-valued inputs. For the complex-valued inputs the transpose operator above is the conjugate transpose.

- Parameters

  **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input matrix b*b* of size (∗,*m*,*k*), where *∗ is zero or more batch dimensions*

  **input2** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input matrix u*u* of size (∗,*m*,*m*), where *∗ is zero of more batch dimensions composed of upper or lower triangular Cholesky factor*

  **upper** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – whether to consider the Cholesky factor as a lower or upper triangular matrix. Default: `False`.

- Keyword Arguments

  **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – the output tensor for c

Example:

```
>>> a = torch.randn(3, 3)
>>> a = torch.mm(a, a.t()) # make symmetric positive definite
>>> u = torch.cholesky(a)
>>> a
tensor([[ 0.7747, -1.9549,  1.3086],
        [-1.9549,  6.7546, -5.4114],
        [ 1.3086, -5.4114,  4.8733]])
>>> b = torch.randn(3, 2)
>>> b
tensor([[-0.6355,  0.9891],
        [ 0.1974,  1.4706],
        [-0.4115, -0.6225]])
>>> torch.cholesky_solve(b, u)
tensor([[ -8.1625,  19.6097],
        [ -5.8398,  14.2387],
        [ -4.3771,  10.4173]])
>>> torch.mm(a.inverse(), b)
tensor([[ -8.1626,  19.6097],
        [ -5.8398,  14.2387],
        [ -4.3771,  10.4173]])
```

**代码实现：**

cpu端实现：

代码实现路径在https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/BatchLinearAlgebra.cpp#L1244-L1307

```
Tensor cholesky_solve(const Tensor& self, const Tensor& A, bool upper)

其中self为方程右边的矩阵b，矩阵A的系数矩阵
1）检查Tensor输入纬度>2
2）对self、A进行boardcast
3）构造self、A clone矩阵
4）执行lapackCholeskySolve
```

分别调用zpotrs_、cpotrs_、dpotrs_、spotrs_实现不同数据类型，调用lapack库实现


gpu端实现：

```
Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {

1）当batchsize==1或者不使用magma，调用cusolver方法，执行_cholesky_solve_helper_cuda_cusolver
2）若不满足1中条件，调用magma方法，执行_cholesky_solve_helper_cuda_magma
```

**二、tensorflow**

Tensorflow API为：`tf.linalg.cholesky_solve`(  chol, rhs, name=None)

- Parameters

  **chol** ([*Tensor*]) – A Tensor. Must be float32 or float64, shape is [..., M, M]. Cholesky factorization of A

  **rhs** ([*Tensor*]) – A Tensor, same type as chol, shape is [..., M, K].

  **name** ([*bool*]*,* *optional*) – A name to give this Op. Defaults to cholesky_solve.

实现方式：

通过两次调用matrix_triangular_solve实现

```
  with ops.name_scope(name, 'cholesky_solve', [chol, rhs]):
    y = gen_linalg_ops.matrix_triangular_solve(
        chol, rhs, adjoint=False, lower=True)
    x = gen_linalg_ops.matrix_triangular_solve(
        chol, y, adjoint=True, lower=True)
    return x
```

**三、numpy**

numpy无此API

# 四、对比分析

参数upper指示所提供的u是上三角矩阵还是下三角矩阵

PyTorch实现方法与Tensorflow实现方法在Paddle中都可以实现，其中PyTorch调用相应库实现正向计算，利用矩阵操作实现反向计算。

Tensorflow通过调用两次已有API matrix_triangular_solve组合实现，不需实现OP与反向。

综合对比，PyTorch实现虽然复杂，但是一次计算，实现高效。Tensorflow正向、反向都需要调用两次matrix_triangular_solve操作，效率较低。且考虑PyTorch更加主流，优先考虑与PyTorch实现对齐。

numpy无此API

结论：该新增API功能与PyTorch一致，参数设计上与PyTorch一一对应，按Paddle风格命名


# 五、方案设计
## 命名与参数设计
`paddle.linalg.cholesky_solve`(*x*, *y*, *upper=False*, *name=None*) → Tensor

对 A @ X = B 的线性方程求解，其中A是方阵，输入x、y分别是矩阵B和矩阵A的Cholesky分解矩阵u。
输入x、y是2维矩阵，或者2维矩阵以batch形式组成的3维矩阵。如果输入是batch形式的3维矩阵，则输出也是batch形式的3维矩阵。

- Parameters

  For Ax = b,  A=u*u^T
    - **x** (Tensor) - 线性方程中的B矩阵。是2维矩阵或者2维矩阵以batch形式组成的3维矩阵。
    - **y** (Tensor) - 线性方程中A矩阵的Cholesky分解矩阵u，上三角或者下三角矩阵。是2维矩阵或者2维矩阵以batch形式组成的3维矩阵。
    - **upper** (bool, 可选) - 输入x是否是上三角矩阵，True为上三角矩阵，False为下三角矩阵。默认值False。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

- Returns

    - Tensor, 线性方程的解X。


## 底层OP设计
优先考虑实现与PyTorch结果对齐

正向：采用lapack库实现与PyTorch保持一致，调用dpotrs_、spotrs_实现。

反向：由于paddle不支持magma库，采用cusolver实现：cusolverDnpotrfBatched。

详细实现步骤：

1) 对输入Tensor x、y做broadcast shape。
2) 由于所调用库输入矩阵是列优先，paddle输入是行优先，对输入Tensor做转置变换为列优先。
3) Tensor内存按batchsize分解，针对每个输入二维矩阵调用lapack库或者cusolver求解。
4) 输出结果转置转换回行优先存储。

## API实现方案
直接调用cholesky_solve OP实现。

# 六、测试和验收的考量
预期实现效果与scipy保持一致：

1) 测试API 动态图和静态图使用方式精度。
2) 测试CPU、GPU上使用方式精度。
3) 矩阵计算只支持float数据类型，测试float32、float64类型计算精度。
4) 测试x、y不同纬度、大小时计算精度。
5) 测试不同配置条件下，即upper=False、True时计算精度。
6) 测试forward、backward（OP）计算准确性。
7) 测试不支持的Tensor维度、数据类型时异常报错情况。

数据精度与scipy对比，其基础应用方式如下：

```
import paddle
from scipy.linalg import cho_factor, cho_solve
M = 4
N = 3
b = 2
A = np.random.rand(M,N)
b = np.random.rand(M,b)
c, low = cho_factor(A)
x = cho_solve((c, low), b)

px = paddle.cholesky_solve(b, c, (not lower))
np.allclose(x, px)
```

# 七、可行性分析及规划排期

该方案已有实际实现案例，Paddle具备相关依赖环境，切实可行。

# 八、影响面
为独立新增API，对其他模块没有影响



# 名词解释
无
# 附件及参考资料
无