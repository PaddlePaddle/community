# paddle.pca_lowrank 设计文档

| API 名称     | paddle.pca_lowrank` / `paddle.sparse.pca_lowrank |
| ------------ |----------------------------------------------|
| 提交作者     | [Sblue-G](https://github.com/Sblue-G)        |
| 提交时间     | 2023-03-21                                   |
| 版本号       | V1.0                                         |
| 依赖飞桨版本 | develop                                      |
| 文件名       | 20230314_api_design_for_pca_lowrank.md       |

# 一、概述

## 1、相关背景

Paddle 需要扩充 API：paddle.pca_lowrank，paddle.sparse.pca_lowrank。

## 2、功能目标

实现 `pca_lowrank` API，对低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析 (PCA)。

## 3、意义

对低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析 (PCA)。

# 二、飞桨现状

飞桨目前没有直接提供此API，但飞桨提供了实现低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析的工具和模块。

对于低秩矩阵PCA和批次低秩矩阵PCA，可以现存的paddle API组合实现

对于稀疏矩阵PCA，可以使用paddle现存的paddle sparse API组合实现

# 三、业内方案调研

在百度中搜索“pytorch pca_lowrank”、“numpy pca_lowrank”和“tensorflow pca_lowrank”，发现PyTorch中有pca_lowrank这个API，而TensorFlow和numpy中还没有。

## PyTorch

### 实现解读

在PyTorch中，pca_lowrank是由python实现的，核心代码为：

```python
def pca_lowrank(
    A: Tensor, q: Optional[int] = None, center: bool = True, niter: int = 2
) -> Tuple[Tensor, Tensor, Tensor]:

    if not torch.jit.is_scripting():
        if type(A) is not torch.Tensor and has_torch_function((A,)):
            return handle_torch_function(
                pca_lowrank, (A,), A, q=q, center=center, niter=niter
            )

    (m, n) = A.shape[-2:]

    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(
            "q(={}) must be non-negative integer"
            " and not greater than min(m, n)={}".format(q, min(m, n))
        )
    if not (niter >= 0):
        raise ValueError("niter(={}) must be non-negative integer".format(niter))

    dtype = _utils.get_floating_dtype(A)

    if not center:
        return _svd_lowrank(A, q, niter=niter, M=None)

    if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
        # reshape c
        column_indices = c.indices()[0]
        indices = torch.zeros(
            2,
            len(column_indices),
            dtype=column_indices.dtype,
            device=column_indices.device,
        )
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(
            indices, c.values(), (n, 1), dtype=dtype, device=A.device
        )

        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        M = _utils.transpose(torch.sparse.mm(C_t, ones_m1_t))
        return _svd_lowrank(A, q, niter=niter, M=M)
    else:
        C = A.mean(dim=(-2,), keepdim=True)
        return _svd_lowrank(A - C, q, niter=niter, M=None)
```
参数分析：

A ( Tensor ) – 大小的输入张量(*,m,n)
q ( int , optional ) – 稍微高估的排名A。默认情况下，q = min(6,m,m)
center ( bool , optional ) -- 如果为真，则将输入张量居中，否则，假设输入居中。
niter ( int , optional ) -- 要进行的子空间迭代次数；niter 必须是非负整数，默认为2。

返回一个命名元组 (U, S, V) ，它是中心矩阵的奇异值分解的最佳近似A.
U是m×q矩阵
S is q-vector
V为n×q矩阵

#### Tensorflow

Tensorflow没有直接api，但是可以使用tf.linalg.svd函数来计算矩阵的奇异值分解（SVD），从而实现PCA。对于低秩矩阵、批次低秩矩阵或稀疏矩阵，可以使用不同的方式实现PCA。

下面是一个示例代码，演示如何使用tensorflow对一个低秩矩阵进行PCA：
```python
import tensorflow as tf
#构造一个低秩矩阵
low_rank_matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)
# 计算SVD
s, u, v = tf.linalg.svd(low_rank_matrix)
# 取前两个奇异值对应的奇异向量作为新的基
new_basis = tf.transpose(u[:, :2])
# 将数据投影到新的基上
projected_data = tf.matmul(new_basis, tf.transpose(low_rank_matrix))
# 输出结果
print("低秩矩阵：")
print(low_rank_matrix.numpy())
print("投影后的数据：")
print(projected_data.numpy())
```
下面是一个示例代码，演示如何使用tensorflow对一个批次低秩矩阵进行PCA：
```python
import tensorflow as tf
# 构造一个批次低秩矩阵
batch_size = 2
low_rank_matrix1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)
low_rank_matrix2 = tf.constant([[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]], dtype=tf.float32)
batch_low_rank_matrix = tf.stack([low_rank_matrix1, low_rank_matrix2])
# 计算SVD
s, u, v = tf.linalg.svd(batch_low_rank_matrix)
# 取前两个奇异值对应的奇异向量作为新的基
new_basis = tf.transpose(u[:, :, :2], perm=[0, 2, 1])
# 将数据投影到新的基上
projected_data = tf.matmul(new_basis, tf.transpose(batch_low_rank_matrix, perm=[0, 2, 1]))
# 输出结果
print("批次低秩矩阵：")
print(batch_low_rank_matrix.numpy())
print("投影后的数据：")
print(projected_data.numpy())
```
下面是一个示例代码，演示如何使用tensorflow对一个稀疏矩阵进行PCA：
```python
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
# 生成一个 5x5 的随机稀疏矩阵
data = np.random.rand(5)
row = np.random.randint(0, 5, size=5)
col = np.random.randint(0, 5, size=5)
sparse_matrix = coo_matrix((data, (row, col)), shape=(5, 5))
# 将稀疏矩阵转换为Tensorflow稠密张量
dense_matrix = tf.sparse.to_dense(sparse_matrix)
# 计算PCA
pca = tf.linalg.eigh(tf.linalg.matmul(dense_matrix, dense_matrix, transpose_a=True))
```
#### Numpy

Numpy没有直接api，但是NumPy中包含了许多常用的数学工具，其中包括了PCA（Principal Component Analysis，主成分分析）的实现。
对于低秩矩阵和批次低秩矩阵的PCA，可以使用NumPy中的linalg.svd函数来实现。linalg.svd函数可以对一个矩阵进行奇异值分解（Singular Value Decomposition，SVD），并返回其左奇异向量、奇异值以及右奇异向量。
下面是一个示例代码，演示如何使用NumPy对一个低秩矩阵进行PCA：
```python
import numpy as np
# 生成一个低秩矩阵
X = np.random.rand(100, 10)
X[:, 5:] = 0
# 对矩阵进行PCA
U, s, V = np.linalg.svd(X)
pc = U[:, :2]
# 将数据投影到主成分上
proj = np.dot(X, pc)
```
对于稀疏矩阵的PCA，可以使用scikit-learn库中的TruncatedSVD类来实现。TruncatedSVD类实现了对稀疏矩阵的奇异值分解，可以用于降维、特征提取等任务。
下面是一个示例代码，演示如何使用TruncatedSVD对一个稀疏矩阵进行PCA：
```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random
# 生成一个稀疏矩阵
X = random(1000, 100, density=0.01)
# 对矩阵进行PCA
svd = TruncatedSVD(n_components=2)
svd.fit(X)
pc = svd.components_
# 将数据投影到主成分上
proj = svd.transform(X)
```
需要注意的是，在使用TruncatedSVD对稀疏矩阵进行PCA时，需要将稀疏矩阵转换为压缩稀疏列（Compressed Sparse Column，CSC）格式或压缩稀疏行（Compressed Sparse Row，CSR）格式。这可以通过SciPy库中的to_csc或to_csr函数来实现。

# 四、对比分析

Tensorflow, Numpy虽然对于稠密张量支持isnan操作，但都没有直接实现对稀疏张量的 isnan　的一元操作。
在　PaddlePaddle　中可以参考Ｐytorch将对这个算子操作进行支持。
Tensorflow, Numpy虽然可以分别对低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析 (PCA)，但都没有直接实现pca_lowrank的一元操作。
在paddlepaddle中可以参考pytorch对这个操作进行支持。

paddle.pca_lowrank API 的设计主要参考 PyTorch 中的实现，PyTorch 中`pca_lowrank`具体逻辑如下：

- 验证输入参数
- 对数据进行中心化
- 使用torch.svd方法计算协方差矩阵的低秩分解
- 计算投影矩阵
- 将数据投影到低维空间
- 返回输出元组

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.pca_lowrank(x, q=None, center=True, niter=2, name=None)` 
参数说明如下：

- **x** (Tensor) – 输入张量的尺寸，一般为`(*, m, n)`
- **q** (int, optional) – 稍微高估的x矩阵的秩，默认情况下q = min(6,m,n)
- **center** (bool, optional) – 如果为真，则将输入张量居中
- **niter** (int, optional) –  要进行的子空间迭代次数；niter 必须是非负整数，默认为2

返回：低秩矩阵或批低秩矩阵的奇异值分解的结果U、S、V。

`paddle.sparse.pca_lowrank(x, q=None, center=True, niter=2, name=None)` 
参数说明如下：

- **x** (Tensor) – 输入张量的尺寸，一般为`(m, n)`
- **q** (int, optional) – 稍微高估的x矩阵的秩，默认情况下q = min(6,m,n)
- **center** (bool, optional) – 如果为真，则将输入张量居中
- **niter** (int, optional) –  要进行的子空间迭代次数；niter 必须是非负整数，默认为2

返回：稀疏矩阵的奇异值分解的结果U、S、V。

## 底层 OP 设计

使用 paddle现有 API 进行设计，不涉及底层OP新增与开发。

## API 实现方案
这个方法的目的是对低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析（PCA）。

- 导入相关包
```
import paddle
from paddle import Tensor
from typing import Optional
```

- 对数据进行中心化
```
    if center:
        A -= paddle.mean(A, axis=0)
```

- 计算矩阵A的SVD
`U, S, V = paddle.svd(A)`

- 计算低秩近似
```
    if q is None:
        # 如果未给出 q，则使用所有奇异值
        q = min(A.shape)
    Ur = U[:, :q]
    Sr = paddle.diag(S[:q])
    Vr = V[:, :q].T
    Ar = Ur @ Sr @ Vr
```

- 迭代以改进近似:
```
    for i in range(niter):
        Ap = Ar
        B = A @ Vr @ Sr.inverse()
        Ur, S, V = paddle.svd(B)
        Ur = Ur[:, :q]
        Sr = paddle.diag(S[:q])
        Vr = V[:, :q].T
        Ar = Ur @ Sr @ Vr
        if paddle.norm(Ap - Ar) / paddle.norm(Ar) < 1e-6:
            break
```

- 最后，这个方法返回元组`Ar`。

# 六、测试和验收的考量

1. 结果正确性:

   - 这个函数的测试和验收的目标是确保它能对低秩矩阵、批次低秩矩阵或稀疏矩阵进行线性主成分分析(PCA)。
   - 前向计算: `paddle.pca_lowrank`计算结果与 `torch.pca_lowrank` 计算结果一致。
   - 反向计算:由 Python 组合新增 API 无需验证反向计算。
2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

# 七、可行性分析和排期规划

方案主要依赖现有 paddle api 组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无