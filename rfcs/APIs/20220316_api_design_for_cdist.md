# paddle.cdist 设计文档

|API名称 | paddle.cdist | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 汪昕([GreatV](https://github.com/GreatV)) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-02 | 
|版本号 | V2.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20200316_api_design_for_cdist.md | 


# 一、概述
## 1、相关背景

`cdist` 用于计算两组行向量集合中每一对之间的 `p-norm` 距离。

## 2、功能目标

为 Paddle 新增 cdist API。dist API 用于计算两个输入 Tensor 的 p 范数（p-norm），计算结果为形状为 [] 的 Tensor，而 cdist API 则用于计算两个输入 Tensor 的所有行向量对的 p 范数（p-norm），输出结果的形状和两个 Tensor 乘积的形状一致。

## 3、意义

为 Paddle 新增 cdist API，提供距离计算功能。

# 二、飞桨现状

对飞桨框架目前不支持此功能，可用其他API组合实现的此功能，对于距离度量为欧氏距离（p=2），代码如下；

```Python
import paddle


def pairwise_dist (A, B):
    A2 = (A ** 2).sum(axis=1).reshape((-1, 1))
    B2 = (B ** 2).sum(axis=1)
    D = A2 + B2 - 2 * A.mm(B.transpose((1, 0)))
    D = D.sqrt()
    return D

a = paddle.to_tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = paddle.to_tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
pairwise_dist(a, b)


Tensor(shape=[3, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[3.11927032, 2.09589314],
        [2.71384096, 3.83217263],
        [2.28300953, 0.37910157]])
```

# 三、业内方案调研

## 1. Scipy

在 Scipy 中使用的 API 格式如下：

`scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)`

上述函数参数中，`metric` 表示距离度量方式。Scipy 支持丰富的距离度量方式，如：'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation','cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon','kulsinski', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski','rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener','sokalsneath', 'sqeuclidean', 'yule' 等。

实现的伪代码如下：

```Python
def cdist(XA, XB, metric, **kwargs):
    mA = XA.shape[0]
    mB = XB.shape[0]
    for i in range(mA):
        for j in range(mB):
            dm[i, j] = metric(XA[i], XB[j], **kwargs)
    return dm
```

## 2. MatLab

在 MatLab 中使用的 API 格式如下：

`D = pdist2(X,Y,Distance)`

在 MatLab 版中，`Distance` 表示距离度量方式。MatLab 同样支持丰富的距离度量方式，如：'euclidean'，'squaredeuclidean', 'seuclidean', 'mahalanobis', 'cityblock', 'minkowski' 等等。

实现 `minkowski` 距离的伪代码如下：

```matlab
function [D,I] = pdist2(X,Y,dist,varargin)
    additionalArg = varargin{1};
    outClass = 'double';

    [nx,p] = size(X);
    [ny,py] = size(Y);
    
    expon = additionalArg;
    for i = 1:ny
        dpow = zeros(nx,1,outClass);
        for q = 1:p
            dpow = dpow + abs(X(:,q) - Y(i,q)).^expon;
        end
        dpow = dpow .^ (1./expon);
```
## 3. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')`

在 Pytorch 中， 如果 p∈[1, ∞)，则函数与 `scipy.spatial.distance.cdist(input,'minkowski', p=p)` 等价；如果 p=0，则函数与 `scipy.spatial.distance.cdist(input, 'hamming') * M` 等价；如果 p=∞，则函数与 `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())` 意义最接近。

## 4. Tensorflow

没有对应的API, 可以用其他 API 组合实现。[#issues 30659](https://github.com/tensorflow/tensorflow/issues/30659)

```Python
def pairwise_dist (A, B):  
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
    A,    [m,d] matrix
    B,    [n,d] matrix
    Returns:
    D,    [m,n] matrix of pairwise distances
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D
```

# 四、对比分析

## 1. 不同框架API使用方式

### 1. Scipy

```Python
from scipy.spatial import distance
import numpy as np

a = np.array([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = np.array([[-2.1763, -0.4713], [-0.6986, 1.3702]])

distance.cdist(a, b, "euclidean")


array([[3.11927026, 2.09589304],
       [2.71384068, 3.83217237],
       [2.28300936, 0.37910116]])
```

### 2. MatLab

```matlab
a = [0.9041, 0.0196; -0.3108, -2.4423; -0.4821, 1.059];
b = [-2.1763, -0.4713; -0.6986, 1.3702];
c = pdist2(a, b)


c =

    3.1193    2.0959
    2.7138    3.8322
    2.2830    0.3791
```

### 3. PyTorch

```Python
import torch
​
a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
torch.cdist(a, b, p=2)
​
tensor([[3.1193, 2.0959],
        [2.7138, 3.8322],
        [2.2830, 0.3791]])
```

### 4. Tensorflow

```Python
import tensorflow as tf
​
def pairwise_dist (A, B):  
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
    A,    [m,d] matrix
    B,    [n,d] matrix
    Returns:
    D,    [m,n] matrix of pairwise distances
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
​
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
​
    # return pairwise euclidean difference matrix
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D
​
a = tf.constant([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = tf.constant([[-2.1763, -0.4713], [-0.6986, 1.3702]])
pairwise_dist(a, b)
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[3.1192703 , 2.0958931 ],
       [2.7138407 , 3.8321724 ],
       [2.2830095 , 0.37910157]], dtype=float32)>
```

上述框架从使用体验来说，**Scipy** 支持的距离度量方式最多，调用API方便，易于使用，但不支持 Tensor。**MatLab** 同 **Scipy** 也存在相同的问题。**PyTorch** 则支持 Tensor，函数与 `scipy.spatial.distance.cdist(input,'minkowski', p=p)` 等价。**Tensorflow** 则支持 Tensor，但没有此API，需要手动实现，使用较为繁琐。

# 五、设计思路与实现方案

## 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')`。其中 `x` 为 `B1 X ... X Bn X P X M` 张量，`y` 为 `B1 X ... X Bn X R X M` 张量。`p` 为 p-范数对应的 p 值，p ∈[0,∞]。输出张量的形状为 `B1 X ... X Bn X P X R`。`compute_mode` 参数的作用是使用矩阵乘法加速欧氏距离（p=2）的计算。当参数 `compute_mode='use_mm_for_euclid_dist_if_necessary'`，当 P > 25 或 R > 25 时，则使用矩阵乘法加速计算，否则使用普通的欧氏距离计算。

## API实现方案

参考 PyTorch 采用现有 PYTHON API 组合实现，实现位置为 Paddle repo `python/paddle/tensor/linalg.py` 目录。

# 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑一下场景：

1. 当`x`、`y` 为空张量，输出为空张量，且输出张量形状正确；
2. 结果一致性，和 SciPy 以及 PyTorch 结果的数值的一致性, `paddle.cdist(x, y, p=2.0)` , `scipy.spatial.distance.cdist(input,'minkowski', p=2)` 和 `torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')` 结果是否一致；
3. 当 p < 1 时, 确保反向传播不会产生 NaNs；
4. 异常测试，对于参数异常值输入，应该有友好的报错信息及异常反馈，需要有相关测试Case验证。

# 七、可行性分析和排期规划

本 API 主要参考 PyTorch 实现，难度适中，工期上能满足要求。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料
