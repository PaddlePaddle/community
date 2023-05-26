# paddle.cdist 设计文档
|API名称 | paddle.cdist                                 | 
|---|----------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | [yangguohao](https://github.com/yangguohao/) |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-05-19                                   | 
|版本号 | V2.0                                         | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                      | 
|文件名 | 20230519_api_design_for_cdist.md             | 


# 一、概述
## 1、相关背景

Issue: 【PaddlePaddle Hackathon 4】2、为 Paddle 新增 cdist API 
`cdist` 用于计算两组行向量集合中每一对之间的 `p-norm` 距离。

## 2、功能目标

为 Paddle 新增 cdist API。dist API 用于计算两个输入 Tensor 的 p 范数（p-norm），计算结果为形状为 [] 的 0-D Tensor，而 cdist API 则用于计算两个输入 Tensor 的所有行向量对的 p 范数（p-norm），输出结果的形状和两个 Tensor 乘积的形状一致。
## 3、意义
为 Paddle 新增 cdist API，提供距离计算功能。
# 二、飞桨现状
对飞桨框架目前不支持此功能，可用其他API组合实现的此功能。
# 三、业内方案调研
## 1. Scipy

在 Scipy 中使用的 API 格式如下：

`scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)`

上述函数参数中，`metric` 表示距离度量方式。

Scipy 支持丰富的距离度量方式，如：'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 
'euclidean', 'hamming', 'jaccard', 'jensenshannon','kulsinski', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener','sokalsneath', 'sqeuclidean', 'yule' 等。


## 2. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')`

计算的方式与 `dist` 算子相似，只是变为 x1 和 x2 各自每一行两两之间计算 dist。
这里的 `compute_mode` 参数，其作用是使用矩阵乘法加速欧氏距离（p=2）的计算。
在 PyTorch 中，参数 `compute_mode='use_mm_for_euclid_dist_if_necessary'`，当 P > 25 或 R > 25 时，则使用矩阵乘法加速计算，否则使用普通的欧氏距离计算。

## 3. Tensorflow

没有对应的API, 对于 dist 距离的计算可以用其他 API 组合实现。

# 四、对比分析

## 1. 不同框架API使用方式

### 1. Scipy

```Python
from scipy.spatial import distance
import numpy as np

a = np.array([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = np.array([[-2.1763, -0.4713], [-0.6986, 1.3702]])

distance.cdist(a, b, "euclidean")


# array([[3.11927026, 2.09589304],
#        [2.71384068, 3.83217237],
#        [2.28300936, 0.37910116]])
```

### 2. PyTorch

```Python
import torch

a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
torch.cdist(a, b, p=2)

# tensor([[3.1193, 2.0959],
#         [2.7138, 3.8322],
#         [2.2830, 0.3791]])
```

### 3. Tensorflow
无对应的 api，需要手动实现。

### 4. 总结
综上所述**Scipy** 对应的计算方法最全面

**Pytorch** 计算的公式与 dist 算子相同，通过不同的参数 p 来控制对应的计算结果。该方法更简洁更友好，对于大部分的情况下足够使用。

**Tensorflow** 没有此API，需要手动实现，使用较为繁琐。



# 五、设计思路与实现方案
## 命名与参数设计

该算子与 Pytorch 保持一致，方便使用用户使用。

API设计为 `paddle.cdist(x, y, p=2, compute_mode=compute_mode='use_mm_for_euclid_dist_if_necessary)`。其中 `p` 为 p-范数对应的 p 值，p 大于等于零。

对于 torch 的 api 中 参数 `compute_mode`， PaddlePaddle 考虑添加这一功能，且设计思路同 torch 相似，默认为：
use_mm_for_euclid_dist_if_necessary，代表我们会根据 P or R 的大小选择是否使用矩阵加速，
donot_use_mm_for_euclid_dist：永远不会使用矩阵乘加速，
use_mm_for_euclid_dist：始终使用矩阵乘加速。原因如下：

torch 中的该方法如下
```
Tensor _euclidean_dist(const Tensor& x1, const Tensor& x2) {
  /** This function does the fist part of the euclidean distance calculation
   * We divide it in two steps to simplify dealing with subgradients in the
   * backward step */
  Tensor x1_norm = x1.pow(2).sum(-1, true);
  Tensor x1_pad = at::ones_like(x1_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor x2_norm = x2.pow(2).sum(-1, true);
  Tensor x2_pad = at::ones_like(x2_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor x1_ = at::cat({x1.mul(-2), std::move(x1_norm), std::move(x1_pad)}, -1);
  Tensor x2_ = at::cat({x2, std::move(x2_pad), std::move(x2_norm)}, -1);
  Tensor result = x1_.matmul(x2_.mT());
  result.clamp_min_(0).sqrt_();
  return result;
}
```
上述算法在 paddle 下通过 python 端进行组合，实现如下
```
x_norm = paddle.sum(x.pow(2), keepdim=True, axis=-1)
y_norm = paddle.sum(y.pow(2), keepdim=True, axis=-1)
x_pad = paddle.ones_like(x_norm)
y_pad = paddle.ones_like(y_norm)
x_ = paddle.concat([x * -2, x_norm, x_pad], -1)
y_ = paddle.concat([y, y_pad, y_norm], -1)
y_perm = list(range(len(y_.shape)))
y_perm[-1], y_perm[-2] = y_perm[-2], y_perm[-1]
out = paddle.clip(x_.matmul(paddle.transpose(y_, y_perm)), min=0).sqrt()
```
该方法与普通方法在不同大小的数据测试结果如下

### 测试1:

x 的 shape 为 (10, 1000, 10), y 的 shape 为 (10, 1000, 10)

初次运行

|         | 矩阵运算                                                     | 普通运算                   | 
|---------|----------------------------------------------------------|------------------------|
| 动态图前向计算 | 0.5474686622619629s                                      | 0.0009992122650146484s |
| 动态图反向计算 | 0.0010619163513183594s                                   | 0.0010509490966796875s | 
| 静态图前向计算 | 2.606377363204956s                                      | 0.07554912567138672s    | 
| 静态图反向计算 | 0.0020089149475097656s                                      | 0.0009915828704833984s   | 

去除第一次运行后，重复运行 100 次的平均时间。

|         | 矩阵运算                    | 普通运算                             | 
|---------|-------------------------|----------------------------------|
| 动态图前向计算 | 0.00034404993057250975s | 0.0016649532318115234s            |
| 动态图反向计算 | 0.0012914729118347168s | 0.0014135384559631349s             | 
| 静态图前向计算 | 0.00047049283981323245s  | 0.004856042861938477s             | 
| 静态图反向计算 | 0.0010589122772216796s  | 0.0014948368072509766s             | 

### 测试2：

x 的 shape 为 (10, 10, 10), y 的 shape 为 (10, 10, 10)

去除第一次运行后，重复运行 100 次的平均时间。

|         | 矩阵运算                    | 普通运算                     | 
|---------|-------------------------|--------------------------|
| 动态图前向计算 | 0.00016795086860656739s | 0.000046715021133422854s |
| 动态图反向计算 | 0.00015584778785705566s | 0.00006124091148376465s  | 
| 静态图前向计算 | 0.00025142621994018555s  | 0.00014510750770568848s  | 
| 静态图反向计算 | 0.00019922804832458497s  | 0.00009114623069763184s | 


## 底层OP设计
无，通过已有的算子在 python 端进行组合。
## API实现方案
通过已有的算子在 python 端进行组合。通过对 x 和 y 的参数进行拼接扩展，之后将扩展后的 x 和 y 根据公式计算距离。
实现位置为 Paddle repo `python/paddle/tensor/linalg.py` 目录。
# 六、测试和验收的考量
测试 api 功能的准确性。
1. 不同 shape 的计算结果测试
2. 不同 p 下的计算结果测试
3. shape 错误时报错测试
4. p < 0 时报错测试
5. 动静态图下计算的测试
# 七、可行性分析和排期规划
本 API 难度适中，工期上能满足要求。