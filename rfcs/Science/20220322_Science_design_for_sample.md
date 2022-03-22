# 增加 paddle 作为 DeepXDE 的 backend

|              |                            |
| ------------ | -------------------------- |
| 提交作者     | 梁嘉铭                     |
| 提交时间     | 2022-03-22                 |
| 版本号       | V1.0                       |
| 依赖飞桨版本 | develop 版本               |
| 文件名       | 20220322_Science_design_for_sample.md |

# 一、概述

## 1、相关背景

https://github.com/PaddlePaddle/PaddleScience/issues/40

## 2、功能目标

1. 为 PaddleScience 新增支持随机/Quasi 采样法
2. 支持重点区域增强采样

## 3、意义

传统方法是画网格方法，而 PINN 方法无需画网格，可以用随机采样法，目前 paddlescience 需要支持随机/Quasi 采样法

# 二、飞桨现状

1. PaddleScience 现在仅支持空间维度的网格采样法。

# 三、业内方案调研

## Nvidia-SimNet

在 Nvidia 发布的 SimNet 支持了 Quasi 采样法，分别有如下两个函数实现，分别对几何的边界，内部进行采样：

- `sample_boundary(nr_points_per_area, criteria=None, param_ranges={}, discard_criteria=True,quasirandom=False)`

- `sample_interior(nr_points_per_volume, bounds, criteria=None,param_ranges={}, discard_criteria=True, quasirandom=False)`

其中`quasirandom`是一个 bool 值，表示是否使用 Quasi 采样法，默认为 False。

## DeepXDE

deepxde 在[sampler](https://github.com/lululxvi/deepxde/blob/master/deepxde/geometry/sampler.py)中实现了采样，支持 Pseudo random 采样与由 skopt.sampler 提供的 Quasi random 采样。

并且分别在各种几何图形中分别实现了采样与边界上的采样，如：

```python
def random_boundary_points(self, n, random="pseudo"):
    if n == 2:
        return np.array([[self.l], [self.r]]).astype(config.real(np))
    return np.random.choice([self.l, self.r], n)[:, None].astype(config.real(np))
```

实现了在边界上的随机采样。

# 四、设计思路与实现方案

## 命名与参数设计

此次将会在 paddlescience.geometry.sampler 中实现 Quasi random 采样法，并且支持重点区域增强采样。

下面是该 API 的参数设置：

```python
def sample(n_samples, dim, sampler, **key):
    """Gunerate samples from a given distribution.

    Parameters
    ----------
    n_samples: int
        Number of samples to generate.
    dim: int
        Number of dimensions.
    sampler: str
        Sampler to use.
    key: dict
        Parameters for the sampler.

    Returns
    -------
    np.ndarray
        Samples.
    """
```

## API 实现方案

对于采样将会使用 scipy.stats 中函数，如：

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

r = st.uniform.rvs(size=(1000, 2))
plt.scatter(r[:, 0], r[:, 1], s=0.8)
plt.show()
```

即可产生随机二维点集。

同时考虑到随机采样将会影响 be_index 的计算，从而影响边界条件的设置，这里提出两种方法：

1. 在各种几何图形中重写 bc_index 的生成方式。

```python
point = geod.get_space_domain()
plt.scatter(point[:, 0], point[:, 1], s=0.6)
bc_index = []
for i in range(10201):
    for j in range(2):
        if abs(point[i][j]) - 0.048 > 1e-10:
            bc_index.append(i)

plt.scatter(point[bc_index, 0], point[bc_index, 1], s=0.6, c='r')
```

![](https://img1.imgtp.com/2022/03/22/KbeD877L.png)

同时对 BC 初始化参数时也需要适应性调整，否则将不会被选中设置：

```python
# Generate BC value
def GenBC(xy, bc_index):
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-3:
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value

# visual bc value
plt.scatter(point[bc_index, 0], point[bc_index, 1], s=0.1, c='r')
for i in range(len(bc_index)):
    if bc_value[i][0] == 1.0:
        plt.scatter(point[bc_index[i], 0], point[bc_index[i], 1], s=0.6, c='b')
plt.show()
```

![](https://img1.imgtp.com/2022/03/22/e1iFC8wl.png)

2. 第二种方式是将 sample_boundary 和 sample_interior 分别实现。

即像 deepxde 中 boundary_sampler 中单独对边缘采样。

第一种随机性过高，可能需要重复调整边界取点条件。所以推荐第二种方式。

## 增强采样

对于一些任务的特殊性，可以在采样时增强采样，如下采样点生成方式

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

c = 1.2
r = st.rdist.rvs(size=(10000, 2), c=c)
plt.scatter(r[:, 0], r[:, 1], s=0.8)
plt.show()

c = 5
x = st.uniform.rvs(size=(10000, 1))
y = st.bradford.rvs(size=(10000, 1), c=c)
plt.scatter(x, y, s=0.8)
plt.show()
```

![](https://i.bmp.ovh/imgs/2022/03/a87e271a7a79ab09.png)

scipy.stats支持多种随机分布采样，可以参考[scipy](https://docs.scipy.org/doc/scipy/reference/stats.html)

# 五、测试和验收的考量

对比网格采样与随机采样效果与优劣，以及网络的收敛性，达到任务要求

# 六、可行性分析和排期规划

scipy 中函数直接可以调用。相关效果已经完成 demo，见如下图。

![](https://img1.imgtp.com/2022/03/22/YacAtbHy.png)

但是可视化由于随机采样是非结构网格，在进行可视化之前需要插值或者选用scatter绘制。

大致可以在任务时间内完成要求

# 七、影响面

将在 PaddleScience 仓库添加 sample 方法，以及修改几何体的边界生成方式与内部点的生成。
