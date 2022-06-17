# 为 PaddleScience 新增支持随机/Quasi 采样法&&增加（2D/3D/ND ）Geometry

|              |                                       |
| ------------ | ------------------------------------- |
| 提交作者     | 梁嘉铭                                |
| 提交时间     | 2022-04-24                            |
| 版本号       | V1.1                                  |
| 依赖飞桨版本 | develop 版本                          |
| 文件名       | 20220322_Science_design_for_sample.md |

# 一、概述

## 1、相关背景

https://github.com/PaddlePaddle/PaddleScience/issues/40

https://github.com/PaddlePaddle/PaddleScience/issues/38

## 2、功能目标

1. 为 PaddleScience 新增支持随机/Quasi 采样法
2. 支持重点区域增强采样
3. 增加 Circle 类型 Geometry（2D/3D）
4. 增加 （2D/3D/ND ）Geometry

## 3、意义

传统方法是画网格方法，而 PINN 方法无需画网格，可以用随机采样法，目前 paddlescience 需要支持随机/Quasi 采样法

增加多种 Geometry 类型以提高 PaddleScience 的灵活性

# 二、飞桨现状

1. PaddleScience 现在仅支持空间维度的网格采样法。
2. 目前 PaddleScience 的 Geometry 类型仅支持`cylinder_in_cube`，`rectangular`等几何体，并且均不支持随机化采样。

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

deepxde 在[geometry](https://github.com/lululxvi/deepxde/tree/master/deepxde/geometry)中将 geometry 抽象为 1D/2D/3D 的几何体，并且提供了一些几何体的构造函数，如：

```python
class Disk(Geometry):
    def __init__(self, center, radius):

    def inside(self, x):

    def on_boundary(self, x):

    def distance2boundary_unitdirn(self, x, dirn):

    def distance2boundary(self, x, dirn):


    def mindist2boundary(self, x):

    def boundary_normal(self, x):
```

并且在[csg](https://github.com/lululxvi/deepxde/blob/master/deepxde/geometry/csg.py)中实现了几何体的合并。

# 四、设计思路与实现方案

## 几何体

因原有仓库中的 Gemetry 仅提供一个 discretize 方法进行离散化，非常不利于其他功能的实现，在该 RFC 将重新实现该类。

如下是该类的抽象类型：

```python
class Geometry(abc.ABC):
    def __init__(self, time_dependent):
        self.time_dependent = time_dependent
        self.points = []
        self.time_points = []
        self.shapes = []
        self.boundary_points = dict()

    @abc.abstractmethod
    def is_internal(self, x):
        """
        Returns True if the point x is internal to the Geometry.
        """
        pass

    @abc.abstractmethod
    def is_boundary(self, x):
        """
        Returns True if the point x is on the boundary of the Geometry.
        """
        pass

    @abc.abstractmethod
    def grid_points(self, n):
        """
        Returns a list of n grid points.
        """
        pass

    @abc.abstractmethod
    def grid_points_on_boundary(self, n):
        """
        Returns a list of n grid points on the boundary.
        """
        pass

    @abc.abstractmethod
    def random_points(self, n):
        """
        Returns a list of n random points.
        """
        pass

    @abc.abstractmethod
    def random_points_on_boundary(self, n):
        """
        Returns a list of n random points on the boundary.
        """
        pass

    def uniform_time_dependent(self, n, time_start, time_end):
        """
        Returns a list of n time-dependent points.
        """
        pass

    def random_time_dependent(self,
                              n,
                              time_start,
                              time_end,
                              samplingtype=None):

    def union(self, other):
        """
        Returns the union of the two geometries.
        """
        pass

    def intersection(self, other):
        """
        Returns the intersection of the two geometries.
        """
        pass

    def difference(self, other):
```

### 说明

1. 为了方便对几何体的不同边界设定初始条件，该类提供了一个 `boundary_points` 属性，该属性是一个字典，其中的键是边界的名称，值是一个列表，列表中的元素是边界上的点。这样可以方便地设定不同边界上的初始条件。完成之后可以使用如下方式进行设定：

```python
import numpy as np
def GenBCWeight(boundary_points):
    bc_weight = {}
    for key in boundary_points:
        bc_weight[key] = np.zeros((len(boundary_points[key]), 2))

    bc_weight['b0'][0] = 1 * np.ones((len(boundary_points['b0'])))
    bc_weight['b0'][1] = 1 * np.ones((len(boundary_points['b0'])))

    bc_weight['b1'][0] = 1 * np.ones((len(boundary_points['b1'])))

    return bc_weight
```

2. 我们计划实例化如下几种几何体：

- Point(圆面，球体)(已有 demo)
- Polygon(非凸多边形)(已有 demo)
- Box(矩形)(已有 demo)
- Cone(圆锥体)
- Cylinder(圆柱体)
- Ellipsoid(椭球体)
- pyramid(棱锥体)
- ring(环形)

3. 支持几何图像的并集，差集，交集运算方法。
   为了实现几何图形的运算，我们需要在不同的几何体中实现 is_internal 和 is_boundary 方法。下面是 Point 几何体一个简单的实现：

   ```python
    def is_internal(self, x):
        return np.linalg.norm(x - self.center, axis=1) <= self.radius

    def is_boundary(self, x):
        return np.linalg.norm(x - self.center, axis=1) == self.radius
   ```

4. 支持便捷“ 挖空 ”几何体的实现，所有几何体均支持输入一个 Geometry 对象列表，返回一个新的几何体，该几何体是原几何体列表中所有几何体的差集。

```python
point = Point([0, 0, 0], 1)
box = Box([0, 0, 0], 4, 4, 10, [point])
```

5. 多次运算实现逻辑

   > 对于所有的几何体，将保留全部的图形运算记录，这样判断是否在内部或者边界的时候只需要判断记录中的图形即可。即有如下逻辑关系

   ![](https://images.puqing.work/image4d6d54ed1cddd9e2.png)

## 随机采样

不同几何体在类中分别实现了 `random_points` 和 `random_points_on_boundary` 方法

## 增强采样

注意该增强方法仅对几何体内部中的挖孔几何体进行讨论，并且对于多个挖空几何体的情况，会分别对其增强采样。

### 中心十字形区域类型

该方法将会在几何体周围生成一个中心十字形区域，该区域的大小为 `width` 参数，该区域的中心为 `center` 参数(该参数默认为几何体中心), 密度为 `density` 参数。

### 中心方向区域

该方法将会在几何体周围生成一个中心区域，该区域的大小为 `width` 参数，该区域的中心为 `center` 参数(该参数默认为几何体中心), 密度为 `density` 参数。

# 五、测试和验收的考量

上述提案涉及众多，现阶段，将考察在 2D/3D 圆柱绕流问题中的网格采样与随机采样的效果，以及网络的收敛性。

# 六、可行性分析和排期规划

本次 RFC 对 Geometry 进行优化，对于 Geometry 的部分实现可在本人[Geometry](https://github.com/AndPuQing/Geometry)库中查看。

本计划改动颇大，计划将在 5 月 10 日之前完成验收部分内容。

# 七、影响面

将优化 Geometry 的实现，将会影响到以下几个面：

- 几何体的构造
- 几何体采样
- 网络计算 Loss 部分
