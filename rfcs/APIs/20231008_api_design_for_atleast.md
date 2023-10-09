# 标题 paddle.atleast 设计文档

|API名称 | paddle.combination | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 吴俊([bapijun] (https://github.com/bapijun)) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-27 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 20231008_api_design_for_atleast.md | 


# 一、概述
## 1、相关背景
为 Paddle 新增 atleast_1d / atleast_2d / atleast_3d API，目前paddle中并没有对应的api,需要开发。

## 2、功能目标

paddle.atleast_1d 作为独立的函数调用，返回每个零维输入张量的一维视图，有一个或多个维度的输入张量将按原样返回。
paddle.atleast_2d 作为独立的函数调用，返回每个零维输入张量的二维视图，有两个或多个维度的输入张量将按原样返回。
paddle.atleast_3d 作为独立的函数调用，返回每个零维输入张量的三维视图，有三个或多个维度的输入张量将按原样返回。

## 3、意义

为 Paddle 新增 `paddle.atleast` API，提供tensor的atleast功能。

# 二、飞桨现状

目前飞桨框架并不存在对应的api，缺少实现方式。


# 三、业内方案调研

### 1. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.atleast_*d(x) `

- `x` 为 输入tensor或者是有tensor组成的list/tuple 。
- `r` 为 `int` 类型，组合的元素数目。

其实现的代码如下

```cpp
Tensor atleast_1d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1});
    default:
      return self;
  }
}

std::vector<Tensor> atleast_1d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_1d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_2d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1});
    case 1: {
      return self.unsqueeze(0);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_2d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_2d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_3d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1, 1});
    case 1: {
      return self.unsqueeze(0).unsqueeze(-1);
    }
    case 2: {
      return self.unsqueeze(-1);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_3d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_3d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}
```

### 2. TensorFlow

在 TensorFlow 中使用的 API 格式如下：

`tf.experimental.numpy.atleast_*d(*arys)`

根据定义,使用的是tensorflow中的numpy格式,实现 numpy.atleast_*d功能

其实现的代码如下

``` python
def _atleast_nd(n, new_shape, *arys):
  """Reshape arrays to be at least `n`-dimensional.

  Args:
    n: The minimal rank.
    new_shape: a function that takes `n` and the old shape and returns the
      desired new shape.
    *arys: ndarray(s) to be reshaped.

  Returns:
    The reshaped array(s).
  """

  def f(x):
    # pylint: disable=g-long-lambda
    x = asarray(x)
    return asarray(
        np_utils.cond(
            np_utils.greater(n, array_ops.rank(x)),
            lambda: reshape(x, new_shape(n, array_ops.shape(x))),
            lambda: x))

  arys = list(map(f, arys))
  if len(arys) == 1:
    return arys[0]
  else:
    return arys


@np_utils.np_doc('atleast_1d')
def atleast_1d(*arys):
  return _atleast_nd(1, _pad_left_to, *arys)


@np_utils.np_doc('atleast_2d')
def atleast_2d(*arys):
  return _atleast_nd(2, _pad_left_to, *arys)


@np_utils.np_doc('atleast_3d')
def atleast_3d(*arys):  # pylint: disable=missing-docstring

  def new_shape(_, old_shape):
    # pylint: disable=g-long-lambda
    ndim_ = array_ops.size(old_shape)
    return np_utils.cond(
        math_ops.equal(ndim_, 0),
        lambda: constant_op.constant([1, 1, 1], dtype=dtypes.int32),
        lambda: np_utils.cond(
            math_ops.equal(ndim_, 1), lambda: array_ops.pad(
                old_shape, [[1, 1]], constant_values=1), lambda: array_ops.pad(
                    old_shape, [[0, 1]], constant_values=1)))

  return _atleast_nd(3, new_shape, *arys)


```

### 3. mindspore

`mindspore.ops.atleast_*d(input, r=2, with_replacement=False)`

调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于*。



参数：
- input (Tensor) - 一维Tensor。

- r (int，可选) - 进行组合的元素个数。默认值： 2 。

- with_replacement (bool，可选) - 是否允许组合存在重复值。默认值： False 。
  
``` python
def atleast_3d(inputs):

    def _expand3(arr):
        ndim = P.Rank()(arr)
        if ndim == 0:
            return P.Reshape()(arr, (1, 1, 1))
        if ndim == 1:
            return P.Reshape()(arr, (1, P.Size()(arr), 1))
        if ndim == 2:
            return P.Reshape()(arr, P.Shape()(arr) + (1,))
        return arr

    if isinstance(inputs, Tensor):
        return _expand3(inputs)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'atleast_3d', each element of 'inputs' must be a tensor, but got {type(tensor)}")
    return tuple([_expand3(arr) for arr in inputs])

```

# 四、对比分析

目前在pytorch中使用c++实现，可以参考他的代码逻辑。参考MindSpore的python的话，和tensorflow的python实现版本，最终实现的结果与pytorch的结果一致。
# 五、设计思路与实现方案

## 命名与参数设计

API设计为`paddle.atleast_1d(x)`
paddle.atleast_1d
----------------------
参数
:::::::::
- x (Tensor) - Tensor 或者是Tensors的list或tuple


API设计为`paddle.atleast_2d(x)`
paddle.atleast_2d
----------------------
参数
:::::::::
- x (Tensor) - Tensor 或者是Tensors的list或tuple

API设计为`paddle.atleast_3d(x)`
paddle.atleast_3d
----------------------
参数
:::::::::
- x (Tensor) - Tensor 或者是Tensors的list或tuple


## API实现方案
同时参考pytorch和MindSpore的方案,以MindSpore为基础进行编写。

# 六、测试和验收的考量

可考虑一下场景：

- 输出数值结果的一致性和数据类型是否正确，使用 pytorch 作为参考标准
- 对不同 shape进行检查
- 输入输出的容错性与错误提示信息
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景
  
# 七、可行性分析和排期规划
方案主要根据相关数学原理并参考 MindSpore 的工程实现方法，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
由于采用独立的模块开发，对其他模块是否有影响。

# 名词解释
无
# 附件及参考资料
无