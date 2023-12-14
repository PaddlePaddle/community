
#为 Paddle 新增 apply API 设计文档

| API名称 | tensor.apply                     |
|---|----------------------------------|
| 提交作者 | yangguohao                       |
| 提交时间 | 2023-11-26                       |
| 版本号 | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名 | 20231126_api_design_for_apply.md |

# 一、概述

## 1、相关背景

为了增强Paddle框架的功能并与其他框架保持竞争力，我们决定为Tensor添加一个新的API，它允许用户应用Python函数到Tensor的每个元素。

## 2、功能目标

- 为Tensor添加 `apply(callable)` 方法，返回新的Tensor，存放计算结果。
- 为Tensor添加 `apply_(callable)` 方法，inplace修改输入Tensor。

## 3、意义

该功能将提供更大的灵活性，使开发人员能够轻松地应用自定义函数到Tensor的每个元素，从而实现更复杂的操作。

# 二、飞桨现状

Paddle当前不支持此功能。

# 三、业内方案调研

TensorFlow、PyTorch 和 NumPy 都提供了允许用户对数组或张量元素应用函数的功能。

- **TensorFlow**: TensorFlow 提供了 `tf.map_fn` 方法，使用户能够对张量的元素应用函数, 函数的范围不限于 python 函数还包括 tf 函数，同时支持 backpropagation。
```
tf.map_fn(
    fn,
    elems,
    dtype=None,
    parallel_iterations=None,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    name=None,
    fn_output_signature=None
)
```

- **PyTorch**: PyTorch 通过 `torch.Tensor.apply_` 方法，允许用户在 inplace 应用函数到张量的元素, 无法应用 torch 函数。但需要注意，此方法仅限于CPU张量，并且性能较差，运行的速度较慢。
```
Tensor.apply_(callable)
```

# 四、对比分析

- **TensorFlow**: `tf.map_fn` 方法提供了强大的功能，但其接口相对复杂，不太适合初学者。

- **PyTorch**: 虽然 `torch.Tensor.apply_` 方法提供了所需的功能，但由于其局限性，它在新代码中不被推荐使用。

Paddle 的 `apply` 和 `apply_` 方法的设计目标以 pytorch 的实现功能类似。

# 五、设计思路与实现方案

## 命名与参数设计

tensor.apply(callable)
tensor.apply_(callable)

参数:
- `callable`: 用户提供的Python函数，将应用于Tensor的每个元素。

## 底层OP设计

无需实现底层 OP 及 Kernel

## API实现方案
采取与 pytorch 相似的设计，以下对应的实现其本质都是在 C++ 端调用 python 函数，可以看作是 python 函数 f 作用在张量 x 上，即 python 中 f(x) 的调用。因此该方案对 GPU 上的 tensor 也同样支持。

### 动态图
在 tensor_patch_methods 内增加 apply 和 apply_ 的两个 api，对于求梯度的 tensor 则无法使用 apply 并报错
```
@framework.dygraph_only
def apply_(self, func):
    """
    Inplace apply the python function to the tensor.
    Returns:
        None
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> x = paddle.to_tensor(5.)
            >>> f = lambda x: 3*x+2
            >>> x.apply_(f)
            >>> print(x) # 17
    """
    if self.stop_gradient is True:
        raise RuntimeError(
            "Cannot apply function on a tensor that stop gradient."
        )
    self._apply_(func)

def apply(self, func):
    """
    Apply the python function to the tensor.
    Returns:
        None
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> x = paddle.to_tensor(5.)
            >>> f = lambda x: 3*x+2
            >>> y = x.apply(f)
            >>> print(y) # 17
            >>> print(x) # 5
    """
    if self.stop_gradient is True:
        raise RuntimeError(
            "Cannot apply function on a tensor that stop gradient."
        )
    return self._apply(func)
```

C++ 端则是在 eager_method.cc 中增加相应的逻辑，可以复用 PyTensorHook 来对 tensor 作用 callable python function。

```
static PyObject* tensor_apply(TensorObject* self,
                              PyObject* args,
                              PyObject* kwargs) {
  EAGER_TRY
  PyObject* apply_func = PyTuple_GET_ITEM(args, 0);
  PyTensorHook func = PyTensorHook(apply_func);
  paddle::Tensor out = func(self->tensor);
  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_apply_(TensorObject* self,
                               PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  PyObject* apply_func = PyTuple_GET_ITEM(args, 0);
  PyTensorHook func = PyTensorHook(apply_func);
  paddle::Tensor out = func(self->tensor);
  self->tensor.set_impl(out.impl());
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}
```

### 静态图

Legacy IR 在 python/paddle/base/framework.py 的 Variable 下添加 apply 函数
```
def apply(self, func):
    if not self.stop_gradient:
        raise RuntimeError(
            "Cannot apply function on a tensor that required gradient."
        )
    try:
        return func(self)
    except:
        raise ValueError(f"The PyFunc {func.__name__} could not be applied")
```

PIR 则在 paddle/fluid/pybind/pir.cc 中实现 apply 并 bind 到 Value 上

```
pir::OpResult apply(Value self, py::object func) {
  py::gil_scoped_acquire gil;
  PyObject *py_func = func.release().ptr();
  Py_INCREF(py_func);
  PyObject *res = nullptr;
  try {
    py::object obj = py::cast(self);
    PyObject *tmp_self = obj.release().ptr();
    Py_INCREF(tmp_self);
    res = PyObject_CallFunctionObjArgs(py_func, tmp_self, nullptr);
    Py_DECREF(tmp_self);
  } catch (std::exception &e) {
    PADDLE_THROW(phi::errors::Unavailable(
        "Hook function of Tensor raises an exception: %s.", e.what()));
  } catch (...) {
    PADDLE_THROW(phi::errors::Fatal(
        "Hook function of Tensor raises an unknown exception."));
  }
  if (res == Py_None) {
    return self.dyn_cast<OpResult>();
  }
  auto out = CastPyArg2Value(res, "apply", 0);
  Py_DECREF(py_func);
  Py_DECREF(res);
  return out.dyn_cast<OpResult>();
}
```

# 六、测试和验收的考量

- 单测代码，apply api 测试
- inplace 测试
- 对 stop_gradient=False 的 tensor 报错测试
- 动转静下测试

# 七、可行性分析和排期规划

已大部分完成代码开发工作，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释
无
