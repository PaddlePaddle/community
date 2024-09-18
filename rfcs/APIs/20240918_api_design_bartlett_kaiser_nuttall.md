# paddle.nn.ParameterDict 设计文档

| API名称      | paddle.nn.bartlett_kaiser_nuttall              |
| ------------ | ---------------------------------------------- |
| 提交作者     | Micalling                                      |
| 提交时间     | 2024-09-18                                     |
| 版本号       | V1.0                                           |
| 依赖飞桨版本 | develop版本                                    |
| 文件名       | 20240918_api_design_bartlett_kaiser_nuttall.md |

# 一、概述

## 1、相关背景

当前 paddle.audio.functional.get_window 中已支持 hamming，hann，blackman 等窗函数，需扩充支持 bartlett 、 kaiser 和 nuttall 窗函数

> 参考赛题：[NO.22 在 paddle.audio.functional.get_window 中支持 bartlett 、 kaiser 和 nuttall 窗函数](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no22-%E5%9C%A8-paddleaudiofunctionalget_window-%E4%B8%AD%E6%94%AF%E6%8C%81-bartlett--kaiser-%E5%92%8C-nuttall-%E7%AA%97%E5%87%BD%E6%95%B0)

## 2、功能目标

实现对应函数，并可通过get_window函数进行调用。

## 3、意义

丰富 Paddle 的 Parameter 相关的 API。

# 二、飞桨现状

目前 Paddle 已经实现了 `hamming，hann，blackman`  。

# 三、业内方案调研

PyTorch 实现了相关接口

- [torch.signal.windows.windows](https://pytorch.org/docs/stable/_modules/torch/signal/windows/windows.html)

具体实现逻辑为在 `torch/signal/windows/windows.py`

```python
def nuttall(
        M: int,
        *,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    return general_cosine(M,
                          a=[0.3635819, 0.4891775, 0.1365995, 0.0106411],
                          sym=sym,
                          dtype=dtype,
                          layout=layout,
                          device=device,
                          requires_grad=requires_grad)

def bartlett(M: int,
             *,
             sym: bool = True,
             dtype: Optional[torch.dtype] = None,
             layout: torch.layout = torch.strided,
             device: Optional[torch.device] = None,
             requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('bartlett', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = -1
    constant = 2 / (M if not sym else M - 1)

    k = torch.linspace(start=start,
                       end=start + (M - 1) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return 1 - torch.abs(k)

def kaiser(
        M: int,
        *,
        beta: float = 12.0,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('kaiser', M, dtype, layout)

    if beta < 0:
        raise ValueError(f'beta must be non-negative, got: {beta} instead.')

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    # Avoid NaNs by casting `beta` to the appropriate dtype.
    beta = torch.tensor(beta, dtype=dtype, device=device)

    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)
    end = torch.minimum(beta, start + (M - 1) * constant)

    k = torch.linspace(start=start,
                       end=end,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return torch.i0(torch.sqrt(beta * beta - torch.pow(k, 2))) / torch.i0(beta)
```

# 四、对比分析

Paddle 目前实现了 `hamming，hann，blackman` ，`paddle/audio/functional/window.py` 中：

```python
@window_function_register.register()
def _hamming(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Hamming window.
    The Hamming window is a taper formed by using a raised cosine with
    non-zero endpoints, optimized to minimize the nearest side lobe.
    """
    return _general_hamming(M, 0.54, sym, dtype=dtype)

@window_function_register.register()
def _hann(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Hann window.
    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.
    """
    return _general_hamming(M, 0.5, sym, dtype=dtype)

@window_function_register.register()
def _blackman(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Blackman window.
    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.
    """
    return _general_cosine(M, [0.42, 0.50, 0.08], sym, dtype=dtype)
```

其中某些函数可以进行复用

# 五、设计思路与实现方案

## 命名与参数设计

```python
def _bartlett(M: int, sym: bool = True, dtype: str = 'float64')
```

其中:

* `M` (int): 窗口函数的长度。这是必须的参数，因为它决定了窗口的大小。
* `sym` (bool, optional): 控制窗口的对称性。默认值为 `True`，表示窗口是对称的。如果设置为 `False`，则窗口将不会是对称的。
* `dtype` (str, optional): 指定返回的张量类型。默认值为 `'float64'`，表示返回的双精度浮点数类型的张量。

```python
def _kaiser(M: int, beta: float = 12.0, sym: bool = True, dtype: str = 'float64')
```

其中:

* `M` (int): 窗口函数的长度。这是必须的参数，用于定义窗口的大小。
* `beta` (float, optional): Kaiser窗口的形状参数。默认值为 `12.0`。这个参数影响窗口的形状，值越大，窗口在边缘的衰减越快。
* `sym` (bool, optional): 控制窗口的对称性。与 `_bartlett` 函数相同，默认为 `True`。
* `dtype` (str, optional): 指定返回的张量类型。默认为 `'float64'`。

```python
def _nuttall(M: int, sym: bool = True, dtype: str = 'float64')
```

其中:

* `M` (int): 窗口函数的长度。这是一个必须的参数，用于定义窗口的长度。
* `sym` (bool, optional): 控制窗口的对称性。默认值为 `True`。
* `dtype` (str, optional): 指定返回的张量类型。默认为 `'float64'`。

## 底层OP设计

直接在 Python 层实现，不涉及底层算子。

## API实现方案

参考代码：

```python
@window_function_register.register()
def _bartlett(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """
    Computes the Bartlett window.
    This function is consistent with scipy.signal.windows.bartlett().
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(0, M, dtype=dtype)
    w = paddle.where(paddle.less_equal(n, (M - 1) / 2.0),
                 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _kaiser(M: int, beta: float = 12.0, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute the Kaiser window.
    This function is consistent with scipy.signal.windows.kaiser().
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(0, M, dtype=dtype)
    alpha = (M - 1) / 2.0
    w = (paddle.i0(beta * paddle.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         paddle.i0(beta))

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _nuttall(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Nuttall window.
    This function is consistent with scipy.signal.windows.nuttall().
    """
    return _general_cosine(M, a=[0.3635819, 0.4891775, 0.1365995, 0.0106411], sym=sym, dtype=dtype)
```

# 六、测试和验收的考量

目前 Paddle 对于 `window` 函数的单测，与 scipy 保持一致。

- **编程范式场景**

  - 常规覆盖动态图 (和静态图) 的测试场景。
- **硬件场景**

  - 常规需覆盖 CPU、GPU 两种测试场景。
- **输入参数**

  - 常规覆盖默认参数，常用参数，错误参数。
  - 常规数据类型 bfloat16, float16, float32 or float64

# 七、可行性分析和排期规划

- 第一周，实现相关代码
- 第二周，测试用例和文档
- 第三周，Review

# 八、影响面

丰富 paddle API，对其他模块没有影响
