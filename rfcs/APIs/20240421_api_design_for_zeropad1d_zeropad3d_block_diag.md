# paddle_zeropad1d_zeropad3d_block_diag 设计文档

| API 名称     | paddle.nn.zeropad1d/zeropad3d/paddle.block_diag            |
| ------------ | ---------------------------------------------------------- |
| 提交作者     | Chen-Lun-Hao                                               |
| 提交时间     | 2024-04-21                                                 |
| 版本号       | V2.0                                                       |
| 依赖飞桨版本 | develop                                                    |
| 文件名       | 20240421_api_design_for_zeropad1d_zeropad3d_block_diag.md  |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，需要为飞桨扩充 API `paddle.nn.ZeroPad1d paddle.nn.ZeroPad3d paddle.block_diag`

本 API 属于飞桨开源个人贡献赛 API 开发任务[No.3：为 Paddle 新增 ZeroPad1D / ZeroPad3D / block_diag API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no3-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-zeropad1d--zeropad3d--block_diag--api)的任务。

## 2、功能目标

- ZeroPad1D/ ZeroPad3D
  用零填充输入张量边界，1D 填充最后一个维度，3D 填充最后三个维度即可。
- block_diag
  从提供的张量列表中创建一个块对角矩阵,返回一个二维张量，所有输入张量按顺序排列，使得它们的左上角和右下角对角相邻。所有其他元素都被设置为 0

预期该 API 支持

- paddle.nn.ZeroPad1d/paddle.nn.ZeroPad3d 作为独立的函数调用
- paddle.block_diag 作为独立的函数调用

## 3、意义

提升飞桨 API 丰富度。

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

PyTorch 中有 API `torch.nn.ZeroPad1d(padding)/torch.nn.ZeroPad3d(padding)`

其介绍为：

> Pads the input tensor boundaries with zero.
> ZeroPad1d 参数表为：

- `(int, tuple)` padding: the size of the padding. If is int, uses the same padding in both boundaries. If a 2-tuple, uses (padding_left, padding_right).

> ZeroPad3d 参数表为：

- `(int, tuple)` the size of the padding. If is int, uses the same padding in all boundaries. If a 6-tuple, uses (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).

### 具体实现

PyTorch 通过调用 torch.nn.functional.pad()实现每个维度的零填充，可以参考[pytorch/torch/nn/modules
/padding.py](https://github.com/pytorch/pytorch/blob/f34905f61d614fb5b0551ec363d7945fdcd06268/torch/nn/modules/padding.py#L657)。

```python
def pad(input: Tensor, pad: List[int], mode: str = "constant", value: Optional[float] = None) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            torch.nn.functional.pad, (input,), input, pad, mode=mode, value=value)
    if not torch.jit.is_scripting():
        if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
            if mode == 'replicate':
                # Use slow decomp whose backward will be in terms of index_put.
                # importlib is required because the import cannot be top level
                # (cycle) and cannot be nested (TS doesn't support)
                return importlib.import_module('torch._decomp.decompositions')._replication_pad(
                    input, pad
                )
    return torch._C._nn.pad(input, pad, mode, value)

class _ConstantPadNd(Module):
    __constants__ = ['padding', 'value']
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self) -> str:
        return f'padding={self.padding}, value={self.value}'

class ConstantPad1d(_ConstantPadNd):
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t, value: float):
        super().__init__(value)
        self.padding = _pair(padding)

class ZeroPad1d(ConstantPad1d):
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__(padding, 0.)

    def extra_repr(self) -> str:
        return f'{self.padding}'

class ConstantPad3d(_ConstantPadNd):
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t, value: float) -> None:
        super().__init__(value)
        self.padding = _ntuple(6)(padding)

class ZeroPad3d(ConstantPad3d):
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__(padding, 0.)

    def extra_repr(self) -> str:
        return f'{self.padding}'

```

PyTorch 中有 API `torch.block_diag(*tensors)`

其介绍为：

> Create a block diagonal matrix from provided tensors.
> 参数表为：

- `(Tensor)` \*tensors: One or more tensors with 0, 1, or 2 dimensions.

### 具体实现

PyTorch 通过调用底层 C++实现，可以参考[torch/\_refs/**init**.py](https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py)。

```cpp
@register_decomposition(aten.block_diag)
@out_wrapper()
def _block_diag_iterable(tensors: List[TensorLikeType]) -> TensorLikeType:
    """
    Reference implementation of torch.block_diag
    """
    tensors_2d = [
        tensor.view(1, -1) if tensor.dim() <= 1 else tensor for tensor in tensors
    ]

    ncols = builtins.sum(tensor.shape[1] for tensor in tensors_2d)
    device = tensors_2d[0].device

    result = []

    col_start = 0
    for i, tensor in enumerate(tensors_2d):
        torch._check(
            tensor.dim() == 2,
            lambda: "Input tensors must have 2 or fewer dimensions. "
            "Input tensors must have 2 or fewer dimensions. "
        )
        torch._check(
            tensor.device == device,
            lambda: "Input tensors must all be on the same device. "
            f"Input 0 is on device {device} and input {i} is on device {tensor.device}.",
        )
        row, col = tensor.shape
        left = torch.zeros((row, col_start), device=device, dtype=tensor.dtype)
        right = torch.zeros(
            (row, ncols - col_start - col), device=device, dtype=tensor.dtype
        )
        result += [torch.cat((left, tensor, right), dim=1)]
        col_start += col

    return torch.cat(result, dim=0)


def block_diag(*tensors: List[TensorLikeType]) -> TensorLikeType:
    """
    This is used as an input to PythonRefInfo. `torch.block_diag`
    expects arguments splatted, but `aten.block_diag` expects only
    one argument that is a list of Tensors.
    """
    return _block_diag_iterable(tensors)

```

# 四、对比分析

对于上述函数，PyTorch 都是是使用 C++ API 实现的，Python 端直接调用 C++ 接口。而在 paddle 能够通过算子组合实现该 api，通过观察，算子的底层也是 C++，因此，在实现上，paddle 的实现方案与 PyTorch 的实现方案是类似的。

# 五、设计思路与实现方案

paddle 目前的 `pad` 算子已经支持 `x`, `pad`, `mode`, `value`, `data_format` 等参数，因此只需指定 value 为 0 以及 mode 为 constant 即可使用 `pad` 算子实现 `zeropad1d/zeropad3d` 。而针对需要实现 `block_diag`的 API，只需在 `zeros` 算子和 `concat`算子即可实现。

## 命名与参数设计

添加 Python API:

```python
paddle.nn.ZeroPad1D(padding, data_format="NCL", name=None)
paddle.nn.ZeroPad3D(padding, data_format="NCDHW", name=None)
paddle.block_diag(*inputs, name=None)
```

ZeroPad1D 参数表：

- padding: (int | Tensor | List[int] | Tuple[int]) 如果为 'int'，使所有维度的填充相同。填充的形式为 （pad_left， pad_right）。
- data_format: (str, optional) 指定输入数据的数据格式。默认值：'NCL'。
- name: (Optional[str]) op 名称

ZeroPad3D 参数表：

- padding: (int | Tensor | List[int] | Tuple[int]) 如果为 'int'，使所有维度的填充相同。填充的形式为 (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)。
- data_format: (str, optional) 指定输入数据的数据格式。默认值：'NCDHW'。
- name: (Optional[str]) op 名称

block_diag 参数表：

- inputs: (List[Tensor]) 输入的 Tensor 列表。列表中的 Tensor 维度应该在 0 到 2 之间
- name: (Optional[str]) op 操作。

## 底层 OP 设计

不涉及底层 OP。

## API 实现方案

使用 `pad` 算子实现相应接口：

```python
class ZeroPad1D(Layer):
    def __init__(self, padding, data_format="NCL", name=None):
        super().__init__()
        self._pad = _npairs(padding, 1)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'

class ZeroPad3D(Layer):
    def __init__(self, padding, data_format="NCDHW", name=None):
        super().__init__()
        self._pad = _npairs(padding, 3)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'

```

使用 `zeros` 和 `concat` 算子实现接口：

```python

def block_diag(*inputs, name=None):
    def to_col_block(arys, i, a):
        return [
            a if idx == i else paddle.zeros([ary.shape[0], a.shape[1]], dtype=a.dtype)
            for idx, ary in enumerate(arys)
        ]

    def to_2d(ary):
        if not isinstance(ary, paddle.Tensor):
            raise TypeError(
                f"For 'block_diag', each element of 'inputs' must be a tensor, but got {type(ary)}"
            )
        if ary.ndim == 0:
            return ary.unsqueeze(axis=0).unsqueeze(axis=0)
        if ary.ndim == 1:
            return ary.unsqueeze(axis=0)
        if ary.ndim == 2:
            return ary
        raise ValueError(
            "For 'block_diag', the dimension of each elements in 'inputs' must be 0, 1, or 2, but got "
            f"{ary.ndim}"
        )

    arys = [to_2d(ary) for ary in inputs]

    matrix = [paddle.concat(to_col_block(arys, idx, ary), axis=0) for idx, ary in enumerate(arys)]
    return paddle.concat(matrix, axis=1)

```

# 六、测试和验收的考量

- 覆盖动态图和静态图的测试场景
- 覆盖 CPU、GPU 两种测试场景
- 支持精度 FP32、FP64 等
- 需要检查计算正确性
- 需要检查多维的情况

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关 PythonAPI 均有实现，可以在开源贡献个人挑战赛期间完成。

# 八、影响面

对其他模块暂无影响

# 名词解释

# 附件及参考资料

[【Hackathon 6th No.3】Paddle 新增 ZeroPad1D / ZeroPad3D / block_diag API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no3-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-zeropad1d--zeropad3d--block_diag--api)
