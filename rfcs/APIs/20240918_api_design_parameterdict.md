# paddle.nn.ParameterDict 设计文档

| API名称      | paddle.nn.ParameterDict              |
| ------------ | ------------------------------------ |
| 提交作者     | Micalling                            |
| 提交时间     | 2024-09-18                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | develop版本                          |
| 文件名       | 20240918_api_design_parameterdict.md |

# 一、概述

## 1、相关背景

paddle.nn.ParameterDict 提供参数字典容器。此容器的行为类似于 Python 字典，但它包含的参数将被正确地注册和添加。使用方式为：

```python
import paddle
import paddle.nn as nn

class MyLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'p1': nn.Parameter(paddle.create_parameter(shape=[2, 2], dtype='float32')),
            'p2': nn.Parameter

(paddle.create_parameter(shape=[2, 2], dtype='float32'))
        })

    def forward(self, x, px):  # px can use 'p1' or 'p2'
        x = self.params[px].add(x)
        return x
```

。

> 参考赛题：[NO.23 为 Paddle 新增 ParameterDict API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no23-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-parameterdict-api)

## 2、功能目标

实现 `paddle.nn.ParameterDict` 作为独立的 Layer 调用。

## 3、意义

丰富 Paddle 的 Parameter 相关的 API。

# 二、飞桨现状

目前 Paddle 已经实现了 `paddle.nn.ParameterList`  ，暂无针对Dict的 `paddle.nn.ParameterDict` 接口。

# 三、业内方案调研

PyTorch 实现了相关接口

- [torch.nn.ParameterDict](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html)

具体实现逻辑为在 `torch/nn/modules/container.py`

```python
class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Module methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
    :class:`~torch.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Any = None) -> None:
        super().__init__()
        self._keys: Dict[str, None] = {}
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(
                "Index given to ParameterDict cannot be used as a key as it is "
                f"not a string (type is '{type(key).__name__}'). Open an issue on "
                "github if you need non-string keys."
            )
        else:
            # Use the key as-is so that `.named_parameters()` returns the right thing
            return key

    def __getitem__(self, key: str) -> Any:
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        # Note that all other function that add an entry to the dictionary part of
        # the ParameterDict end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the dictionary part and thus won't
        # call into this function.
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, torch.Tensor) and not isinstance(value, Parameter):
            value = Parameter(value)
        setattr(self, attr, value)

    def __delitem__(self, key: str) -> None:
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        return reversed(list(self._keys))

    def copy(self) -> "ParameterDict":
        """Return a copy of this :class:`~torch.nn.ParameterDict` instance."""
        # We have to use an OrderedDict because the ParameterDict constructor
        # behaves differently on plain dict vs OrderedDict
        return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def setdefault(self, key: str, default: Optional[Any] = None) -> Any:
        """Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        r"""Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def popitem(self) -> Tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
        k, _ = self._keys.popitem()
        # We need the key in the _keys to be able to access/del
        self._keys[k] = None
        val = self[k]
        del self[k]
        return k, val

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        r"""Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
        return self[key] if key in self else default

    def fromkeys(
        self, keys: Iterable[str], default: Optional[Any] = None
    ) -> "ParameterDict":
        r"""Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
        return ParameterDict((k, default) for k in keys)

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys."""
        return self._keys.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        r"""Return an iterable of the ParameterDict key/value pairs."""
        return ((k, self[k]) for k in self._keys)

    def values(self) -> Iterable[Any]:
        r"""Return an iterable of the ParameterDict values."""
        return (self[k] for k in self._keys)

    def update(self, parameters: Union[Mapping[str, Any], "ParameterDict"]) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParametersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(parameters).__name__
            )

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                # parameters as length-2 list too cumbersome to type, see ModuleDict.update comment
                self[p[0]] = p[1]  # type: ignore[assignment]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self.items():
            if isinstance(p, torch.Tensor):
                size_str = "x".join(str(size) for size in p.size())
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f" ({p.device})"
                else:
                    device_str = ""
                parastr = "{} containing: [{} of size {}{}]".format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    torch.typename(p),
                    size_str,
                    device_str,
                )
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                child_lines.append(
                    "  (" + str(k) + "): Object of type: " + type(p).__name__
                )
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("ParameterDict should not be called.")

    def __or__(self, other: "ParameterDict") -> "ParameterDict":
        copy = self.copy()
        copy.update(other)
        return copy

    def __ror__(self, other: "ParameterDict") -> "ParameterDict":
        copy = other.copy()
        copy.update(self)
        return copy

    def __ior__(self, other: "ParameterDict") -> Self:
        self.update(other)
        return self
```

# 四、对比分析

Paddle 目前实现了 `ParameterList` ，`paddle/nn/layer/container.py` 中：

```python
class ParameterList(Layer):
    """ParameterList Container.

    This container acts like a Python list, but parameters it contains will be properly added.

    Parameters:
        parameters (iterable, optional): Iterable Parameters to be added.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> class MyLayer(paddle.nn.Layer):
            ...     def __init__(self, num_stacked_param):
            ...         super().__init__()
            ...         # create ParameterList with iterable Parameters
            ...         self.params = paddle.nn.ParameterList(
            ...             [paddle.create_parameter(
            ...                 shape=[2, 2], dtype='float32')] * num_stacked_param)
            ...
            ...     def forward(self, x):
            ...         for i, p in enumerate(self.params):
            ...             tmp = self._helper.create_variable_for_type_inference('float32')
            ...             self._helper.append_op(
            ...                 type="mul",
            ...                 inputs={"X": x,
            ...                         "Y": p},
            ...                 outputs={"Out": tmp},
            ...                 attrs={"x_num_col_dims": 1,
            ...                         "y_num_col_dims": 1})
            ...             x = tmp
            ...         return x
            ...
            >>> x = paddle.uniform(shape=[5, 2], dtype='float32')
            >>> num_stacked_param = 4
            >>> model = MyLayer(num_stacked_param)
            >>> print(len(model.params))
            4
            >>> res = model(x)
            >>> print(res.shape)
            [5, 2]

            >>> replaced_param = paddle.create_parameter(shape=[2, 3], dtype='float32')
            >>> model.params[num_stacked_param - 1] = replaced_param  # replace last param
            >>> res = model(x)
            >>> print(res.shape)
            [5, 3]
            >>> model.params.append(paddle.create_parameter(shape=[3, 4], dtype='float32'))  # append param
            >>> print(len(model.params))
            5
            >>> res = model(x)
            >>> print(res.shape)
            [5, 4]
    """

    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            for idx, param in enumerate(parameters):
                assert isinstance(param, Parameter)
                self.add_parameter(str(idx), param)

    def __getitem__(self, idx):
        with param_guard(self._parameters):
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        assert isinstance(param, Parameter)
        setattr(self, str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        with param_guard(self._parameters):
            return iter(self._parameters.values())

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Parameters:
            parameter (Parameter): parameter to append
        """
        idx = len(self._parameters)
        self.add_parameter(str(idx), parameter)
        return self
```

其实现逻辑与 ParameterDict 一致，可进行借鉴。

# 五、设计思路与实现方案

由于 `ParameterDict` 与 `ParameterList` 的实现逻辑一致，因此，可以使用 `ParameterList` 的逻辑进行实现：

- `paddle.nn.ParameterDict` 作为独立的函数调用。

## 命名与参数设计

```python
class ParameterDict(parameters): ...

```

其中:

- parameters (dict)，输入的字典数据

## 底层OP设计

直接在 Python 层实现，不涉及底层算子。

## API实现方案

参考代码：

```python
class ParameterDict(paddle.nn.Layer):

    def __init__(self, parameters=None):
        super().__init__()
        self._keys = OrderedDict()
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key):
        if not isinstance(key, str):
            raise TypeError("ParameterDict的索引必须是字符串类型。")
        return key

    def __getitem__(self, key):
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key, value):
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, paddle.Tensor) and not isinstance(value, paddle.nn.Parameter):
            value = paddle.nn.Parameter(value)
        setattr(self, attr, value)

    def __delitem__(self, key):
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __reversed__(self):
        return reversed(self._keys)

    def copy(self):
        return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key):
        return key in self._keys

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self):
        for k in list(self._keys):
            del self[k]

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def popitem(self):
        k, _ = self._keys.popitem()
        val = self[k]
        del self[k]
        return k, val

    def get(self, key, default=None):
        return self[key] if key in self else default

    def fromkeys(self, keys, default=None):
        return ParameterDict((k, default) for k in keys)

    def keys(self):
        return self._keys.keys()

    def items(self):
        return ((k, self[k]) for k in self._keys)

    def values(self):
        return (self[k] for k in self._keys)

    def update(self, parameters):
        if not isinstance(parameters, (OrderedDict, ParameterDict, container_abcs.Iterable)):
            raise TypeError("ParameterDict.update should be called with an iterable of key/value pairs.")

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError("ParameterDict update sequence element #{} should be Iterable; is {}".format(j, type(p).__name__))
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element #{} has length {}; 2 is required".format(j, len(p)))
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self.items():
            if isinstance(p, paddle.Tensor):
                size_str = 'x'.join(str(size) for size in p.shape)
                parastr = '{} containing: [{} of size {}]'.format(
                    "Parameter" if isinstance(p, paddle.nn.Parameter) else "Tensor",
                    paddle.typename(p), size_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child
```

# 六、测试和验收的考量

目前 Paddle 对于 `alpha_dropout` 的单测，与其他 dropout 的单测，放置于：`test/legacy_test/test_dropout_op.py`，此处保持一致。

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
