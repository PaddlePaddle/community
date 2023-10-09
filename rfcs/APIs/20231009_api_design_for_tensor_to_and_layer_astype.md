# paddle.Tensor.to / paddle.Layer.astype设计文档

| API名称                                                      | paddle.Tensor.to / paddle.Layer.astype                    |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| 提交作者  | YibinLiu666                                               |
| 提交时间  | 2023-10-09                                                |
| 版本号                                                       | V1.0                                                      |
| 依赖飞桨版本 | v2.5.0                                                    |
| 文件名                                                       | 20231009_api_design_for_tensor_to_and_layer_astype.md<br> |

# 一、概述

## 1、相关背景

在深度学习中，使用张量（tensor）作为数据的基本单位。张量可以在不同的设备上进行计算，如 CPU 或 GPU。`Tensor.to()` 方法允许将张量从一个设备转移到另一个设备。这对于利用 GPU 进行加速或在多个设备之间进行数据传输非常有用。此外，深度学习模型通常对输入数据的数据类型有要求，例如要求输入是 float32 类型。因此，`Tensor.to()` 也可以用于将张量的数据类型转换为满足模型要求的类型。

层（layer）是模型的基本组成部分，由权重和偏置参数组成。这些参数的数据类型可能在不同的模型或框架中有所不同。`Layer.astype()` 用于将层的参数以及层接收的输入数据的数据类型转换为特定的类型。这在处理不同数据类型之间的兼容性问题时非常有用。例如，某些情况下，使用较低精度的数据类型（如 float16）可以减少模型的内存占用和计算量，从而提高训练和推理的效率。

## 2、功能目标

- `Tensor.to`：对Tensor进行设备类型或数据类型的转换，输入参数需要支持多种形式，需支持多种用法，例如： `x.to('float64')`、`x.to('cpu')`、`x.to('cpu', 'float32')`、`x.to(y)`，同时上述例子均可设置blocking来控制是否同步阻塞拷贝。通过(*args, **kwargs)的参数设置可实现上述所有功能。
- `Layer.astype`：支持对网络层进行数据类型的转换，例如`Linear.astype('float64')`



## 3、意义

`Tensor.to()` 和 `Layer.astype()` 提供了在深度学习中进行数据转换和类型转换的便利方法，以满足模型和设备之间的要求，并处理不同数据类型之间的兼容性问题。

# 二、飞桨现状

目前paddle没有顶层接口的实现。但是已经实现了`Tensor._to(device, dtype, blocking)`的方式将张量转移到另一个设备或者类型。

# 三、业内方案调研

## Pytorch

Pytorch中有API`Tensor.to(*args, **kwargs) → Tensor`，目前支持三种调用方式：

- ` to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) → Tensor `
- `to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) → Tensor`
- `to(other, non_blocking=False, copy=False) → Tensor`这里的other为另一个tensor

对于层的类型转换，pytorch的`torch.nn.module`提供了`module.type(dst_type)`将层的参数全部转为目标数据类型。

### 实现方法

由于paddle已经有了`Tensor._to(device, dtype, blocking)`接口因此只需要参考pytorch的顶层接口`Tensor.to(*args, **kwargs) → Tensor`是怎么处理接收到的参数即可，pytorch的`ShardedTensor`处理方式为：

```c++
def to(self, *args, **kwargs) -> ShardedTensor:
        current_device: torch.device
        if self._local_shards:
            current_device = self._local_shards[0].tensor.device
        elif self._process_group._get_backend_name() == "gloo":
            current_device = torch.device("cpu")
        else:
            current_device = torch.device(torch.cuda.current_device())
        current_dtype = self.dtype
        device_to = current_device
        dtype_to = current_dtype
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_to = args[0]
            elif isinstance(args[0], torch.device):
                device_to = args[0]
            elif isinstance(args[0], (str, int)):
                device_to = torch.device(args[0])
            elif isinstance(args[0], torch.Tensor):
                dtype_to = args[0].dtype
                device_to = args[0].device
            else:
                raise RuntimeError(f"ShardedTensor.to() have wrong arguments: {args}")
        elif len(args) == 2:
            device_to, dtype_to = args
        else:
            dtype_to = kwargs.get("dtype", current_dtype)
            device_to = kwargs.get("device", current_device)

        device_to = torch.device(device_to) if isinstance(device_to, (str, int)) else device_to

        if device_to.type == "cuda":
            # if device_to set to cuda, set to current device even
            # if user specify the device index.
            current_idx = torch.cuda.current_device()
            if device_to.index != current_idx:
                warnings.warn("ShardedTensor.to only move tensor to its current device"
                              "If you want to put to different device, use `reshard` instead.")
            device_to = torch.device(current_idx)

        copy_tensor = kwargs.get("copy", False)
        non_blocking = kwargs.get("non_blocking", False)
        memory_format = kwargs.get("memory_format", torch.preserve_format)
        process_group = kwargs.get("process_group", None)

        if not copy_tensor and dtype_to == current_dtype and device_to == current_device:
            # already have correct dtype and device, return itself
            return self

        # returns a copy of ShardedTensor on CUDA current device
        list_shards: List[Shard] = []

        for shard in self._local_shards:
            new_tensor = shard.tensor.to(  # type: ignore[call-overload]
                device=device_to,
                dtype=dtype_to,
                non_blocking=non_blocking,
                copy=copy_tensor,
                memory_format=memory_format
            )
            metadata = copy.deepcopy(shard.metadata)
            if metadata.placement is not None:
                metadata.placement._device = device_to
            list_shards.append(Shard(new_tensor, metadata))

        # update metadata
        st_meta = copy.deepcopy(self.metadata())
        st_meta.tensor_properties.dtype = dtype_to
        for meta in st_meta.shards_metadata:
            meta.placement._device = device_to  # type: ignore[union-attr]

        pg = self._process_group if process_group is None else process_group
        # we need to use `init_from_local_shards` to communicate between ranks
        # and update the sharding spec/shards metadata.
        st_to = ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs
        )
        return st_to
```

整体逻辑为：

- 先根据第`args`的长度和第一个参数来判断是哪种输入形式。
- 如果第一个参数是`device`或者是`dtype`，则直接将参数作为device或者dtype的值
- 如果第一个参数是`tensor`，则需要从tensor中获取`device`和`dtype`的值
- 获取`blocking`等其他参数的值
- 将这些值传给底层接口，实现`Tensor`的转换

对于pytorch的`module.type`，实现方式为：

```python
def type(self, dst_type):
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))
```

整体逻辑就是将模型的所有参数拿出来，每个参数都tensor，将这些tensor的类型转为目标类型即可。

# 四、对比分析

pytorch的`Tensor.to`和`module.type`的实现方式对于paddle来说具有较强的参考性，可以利用其实现思路来实现paddle的这两个功能。


# 五、方案设计

## 命名与参数设计

添加python API

```python
class Tensor:
    ...
    to(
        self,
        *args,
        **kwargs
    )
```

类似于pytorch，支持三种调用方式：

- ` to(dtype, blocking=None) → Tensor `
- `to(device, dtype=None, blocking=None) → Tensor`
- `to(other, blocking=None) → Tensor`

同时添加python API

```python
class Layer:
	...
	astype(
		self,
		dst_type
	)
```

- `dst_type`可以是`["bfloat16","float16","float32","float64", "int8","int16","int32","int64","uint8","uint16","complex64","complex128", "bool"]`中的任意一个字符串，也可以是`paddle.dtype`


## 底层OP设计

使用已有API组合实现，不再单独设计OP。

## API实现方案

对于`Tensor.to`: 先根据输入的参数长度和第一个参数的类型来判断是哪种输入形式，这里有一点与pytorch不同的是，paddle支持`str`类型的`type`输入，`type`如果是`str`的话需要是`["bfloat16","float16","float32","float64", "int8","int16","int32","int64","uint8","uint16","complex64","complex128", "bool"]`中的任意一个字符串。然后根据输入的形式提取出`device`，`type`以及`blocking`这三个参数的值，将这三个参数传入到`Tensor._to`接口中实现tensor的转换。

对于`Layer.astype`，由于每个参数的类型都是tensor，因此可以参考pytorch，将层的每一个参数拿出来进行类型转换即可。

# 六、测试和验收的考量

测试考虑的case如下：

- tensor仅进行设备的切换，包括cpu到gpu，gpu到cpu等。
- tensor仅进行数据类型的切换，float32到float64等。
- tensor切换数据类型和设备。
- 同步阻塞拷贝测试。
- layer数据类型的切换测试。

# 七、可行性分析及规划排期

方案主要依赖现有paddle api，仅需要对输入参数进行处理，`layer.astype`也只需要对网络的每个参数进行类型切换，实现较为简单，预计可以在11月之前完成。

# 八、影响面

为`Tensor`独立新增方法，对其他模块没有影响



# 名词解释

无

# 附件及参考资料

无
