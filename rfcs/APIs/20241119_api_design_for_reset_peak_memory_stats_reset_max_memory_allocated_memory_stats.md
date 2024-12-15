# paddle.device.cuda.reset_max_memory_allocated / paddle.device.cuda.reset_max_memory_reserved 设计文档

| API名称                                                            | paddle.device.cuda.reset_max_memory_allocated / paddle.device.cuda.reset_max_memory_reserved |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | Qin-sx                                                                                                                       |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2024-11-19                                                                                                                   |
| 版本号                                                             | V1.0                                                                                                                         |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本                                                                                                                  |
| 文件名                                                             | 20241119_api_design_for_reset_max_memory_allocated_reset_max_memory_reserved.md<br>                           |

# 一、概述

## 1、相关背景

https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no21-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-reset_peak_memory_statsreset_max_memory_allocatedmemory_stats-api

## 2、功能目标

在 paddle.device.cuda 包中，增加对 CUDA 张量类型的以下两个支持  

1. **重置最大分配的GPU内存的跟踪起点**：新增API `reset_max_memory_allocated`，位于`paddle.device.cuda`路径下，用于重置特定设备上张量占用的最大GPU内存的跟踪起点。

2. **重置最大保留的GPU内存的跟踪起点**：新增API `reset_max_memory_reserved`，位于`paddle.device.cuda`路径下，用于重置特定设备上分配器持有的最大GPU内存的跟踪起点。

## 3、意义

新增paddle.device.cuda.reset_max_memory_allocated和paddle.device.cuda.reset_max_memory_reserved丰富 paddle API。

# 二、飞桨现状

飞桨（PaddlePaddle）目前提供了几个关于CUDA设备端内存信息的API，包括：

- `max_memory_allocated`：用于获取给定设备上分配给Tensor的显存峰值。[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/max_memory_allocated_cn.html#cn-api-paddle-device-cuda-max-memory-allocated)
- `max_memory_reserved`：用于获取给定设备上由Allocator管理的显存峰值。[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/max_memory_reserved_cn.html#cn-api-paddle-device-cuda-max-memory-reserved)
- `memory_allocated`：用于获取给定设备上当前分配给Tensor的显存大小。[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/memory_allocated_cn.html#cn-api-paddle-device-cuda-memory-allocated)
- `memory_reserved`：用于获取给定设备上当前由Allocator管理的显存大小。[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/memory_reserved_cn.html#cn-api-paddle-device-cuda-memory-reserved)

然而，飞桨尚未提供类似于`reset_max_memory_allocated`和`reset_max_memory_reserved`这样的API，这些API能够重置特定设备上GPU内存的峰值统计信息。为了进一步完善内存管理功能，飞桨应该加入`reset_max_memory_allocated`和`reset_max_memory_reserved`API。这些新增的API将有助于开发者更精确地监控和控制内存使用情况。

# 三、业内方案调研

## PyTorch

### `cuda.reset_peak_memory_stats`的实现

#### Python 接口

`reset_peak_memory_stats`函数主要通过`torch._C._cuda_resetPeakMemoryStats` 函数实现。

```python
def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Reset the "peak" stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetPeakMemoryStats(device)
```

#### C++ 实现

`torch._C._cuda_resetPeakMemoryStats`函数位于 `torch/csrc/cuda/Module.cpp`中，并通过 Python C API 注册到 Python 模块中。

```C++
static struct PyMethodDef _THCPModule_methods[] = {
    {"_cuda_resetPeakMemoryStats",
     THCPModule_resetPeakMemoryStats,
     METH_O,
     nullptr}
     // others...
}
```
`THCPModule_resetPeakMemoryStats`函数主要通过调用`c10::cuda::CUDACachingAllocator::resetPeakStats`函数来实现。

```C++
PyObject* THCPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}
```

`c10::cuda::CUDACachingAllocator::resetPeakStats`函数主要是调用`NativeCachingAllocator`类的`resetPeakStats`函数。

```C++
C10_CUDA_API extern std::atomic<CUDAAllocator*> allocator;

inline CUDAAllocator* get() {
  return allocator.load();
}

inline void resetPeakStats(c10::DeviceIndex device) {
  return get()->resetPeakStats(device);
}
```

`NativeCachingAllocator`类的`resetPeakStats`函数主要是调用`DeviceCachingAllocator`类的`resetPeakStats`函数。

```C++
class NativeCachingAllocator : public CUDAAllocator {
public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  void resetPeakStats(c10::DeviceIndex device) override {
    assertValidDevice(device);
    device_allocator[device]->resetPeakStats();
  }
}
```

`DeviceCachingAllocator`类的`resetPeakStats`函数将`DeviceStats`类中的所有相关参数从`peak`改为`current`。

```C++
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from device memory allocation.
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via device memory deallocation)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory alocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // others ...
}

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocation[statType].reset_peak();
      stats.segment[statType].reset_peak();
      stats.active[statType].reset_peak();
      stats.inactive_split[statType].reset_peak();
      stats.allocated_bytes[statType].reset_peak();
      stats.reserved_bytes[statType].reset_peak();
      stats.active_bytes[statType].reset_peak();
      stats.inactive_split_bytes[statType].reset_peak();
      stats.requested_bytes[statType].reset_peak();
    }
    stats.oversize_allocations.reset_peak();
    stats.oversize_segments.reset_peak();
  }
}
```

### `cuda.reset_max_memory_allocated`的实现

#### Python 接口

PyTorch中的`reset_max_memory_allocated`函数是通过调用`reset_peak_memory_stats`函数实现，即会将所有的内存状态重置。

```python
def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

    See :func:`~torch.cuda.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)
```

#### 备注
PyTorch中的`reset_max_memory_cached`函数也是通过调用`reset_peak_memory_stats`函数实现。

```python
def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.
    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "torch.cuda.reset_max_memory_cached now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)
```

PyTorch中的`reset_max_memory_reserved`函数应该是被弃用了。

```
.. FIXME The following doesn't seem to exist. Is it supposed to?
   https://github.com/pytorch/pytorch/issues/27785
   .. autofunction:: reset_max_memory_reserved
```

### `cuda.memory_stats`的实现

#### Python 接口

`memory_stats`函数主要通过`memory_stats_as_nested_dict`函数收集有关内存管理的信息。

```python
def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return a dictionary of CUDA memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cudaMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cudaMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of CUDA allocation calls. This includes both
      cuMemMap and cudaMalloc.
    - ``"num_device_free"``: number of CUDA free calls. This includes both cuMemUnmap
      and cudaFree.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the Cuda Semantics documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cudaMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.

    .. note::
        With :ref:`backend:cudaMallocAsync<cuda-memory-envvars>`, some stats are not
        meaningful, and are always reported as zero.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)
```

`memory_stats_as_nested_dict`函数主要通过`torch._C._cuda_memoryStats`函数实现。

```python
def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_memoryStats(device)
```

#### C++ 实现

`torch._C._cuda_memoryStats`函数位于 `torch/csrc/cuda/Module.cpp`中，并通过 Python C API 注册到 Python 模块中。

```C++
static struct PyMethodDef _THCPModule_methods[] = {
  {"_cuda_memoryStats", THCPModule_memoryStats, METH_O, nullptr}
  // others...
}
```

`THCPModule_memoryStats`函数主要将`DeviceStats`的信息存入字典中并返回。

```C++
PyObject* THCPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);

  using c10::CachingDeviceAllocator::DeviceStats;
  using c10::CachingDeviceAllocator::Stat;
  using c10::CachingDeviceAllocator::StatArray;
  using c10::CachingDeviceAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["num_sync_all_streams"] = stats.num_sync_all_streams;
  result["num_device_alloc"] = stats.num_device_alloc;
  result["num_device_free"] = stats.num_device_free;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}
```

## TensorFlow

TensorFlow中关于GPU内存使用信息的函数主要是`reset_memory_stats`函数。

### `reset_memory_stats`的实现

#### Python 接口

```python
@tf_export('config.experimental.reset_memory_stats')
def reset_memory_stats(device):
  """Resets the tracked memory stats for the chosen device.

  This function sets the tracked peak memory for a device to the device's
  current memory usage. This allows you to measure the peak memory usage for a
  specific part of your program. For example:

  >>> if tf.config.list_physical_devices('GPU'):
  ...   # Sets the peak memory to the current memory.
  ...   tf.config.experimental.reset_memory_stats('GPU:0')
  ...   # Creates the first peak memory usage.
  ...   x1 = tf.ones(1000 * 1000, dtype=tf.float64)
  ...   del x1 # Frees the memory referenced by `x1`.
  ...   peak1 = tf.config.experimental.get_memory_info('GPU:0')['peak']
  ...   # Sets the peak memory to the current memory again.
  ...   tf.config.experimental.reset_memory_stats('GPU:0')
  ...   # Creates the second peak memory usage.
  ...   x2 = tf.ones(1000 * 1000, dtype=tf.float32)
  ...   del x2
  ...   peak2 = tf.config.experimental.get_memory_info('GPU:0')['peak']
  ...   assert peak2 < peak1  # tf.float32 consumes less memory than tf.float64.

  Currently only supports GPU and TPU. If called on a CPU device, an exception
  will be raised.

  Args:
    device: Device string to reset the memory stats, e.g. `"GPU:0"`, `"TPU:0"`.
      See https://www.tensorflow.org/api_docs/python/tf/device for specifying
      device strings.

  Raises:
    ValueError: No device found with the device name, like '"nonexistent"'.
    ValueError: Invalid device name, like '"GPU"', '"CPU:GPU"', '"CPU:"'.
    ValueError: Multiple devices matched with the device name.
    ValueError: Memory statistics not tracked or clearing memory statistics not
      supported, like '"CPU:0"'.
  """
  context.context().reset_memory_stats(device)
```

`reset_memory_stats`函数主要是由`Context`类的`reset_memory_stats`函数实现。

```python
class Context:
  def reset_memory_stats(self, dev):
    """Resets the tracked memory stats for the device."""
    self._initialize_physical_devices()
    self.ensure_initialized()
    pywrap_tfe.TFE_ResetMemoryStats(self._context_handle, dev)
```

`reset_memory_stats`函数主要通过`TFE_ResetMemoryStats`函数实现。

#### C++ 实现

`TFE_ResetMemoryStats`函数通过 PYBIND11 注册到 Python 模块中。

```C++
PYBIND11_MODULE(_pywrap_tfe, m) {
  m.def("TFE_ResetMemoryStats", [](py::handle& ctx, const char* device_name) {
    tensorflow::Device* matched_device =
        tensorflow::GetMatchedDevice(ctx, device_name);

    tensorflow::AllocatorAttributes attrs;
    tensorflow::Allocator* allocator = matched_device->GetAllocator(attrs);

    if (!allocator->ClearStats()) {
      tensorflow::ThrowValueError(
          absl::StrFormat("Cannot reset memory stats for device '%s'",
                          device_name)
              .c_str());
    }
  });
  // others...
}
```

`TFE_ResetMemoryStats`函数主要通过`tensorflow::Allocator`的`ClearStats`函数实现。

# 四、对比分析

由于 PyTorch 中的函数结构更符合飞桨的要求，因此可以参考 PyTorch 中的函数实现。

# 五、设计思路与实现方案

## 命名与参数设计

API `paddle.device.cuda.reset_max_memory_allocated(device: _CudaPlaceLike | None = None) -> None`
paddle.device.cuda.reset_max_memory_allocated
----------------------
参数
- device (_CudaPlaceLike) - 输入device名称或者序号。
- None 无返回值。

API `paddle.device.cuda.reset_max_memory_reserved(device: _CudaPlaceLike | None = None) -> None`
paddle.device.cuda.reset_max_memory_reserved
----------------------
参数
- device (_CudaPlaceLike) - 输入device名称或者序号。
- None 无返回值。

## 底层设计

### `cuda.reset_max_memory_allocated`的实现

在`Stat`类中，加入`ResetPeakValue`函数，将类中记录的`peak_value_`改为当前的`current_value`的值。同时将各个线程中存储的`peak`值改为`current`值。  
在`StatBase`类中加入`ResetPeakValue`虚函数接口。

```C++
class StatBase {
 public:
  virtual void ResetPeakValue() = 0;
};

template <typename ThreadLocalStatType>
class Stat : public StatBase {
  // 新增函数，用于将 peak_value_ 设置为 current_value_
  void ResetPeakValue() override {
    int64_t current_value = GetCurrentValue();
    peak_value_.store(current_value, std::memory_order_relaxed);

    std::unordered_map<uint64_t, std::reference_wrapper<ThreadLocalStatType>> thread_local_stats =
      ThreadDataRegistry<ThreadLocalStatType>::GetInstance().GetAllThreadDataByRef();

    for (auto pair : thread_local_stats) {
      pair.second.get().peak = pair.second.get().current;
    }

    VLOG(8) << "Reset peak_value to current_value = " << current_value;
  }
};

void DeviceMemoryStatResetPeakValue(const std::string& stat_type, int dev_id) {
  StatRegistry::GetInstance()->ResetPeakValue("Device" + stat_type, dev_id);
}
```

调用函数时参数输入为`Allocated`，重置`分配给Tensor的显存峰值`。

### `cuda.reset_max_memory_reserved`的实现

与`reset_peak_memory_stats`函数的实现相似，但是调用函数时参数输入为`Reserved`，重置`由Allocator管理的显存峰值`。

## API实现方案

通过PYBIND11将C++函数注册到Python模块中。

```C++
PYBIND11_MODULE(libpaddle, m){
  m.def("device_memory_stat_reset_peak_value", memory::DeviceMemoryStatResetPeakValue);
}
```

### `cuda.reset_max_memory_allocated`的实现

调用PYBIND11中注册的`device_memory_stat_reset_peak_value`函数，参数输入为`Allocated`（仅重置`分配给Tensor的显存峰值`）。

```python
def reset_max_memory_allocated(device: _CudaPlaceLike | None = None) -> None:
    device_id = extract_cuda_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Allocated", device_id)
```

### `cuda.reset_max_memory_reserved`的实现

调用PYBIND11中注册的`device_memory_stat_reset_peak_value`函数，参数输入为`Reserved`（仅重置`由Allocator管理的显存峰值`）。

```python
def reset_max_memory_reserved(device: _CudaPlaceLike | None = None) -> None:
    device_id = extract_cuda_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Reserved", device_id)
```

# 六、测试和验收的考量

## 单元测试

### `DeviceMemoryStatResetPeakValue`和`HostMemoryStatResetPeakValue`

#### 功能测试
  - 更新一定量的内存。
  - 调用`DeviceMemoryStatResetPeakValue`或者`HostMemoryStatResetPeakValue`函数。
  - 确保峰值被重置为当前值。

#### 验证重置效果
  - 重置后，获取当前的峰值内存使用量，应该等于当前的内存使用量。

### `reset_max_memory_allocated`

#### 功能测试
  - 分配内存，记录`分配给Tensor的显存峰值`。
  - 调用 `reset_max_memory_allocated` 函数。
  - 再次分配内存，验证`分配给Tensor的显存峰值`从重置后开始统计。

#### 验证重置效果
  - 重置后，获取`分配给Tensor的显存峰值`，应该等于`当前分配给Tensor的显存大小`。

### `reset_max_memory_reserved`

#### 功能测试
  - 分配内存，记录`由Allocator管理的显存峰值`。
  - 调用 `reset_max_memory_allocated` 函数。
  - 再次分配内存，验证`由Allocator管理的显存峰值`从重置后开始统计。

#### 验证重置效果
  - 重置后，获取`由Allocator管理的显存峰值`，应该等于`当前由Allocator管理的显存大小`。


# 七、影响面

## 需要进一步讨论的问题
1. 是否对`Stat`类进行修改。（对`Stat`类进行了修改）
2. 目前`StatRegistry`类中是否只注册了`Allocated`和`Reserved`。（目前只注册了`Allocated`和`Reserved`）
3. 是否参考Pytorch注册更多的内存信息标签。（目前不需要注册其他内存信息标签）

## 对二次开发用户的影响

`Stat`类中新增的函数`ResetPeakValue`会暴露给二次开发用户。

# 八、排期规划

1. 2024/11/30前提交完善后的RFC文档。
2. 2024/12/08前提交第一版代码。
3. 2024/12/14前提交优化后的代码和API文档。

# 九、参考资料

飞桨内存池统计方案：
https://patentimages.storage.googleapis.com/86/71/69/a6a95a1bcb9b05/CN114610575B.pdf

# 十、致谢

感谢陈锐彪老师和骆涛老师在飞桨GPU内存管理方面的指导。