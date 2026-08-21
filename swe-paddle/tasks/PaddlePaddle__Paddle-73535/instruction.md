# 修复 `paddle.nn.functional.conv1d` 在 CPU 上对 float16 weight/bias 的支持

## 详细描述

当 `paddle.nn.functional.conv1d(x, weight, bias, ...)` 在 CPU 设备上运行时，如果 `weight` 或 `bias` 的 dtype 为 `float16`，当前实现会直接调用底层 `conv2d` 算子，但 CPU 的 `conv2d` 算子不支持 `float16` 输入，导致运行时报错或精度异常。

典型表现包括：

- CPU 环境下传入 float16 的 weight/bias 时，conv1d 计算报错或产生错误结果
- PaddleAPITest 精度对比测试中 `paddle.nn.functional.conv1d` 在 CPU 上出现 accuracy diff

例如：

```python
import paddle
import paddle.nn.functional as F

with paddle.base.device_guard("cpu"):
    x = paddle.ones([1, 1, 1])
    w = paddle.ones([1, 1, 1]).astype(paddle.float16)
    b = paddle.ones([1]).astype(paddle.float16)
    y = F.conv1d(x, w, b)
    # 期望: y 的值为 [[[2]]]，dtype 为 float16
```

上述调用中 weight 和 bias 均为 float16，在 CPU 上应能正确执行并返回 float16 类型的结果。

当前 `conv1d` 函数在将 1D 卷积转换为 2D 卷积之前，没有对 CPU 设备上的 float16 输入进行类型转换处理。需要在调用底层 conv2d 算子之前，检测 CPU 设备并将 float16 的 weight 和 bias 临时转换为 x 的 dtype（通常为 float32），完成计算后再将结果转换回 float16。

## 验收说明

- 当 CPU 设备上 weight 或 bias 为 float16 时，`paddle.nn.functional.conv1d` 应正常完成计算
- 输出的 dtype 应保持为 float16（与输入 weight 的 dtype 一致）
- 输出的数值应与 GPU 上 float16 conv1d 的结果一致（精度在可接受范围内）
- 非 float16 输入（float32、float64 等）下的 conv1d 行为不得退化
- 非 CPU 设备（GPU）上的 conv1d 行为不受影响

## 技术要求

- 熟悉 Python 和 Paddle 动态图 API 开发
- 了解 Paddle 设备（device）检测和 dtype 转换机制
- 了解 conv1d 到 conv2d 的转换实现
- 不需要修改 C++ kernel，仅需修改 Python 层代码
