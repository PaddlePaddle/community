# 修复 `normal` 单元测试在动态图模式下运行静态用例时报错的问题

## 详细描述

`test/legacy_test/test_normal.py` 同时包含 `paddle.normal` 的静态模式和动态图模式测试。

目前，静态测试默认调用前已经处于静态模式。当 Paddle 当前处于动态图模式时，直接调用 `static_api()` 会出现以下问题：

* 使用标量 `mean` 和 `std` 时，`paddle.normal` 返回动态图 Tensor，之后传给静态 `Executor` 会报错：

```text
AttributeError: 'Tensor' object has no attribute 'desc'
```

* 使用 Tensor 形式的 `mean` 或 `std` 时，调用 `paddle.static.data()` 会报错：

```text
AssertionError: 'data()' is only supported in static graph mode
```

因此，这些测试能否通过会受到调用顺序影响，单独运行静态用例时也可能失败。

此外，复数用例中，当 `mean` 是复数标量、`std` 是数组时，静态输入的 dtype、shape 和实际 feed 数据需要保持一致，否则会出现 dtype 或 shape 不匹配。

需要修正这些测试，使静态用例无论从哪种执行模式开始都能正常构图和运行，并且不会影响后续动态图用例。

## 验收说明

* Paddle 当前处于动态图模式时，`static_api()` 仍能正常构建并执行静态程序
* 标量 `mean`、标量 `std` 的静态用例可以正常运行
* Tensor 形式的 `mean` 或 `std` 可以正常创建静态输入并执行
* 实数和复数相关用例都能正常运行
* 复数标量 `mean` 与数组 `std` 组合使用时，feed 数据的 shape 和 dtype 应与静态输入一致
* 静态用例执行完成后，后续动态图用例可以继续正常运行
* 原有的输出 shape 和随机分布数值检查保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle 的静态模式和动态图模式
* 了解 `paddle.static.data`、`Program` 和 `Executor`
* 了解 NumPy 数组的 dtype 和广播规则
