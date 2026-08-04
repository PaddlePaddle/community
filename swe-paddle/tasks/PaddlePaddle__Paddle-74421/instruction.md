# 新增 `paddle.msort` API

## 详细描述

Paddle 当前没有 `msort` 接口。需要新增 `paddle.msort(input)`，沿输入 Tensor 的第 0 维进行升序排序。

该接口只接收一个名为 `input` 的 Tensor 参数，不提供 `axis`、`descending` 或 `out` 参数。返回结果的 shape 和 dtype 应与输入保持一致。

除了 `paddle.msort`，还应支持以下调用方式：

```python
paddle.tensor.msort(input)
input.msort()
````

## 验收说明

* `paddle.msort`、`paddle.tensor.msort` 和 `Tensor.msort()` 均可正常调用
* 支持位置参数和 `input=` 关键字参数
* 无论输入有多少个维度，都固定沿第 0 维升序排序
* 返回结果的 shape 和 dtype 与输入一致
* 在动态图和 `static mode` 下均可正常使用
* 现有 `paddle.sort` 的行为保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle Tensor API
* 了解 Paddle API 的导出和 Tensor 方法注册方式
