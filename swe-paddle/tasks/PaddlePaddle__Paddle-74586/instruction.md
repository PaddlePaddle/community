# 新增 `paddle.scatter_add` API

## 详细描述

Paddle 目前缺少 `scatter_add` 接口。

现在需要新增 `paddle.scatter_add(input, dim, index, src)`，以 `input` 为初始结果，按照 `index` 指定的位置，将 `src` 中的值沿 `dim` 维累加进去。

当多个索引指向同一位置时，这些值都应累加，不能相互覆盖。该接口返回新的 Tensor，不修改输入的 `input`。

同时支持以下调用方式：

```python
paddle.scatter_add(input, dim, index, src)
paddle.tensor.scatter_add(input, dim, index, src)
input.scatter_add(dim, index, src)
````

## 验收说明

* 支持 `input`、`dim`、`index` 和 `src` 参数
* 应在 `input` 原有数据的基础上累加 `src`
* 多个索引指向同一位置时，应正确累加所有对应值
* 返回结果的 shape 和 dtype 应与 `input` 保持一致
* `index` 支持 `int32` 和 `int64`
* `src` 的 dtype 应与 `input` 一致
* 调用后不应修改 `input`
* 三种调用方式应得到相同结果
* 现有 Tensor 操作接口的行为保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle Tensor API
* 了解按索引进行数据更新和累加的方式
