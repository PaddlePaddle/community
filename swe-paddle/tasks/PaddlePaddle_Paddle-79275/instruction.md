# 新增 flex_attention mask 组合 API

## 详细描述

Paddle 需要提供与 PyTorch 对齐的 flex attention mask 组合接口，使调用方能够将一个或多个 mask 函数组合成新的 mask 函数。

每个 mask 函数接收 batch、head、query index 和 key/value index，并返回对应位置的布尔结果。新增接口需要分别支持逻辑 OR 和逻辑 AND 组合，并能够在动态图和静态图模式下正常使用。

## 验收说明

* 提供 `paddle.nn.attention.flex_attention.or_masks` 和 `paddle.nn.attention.flex_attention.and_masks`
* 组合后的函数应继续接收 batch、head、query index 和 key/value index，并返回布尔 Tensor
* `or_masks` 应返回所有输入 mask 结果的逻辑 OR
* `and_masks` 应返回所有输入 mask 结果的逻辑 AND
* 仅传入一个 mask 时，组合结果应与该 mask 的结果一致
* 未传入 mask 时，`or_masks` 应返回 `False`，`and_masks` 应返回 `True`
* 任一输入不是 callable 时，应抛出 `RuntimeError`
* 相关接口应支持动态图和静态图模式
* `paddle.nn.attention` 中已有的公共 API 和合法调用方式不得受到影响

## 技术要求

* 熟悉 Python callable 和布尔 Tensor 运算
* 了解 Paddle attention API 的模块组织及公共接口导出方式
* 了解 Paddle 动态图和静态图模式
* 能够为组合结果、边界输入和异常输入设计稳定的 CPU 测试
