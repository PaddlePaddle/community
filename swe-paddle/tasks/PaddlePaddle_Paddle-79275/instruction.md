# 增加 flex_attention mask 组合 API

## 详细描述

Paddle 的 flex attention 接口需要提供 mask 组合能力，使调用方可以将一个或多个满足 attention mask 调用约定的函数组合成新的 mask callable。组合后的 callable 应继续接收 batch、head、query index 和 key/value index 参数，并返回对应的布尔 mask 结果。

需要同时支持逻辑 OR 和逻辑 AND 两种组合方式。单个 mask 的组合结果应与原 mask 等价；空参数调用应具有合理的逻辑恒等行为；传入不可调用对象时应明确拒绝。现有 `paddle.nn.attention` 公共接口必须保持兼容。

## 验收说明

- 提供 `paddle.nn.attention.flex_attention.or_masks` 和 `and_masks`，能够正确组合一个或多个合法 mask callable 的逻辑结果。
- 单 mask、空 mask 集合以及包含非 callable 输入时，应分别表现出一致的逻辑恒等与输入校验行为。
- `paddle.nn.attention` 中已有的公共 API 和合法调用方式应保持不变。

## 技术要求

- 熟悉 Python callable、闭包以及布尔 Tensor 运算语义。
- 了解 Paddle attention API 的模块组织与公共导出方式。
- 能够为 API 边界、组合行为和异常输入设计稳定的 CPU-only 回归测试。
