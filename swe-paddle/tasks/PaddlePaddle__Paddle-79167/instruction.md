# 为 paddle.random.initial_seed 增加顶层公共别名

## 详细描述

Paddle 已提供 `paddle.random.initial_seed` 用于读取初始随机种子，但该 API 尚未从 `paddle` 顶层命名空间公开。调用方和自动迁移工具通常期望通过 `paddle.initial_seed` 访问同一能力。需要增加该顶层公共 API，同时保持现有随机数 API 的对象关系和导出行为稳定。

## 验收说明

- `paddle.initial_seed` 应可从 Paddle 顶层命名空间访问。
- `paddle.initial_seed` 应与 `paddle.random.initial_seed` 指向同一个函数对象，而不是行为相似的重复实现。
- 新名称应纳入 Paddle 的标准公共导出集合。
- 现有 `paddle.seed` 和 `paddle.manual_seed` 行为应保持不变。

## 技术要求

- 熟悉 Python package 顶层 API 导出机制。
- 了解 `__all__` 与 API alias 的语义。
- 能够维护既有公共 API 的向后兼容性。
