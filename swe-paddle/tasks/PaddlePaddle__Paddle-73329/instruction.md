# 支持 SOT 中的 dataclass 构造和基本访问

## 详细描述

在 SOT strict mode 中，程序无法完整执行 Python `dataclass` 的构造和基本实例操作。带有必填字段、默认值、`default_factory` 或 keyword-only 字段的 dataclass 构造可能触发 fallback；实例字段读取、临时更新与恢复以及不同 dataclass 实例之间的比较也无法稳定保持 eager execution 的行为。这限制了使用结构化 Python 数据对象的 dynamic-to-static 程序。

## 验收说明

- SOT strict mode 应支持使用位置参数和关键字参数构造 dataclass，并正确处理默认值、`field`、`default_factory` 和 keyword-only 字段。
- dataclass 实例的字段读取、字段更新与恢复以及相等性比较应与 eager execution 结果一致。
- 已有普通容器、对象属性和 side effect 处理行为应保持兼容。

## 技术要求

- 熟悉 Python dataclass、字段默认值与对象比较语义
- 熟悉 Paddle SOT / dynamic-to-static 执行机制
- 熟悉变量追踪、对象重建与 side effect 管理
