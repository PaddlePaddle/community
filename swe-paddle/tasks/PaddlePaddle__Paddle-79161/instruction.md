# 为 `paddle.set_rng_state` 的 `state_list` 参数添加 `new_state` 别名

## 详细描述

`paddle.set_rng_state` 当前使用 `state_list` 接收需要恢复的随机数生成器状态。为提升接口兼容性，该函数还需要支持与 `state_list` 语义相同的 `new_state` 参数名称。

通过位置参数、`state_list=` 或 `new_state=` 传入相同状态时，应产生一致的状态设置结果。原有的参数形式以及设备选择行为不得受到影响。

当一次调用同时使用 `state_list` 和 `new_state` 时，应将其视为冲突输入并给出明确的异常。

## 验收说明

* `paddle.set_rng_state` 应支持通过 `new_state` 传入随机数生成器状态
* 使用 `new_state=` 和 `state_list=` 传入相同状态时，应产生一致的结果
* 原有的位置参数调用方式应保持有效
* 原有的 `state_list=` 关键字调用方式应保持有效
* 同时传入 `state_list` 和 `new_state` 时，应抛出 `ValueError`
* 现有 `device` 参数及其他合法调用方式的行为不得发生变化

## 技术要求

* 熟悉 Python 函数参数和关键字参数处理
* 了解 Paddle 随机数生成器状态相关 API
* 理解参数别名和冲突参数的兼容性要求
