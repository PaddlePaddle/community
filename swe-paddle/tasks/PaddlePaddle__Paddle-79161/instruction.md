# 为 `paddle.set_rng_state` 的 `state_list` 参数增加 `new_state` 别名

## 详细描述

`paddle.set_rng_state` 目前只支持通过 `state_list` 传入随机数生成器状态。部分代码使用 `new_state` 作为参数名，无法直接调用该接口。

`new_state` 应作为 `state_list` 的别名。下面三种调用方式应具有相同的效果：

```python
paddle.set_rng_state(states)
paddle.set_rng_state(state_list=states)
paddle.set_rng_state(new_state=states)
```

如果同时传入 `state_list` 和 `new_state`，应抛出 `ValueError`。现有的 `device` 参数和调用方式不应受到影响。

## 验收说明

* 支持通过 `new_state=` 设置随机数生成器状态
* 位置参数、`state_list=` 和 `new_state=` 三种调用方式结果一致
* 同时传入 `state_list` 和 `new_state` 时抛出 `ValueError`
* `device` 参数和现有调用方式保持不变

## 技术要求

* 熟悉 Python 函数参数处理
* 了解 Paddle 随机数生成器状态 API
