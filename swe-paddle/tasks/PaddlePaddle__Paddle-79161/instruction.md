# 为 paddle.set_rng_state 增加 new_state 参数别名

## 详细描述

`paddle.set_rng_state` 当前仅接受 `state_list` 作为随机数生成器状态参数。使用其他框架风格代码或自动迁移工具时，调用方可能使用语义等价的 `new_state` 参数，导致现有 API 无法直接兼容。该 API 应支持 `new_state` 作为 `state_list` 的等价别名，同时保持原有调用方式和错误语义稳定。

## 验收说明

- `paddle.set_rng_state(new_state=...)` 应与使用 `state_list=...` 产生相同的状态设置行为。
- positional 参数和原有 `state_list=` 关键字调用应保持兼容。
- 同时传入 `state_list` 与 `new_state` 时，应明确拒绝冲突输入。

## 技术要求

- 熟悉 Python function signature 和 decorator 行为。
- 了解 Paddle random generator state API。
- 能够编写稳定的 API compatibility behavior tests。
