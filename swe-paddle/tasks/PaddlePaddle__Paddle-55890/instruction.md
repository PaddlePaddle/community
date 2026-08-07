# 修复 VPP 与 sharding/DP overlap 同时开启时的梯度同步问题

## 详细描述

在 virtual pipeline parallel（VPP）训练中开启 sharding 或 data parallel 的梯度通信 overlap 后，部分 model chunk 的梯度可能没有在正确的 backward step 进行同步。

当前代码使用 `accumulate_steps` 判断何时发起通信。当 `accumulate_steps` 和 `num_stages` 不同时，部分 model chunk 的梯度通信可能延迟或遗漏。对于 `stage_id != 0` 的情况，backward 全部完成后，chunk 0 的梯度也可能仍未同步。

需要修正 VPP 下的梯度通信调度，使各个 model chunk 都能在正确的时机完成通信。

## 验收说明

* 每个 model chunk 应在对应的 backward 完成后正常发起梯度通信
* `accumulate_steps` 和 `num_stages` 不同时，不应遗漏或重复梯度通信
* `stage_id != 0` 时，backward 结束前应完成 chunk 0 的梯度通信
* 原有能够正常运行的 VPP overlap 配置应保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle VPP
* 了解 sharding 和 data parallel 的梯度通信 overlap
* 了解 model chunk 和 backward 调度
