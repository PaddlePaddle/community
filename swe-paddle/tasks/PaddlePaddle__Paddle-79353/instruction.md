# 修复 Pipeline P2P overlap 在无需通信时访问未定义变量的问题

## 详细描述

开启 P2P overlap 后，当 `pp_first_stage=True` 时，`recv_forward` 和 `send_backward` 不需要执行实际通信。

目前，这两个接口在该分支返回时会访问尚未赋值的 `wait_handles`，导致程序报错：

```text
UnboundLocalError: cannot access local variable 'wait_handles'
````

没有通信任务时，应正常返回 `None`，而不是因为 `wait_handles` 未定义而中断执行。

## 验收说明

* `pp_first_stage=True` 且开启 overlap 时，`recv_forward` 不应报错，并返回 `(None, None)`
* `pp_first_stage=True` 且开启 overlap 时，`send_backward` 不应报错，并返回 `None`
* 未开启 overlap 时，现有返回结果保持不变
* 需要执行 P2P 通信的其他分支保持原有行为

## 技术要求

* 熟悉 Python
* 了解 Pipeline 并行和 P2P 通信
* 了解同步通信与 overlap 通信的区别
