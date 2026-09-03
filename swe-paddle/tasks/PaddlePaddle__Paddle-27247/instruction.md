# 修复 distributed spawn 场景下多进程 DataLoader 无法启动的问题

## 详细描述

通过 `paddle.distributed.spawn` 启动训练进程，并在训练进程中使用开启了多进程读取的 DataLoader 时，DataLoader 在创建 worker 过程中会报错：

```text
TypeError: can't pickle _thread.lock objects
```

该错误会导致训练在开始读取数据前直接退出。

需要让多进程 DataLoader 能够在 `paddle.distributed.spawn` 启动的训练进程中正常创建 worker，同时保持现有的数据读取和进程管理行为不变。

## 验收说明

* 通过 `paddle.distributed.spawn` 启动训练时，多进程 DataLoader 能够正常创建 worker。
* Dataset DataLoader 和 generator DataLoader 的多进程读取流程都应正常工作。
* 现有的 worker 启动、进程注册、数据传递和异常处理行为保持不变。

## 技术要求

* 熟悉 Python multiprocessing 和进程间对象传递。
* 熟悉 PaddlePaddle DataLoader 的多进程读取流程。
* 了解 `paddle.distributed.spawn` 的进程启动方式。
