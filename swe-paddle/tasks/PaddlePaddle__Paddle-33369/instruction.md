# 为 collective 分布式训练增加可选的故障恢复流程

## 详细描述

collective 分布式训练运行时间较长时，任意 worker 或节点异常退出都会让当前任务直接结束，用户只能手动清理并重新启动整组训练。多节点任务还可能因为各节点启动时间不同，出现部分进程已经运行、其他节点尚未就绪的情况。

需要为 fleet launcher 增加可选的 elastic 流程。启用后，各节点在训练开始前先完成组网，训练期间持续关注本地进程和其他节点的状态；worker 失败时给出明确的重启结果，节点成员发生变化时先停止当前进程组并等待重新组网。未配置 elastic 时，原有 collective 和 parameter-server 启动方式应保持不变。

## 验收说明

- 启用 elastic 后，collective 任务应在节点就绪后启动，并根据 worker 退出结果区分完成、重启和错误状态。
- 训练期间节点成员发生变化时，应停止当前进程组并等待下一次组网，不能继续使用过期的集群信息。
- 未启用 elastic 时，已有 collective 和 parameter-server launch 流程保持不变。

## 技术要求

- 熟悉 Python 进程生命周期和信号处理。
- 熟悉 PaddlePaddle fleet collective launcher。
- 了解分布式任务的节点注册、状态同步和故障恢复流程。
