# PaddlePaddle__Paddle-62012

本目录将 Paddle PR #62012 整理为一个可直接验证的 SWE-Paddle 候选任务。

## 来源

| 字段 | 内容 |
| --- | --- |
| 仓库 | `PaddlePaddle/Paddle` |
| 共建 issue | [#79447](https://github.com/PaddlePaddle/Paddle/issues/79447) |
| 来源 PR | [#62012](https://github.com/PaddlePaddle/Paddle/pull/62012) |
| PR 标题 | `[SOT] rewrite resume function generation` |
| Base commit | `1b68a51dbdc6b4e93a0c8e28df74e8d881272501` |
| Gold commit | `e19e3c9435ee71ac844d78f98a34265ac7a73589` |
| PR head | `e506003f01ed7a9e36e56b17946c8f998c9a743b` |
| 合入时间 | `2024-02-26T11:23:52+08:00` |
| 资源 | CPU |

## 任务概述

修复 SOT 在 Tensor 条件分支触发子图打断时错误读取尚未赋值的局部变量，并重构继续执行函数的生成和变量读写分析流程。

该任务为 Python-only。生产补丁横跨 9 个文件，共 1274 行改动；测试补丁完整保留原 PR 对 2 个测试文件的修改。

## 文件说明

- `instruction.md`：面向开发者的任务说明。
- `proposal.md`：SWE-Paddle 候选任务提案。
- `solution/code.patch`：Base 到 Gold 的完整生产代码补丁。
- `tests/test.patch`：原 PR 自带测试的精确补丁。
- `tests/test.sh`：运行两个目标测试文件。
- `environment/README.md`：复现环境和执行顺序。
- `cross_PR62012.sh`：Base/Solution 交叉验证脚本。
- `runtime-overlay/`：让历史 SOT Python 源码使用现有 Paddle CPU ABI 的验证辅助层。
- `logs/`：实际交叉验证日志。

## 验证

在 Git Bash 中运行：

```bash
bash cross_PR62012.sh
```

脚本会校验提交关系、变更范围、测试补丁、Gold 文件内容，并运行 Base/Solution 的 P2P、F2P 和完整目标测试。
