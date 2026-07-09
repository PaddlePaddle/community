# SWE-Paddle 社区任务

SWE-Paddle 将已经合入的 Paddle 框架 PR 转成可复现、可验证的 coding-agent benchmark 任务。每个任务按 PR 粒度组织在一个独立目录中。

社区贡献采用 **proposal-first** 流程：贡献者先建目录提交 `proposal.md`；proposal 被 approve 后，再在同一目录继续补齐完整任务包。

## 贡献流程

1. 从一个已合入的 Paddle PR 出发，在 `swe-paddle/tasks/` 下新建目录，例如 `PaddlePaddle__Paddle-68432/`。
2. 第一阶段只提交 `proposal.md`，用于说明 PR 来源、任务价值、验证思路、环境资源和风险。
3. 维护组 review proposal，并给出 `approved` / `needs-info` / `not-fit` / `deferred`。
4. proposal approved 后，贡献者在同一目录继续补齐完整任务包。
5. 维护组运行 Run/Test/Fix verifier，并由 Paddle domain reviewer + benchmark reviewer 终审。

## 第一阶段目录结构

proposal 阶段只需要：

```text
swe-paddle/
  tasks/
    PaddlePaddle__Paddle-68432/
      proposal.md
```

proposal 模板见 [`proposal_template.md`](proposal_template.md)。除了 PR 链接、`base_commit`、任务类型等必要字段外，正文建议使用中文，方便 Paddle contributor 和 reviewer 快速理解。

## 完整任务包结构

proposal approved 后，再补齐：

```text
swe-paddle/
  README.md
  proposal_template.md
  tasks/
    PaddlePaddle__Paddle-68432/
      README.md
      proposal.md
      instruction.md
      environment/README.md
      solution/code.patch
      tests/test.patch
      tests/test.sh
```

首版先保持目录简洁。后续如果 verifier、看板或数据发布流程需要机器可读字段，再补充轻量 metadata 文件。

## 首个完整样例

| Instance | PR | Title | Type | Why this sample |
| --- | --- | --- | --- | --- |
| `PaddlePaddle__Paddle-68432` | [#68432](https://github.com/PaddlePaddle/Paddle/pull/68432) | 【Hackathon 7th No.18】为稀疏计算添加复数支持2 -part | `feature_enhancement` | Opus 4.8 passed while ERNIE 5.1 failed in the CPU10 pilot; the task exercises sparse kernels, complex dtype support, and complex autograd formulas. |

这个样例比简单 API 边界修复更适合作为首个展示样本：模型不仅要找到 dtype 注册路径，还要正确处理复数梯度公式中的 conjugation。

## 文件用途

- `proposal.md`：给维护组和 reviewer 的初筛材料，建议中文填写。
- `instruction.md`：给 coding agent 的自包含问题描述，不能泄露答案。
- `solution/code.patch`：来自已合入 PR 的 gold patch。
- `tests/test.patch`：暴露目标行为的测试补丁。
- `tests/test.sh`：最小目标测试命令。
- `environment/README.md`：复现环境和运行顺序。

## 收录原则

有效样本应满足：修复前失败，应用 gold patch 后通过，并包含足够的 P2P 回归护栏，避免 agent 只修目标测试却破坏其他行为。
