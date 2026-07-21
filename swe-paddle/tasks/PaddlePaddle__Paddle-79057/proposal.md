# Task Proposal: PaddlePaddle__Paddle-79057

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79057`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79057
- PR 标题：`[Security] Fix RCE vulnerability in RestrictedUnpickler MRO check`
- `base_commit`：`f4014bfa7b9acddfcfcaffb57b57b2a5c8fe9e7a`
- merged 时间：`2026-05-22`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `RestrictedUnpickler` 的 safe-class check 未检查完整 MRO，导致继承 unsafe pickle hooks 的 subclass 绕过安全校验的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：任务来自已合入的 Paddle security bug-fix PR，不是合成问题。
* **边界清楚**：production change 仅涉及一个 Python safe-class check，代码范围小且职责单一。
* **可验证性**：修复前，继承 unsafe pickle hooks 的 subclass 会被错误接受；修复后应被拒绝。
* **行为导向**：测试覆盖 safe-class check 和实际 restricted unpickling path，不依赖 source-code text matching。
* **安全可控**：测试使用无外部副作用的 in-memory flag 模拟 unsafe state restoration，不执行命令、不访问网络。
* **环境稳定**：测试仅依赖 Python standard library 和 pytest，不要求编译 Paddle 或加载 device runtime。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[security, pickle, restricted_unpickler, inheritance, mro, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_restricted_unpickler_mro.py`
- P2P：普通 user-defined class 继续被接受，直接声明 unsafe pickle hooks 的 class 继续被拒绝。
- F2P 1：继承 `__reduce__`、`__reduce_ex__`、`__getstate__` 或 `__setstate__` 的 subclass 在修复前被错误接受，修复后应被拒绝。
- F2P 2：带有 inherited `__setstate__` hook 的 pickle 在修复前会执行该 hook，修复后应在 hook 执行前抛出 `pickle.UnpicklingError`。
- 修复后预期：目标测试文件全部通过。

## 6. 环境与资源

- 资源需求：CPU
- patch 类型：Python-only
- Python 依赖：pytest
- Paddle 构建：不需要
- GPU、分布式运行、外部服务和额外数据集：不需要

## 7. 风险自查

- 泄露风险：`instruction.md` 描述安全行为和验收标准，不指定具体循环结构或代码位置。
- 安全风险：测试不执行系统命令；模拟钩子仅修改进程内布尔标记。
- flaky 风险：测试不依赖随机数、时间、网络或设备状态。
- 兼容风险：测试直接加载当前源码修订中的目标模块，避免已安装 Paddle 版本与目标提交不一致。
