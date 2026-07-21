# 修复 model-parallel 操作重复调用导致的内存增长

## 详细描述

修复 dynamic graph 模式下 model-parallel identity 和 all-reduce 操作被重复调用时产生的 Python-side runtime type 累积问题，避免长时间训练过程中出现不必要的内存持续增长。

## 验收说明

- 重复调用 model-parallel identity 操作时，不应按调用次数持续创建新的 Python-side autograd runtime type
- 重复调用 model-parallel all-reduce 操作时，不应按调用次数持续创建新的 Python-side autograd runtime type
- identity 和 all-reduce 原有的 forward/backward communication behavior 必须保持不变

## 技术要求

- 熟悉 Python
- 熟悉 Paddle dynamic graph 和 autograd
- 了解 model-parallel process group、identity 和 all-reduce 的执行语义

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
