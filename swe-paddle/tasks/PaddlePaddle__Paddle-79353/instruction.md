# 完善 pipeline P2P overlap no-communication path 的返回行为

## 详细描述

完善 pipeline parallel P2P communication helper 在 first pipeline stage 无需执行实际 communication 时的 overlap return behavior。

## 验收说明

- first pipeline stage 调用 overlap recv_forward 时，应返回 (None, None)
- first pipeline stage 调用 overlap send_backward 时，应返回 None
- non-overlap path 的现有 return behavior 保持不变
- no-communication path 不得因 wait handle 状态缺失而异常
- 其他需要执行实际 P2P communication 的路径不得退化

## 技术要求

- 熟悉 Python
- 了解 pipeline parallel 和 P2P communication flow
- 了解 synchronous 与 overlap communication API 的 return contract

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
