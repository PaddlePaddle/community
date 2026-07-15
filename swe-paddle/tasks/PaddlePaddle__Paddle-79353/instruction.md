# 完善 Pipeline P2P overlap 无通信路径的返回行为

## 详细描述

完善 Pipeline 并行 P2P 通信辅助接口在无需执行实际通信的 stage 边界场景下，对 overlap 模式返回值的处理。

## 验收说明

- 首个 Pipeline stage 调用 overlap `recv_forward` 时，应返回空输入及空等待句柄
- 首个 Pipeline stage 调用 overlap `send_backward` 时，应返回空等待句柄
- 非 overlap 模式的现有返回行为保持不变
- 不需要执行通信的路径不得因局部状态未正确处理而异常

## 技术要求

- 熟悉 Python
- 了解 Pipeline 并行与 P2P 通信流程
- 了解同步与 overlap 通信接口的返回约定

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
