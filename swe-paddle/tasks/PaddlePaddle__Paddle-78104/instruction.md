# 修复 device context 对 generic Place 的解析

## 详细描述

Tensor 的 `place` 可能表现为 generic `Place` 对象，而不是直接暴露具体的 CPU、CUDA、XPU 或 custom Place 子类。将该对象传给 device context 时，系统可能无法保留其真实 device type 和 device id，从而选择错误设备。

## 验收说明

- generic Place 应根据其实际 device type 转换为对应 concrete Place
- CUDA、XPU 和 custom device 的 device id 应保持不变
- 已有 string device 和 concrete Place 输入行为不得退化

## 技术要求

- 熟悉 Python
- 了解 Paddle Place 与 device context
- 了解 CPU、CUDA、XPU 和 custom device 表示

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
