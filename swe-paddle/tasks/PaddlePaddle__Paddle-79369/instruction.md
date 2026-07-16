# 完善 Fleet 内存使用日志

## 详细描述

完善 Fleet `check_memory_usage` 在 CPU 运行环境中的内存信息采集和日志输出，确保不同设备能力下能够稳定执行。

## 验收说明

- 保留已有的 GPU、pinned memory 和系统内存日志行为
- CPU 运行环境下的内存日志流程应正常完成
- 系统内存信息应继续被正确采集和记录
- 在对应单测中覆盖目标场景和已有日志行为

## 技术要求

- 熟悉 Python 和单元测试
- 了解 Paddle 设备 API 与 Fleet 工具模块
- 了解 mock 在外部接口隔离中的使用

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
