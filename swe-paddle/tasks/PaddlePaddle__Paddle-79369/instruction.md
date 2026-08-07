# 修复 `check_memory_usage` 在 CPU 环境下报错的问题

## 详细描述

Fleet 的 `check_memory_usage` 会记录 GPU、pinned memory 和系统内存的使用情况。

目前，该函数还会调用 `paddle.device.cpu.max_memory_allocated()` 等 CPU 内存接口。这些接口在当前环境中并不支持，调用后会抛出异常，导致内存日志没有记录完，训练也可能因此中断。

CPU 内存继续通过系统命令 `free -h` 记录即可，不应依赖这些不支持的接口。现有的 GPU、pinned memory 和系统内存日志需要保持不变。

## 验收说明

- CPU 内存接口不支持时，调用 `check_memory_usage` 不应报错
- 系统内存信息应继续通过 `free -h` 获取并写入日志
- GPU 和 pinned memory 的现有日志保持不变
- 传入的日志标识信息应正常保留

## 技术要求

- 熟悉 Python
- 了解 Paddle Fleet 的日志工具
- 了解 Paddle 设备内存接口和系统内存命令
