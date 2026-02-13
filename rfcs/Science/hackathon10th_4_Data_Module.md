# Science 4 Paddle Geometric Data 模块实现

|           |                                |
| --------- |--------------------------------|
| 提交作者     | MinazukiHotaru                 |
| 提交时间     | 2026-02-01                     |
| RFC 版本号   | v1.0                           |
| 依赖飞桨版本 | develop                        |
| 文件名       | hackathon10th_4_Data_Module.md |

## 1. 概述

## 1.1 相关背景

Paddle Geometric 是飞桨框架的图神经网络扩展库，旨在提供类似 PyTorch Geometric 的功能。当前 Paddle Geometric 的 data 模块已有部分实现，但与 PyTorch Geometric 2.6.1 版本相比仍存在功能缺失和差异。

本次任务的目标是实现 PyTorch Geometric 库（2.6.1版本）的 data 模块，并将实现结果合入到 paddle_geometric 仓库。

## 1.2 功能目标

实现/完善以下模块：
- `collate` - 数据批处理合并
- `database` - 数据库存储后端
- `datapipes` - 数据管道
- `download` - 文件下载
- `extract` - 文件解压
- `hypergraph_data` - 超图数据结构
- `on_disk_dataset` - 磁盘数据集
- `remote_backend_utils` - 远程后端工具
- `temporal` - 时序图数据

## 1.3 意义

本次升级将使 Paddle Geometric 的 data 模块功能与 PyTorch Geometric 保持一致，为用户提供完整的图数据处理能力，降低从 PyTorch 迁移到飞桨的门槛。

## 2. PaddleScience 现状

对 Paddle Geometric 目前支持此功能的现状调研：

1. **已完成模块**：`datapipes.py`, `download.py`, `on_disk_dataset.py`, `remote_backend_utils.py`
2. **需要完善的模块**：`database.py`, `extract.py`, `temporal.py`
3. **可优化的模块**：`collate.py`, `hypergraph_data.py`

## 3. 目标调研

PyTorch Geometric 是目前主流的图神经网络库之一，其 data 模块提供了完整的图数据处理能力，包括：
- 批处理和合并机制
- 多种数据库后端支持（SQLite、RocksDB）
- 时序图数据结构
- 完善的数据下载和解压工具

本次任务采用对齐策略，将 PyTorch Geometric 2.6.1 版本的 data 模块功能完整迁移到 Paddle Geometric。以下是各模块的对比分析：

### 3.1 collate.py

| 项目 | PyTorch Geometric | Paddle Geometric | 差异说明 |
|------|-------------------|------------------|----------|
| **嵌套张量支持** | ✅ 支持 | ❌ 不支持 | PyG:178-184 有 `getattr(elem, 'is_nested', False)` 处理 |
| **共享内存优化** | ✅ 支持 (多worker) | ❌ 不支持 | PyG:186-201 实现了共享内存写入优化 |

**优势劣势分析**：
- PyG 的嵌套张量支持提供了更灵活的数据处理能力
- 共享内存优化可显著提升多进程数据加载性能

### 3.2 database.py

| 项目 | PyTorch Geometric | Paddle Geometric | 差异说明 |
|------|-------------------|------------------|----------|
| **Database 抽象类** | ✅ 完整 | ⚠️ 简化版 | Paddle版本缺少multi等操作 |
| **SQLiteDatabase** | ✅ 完整 | ⚠️ 基础版 | 补全multi,序列化/反序列化,辅助功能 |
| **RocksDatabase** | ✅ 完整 | ✅ 基本完整 | 需要补全multi功能 |

**优势劣势分析**：
- PyG 的序列化机制更完善，避免了直接使用 pickle
- 批量操作优化提升了大规模数据处理效率

### 3.3 extract.py

| 项目 | PyTorch Geometric | Paddle Geometric | 差异说明 |
|------|-------------------|------------------|----------|
| **extract_tar** | ✅ 完整 | ⚠️ 缺少安全参数 | PyG 使用 `filter='data'` |




### 3.4 temporal.py

| 项目 | PyTorch Geometric | Paddle Geometric | 差异说明 |
|------|-------------------|------------------|----------|
| **TemporalData 类** | ✅ 完整 | ⚠️ 缺少部分方法 | Paddle版本缺少部分方法 |

**优势劣势分析**：
- PyG 提供了完整的时序图数据接口
- 缺失方法影响批处理和数据集操作的兼容性

## 4. 设计思路与实现方案

本次实现采用对齐策略，核心思路是在保持 API 兼容的前提下，将 PyTorch Geometric 2.6.1 版本的 data 模块功能完整迁移到 Paddle Geometric。

涉及修改的模块及代码位置：
- `paddle_geometric/data/collate.py` - 添加嵌套张量和共享内存支持
- `paddle_geometric/data/database.py` - 完善序列化和批量操作
- `paddle_geometric/data/extract.py` - 添加安全参数
- `paddle_geometric/data/temporal.py` - 补全缺失方法

参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API进行改写

## 5. 测试和验收的考量

### 自测方案

每个模块按照要求编写以下测试：
- 基本功能测试
- 边界条件测试
- 异常处理测试
- 与 PyTorch Geometric 的对比测试

### 目标达成验收的度量方式

1. 所有单元测试通过
2. 与 PyTorch Geometric 行为一致性验证通过
3. 性能测试满足要求（共享内存优化需有性能提升数据）

## 6. 可行性分析和排期规划

1. 提交RFC 26年2月
2. 完成PR合入 26年2月