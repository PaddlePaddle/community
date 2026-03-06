# Science 4 Paddle Geometric Data 模块实现

|           |                                |
| --------- |--------------------------------|
| 提交作者     | ADream-ki                 |
| 提交时间     | 2026-02-12                     |
| RFC 版本号   | v2.0                           |
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

以下是各模块的详细对比分析：

### 3.1 collate.py - 数据批处理合并

#### API 差异对比

| 功能 | PyTorch Geometric | Paddle Geometric (修改前) | Paddle Geometric (修改后) |
|------|-------------------|--------------------------|--------------------------|
| 嵌套张量支持 | `getattr(elem, 'is_nested', False)` |  不支持 |  支持 |
| 共享内存优化 | `torch.get_worker_info()` + `_to_shared()` |  不支持 |  支持 |
| CUDA 设备 | `torch.device` | 仅支持 CPU |  支持 CUDAPlace |

#### 实现方案

**1. 嵌套张量支持**
```python
# 修改前：直接拼接
value = torch.cat(values, axis=cat_dim or 0)

# 修改后：检测并展开嵌套张量
if getattr(elem, 'is_nested', False):
    try:
        tensors = []
        for nested_tensor in values:
            if hasattr(nested_tensor, 'unbind'):
                tensors.extend(nested_tensor.unbind())
            elif hasattr(nested_tensor, '__iter__'):
                for sub_tensor in nested_tensor:
                    tensors.append(sub_tensor)
            else:
                tensors.append(nested_tensor)
        value = paddle.concat(tensors, axis=cat_dim or 0)
```

**2. 共享内存优化**
```python
# 修改前：直接拼接
value = paddle.concat(values, axis=cat_dim or 0)

# 修改后：多 worker 环境使用共享内存
if get_worker_info is not None and not isinstance(elem, (Index, EdgeIndex)):
    numel = sum(value.numel().item() for value in values)
    shape = list(elem.shape)
    if cat_dim is None or elem.ndim == 0:
        shape = [len(values)] + shape
    else:
        shape[cat_dim] = int(slices[-1].item())
    try:
        if hasattr(elem, '_to_shared'):
            storage = elem._to_shared(numel * elem.element_size(), place=elem.place)
            out = elem.new(storage).reshape_(*shape)
        else:
            out = paddle.empty(shape, dtype=elem.dtype, place=elem.place)
    except Exception:
        out = None
```

**3. CUDA 设备支持**
```python
# 修改前
device: Optional[paddle.CPUPlace] = None

# 修改后
device: Optional[Union[paddle.CPUPlace, paddle.CUDAPlace]] = None
```

---

### 3.2 database.py - 数据库存储后端

#### API 差异对比

| 功能 | PyTorch Geometric | Paddle Geometric (修改前) | Paddle Geometric (修改后) |
|------|-------------------|--------------------------|--------------------------|
| connect/close |  支持 |  不支持 |  支持 |
| multi_insert |  支持 |  基础版 |  完整版 |
| multi_get |  支持 |  基础版 |  完整版 |
| 序列化 | `torch.save/load` | `pickle` | `paddle.save/load` + fallback |
| Index 序列化 |  支持 |  不支持 |  支持 |
| EdgeIndex 序列化 |  支持 |  不支持 |  支持 |

#### 实现方案

**1. 批量操作完善**
```python
# 修改前：简单循环
def multi_insert(self, indices, data_list):
    for index, data in zip(indices, data_list):
        self.insert(index, data)

# 修改后：支持批处理和日志
def multi_insert(self, indices, data_list, batch_size=None, log=False):
    if isinstance(indices, slice):
        indices = self.slice_to_range(indices)
    length = min(len(indices), len(data_list))
    batch_size = length if batch_size is None else batch_size
    if log and length > batch_size:
        offsets = tqdm(range(0, length, batch_size), desc=f'Insert {length} entries')
    else:
        offsets = range(0, length, batch_size)
    for start in offsets:
        self._multi_insert(indices[start:start + batch_size], data_list[start:start + batch_size])
```

**2. SQLiteDatabase 批量查询优化**
```python
# 修改前：使用 IN 子句
SELECT * FROM table WHERE id IN (1, 2, 3, ...)

# 修改后：使用临时表 JOIN
CREATE TEMP TABLE table__join (id INTEGER, row_id INTEGER)
INSERT INTO table__join VALUES (1, 0), (2, 1), (3, 2)
SELECT * FROM table INNER JOIN table__join ON table.id = table__join.id ORDER BY table__join.row_id
DROP TABLE table__join
```

**3. 序列化机制升级**
```python
# 修改前：直接使用 pickle
def _serialize(self, data):
    return [pickle.dumps(data.get(key)) for key in self.schema.keys()]

# 修改后：优先使用 paddle.save
def _serialize(self, row):
    out = []
    row_dict = self._to_dict(row)
    for key, schema in self.schema.items():
        col = row_dict[key]
        if isinstance(schema, TensorInfo):
            # 特殊处理 Index 和 EdgeIndex
            if schema.is_index:
                meta = paddle.to_tensor([col.dim_size or -1, col.is_sorted], dtype=paddle.int64)
                out.append(meta.numpy().tobytes() + col.as_tensor().numpy().tobytes())
            elif schema.is_edge_index:
                meta = paddle.to_tensor([
                    col.sparse_size()[0] or -1,
                    col.sparse_size()[1] or -1,
                    SORT_ORDER_TO_INDEX[col._sort_order],
                    col.is_undirected
                ], dtype=paddle.int64)
                out.append(meta.numpy().tobytes() + col.as_tensor().numpy().tobytes())
            else:
                out.append(col.numpy().tobytes())
        else:
            buffer = io.BytesIO()
            try:
                paddle.save(col, buffer)
            except (AttributeError, RuntimeError):
                pickle.dump(col, buffer)
            out.append(buffer.getvalue())
    return out
```

---

### 3.3 temporal.py - 时序图数据

#### API 差异对比

| 方法 | PyTorch Geometric | Paddle Geometric (修改前) | Paddle Geometric (修改后) |
|------|-------------------|--------------------------|--------------------------|
| from_dict |  支持 |  不支持 |  支持 |
| index_select |  支持 |  不支持 |  支持 |
| train_val_test_split |  支持 |  不支持 |  支持 |
| size |  支持 |  不支持 |  支持 |
| __cat_dim__ |  支持 |  不支持 |  支持 |
| __inc__ |  支持 |  不支持 |  支持 |
| num_events |  支持 |  不支持 |  支持 |
| edge_index |  支持 |  不支持 |  支持 |

#### 实现方案

**1. from_dict 类方法**
```python
@classmethod
def from_dict(cls, mapping: Dict[str, Any]) -> 'TemporalData':
    """从字典创建 TemporalData 对象"""
    return cls(**mapping)
```

**2. index_select 方法**
```python
def index_select(self, idx: Any) -> 'TemporalData':
    idx = prepare_idx(idx)
    data = copy.copy(self)
    for key, value in data._store.items():
        if value.shape[0] == self.num_events:
            data[key] = value[idx]
    return data
```

**3. train_val_test_split 方法**
```python
def train_val_test_split(self, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """基于时间划分训练/验证/测试集"""
    val_time, test_time = np.quantile(
        self.t.numpy(),
        [1. - val_ratio - test_ratio, 1. - test_ratio])
    val_idx = int((self.t <= val_time).sum().item())
    test_idx = int((self.t <= test_time).sum().item())
    return self[:val_idx], self[val_idx:test_idx], self[test_idx:]
```

**4. size 方法**
```python
def size(self, dim: Optional[int] = None):
    """返回邻接矩阵大小"""
    size = (int(self.src.max()), int(self.dst.max()))
    return size if dim is None else size[dim]
```

---

### 3.4 hypergraph_data.py - 超图数据结构

#### API 差异对比

| 方法 | PyTorch Geometric | Paddle Geometric (修改前) | Paddle Geometric (修改后) |
|------|-------------------|--------------------------|--------------------------|
| subgraph |  支持 |  不支持 |  支持 |
| validate |  支持 |  不支持 |  支持 |
| has_isolated_nodes |  支持 |  不支持 |  支持 |
| num_edges |  支持 |  不支持 |  支持 |

#### 实现方案

**1. subgraph 方法**
```python
def subgraph(self, subset: Tensor) -> 'HyperGraphData':
    """返回给定节点索引的诱导子图"""
    assert self.edge_index is not None
    out = hyper_subgraph(subset, self.edge_index, relabel_nodes=True,
                         num_nodes=self.num_nodes, return_edge_mask=True)
    edge_index, _, edge_mask = out
    data = copy.copy(self)
    for key, value in self.items():
        if key == 'edge_index':
            data.edge_index = edge_index
        elif key == 'num_nodes':
            if subset.dtype == paddle.bool:
                data.num_nodes = int(subset.sum())
            else:
                data.num_nodes = subset.size(0)
        elif self.is_node_attr(key):
            cat_dim = self.__cat_dim__(key, value)
            data[key] = select(value, subset, dim=cat_dim)
        elif self.is_edge_attr(key):
            cat_dim = self.__cat_dim__(key, value)
            data[key] = select(value, edge_mask, dim=cat_dim)
    return data
```

**2. validate 方法**
```python
def validate(self, raise_on_error: bool = True) -> bool:
    """验证数据的正确性"""
    cls_name = self.__class__.__name__
    status = True
    num_nodes = self.num_nodes
    if num_nodes is None:
        status = False
        warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'", raise_on_error)
    if self.edge_index is not None:
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            status = False
            warn_or_raise(f"'edge_index' needs to be of shape [2, num_edges] in '{cls_name}'", raise_on_error)
    return status
```

**3. has_isolated_nodes 方法**
```python
def has_isolated_nodes(self) -> bool:
    """返回图中是否有孤立节点"""
    if self.edge_index is None:
        return False
    return paddle.unique(self.edge_index[0]).shape[0] < self.num_nodes
```

---

### 3.5 datapipes.py - 数据管道

#### API 差异对比

| 功能 | PyTorch Geometric | Paddle Geometric (修改前) | Paddle Geometric (修改后) |
|------|-------------------|--------------------------|--------------------------|
| 数据管道 |  完整 |  基础版 |  完整 |

#### 实现方案

完整实现数据处理流水线，支持链式操作和数据转换。

## 4. 设计思路与实现方案

本次实现采用对齐策略，核心思路是在保持 API 兼容的前提下，将 PyTorch Geometric 2.6.1 版本的 data 模块功能完整迁移到 Paddle Geometric。

### 4.1 核心设计原则

1. **API 兼容性**：与 PyTorch Geometric 2.6.1 保持 API 一致
2. **性能优化**：引入共享内存机制提升多进程数据加载效率
3. **序列化兼容**：优先使用 paddle.save/paddle.load，pickle 作为 fallback
4. **设备支持**：完整支持 CPU 和 CUDA 设备

### 4.2 修改文件列表

#### 核心模块（8个）
- `paddle_geometric/data/collate.py` - 数据批处理合并
- `paddle_geometric/data/database.py` - 数据库存储后端
- `paddle_geometric/data/datapipes.py` - 数据管道
- `paddle_geometric/data/hetero_data.py` - 异构图数据
- `paddle_geometric/data/hypergraph_data.py` - 超图数据结构
- `paddle_geometric/data/on_disk_dataset.py` - 磁盘数据集
- `paddle_geometric/data/temporal.py` - 时序图数据
- `paddle_geometric/data/storage.py` - 存储层

#### 测试文件（7个）
- `test/data/test_collate.py`
- `test/data/test_database.py`
- `test/data/test_datapipes.py`
- `test/data/test_hypergraph_data.py`
- `test/data/test_on_disk_dataset.py`
- `test/data/test_remote_backend_utils.py`
- `test/data/test_temporal.py`

### 4.3 关键技术点

#### 4.3.1 API 转换策略
参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API 进行改写：
- `torch.cat` → `paddle.concat`
- `torch.device` → `paddle.CPUPlace` / `paddle.CUDAPlace`
- `torch.save/load` → `paddle.save/load`
- `torch.get_worker_info()` → 使用相同机制获取 worker 信息

#### 4.3.2 特殊类型处理
- **Index 和 EdgeIndex**：需要序列化元数据（排序状态、稀疏大小等）
- **嵌套张量**：检测 `is_nested` 属性并展开
- **共享内存**：在多 worker 环境下使用 `_to_shared()` 方法优化

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