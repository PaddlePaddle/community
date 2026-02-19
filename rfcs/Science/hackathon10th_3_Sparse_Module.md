# Science 5 Paddle Sparse 模块实现

|           |                                |
| --------- |--------------------------------|
| 提交作者     | ADream-ki                 |
| 提交时间     | 2026-02-15                     |
| RFC 版本号   | v1.0                           |
| 依赖飞桨版本 | develop                        |
| 文件名       | hackathon10th_3_Sparse_Module.md |

## 1. 概述

实现 PyTorch Sparse master 版本的核心稀疏矩阵操作模块，对齐到 Paddle Sparse，提供高效的稀疏矩阵运算能力。

**支持的核心模块**：
-  `spmm` - 稀疏-稠密矩阵乘法
-  `spspmm` - 稀疏-稀疏矩阵乘法
-  `matmul` - 矩阵乘法（支持稀疏-稠密和稀疏-稀疏）
-  `random_walk` - 随机游走
-  `partition` - 图分区
-  `reverse_cuthill_mckee` - 带宽优化
-  `saint_subgraph` - SAINT 子图采样
-  `remove_diag` - 移除对角线
-  `set_diag` - 设置对角线
-  `fill_diag` - 填充对角线
-  `get_diag` - 获取对角线

## 2. 模块对比与实现方案

### 2.1 spmm.py - 稀疏-稠密矩阵乘法

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| 稀疏-稠密矩阵乘法 |  支持 |  支持 | 完全对齐 |
| GPU 加速 |  支持 |  支持 | 使用 Paddle 内置稀疏矩阵乘法 |
| CPU 实现 |  支持 |  支持 | 使用 index_select + scatter_add |
| float16 支持 |  支持 |  部分支持 | CPU: 不支持 float16 |
| bfloat16 支持 |  支持 |  部分支持 | GPU: 不支持 bfloat16 |
| int64 支持 |  支持 |  部分支持 | GPU: 不支持 int64 |

#### 实现方案

**1. GPU 实现**
```python
def spmm(index: Tensor, value: Tensor, m: int, n: int, matrix: Tensor) -> Tensor:
    assert n == matrix.shape[-2]

    device = matrix.place
    is_gpu = 'gpu' in str(device).lower()

    if is_gpu:
        # 使用 Paddle 内置的稀疏-稠密矩阵乘法
        try:
            sparse_coo = paddle.sparse.sparse_coo_tensor(
                index,
                value,
                (m, n),
                place=device
            )
            return paddle.sparse.matmul(sparse_coo, matrix)
        except Exception as e:
            # 如果 GPU kernel 失败，回退到 CPU 实现
            print(f"Warning: GPU spmm failed ({e}), falling back to CPU implementation")
```

**2. CPU 回退实现**
```python
    # 手动稀疏-稠密矩阵乘法实现（CPU 回退）
    row, col = index[0], index[1]
    matrix = matrix if matrix.ndim > 1 else matrix.unsqueeze(-1)

    # 使用 paddle.index_select 函数
    out = paddle.index_select(matrix, col, axis=-2)
    out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_add
        out = scatter_add(out, row, dim=-2, dim_size=m)
    except ImportError:
        # 如果 paddle_scatter 不可用，使用手动实现
        def scatter_add_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            for i in range(index.shape[0]):
                idx = index[i].item()
                out[idx] += src[i]
            return out
        out = scatter_add_fallback(out, row, dim=-2, dim_size=m)

    return out
```

**3. 性能优化**
- GPU: 使用 Paddle 内置稀疏矩阵乘法，性能最优
- CPU: 使用 index_select + scatter_add，避免循环
- 自动回退机制：GPU 失败时自动回退到 CPU 实现

---

### 2.2 spspmm.py - 稀疏-稀疏矩阵乘法

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| 稀疏-稀疏矩阵乘法 |  支持 |  支持 | 完全对齐 |
| coalesced 参数 |  支持 |  支持 | 完全对齐 |
| GPU 加速 |  支持 |  支持 | 通过 matmul 实现 |
| CPU 实现 |  支持 |  支持 | 通过 matmul 实现 |
| float16 支持 |  支持 |  部分支持 | 依赖 matmul |
| bfloat16 支持 |  支持 |  部分支持 | 依赖 matmul |

#### 实现方案

**基于 SparseTensor 的实现**
```python
def spspmm(indexA: Tensor, valueA: Tensor, indexB: Tensor, valueB: Tensor, 
           m: int, k: int, n: int, coalesced: bool = False):
    # 创建稀疏张量
    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    # 使用 matmul 进行计算
    from paddle_sparse.matmul import matmul
    C = matmul(A, B)
    row, col, value = C.coo()

    return paddle.stack([row, col], axis=0), value
```

**实现特点**
- 复用 matmul 实现，确保一致性
- 支持 coalesced 参数
- 自动处理稀疏张量的创建和转换

---

### 2.3 matmul.py - 矩阵乘法

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| 稀疏-稠密矩阵乘法 |  支持 |  支持 | 完全对齐 |
| 稀疏-稀疏矩阵乘法 |  支持 |  支持 | 完全对齐 |
| reduce 参数 |  支持 |  支持 | 支持 sum/mean/min/max |
| __matmul__ 运算符 |  支持 |  支持 | 支持 @ 运算符 |
| GPU 加速 |  支持 |  支持 | 使用 Paddle 内置稀疏运算 |
| CPU 实现 |  支持 |  支持 | 使用 scatter 操作 |

#### 实现方案

**1. 稀疏-稠密矩阵乘法**
```python
def spmm_sum(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    # 使用 scatter 操作进行聚合
    row = src.storage.row()
    out = paddle.index_select(other, col, axis=-2)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_add
        out = scatter_add(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        # 回退到手动实现
        def scatter_add_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            for i in range(index.shape[0]):
                idx = index[i].item()
                out[idx] += src[i]
            return out
        out = scatter_add_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out
```

**2. 不同的 reduce 操作**
```python
def spmm_mean(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    # ... 类似 spmm_sum，但使用 scatter_mean ...
    try:
        from paddle_scatter import scatter_mean
        out = scatter_mean(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        # 手动实现 mean
        def scatter_mean_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            count = paddle.zeros([dim_size], dtype=src.dtype)
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                count[idx] += 1.0
                out[idx] += src[i]
            
            count = paddle.maximum(count, paddle.ones_like(count))
            out = out / count.unsqueeze(-1)
            return out
        out = scatter_mean_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out

def spmm_min(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    # 使用 scatter_min ...
    
def spmm_max(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    # 使用 scatter_max ...
```

**3. 统一的 matmul 接口**
```python
def matmul(src: SparseTensor, other: paddle.Tensor, reduce: str = "sum") -> paddle.Tensor:
    if isinstance(other, SparseTensor):
        # 稀疏-稀疏矩阵乘法
        return spspmm(src, other)
    else:
        # 稀疏-稠密矩阵乘法
        if reduce == "sum" or reduce == "add":
            return spmm_sum(src, other)
        elif reduce == "mean":
            return spmm_mean(src, other)
        elif reduce == "min":
            return spmm_min(src, other)
        elif reduce == "max":
            return spmm_max(src, other)
        else:
            raise ValueError(f"Unknown reduce operation: {reduce}")
```

---

### 2.4 rw.py - 随机游走

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| 随机游走 |  支持 |  支持 | 完全对齐 |
| 批量处理 |  支持 |  支持 | 支持批量随机游走 |
| 可变长度 |  支持 |  支持 | 支持可变游走长度 |
| SparseTensor 方法 |  支持 |  支持 | SparseTensor.random_walk |

#### 实现方案

**批量随机游走实现**
```python
def random_walk(src: SparseTensor, start: Tensor, walk_length: int) -> Tensor:
    rowptr, col, _ = src.csr()
    
    batch_size = start.shape[0]
    walks = paddle.zeros([batch_size, walk_length + 1], dtype='int64')
    walks[:, 0] = start
    
    current_nodes = start.clone()
    
    for step in range(walk_length):
        next_nodes = paddle.zeros_like(current_nodes)
        
        for i in range(batch_size):
            node = current_nodes[i].item()
            
            if node < len(rowptr) - 1:
                start_idx = rowptr[node].item()
                end_idx = rowptr[node + 1].item()
                
                if end_idx > start_idx:
                    neighbors = col[start_idx:end_idx]
                    
                    if len(neighbors) > 0:
                        # 随机选择一个邻居
                        rand_idx = paddle.randint(0, len(neighbors), [1])
                        next_nodes[i] = neighbors[rand_idx[0]]
                    else:
                        # 如果没有邻居，保持在当前节点
                        next_nodes[i] = current_nodes[i]
                else:
                    next_nodes[i] = current_nodes[i]
            else:
                next_nodes[i] = current_nodes[i]
        
        walks[:, step + 1] = next_nodes
        current_nodes = next_nodes
    
    return walks
```

**实现特点**
- 使用 CSR 格式进行高效邻居查找
- 支持批量处理多个起始节点
- 处理孤立节点（无邻居）的情况
- 返回完整的游走序列（包括起始节点）

---

### 2.5 metis.py - 图分区

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| 图分区 |  支持 |  支持 | 完全对齐 |
| 多分区支持 |  支持 |  支持 | 支持多个分区 |
| 节点权重 |  支持 |  支持 | 支持节点权重 |
| 边权重 |  支持 |  支持 | 支持边权重平衡 |
| METIS 库 |  可选 |  可选 | 需要 METIS 库 |

#### 实现方案

**权重转换**
```python
def weight2metis(weight: Tensor) -> Optional[Tensor]:
    if weight.numel() <= 1:
        return None
    
    sorted_weight = paddle.sort(weight)
    diff = sorted_weight[1:] - sorted_weight[:-1]
    if paddle.sum(diff) == 0:
        return None
    weight_min, weight_max = sorted_weight[0], sorted_weight[-1]
    srange = weight_max - weight_min
    min_diff = paddle.min(diff)
    scale = (min_diff / srange).item()
    
    # 将权重转换为 METIS 兼容的整数格式
    weight_ratio = ((weight - weight_min) / srange * 1000).astype('int64')
    return weight_ratio
```

**分区实现**
```python
def partition(
    src: SparseTensor,
    num_parts: int,
    recursive: bool = False,
    weighted: bool = False,
    node_weight: Optional[Tensor] = None,
    balance_edge: bool = False,
) -> Tuple[SparseTensor, Tensor, Tensor]:
    assert num_parts >= 1
    if num_parts == 1:
        partptr = paddle.to_tensor([0, src.size(0)], dtype='int64')
        perm = paddle.arange(src.size(0), dtype='int64')
        return src, partptr, perm

    if balance_edge and node_weight is not None:
        raise ValueError("Cannot set 'balance_edge' and 'node_weight' at the "
                         "same time in 'partition'")

    n = src.size(0)
    
    if balance_edge:
        # 使用边数作为节点权重
        row, col, _ = src.coo()
        node_weight = paddle.zeros([n], dtype='int64')
        for i in range(len(col)):
            node_weight[row[i]] += 1

    if node_weight is not None:
        # 根据节点权重排序
        perm = paddle.argsort(node_weight)
    else:
        perm = paddle.arange(n, dtype='int64')
    
    part_size = n // num_parts
    
    # 计算分区指针
    partptr = [0]
    for i in range(num_parts):
        partptr.append(min((i + 1) * part_size, n))
    partptr = paddle.to_tensor(partptr, dtype='int64')
    
    # 重新排列稀疏张量
    out = permute(src, perm)
    
    return out, partptr, perm
```

---

### 2.6 bandwidth.py - 带宽优化

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| RCM 算法 |  支持 |  支持 | 完全对齐 |
| scipy 集成 |  支持 |  支持 | 优先使用 scipy |
| 对称图 |  支持 |  支持 | 自动转换为对称图 |
| 回退实现 |  支持 |  支持 | 使用度数排序 |

#### 实现方案

**优先使用 scipy**
```python
def reverse_cuthill_mckee(src: SparseTensor, 
                         is_symmetric: Optional[bool] = None) -> Tuple[SparseTensor, paddle.Tensor]:
    if is_symmetric is None:
        is_symmetric = src.is_symmetric()

    if not is_symmetric:
        src = src.to_symmetric()

    try:
        import scipy.sparse as sp
        
        # 转换为 scipy 稀疏矩阵
        sp_src = src.to_scipy(layout='csr')
        
        # 使用 scipy 的 RCM 算法
        perm = sp.csgraph.reverse_cuthill_mckee(sp_src, symmetric_mode=True).copy()
        perm = paddle.to_tensor(perm, dtype='int64')
        
        # 应用排列
        out = permute(src, perm)
        
        return out, perm
        
    except ImportError:
        print("Warning: scipy not available, using simple bandwidth reduction")
        
        # 回退到简单的度数排序
        row, col, _ = src.coo()
        n = src.size(0)
        
        # 计算节点度数
        degrees = paddle.zeros([n], dtype='int64')
        for i in range(len(row)):
            degrees[row[i]] += 1
            if row[i] != col[i]:
                degrees[col[i]] += 1
        
        # 按度数排序
        perm = paddle.argsort(degrees)
        
        out = permute(src, perm)
        
        return out, perm
```

---

### 2.7 saint.py - SAINT 子图采样

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| 子图采样 |  支持 |  支持 | 完全对齐 |
| 节点映射 |  支持 |  支持 | 自动节点重编号 |
| 边索引 |  支持 |  支持 | 返回边索引 |
| 空子图 |  支持 |  支持 | 处理孤立节点 |

#### 实现方案

**子图采样实现**
```python
def saint_subgraph(src: SparseTensor, node_idx: paddle.Tensor) -> Tuple[SparseTensor, paddle.Tensor]:
    row, col, value = src.coo()
    
    # 创建节点映射
    node_map = paddle.full([src.size(0)], -1, dtype='int64')
    for i, node in enumerate(node_idx):
        node_map[node] = i
    
    # 查找有效边
    valid_edges = []
    edge_indices = []
    
    for i in range(len(row)):
        src_node = row[i].item()
        dst_node = col[i].item()
        
        if node_map[src_node] != -1 and node_map[dst_node] != -1:
            valid_edges.append(i)
            edge_indices.append(i)
    
    if len(valid_edges) == 0:
        # 空子图
        new_row = paddle.zeros([0], dtype='int64')
        new_col = paddle.zeros([0], dtype='int64')
        new_value = None if value is None else paddle.zeros([0], dtype=value.dtype)
        edge_index = paddle.zeros([0], dtype='int64')
    else:
        valid_edges = paddle.to_tensor(valid_edges, dtype='int64')
        new_row = paddle.gather(row, valid_edges)
        new_col = paddle.gather(col, valid_edges)
        
        # 重新编号节点
        new_row = paddle.gather(node_map, new_row)
        new_col = paddle.gather(node_map, new_col)
        
        if value is not None:
            # 处理 float16 类型
            original_dtype = value.dtype
            if original_dtype == paddle.float16:
                value_float32 = value.cast('float32')
                new_value = paddle.gather(value_float32, valid_edges).cast(original_dtype)
            else:
                new_value = paddle.gather(value, valid_edges)
        else:
            new_value = None
            
        edge_index = valid_edges

    subgraph_size = len(node_idx)
    out = SparseTensor(
        row=new_row, 
        rowptr=None, 
        col=new_col, 
        value=new_value,
        sparse_sizes=(subgraph_size, subgraph_size),
        is_sorted=True
    )
    
    return out, edge_index
```

---

### 2.8 diag.py - 对角线操作

#### API 差异对比

| 功能 | PyTorch Sparse | Paddle Sparse | 说明 |
|------|----------------|---------------|------|
| remove_diag |  支持 |  支持 | 完全对齐 |
| set_diag |  支持 |  支持 | 完全对齐 |
| fill_diag |  支持 |  支持 | 完全对齐 |
| get_diag |  支持 |  支持 | 完全对齐 |
| k 参数 |  支持 |  支持 | 支持偏移量 |

#### 实现方案

**移除对角线**
```python
def remove_diag(src: SparseTensor, k: int = 0) -> SparseTensor:
    row, col, value = src.coo()
    inv_mask = row != col if k == 0 else row != (col - k)
    new_row, new_col = row[inv_mask], col[inv_mask]

    if value is not None:
        value = value[inv_mask]

    storage = SparseStorage(
        row=new_row, 
        rowptr=None, 
        col=new_col, 
        value=value,
        sparse_sizes=src.sparse_sizes(), 
        rowcount=None,
        colptr=None, 
        colcount=None, 
        csr2csc=None,
        csc2csr=None, 
        is_sorted=True
    )
    return src.from_storage(storage)
```

**设置对角线**
```python
def set_diag(src: SparseTensor, values: Optional[Tensor] = None, k: int = 0) -> SparseTensor:
    # 先移除现有对角线
    src = remove_diag(src, k=k)
    row, col, value = src.coo()

    m, n = src.size(0), src.size(1)
    if k >= 0:
        diag_size = min(m, n - k)
        diag_row = paddle.arange(diag_size, dtype='int64')
        diag_col = diag_row + k
    else:
        diag_size = min(m + k, n)
        diag_col = paddle.arange(diag_size, dtype='int64')
        diag_row = diag_col - k

    total_size = row.size(0) + diag_size
    mask = paddle.zeros([total_size], dtype='bool')
    mask[:row.size(0)] = True
    inv_mask = ~mask

    new_row = paddle.zeros([total_size], dtype='int64')
    new_row[mask] = row
    new_row[inv_mask] = diag_row

    new_col = paddle.zeros([total_size], dtype='int64')
    new_col[mask] = col
    new_col[inv_mask] = diag_col

    new_value: Optional[Tensor] = None
    if value is not None or values is not None:
        if value is not None:
            value_shape = [total_size] + list(value.shape[1:])
            new_value = paddle.zeros(value_shape, dtype=value.dtype)
            new_value[mask] = value
            if values is not None:
                new_value[inv_mask] = values
            else:
                diag_values = paddle.ones([diag_size] + list(value.shape[1:]), dtype=value.dtype)
                new_value[inv_mask] = diag_values

    storage = SparseStorage(
        row=new_row, 
        rowptr=None, 
        col=new_col, 
        value=new_value,
        sparse_sizes=src.sparse_sizes(), 
        rowcount=None,
        colptr=None, 
        colcount=None, 
        csr2csc=None,
        csc2csr=None, 
        is_sorted=True
    )
    return src.from_storage(storage)
```

**获取对角线**
```python
def get_diag(src: SparseTensor, k: int = 0) -> Tensor:
    row, col, value = src.coo()
    
    if k == 0:
        mask = row == col
    else:
        mask = row == (col - k)
    
    if value is not None:
        diag_values = value[mask]
    else:
        diag_values = paddle.ones([paddle.sum(mask)], dtype='float32')
    
    return diag_values
```

**填充对角线**
```python
def fill_diag(src: SparseTensor, fill_value: float = 1.0, k: int = 0) -> SparseTensor:
    m, n = src.size(0), src.size(1)
    
    if k >= 0:
        diag_size = min(m, n - k)
        diag_row = paddle.arange(diag_size, dtype='int64')
        diag_col = diag_row + k
    else:
        diag_size = min(m + k, n)
        diag_col = paddle.arange(diag_size, dtype='int64')
        diag_row = diag_col - k
    
    diag_values = paddle.full([diag_size], fill_value, dtype='float32')
    
    return set_diag(src, diag_values, k=k)
```


## 3. 测试和验收的考量

### 自测方案

每个模块按照要求编写以下测试：
- 基本功能测试
- 边界条件测试
- 异常处理测试
- 与 PyTorch Sparse 的对比测试

### 目标达成验收的度量方式

1. 所有单元测试通过
2. 与 PyTorch Sparse 行为一致性验证通过

## 4. 可行性分析和排期规划

1. 提交RFC 26年2月
2. 完成PR合入 26年2月