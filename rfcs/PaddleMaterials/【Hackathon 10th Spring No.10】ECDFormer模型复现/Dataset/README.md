# ECFormer 数据集模块设计总览

| 所属任务 | 【Hackathon 10th Spring No.10】ECDFormer模型复现 |
| --- | --- |
| **提交作者**     | PlumBlossomMaid |
| **提交时间**     | 2026-02-14 |
| **版本号**       | 1.0 |
| **依赖飞桨版本** | paddlepaddle-gpu 3.3.0 |
| **文件名** | README.md |
| **计算平台**     | Windows 10 Python 3.13.1 AMD64 64bit |

---

## 一、概述

### 1、相关背景
ECFormer模型家族包含两个谱图预测任务：**ECD光谱预测**（核心贡献，Nature Computational Science 2025）和**IR光谱预测**（泛化能力演示）。这两个任务的数据格式、处理逻辑、特殊增强各有不同，但都基于相同的底层图结构（atom-bond图 + bond-angle图）。为了在PaddleMaterials中统一管理，需要设计一套清晰、可扩展、用户友好的数据集模块。

### 2、功能目标
- 为ECFormer模型提供两个独立但风格一致的数据集模块：`ECDFormerDataset` 和 `IRDataset`
- 所有数据集模块放置在 `ppmat/datasets/` 目录下，通过 `__init__.py` 统一导出
- 每个数据集模块**完全自治**：包含自己的数据加载、处理、缓存、DataLoader实现
- 解决Paddle DataLoader对非Tensor返回类型的限制（见[Issue #77754](https://github.com/PaddlePaddle/Paddle/issues/77754)），通过自定义collate_fn解决该问题同时支持静态图
- 提供类级别缓存机制，避免重复加载，提升用户体验

### 3、意义
- **架构清晰**：数据集模块自治，避免侵入式修改框架通用代码
- **用户友好**：无需理解化学数据格式，开箱即用
- **扩展性**：新增谱图任务（如质谱）只需新建模块，不影响现有模块
- **框架适配**：通过自定义collate_fn绕过框架限制，同时支持动态图和静态图

---

## 二、飞桨现状

### 1、现有数据加载机制
`ppmat/datasets/__init__.py` 提供了 `build_dataloader` 函数，通过配置文件动态构建DataLoader。其核心流程如下：

```python
def build_dataloader(cfg: Dict):
    # 1. 实例化数据集
    dataset = eval(cls_name)(**init_params)
    
    # 2. 从loader_config获取collate_fn名称
    collate_fn_name = loader_config.pop("collate_fn", "DefaultCollator")
    collate_cls = getattr(collate_fn, collate_fn_name)
    collate_obj = collate_cls(**collate_params)
    
    # 3. 构建DataLoader
    data_loader = DataLoader(..., collate_fn=collate_obj)
```

`ppmat/datasets/collate_fn.py` 提供了多种预定义collator，其中 `DefaultCollator` 支持 `Data` 对象的batch化。

### 2、存在的问题
- **Paddle框架自身限制**：Paddle静态图要求所有模型输入必须是 `paddle.Tensor`，而 `Batch.from_data_list` 返回的是 `Batch` 对象，无法直接用于静态图
- **侵入式扩展**：如果为每个数据集在 `collate_fn.py` 中添加专用collator，会导致文件臃肿，且与数据集逻辑耦合
- **API冗余**：ECFormer模型不需要 `build_dataset_infos` 等复杂统计功能，强行使用会增加不必要的依赖

### 3、绕过方式
通过**数据集模块自治**的方式解决：
- 每个数据集模块自行实现 `DataLoader` 子类，包含自定义 `collate_fn`
- 自定义 `collate_fn` 中将 `Batch` 对象解包为 `Tensor` 字典，满足静态图要求
- 模块内部处理所有数据加载逻辑，对外只暴露 `Dataset` 和 `DataLoader` 类

---

## 三、业内方案调研（ECDFormer原仓库分析）

原ECDFormer仓库的数据处理存在以下问题：

| 问题 | 表现 | 本设计改进 |
|------|------|-----------|
| **代码分散** | 数据处理分散在多个文件中，复用困难 | **模块自治**，每个数据集独立完整 |
| **加载效率低** | IR数据集全量扫描JSON文件（耗时1小时） | **按需读取**，只读需要的文件 |
| **无缓存机制** | 多次实例化重复加载 | **类级别缓存**，秒级返回 |
| **与框架耦合** | 直接使用Data对象，无法静态图导出 | **自定义collate_fn**解包为Tensor |

---

## 四、对比分析

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **方案A（框架集成）**：在 `collate_fn.py` 添加专用collator，统一注册数据集 | 符合现有框架流程 | 侵入式修改，每次新增数据集都需要改框架代码 | ❌ **拒绝** |
| **方案B（模块自治）**：每个数据集独立实现，通过 `__init__.py` 导出 | ✅ **低耦合**<br>✅ **扩展性强**<br>✅ **不侵入框架** | 需自行处理DataLoader | ✅ **采用** |

**选型决策**：**坚决采用方案B**。虽然每个模块需要自行实现DataLoader，但这是**一次性成本**，换来的是**永久的可维护性和扩展性**。

---

## 五、设计思路与实现方案

### 1、主体设计思路

#### 整体架构图

```
ppmat/datasets/
├── __init__.py                 # 统一导出所有数据集
├── collate_fn.py               # 框架通用collator（保持不变）
├── msd_nmr_dataset.py          # 已有NMR数据集
├── ... (其他数据集)
├── ECDFormerDataset/           # ECD数据集模块（自治）
│   ├── __init__.py             # 导出ECDFormerDataset和ECDFormerDataLoader
│   ├── dataloader.py           # 自定义DataLoader + collate_fn
│   ├── compound_tools.py       # 分子特征提取工具
│   ├── eval_func.py            # 峰值检测等评估函数
│   ├── util_func.py            # 归一化等工具函数
│   ├── place_env.py            # 设备上下文管理器
│   └── colored_tqdm.py         # 彩虹进度条（美化）
└── IRDataset/                   # IR数据集模块（自治）
    ├── __init__.py             # 导出IRDataset和IRDataLoader
    ├── compound_tools.py       # 复用ECD的分子工具（或独立实现）
    ├── place_env.py            # 设备上下文管理器（复用）
    └── colored_tqdm.py         # 彩虹进度条（复用）
```

#### 模块自治的核心设计

每个数据集模块都是**独立、完整、可插拔**的：

```python
# ECDFormerDataset/__init__.py
from .dataloader import ECDFormerDataLoader
from . import ECDFormerDataset

__all__ = ['ECDFormerDataset', 'ECDFormerDataLoader']

# IRDataset/__init__.py  
from . import IRDataset, IRDataLoader

__all__ = ['IRDataset', 'IRDataLoader']

# ppmat/datasets/__init__.py
from ppmat.datasets.ECDFormerDataset import ECDFormerDataset, ECDFormerDataLoader
from ppmat.datasets.IRDataset import IRDataset, IRDataLoader

__all__ += ['ECDFormerDataset', 'ECDFormerDataLoader', 'IRDataset', 'IRDataLoader']
```

### 2、关键技术点

#### 技术点1：自定义DataLoader与collate_fn

每个数据集模块实现自己的 `DataLoader` 子类，包含解包逻辑：

```python
class ECDFormerDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=True, **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    @staticmethod
    def _collate_fn(batch):
        # 1. 分别batch化两个图
        batch_atom_bond = Batch.from_data_list([item[0] for item in batch])
        batch_bond_angle = Batch.from_data_list([item[1] for item in batch])
        
        # 2. 解包为Tensor字典（静态图就绪！）
        return {
            "x": batch_atom_bond.x,
            "edge_index": batch_atom_bond.edge_index,
            "edge_attr": batch_atom_bond.edge_attr,
            "batch_data": batch_atom_bond.batch,
            "ba_edge_index": batch_bond_angle.edge_index,
            "ba_edge_attr": batch_bond_angle.edge_attr,
            "query_mask": batch_atom_bond.query_mask
        }, {
            "peak_number_gt": batch_atom_bond.peak_num,
            "peak_position_gt": batch_atom_bond.peak_position,
            "peak_height_gt": batch_atom_bond.peak_height
        }
```

**收益**：模型可直接接收Tensor字典，完美支持 `paddle.jit.to_static`。

#### 技术点2：类级别缓存

```python
class ECDFormerDataset(Dataset):
    _cache = None  # 类级别缓存
    
    def __init__(self, path, ...):
        if self._cache:
            self.graph_atom_bond, self.graph_bond_angle = self._cache
            return
        # ... 首次加载 ...
        self._cache = (self.graph_atom_bond, self.graph_bond_angle)
```

**收益**：用户创建train/val/test三个数据集实例时，只有第一次需要等待（30-60秒），后两次瞬间返回。

#### 技术点3：按需读取（IRDataset关键优化）

```python
def read_ir_spectra_by_ids(sample_path, index_all):
    for fileid in tqdm(index_all):  # 只读需要的文件！
        filepath = os.path.join(sample_path, f"{fileid}.json")
        # ... 处理 ...
```

**收益**：避免扫描全部120,000+个JSON文件，节省约1小时加载时间。

#### 技术点4：设备上下文管理器

```python
@PlaceEnv(paddle.CPUPlace())
def __init__(self, ...):
    # 整个初始化过程在CPU上执行
```

**收益**：确保数据加载在CPU进行，避免GPU资源占用和混合设备错误。

### 3、主要影响的模块接口变化

#### 新增暴露的API

```python
# 用户代码中可直接导入使用
from ppmat.datasets import ECDFormerDataset, ECDFormerDataLoader
from ppmat.datasets import IRDataset, IRDataLoader

# ECD数据集
ecd_dataset = ECDFormerDataset(path="./dataset/ECD")
ecd_loader = ECDFormerDataLoader(ecd_dataset, batch_size=32)

# IR数据集（默认100样本快速测试）
ir_dataset = IRDataset(path="./dataset/IR", mode='100')
ir_loader = IRDataLoader(ir_dataset, batch_size=64)
```

#### 对框架各环节的影响排查

- **网络定义**：✅ 无影响
- **底层数据结构**：✅ 无影响，输出已解包为Tensor
- **数据IO**：✅ 无影响，通过自定义DataLoader实现
- **执行**：✅ 无影响，支持动态图/静态图/分布式
- **模型保存**：✅ 无影响
- **预测部署**：**正向影响**！纯Tensor输入可支持静态图下的训练

---

## 六、测试和验收的考量

### 自测方案
1. **数据加载测试**：验证 `__len__` 和 `__getitem__` 返回值正确
2. **缓存测试**：第二次实例化时间 < 0.1秒
3. **按需读取测试**（IR）：只读index_all中的文件，不扫描全部
4. **静态图测试**：通过 `to_static` 转换后前向传播正常
5. **端到端训练测试**：在ECD和IR小数据集上跑通训练流程

### 目标达成验收
| 验收项 | 通过标准 |
|-------|---------|
| **模块自治** | 每个数据集独立，不依赖框架其他组件 |
| **静态图支持** | `paddle.jit.to_static` 成功转换 |
| **缓存机制** | 第二次实例化时间 < 0.1秒 |
| **按需读取** | IRDataset只读 `index_all` 中的文件 |

---

## 七、影响面

### 对用户的影响
- **正向影响**：开箱即用，无需理解数据格式；默认小数据集（IR 100样本）快速验证
- **学习成本**：只需知道 `mode` 参数，调用方式与Paddle官方Dataset一致

### 对二次开发用户的影响
- 新增谱图任务（如质谱）可参考 `ECDFormerDataset` 模板，30分钟内完成新数据集接入
- 暴露的 `DataLoader` 子类可直接用于训练循环

### 对框架架构的影响
- 为 `ppmat/datasets` 新增两个自治模块，与现有数据集并列
- `place_env.py` 中的设备上下文管理器有潜力贡献给Paddle主框架

### 对性能的影响

| 数据集 | 首次加载 | 二次加载 | **收益** |
|-------|---------|---------|---------|
| **ECDFormerDataset** | 30-60秒 | <0.1秒 | **300-600倍提速** |
| **IRDataset (10000)** | 4分钟 | <0.1秒 | **2400倍提速** |
| **IRDataset (all)** | **1~2小时** | <0.1秒 | **36000-72000倍提速** |


---

## 名词解释

- **模块自治**：每个数据集模块独立包含自己的所有依赖，不依赖框架其他组件
- **类级别缓存**：使用Python类变量实现的缓存，同一类的所有实例共享
- **按需读取**：只读取 `index_all` 中指定的文件，而非扫描目录
- **`PlaceEnv`**：设备上下文管理器，临时改变Paddle默认设备

---

## 附件及参考资料

1. Li, H. et al. Decoupled peak property learning for efficient and interpretable electronic circular dichroism spectrum prediction. *Nature Computational Science*, 2025.
2. ECDFormerDataset 设计文档：`ECDFormerDataset.md`
3. IRDataset 设计文档：`IRDataset.md`
4. PaddlePaddle Issue #77754：DataLoader cannot return non-Tensor data
