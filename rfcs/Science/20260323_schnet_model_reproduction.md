# 【Hackathon 10th Spring No.18】SchNet 模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | cloudforge1        |
| 提交时间      | 2026-03-23         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260323_schnet_model_reproduction.md |

## 1. 概述

### 1.1 相关背景

SchNet 是一种基于连续滤波卷积（Continuous-Filter Convolution）的深度学习模型，用于建模分子和材料体系中的量子化学相互作用。由 Schütt 等人提出（NeurIPS 2017, arXiv:1706.08566；JCP 2018, arXiv:1712.06113）。

核心创新：**连续滤波卷积层（CFConv）** — 利用原子间距离经高斯径向基函数（RBF）展开后，通过可学习的滤波网络动态生成原子间相互作用滤波器。模型天然满足平移、旋转不变性和原子置换等变性。

SchNet 是分子属性预测领域最早且最具影响力的端到端方法之一（Google Scholar 3000+），在 QM9 和 MD17 基准上取得了当时的 SOTA，是 DimeNet++、PaiNN 等后续模型的设计起点。

- 原始代码仓库：https://github.com/atomistic-machine-learning/schnetpack
- 论文：https://arxiv.org/abs/1706.08566 / https://arxiv.org/abs/1712.06113
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194

### 1.2 功能目标

1. 将 SchNet 从 PyTorch（SchNetPack v0.3）迁移至 PaddlePaddle，集成到 `ppmat/models/`
2. 复用已有的 `qm9_dataset.py`；新增 MD17 数据集，接入 `ppmat/datasets/` 工厂函数
3. 基于 BaseTrainer 实现 YAML 配置驱动的训练流程
4. 在 QM9 和 MD17 上复现论文精度（MAE）
5. 提供 PyTorch→Paddle 权重转换与前向对齐验证工具

### 1.3 意义

SchNet 作为连续滤波卷积范式的开创者，是 PaddleMaterials 属性预测谱系的基础。目前已有 DimeNet++（SchNet 的后续工作），但缺少 SchNet 本身。加入后可与 DimeNet++、MEGNet 形成完整的 GNN 属性预测对比矩阵，并为轻量级分子筛选场景提供高效基线。

## 2. PaddleMaterials 现状

已集成模型：CHGNet、ComFormer、DimeNet++、MEGNet、InfGCN、MatENO（属性预测），DiffCSP、MatterGen（结构生成），DiffNMR（谱图），MatterSim（分子动力学）。**尚无 SchNet。**

SchNet vs DimeNet++ 定位差异：

| 对比项 | SchNet | DimeNet++ |
|--------|--------|-----------|
| 核心表征 | 原子间距离（二体） | 距离 + 键角（三体） |
| 计算复杂度 | $O(N \cdot k)$ | $O(N \cdot k^2)$ |
| 精度层级 | 基线级 | 更高精度 |
| 场景 | 快速筛选、MD | 高精度属性预测 |

可复用基础设施：

| 组件 | 路径 | 复用方式 |
|------|------|---------|
| QM9 数据集 | `ppmat/datasets/qm9_dataset.py` | 直接使用 |
| 图构建 | `ppmat/datasets/build_molecule.py` | 直接使用 |
| scatter 操作 | `ppmat/utils/scatter.py` | 邻居聚合 |
| BaseTrainer | `ppmat/trainer/base_trainer.py` | 继承使用 |
| 训练入口 | `property_prediction/train.py` | 直接使用 |

需新增：SchNet 模型（`ppmat/models/schnet/`）、MD17 数据集（`ppmat/datasets/md17_dataset.py`）、YAML 配置（`property_prediction/configs/schnet/`）、权重转换工具、单元测试。

## 3. 目标调研

### 3.1 模型架构

SchNet 的数据流：原子序数 $Z \xrightarrow{\text{Embedding}} \mathbf{x}^0$，距离 $d_{ij} \xrightarrow{\text{GaussianRBF}} \mathbf{e}_{ij}$，经 $T$ 层交互后原子级 MLP 预测，scatter sum 得分子属性。

激活函数：$\text{ssp}(x) = \ln(1 + e^x) - \ln 2$（Shifted Softplus）

#### 论文标准配置

| 参数 | QM9 | MD17 |
|------|-----|------|
| `n_features` | 128 | 64 |
| `n_interactions` | 6 | 6 |
| `cutoff` (Å) | 10.0 | 5.0 |
| `n_gaussians` | 50 | 25 |
| `max_z` | 100 | 100 |
| 参数量 | ~594K | ~150K |

#### 组件层次

```
SchNet
├── embedding: Embedding(max_z, n_features)
├── distance_expansion: GaussianRBF(n_gaussians, cutoff)
├── interactions × n_interactions:
│   ├── filter_network: Dense(n_gaussians→n_features) → SSP → Dense(n_features→n_features)
│   ├── cfconv: in2f(no bias) → element-wise × filter → f2out
│   └── dense: Linear(n_features, n_features) + residual
└── output: Dense(n_features→n_features//2) → SSP → Dense(→1) → scatter_sum
```

### 3.2 数据集与指标

**QM9**（已有 `qm9_dataset.py`）：~134k 有机小分子，12 个量化学属性。论文关键 MAE：

| 属性 | 单位 | SchNet MAE |
|------|------|-----------|
| $U_0$ | meV | 14 |
| HOMO | meV | 41 |
| LUMO | meV | 34 |
| Gap | meV | 63 |
| $\mu$ | mDebye | 33 |

**MD17**（需新增）：8 个分子的从头算 MD 轨迹，含能量和力。来源：http://www.sgdml.org/#datasets

| 分子 | 能量 MAE (meV) | 力 MAE (meV/Å) |
|------|---------------|----------------|
| Ethanol | 0.08 | 0.39 |
| Malonaldehyde | 0.08 | 0.66 |
| Aspirin | 0.37 | 1.35 |

### 3.3 torch→Paddle 迁移

**迁移复杂度：低。** SchNet 不涉及自定义 CUDA 算子。核心 API 均有 1:1 对应（`nn.Embedding`、`nn.Linear`、`nn.ModuleList→LayerList`、`scatter_add→ppmat.utils.scatter`）。

关键注意点：Linear 权重 PyTorch `[out, in]` → Paddle `[in, out]`；`shifted_softplus` 需手动实现。

### 3.4 已有实现

PyTorch 有三个实现（SchNetPack 官方、PyG、OCP），PaddlePaddle/MindSpore 均无。以 SchNetPack v0.3 为参考（论文作者维护，架构与论文最一致）。

## 4. 设计思路与实现方案

### 4.1 代码结构

```
PaddleMaterials/
├── ppmat/models/schnet/
│   ├── __init__.py
│   └── schnet.py                    # 模型定义（~400行）
├── ppmat/datasets/md17_dataset.py   # MD17 数据集
├── property_prediction/configs/schnet/
│   ├── schnet_qm9_energy_U0.yaml
│   ├── schnet_qm9_homo.yaml
│   └── schnet_md17_ethanol.yaml
├── tools/convert_schnet_weights.py  # 权重转换
└── test/test_schnet.py
```

### 4.2 核心模型

遵循 DimeNet++ 的三层方法模式（`_forward`/`forward`/`predict`），关键组件：

```python
import paddle
import paddle.nn as nn
import numpy as np
from ppmat.utils.scatter import scatter

def shifted_softplus(x):
    return paddle.nn.functional.softplus(x) - np.log(2.0)

class GaussianRBF(nn.Layer):
    """高斯径向基函数展开 distances → [n_edges, n_gaussians]"""
    def __init__(self, n_gaussians=50, cutoff=10.0):
        super().__init__()
        offsets = paddle.linspace(0.0, cutoff, n_gaussians)
        self.register_buffer("offsets", offsets)
        width = offsets[1] - offsets[0]
        self.register_buffer("width", paddle.full([n_gaussians], width))

    def forward(self, distances):
        return paddle.exp(-0.5 * ((distances.unsqueeze(-1) - self.offsets) / self.width) ** 2)

class CFConv(nn.Layer):
    """连续滤波卷积：filter_network 生成滤波器 → 逐元素乘 → 邻居聚合"""
    def __init__(self, n_features, n_gaussians):
        super().__init__()
        self.in2f = nn.Linear(n_features, n_features, bias_attr=False)
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, n_features),
            nn.Linear(n_features, n_features),
        )
        self.f2out = nn.Linear(n_features, n_features)

    def forward(self, x, edge_index, edge_attr, n_atoms):
        x_j = self.in2f(x)[edge_index[0]]
        W = shifted_softplus(self.filter_net[0](edge_attr))
        W = self.filter_net[1](W)
        y = scatter(x_j * W, edge_index[1], dim=0, dim_size=n_atoms, reduce="sum")
        return self.f2out(y)

class SchNetInteraction(nn.Layer):
    def __init__(self, n_features, n_gaussians):
        super().__init__()
        self.cfconv = CFConv(n_features, n_gaussians)
        self.dense = nn.Linear(n_features, n_features)

    def forward(self, x, edge_index, edge_attr, n_atoms):
        v = shifted_softplus(self.cfconv(x, edge_index, edge_attr, n_atoms))
        return x + self.dense(v)  # residual

class SchNet(nn.Layer):
    def __init__(self, n_features=128, n_interactions=6, cutoff=10.0,
                 n_gaussians=50, max_z=100, property_names="energy_U0",
                 readout="sum", data_mean=0.0, data_std=1.0, loss_type="l1_loss"):
        super().__init__()
        self.embedding = nn.Embedding(max_z, n_features, padding_idx=0)
        self.distance_expansion = GaussianRBF(n_gaussians, cutoff)
        self.interactions = nn.LayerList([
            SchNetInteraction(n_features, n_gaussians) for _ in range(n_interactions)
        ])
        self.output_network = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.Linear(n_features // 2, 1),
        )
        # normalize/unnormalize, loss_fn 与 DimeNet++ 完全同构

    def _forward(self, data):
        """解析图数据 → RBF 展开 → 交互层循环 → 原子级预测 → scatter 聚合"""
        ...

    def forward(self, data, return_loss=True, return_prediction=True):
        """训练入口，返回 {"loss_dict": ..., "pred_dict": ...}（同 DimeNet++）"""
        ...

    def predict(self, graphs):
        """推理入口（同 DimeNet++）"""
        ...
```

### 4.3 MD17 数据集

参考 `qm9_dataset.py` 实现模式，使用 ASE 读取 `.xyz` 格式 MD 轨迹。接口与 QM9Dataset 一致（`path` + `property_names` + `build_structure_cfg` + `build_graph_cfg`），注册到工厂函数。MD17Dataset 在首次使用时从 http://www.sgdml.org/#datasets 自动下载原始数据，缓存至 `path` 目录（复用 `ppmat.utils.download` 的下载-解压-缓存逻辑）。

### 4.4 YAML 配置

```yaml
# schnet_qm9_energy_U0.yaml（核心字段）
Model:
  __class_name__: SchNet
  __init_params__:
    n_features: 128
    n_interactions: 6
    cutoff: 10.0
    n_gaussians: 50
    max_z: 100
    readout: sum
    property_names: ${Global.label_names}
    loss_type: l1_loss
Optimizer:
  __class_name__: Adam
  __init_params__:
    lr:
      __class_name__: ReduceOnPlateau
      __init_params__:
        learning_rate: 0.0001
        factor: 0.5
        patience: 25
Trainer:
  max_epochs: 1000
  output_dir: ./output/schnet_qm9_energy_U0
Dataset:
  train:
    dataset:
      __class_name__: QM9Dataset
      __init_params__:
        path: "./data/qm9/"
        property_names: ${Global.label_names}
```

### 4.5 权重转换

`tools/convert_schnet_weights.py` 实现 PyTorch→Paddle 参数格式转换（Linear 转置 `[out,in]→[in,out]` + 键名映射），用于**开发阶段**的前向/反向对齐验证——确认 Paddle 实现与原始 PyTorch 代码在相同输入下输出一致。Embedding 直接复制，buffer 直接复制。最终验收精度基于从零训练结果。

## 5. 测试和验收的考量

| 验收项 | 标准 | 方法 |
|--------|------|------|
| 前向精度对齐 | logits diff ≤ 1e-6 | 使用相同初始化参数，Paddle 与 PyTorch 同输入对比 |
| 反向对齐 | loss 趋势一致 | 相同数据超参训练 2+ 轮 |
| QM9 训练精度 | metric 误差 ≤ 1% | 完整训练，MAE 对比论文 |
| MD17 训练精度 | metric 误差 ≤ 1% | 完整训练，能量/力 MAE |
| 数据集覆盖 | QM9 全 12 属性 + MD17 ≥ 4 分子 | 对应论文所有数据集 |

| 转换权重上传 | BCS 上传 `.pdparams` | 提供下载链接，用于社区验证对齐 |
| 训练日志 | 完整日志文件 | QM9 + MD17 训练日志附 PR 或上传 BCS |

测试项：模型前向形状验证、GaussianRBF 正确性、参数格式转换正确性、完整训练流程集成测试、Paddle/PyTorch loss 曲线对比截图。

## 6. 可行性分析和排期规划

| 阶段 | 时间 | 内容 |
|------|------|------|
| 第 1 周 | 模型迁移 | SchNet 代码、权重转换、前向对齐 |
| 第 2 周 | 数据与训练 | MD17 数据集、YAML 配置、QM9 训练复现 |
| 第 3 周 | 完整复现 | MD17 训练、全属性 MAE 表、推理脚本 |
| 第 4 周 | 验收文档 | 单元测试、README、日志上传、提交 PR |

SchNet 架构为标准 GNN，无自定义 CUDA 算子，API 映射全面 1:1 对应，迁移风险极低。QM9 单卡 V100 约 6-8 小时完成训练。

## 7. 影响面

1. **新增模型**：`ppmat/models/schnet/`
2. **新增数据集**：`ppmat/datasets/md17_dataset.py`（可被后续模型如 PaiNN、NequIP 复用）
3. **新增配置**：`property_prediction/configs/schnet/`
4. **无侵入性修改**：仅在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 注册
5. 所有新增文件包含 Apache 2.0 License 头，遵循 PaddleMaterials 既有规范

## 附件及参考资料

1. Schütt et al. *"SchNet: A continuous-filter convolutional neural network for modeling quantum interactions."* NeurIPS 2017
2. Schütt et al. *"SchNet – A deep learning architecture for molecules and materials."* JCP 2018
3. SchNetPack：https://github.com/atomistic-machine-learning/schnetpack
4. PaddleMaterials issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194
5. DimeNet++ 实现（模板）：`ppmat/models/dimenetpp/dimenetpp.py`
