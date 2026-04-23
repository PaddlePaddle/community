# 【Hackathon 10th Spring No.7】wD-MPNN 高分子属性预测模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | bobby-cloudforge        |
| 提交时间      | 2026-05-21         |
| RFC 版本号    | v1.1               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_wd_mpnn_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

wD-MPNN（weighted Directed Message Passing Neural Network）是一种专门用于 **高分子（polymer）属性预测** 的有向消息传递神经网络，由 Aldeghi 和 Coley 在 *Chemical Science* 2022, 13, 10486-10498 发表，论文标题为 "A graph representation of molecular ensembles for polymer property prediction"。该模型在 Chemprop 框架基础上扩展，引入加权有向边和单体连接性表示，实现对高分子集合体（molecular ensemble）的属性预测。

wD-MPNN 的核心创新在于：
1. **高分子图表示**：将共聚物/高分子表示为加权有向图，其中边权反映单体连接频率和方向性（不同于一般分子的无权无向图）
2. **加权有向消息传递**：在 D-MPNN 基础上引入边权重，消息传递时按连接概率加权，捕捉共聚物中单体间的统计共聚关系
3. **扩展 SMILES 输入**：设计专用输入格式，编码单体 SMILES、化学计量比、连接点权重和聚合度信息（`monomer_SMILES|stoichiometry|connectivity~Xn`）
4. **聚合度嵌入**：将聚合度（degree of polymerization, Xn）通过 `1 + log(Xn)` 变换嵌入到图表示中

- 原始代码仓库：https://github.com/coleygroup/polymer-chemprop（基于 https://github.com/chemprop/chemprop 分叉，MIT License，49 stars）
- 论文链接：https://doi.org/10.1039/D2SC02839E
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 2 项）

### 1.2 功能目标

1. 在 `ppmat/models/wd_mpnn/` 下实现 wD-MPNN 模型，包含高分子图构建、加权有向消息传递编码器、属性预测头
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 支持扩展 SMILES 格式的高分子数据加载（单体连接性、化学计量比、聚合度）
4. 前向精度对齐（监督任务要求 logits diff 1e-4 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 监督任务 metric 误差控制在 1% 以内（RMSE/MAE/R²）
7. 覆盖原论文中的高分子属性预测基准（玻璃化转变温度 Tg、电子亲和能 EA、离子电导率等）
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 目前尚无高分子/聚合物属性预测模型（现有模型面向晶体和小分子），wD-MPNN 将填补这一空白，同时引入加权有向图范式。
2. 为共聚物、嵌段聚合物等复杂高分子体系提供图表示能力，拓展 PaddleMaterials 在电池电解质、药物载体、工程塑料等领域的应用场景。
3. wD-MPNN 是高分子信息学领域的标杆方法（Coley 组, MIT），与已有 DimeNet++（晶体）、SchNet（小分子势函数）形成互补。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下属性预测模型：

| 已有模型 | 任务类型 | 架构 | 与 wD-MPNN 关系 |
|---------|---------|------|----------------|
| DimeNet++ | 晶体属性预测 | 三体 GNN | 可参考图构建、scatter 操作 |
| MEGNet | 材料属性预测 | GNN | 可参考全局状态向量设计 |
| SchNet | 原子间势函数 | 连续滤波 GNN | 可参考距离嵌入逻辑 |
| InfGCN | 属性预测 | 图卷积 | 可参考消息传递实现 |
| ComFormer | 属性预测 | Transformer | 可参考 readout 设计 |

**关键现状分析**：

1. **缺少高分子图表示**：现有模型处理晶体（周期性）或小分子（无向图），均不支持共聚物的加权有向图表示。需新增高分子图构建模块。
2. **缺少扩展 SMILES 解析**：现有数据集使用 CIF/POSCAR（晶体）或标准 SMILES（小分子）。wD-MPNN 的扩展 SMILES 格式（含化学计量比和连接权重）需新增解析器。
3. **缺少有向消息传递**：现有 GNN 均使用无向边。wD-MPNN 的加权有向消息传递需新增实现。
4. **缺少 RDKit 特征提取器**：现有特征提取基于原子距离矩阵，缺少化学键特征（键类型、共轭、芳香性等）。需新增基于 RDKit 的原子/键特征提取模块。

已有基础设施可复用：

- `ppmat/trainer/BaseTrainer`：训练/评估循环可直接沿用，支持回归任务的 early stopping 和 checkpoint 管理。
- `ppmat/losses/`：MSELoss、MAELoss 等回归损失函数可直接调用。
- `ppmat/optimizer/build_optimizer`：统一优化器和学习率调度构建。
- YAML + OmegaConf 配置体系：复用全局配置解析逻辑。

## 3. 目标调研

### 3.1 模型架构概述

wD-MPNN 数据流：扩展 SMILES → 高分子图构建（加权有向边） → 原子/键特征提取 → 加权有向消息传递（T 层） → Readout → 属性预测。

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| Polymer Graph Builder | 解析扩展 SMILES，构建加权有向图 | 连接权重、化学计量比、Xn |
| Atom/Bond Featurizer | 原子/键特征提取（RDKit） | atom ~70 维, bond ~14 维 |
| Weighted D-MPNN Encoder | 加权有向消息传递 T 层 | hidden=300, T=3 |
| Readout FFN | 原子→分子表示 | 2 层 FFN, hidden=300 |
| Prediction Head | 属性预测 | 回归（MSE）/ 分类（CE） |

**与原始 D-MPNN 的关键区别**：

| 特性 | 原始 D-MPNN (Yang 2019) | 高分子 wD-MPNN (Aldeghi 2022) |
|-----|----------------------|------------------------------|
| 图类型 | 无权有向图 | **加权**有向图（边权=连接概率） |
| 输入格式 | 标准 SMILES | 扩展 SMILES（含化学计量比+连接权重+Xn） |
| 消息聚合 | 等权求和 | 按边权加权求和 |
| 适用对象 | 小分子 | 共聚物、均聚物、高分子集合体 |
| 聚合度 | 不适用 | `1 + log(Xn)` 嵌入 |

### 3.2 源码结构分析

polymer-chemprop 基于 chemprop 分叉，核心修改集中在消息传递和数据处理：

| 文件 | 功能 | 复现策略 |
|-----|------|---------|
| `chemprop/features/featurization.py` | 原子/键特征 + **高分子图构建** | 核心迁移：扩展 SMILES 解析、加权边构建 |
| `chemprop/models/mpn.py` | D-MPNN 编码器 + **加权消息传递** | 核心迁移：边权加权聚合 |
| `chemprop/nn/message_passing.py` | 消息传递层 | 添加权重支持 |
| `chemprop/data/` | 数据加载、Batch 打包 | 适配 ppmat DataLoader |
| `data.tar.gz` | 高分子属性数据集 | 预处理后上传 BCS |

### 3.3 扩展 SMILES 格式

wD-MPNN 独有的高分子输入格式：

```
[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1ccc([*:4])c(N)c1|0.25|0.75|<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25~502.5
```

解析结构：
- **单体 SMILES**：`[*:N]` 标记连接点，`.` 分隔不同单体
- **化学计量比**：`|0.25|0.75|` — 两种单体的摩尔分数
- **连接权重**：`<1-3:0.25:0.25` — 连接点 1→3 的正向/反向边权
- **聚合度**：`~502.5` — Xn（数均聚合度）

### 3.4 PyTorch → Paddle API 映射

wD-MPNN 仅使用标准 PyTorch NN 操作，无自定义 CUDA kernel，迁移风险极低：

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `nn.Linear` | `paddle.nn.Linear` | 一致 |
| `nn.ModuleList` | `paddle.nn.LayerList` | 命名不同 |
| `nn.Parameter` | `paddle.create_parameter` | 初始化方式不同 |
| `torch.cat` | `paddle.concat` | 命名不同 |
| `scatter` | `ppmat.utils.scatter` | 复用已有工具 |

**不需迁移**：扩展 SMILES 解析（纯 Python）、原子/键特征提取（依赖 RDKit，非 PyTorch）

### 3.5 数据集概况

原论文使用高分子属性数据集（随 repo `data.tar.gz` 提供）：

| 数据集 | 属性 | 任务类型 | 数据量 | 说明 |
|--------|------|---------|--------|------|
| Tg | 玻璃化转变温度 | 回归 | ~7k | 高分子热力学关键参数 |
| EA | 电子亲和能 | 回归 | ~5k | 电化学稳定性指标 |
| Ionic Conductivity | 离子电导率 | 回归 | ~3k | 电池电解质设计核心参数 |

数据集随 repo 提供（`data.tar.gz`），MIT License，可直接使用。

### 3.6 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ coleygroup/polymer-chemprop（MIT License，49 stars） |

目前仅有原始 PyTorch 实现（基于 Chemprop 分叉），无其他框架移植。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/wd_mpnn/
│   ├── __init__.py                  # 模块导出 + 注册
│   ├── wd_mpnn.py                   # 主模型（_forward/forward/predict 三层模式）
│   ├── mpn_encoder.py               # 加权有向消息传递编码器
│   └── polymer_featurizer.py        # 高分子特征提取 + 扩展 SMILES 解析
├── ppmat/datasets/polymer_dataset.py  # 高分子数据集 + build_polymer 工厂函数
├── polymer_property_prediction/       # 高分子属性预测任务
│   ├── README.md                    # 任务说明
│   └── configs/wd_mpnn/
│       ├── wd_mpnn_tg.yaml          # 玻璃化转变温度预测
│       ├── wd_mpnn_ea.yaml          # 电子亲和能预测
│       └── wd_mpnn_conductivity.yaml # 离子电导率预测
├── examples/wd_mpnn/
│   ├── README.md
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 wd_mpnn 导入
# ppmat/datasets/__init__.py   — 新增 build_polymer 工厂函数注册
```

### 4.2 模型实现

核心模型遵循 PaddleMaterials 统一的 `_forward()`/`forward()`/`predict()` 三层设计。

#### 数据流

```
扩展 SMILES "mono1.mono2|s1|s2|<i-j:w_fwd:w_rev..~Xn"
                       ↓
              PolymerGraphBuilder
    （RDKit 解析 → 有向图 + 边权 + Xn）
                       ↓
            {atom_features, bond_features,
             edge_index, edge_weights, Xn}
                       ↓
              atom_embed → h₀[N, D]
                       ↓
         ┌─────────────────────────┐
         │ WeightedMessagePassing  │ × T 层（残差连接）
         │                         │
         │  m_{u→v} = ReLU(W·h_u + W_b·e_uv)
         │  h_v' = h_v + Σ_{u→v} w_{u→v} · m_{u→v}
         │          ↑ 核心创新：有向边权重调制消息
         └─────────────────────────┘
                       ↓
         scatter_sum(batch_index) → mol_repr
                       ↓
         mol_repr × (1 + log Xn)  ← 聚合度缩放
                       ↓
                 MLP → 属性预测值
```

#### 关键设计决策

1. **有向加权边**：单体间连接权重来自扩展 SMILES 规范，单体内键权重恒为 1.0
2. **聚合度缩放**：`mol_repr × (1 + log Xn)` 建模聚合物分子量对属性的影响
3. **消息传递**：`scatter(weighted_messages, target_index, reduce="sum")` 复用 `ppmat.utils.scatter`

#### 类签名

```
WDMPNN(hidden_dim=300, n_layers=3, dropout=0.1, atom_fdim=70, bond_fdim=14)
  ├─ _forward(batch) → logits: Tensor[n_mols, 1]
  ├─ forward(batch)  → (loss_dict, pred_dict)     # 训练入口
  └─ predict(batch)  → {"logits": Tensor}          # 推理入口

WeightedMessagePassing(hidden_dim)
  └─ forward(h, bond_features, edge_index, edge_weights) → aggregated

PolymerGraphBuilder
  └─ parse_extended_smiles(ext_smiles) → graph_dict
```

### 4.3 数据集适配

数据格式：CSV，每行一条高分子记录。关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `smiles` | str | 扩展 SMILES（含单体连接权重和 Xn） |
| `target` | float | 回归目标值（如 Tg、密度等） |

工厂函数签名：

```
build_polymer(config) → PolymerDataset
  config.data_path    — CSV 文件路径
  config.target_col   — 目标列名（默认 'target'）
```

`PolymerDataset.__getitem__` 通过 `PolymerGraphBuilder.parse_extended_smiles` 将 SMILES 转换为图结构字典。

### 4.4 训练流程

直接使用 `ppmat/trainer/BaseTrainer`，**不新增自定义训练器文件**，关键适配点：

1. **损失函数**：MSE/MAE（回归任务）、16. §CrossEntropyLoss（分类任务）
2. **优化器**：Adam，学习率 1e-4，Noam 调度器
3. **评估指标**：RMSE、MAE、R²

### 4.5 YAML 配置示例

```yaml
Global:
  task: polymer_property_prediction
  seed: 42
  output_dir: output/wd_mpnn_tg/

Model:
  __class_name__: WDMPNN
  __init_params__:
    hidden_dim: 300
    n_message_passing: 3
    n_ffn_layers: 2

Dataset:
  __class_name__: polymer
  __init_params__:
    data_path: data/wd_mpnn/glass_transition_temp.csv
    target_col: Tg
    split_ratio: [0.8, 0.1, 0.1]

Optimizer:
  __class_name__: Adam
  __init_params__:
    lr: 1.0e-4
    weight_decay: 0.0
    lr_scheduler: NoamDecay
    warmup_steps: 2000
```

### 4.6 补充说明

- 本次复现不包含 Chemprop 中的多任务共训练（multi-task）功能，仅复现单任务预测。
- 不迁移 Chemprop 的超参数搜索模块（hyperopt），使用固定超参。

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同扩展 SMILES 输入和随机种子，对比 PyTorch（polymer-chemprop）和 Paddle 实现的 logits 输出
- 要求：logits diff ≤ 1e-4

### 5.2 反向训练对齐

- 使用相同高分子数据子集，对比两个框架训练 2 轮以上的 loss 曲线
- 要求 loss 逐 epoch 一致

### 5.3 监督任务指标

| 属性 | 指标 | 原论文值 | 允许误差 |
|------|------|---------|---------|
| Tg（玻璃化转变温度） | RMSE / R² | 论文 Table 2 | ±1% |
| EA（电子亲和能） | RMSE / R² | 论文 Table 2 | ±1% |
| Ionic Conductivity（离子电导率） | RMSE / R² | 论文 Table 2 | ±1% |

### 5.4 扩展 SMILES 解析验证

- 验证扩展 SMILES 解析器正确处理：多单体、化学计量比、双向连接权重、聚合度
- 边权归一化校验：出边权重之和应反映化学计量

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **中** | 基于 Chemprop 标准 GNN，加权消息传递逻辑额外 ~100 行 |
| 自定义算子 | **无** | 无 CUDA kernel，全部为标准 NN 操作 + scatter |
| 依赖项风险 | **低** | RDKit（标准化学信息学库）、rdkit 已在 ppmat 生态中使用 |
| 数据可获取性 | **高** | 数据随 repo 提供（`data.tar.gz`），MIT License |
| API 映射完整性 | **高** | 所有 PyTorch API 均有 Paddle 对应实现 |
| 关键风险 | **中** | 扩展 SMILES 解析逻辑需仔细迁移（字符串处理，非 tensor 操作） |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | GPU 环境配置、原始 polymer-chemprop 验证 | 2 天 |
| Phase 1：数据处理 | 扩展 SMILES 解析器、高分子图构建 | 3 天 |
| Phase 2：模型迁移 | 加权有向消息传递、编码器、预测头 → Paddle | 4 天 |
| Phase 3：训练对齐 | Trainer 适配、loss 曲线对齐 | 3 天 |
| Phase 4：评估验证 | Tg/EA/Conductivity 全量训练+评估 | 4 天 |
| Phase 5：文档与合入 | README 编写、提交 PR | 2 天 |

**合计**：~18 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `wd_mpnn/` 目录（~500 行代码），包含加权有向消息传递编码器和高分子图构建器。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `polymer_dataset.py`（~100 行），支持扩展 SMILES 格式的高分子表格数据加载，提供开箱即用的 Tg、EA、离子电导率预测能力。
3. **新增任务类型**：新建 `polymer_property_prediction/` 目录，引入“高分子属性预测”任务类型，填补 PaddleMaterials 在聚合物材料领域的空白。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| wD-MPNN | weighted Directed Message Passing Neural Network，加权有向消息传递神经网络 |
| 扩展 SMILES | 包含单体连接权重、化学计量比和聚合度信息的 SMILES 编码格式 |
| 聚合度 (Xn) | 高分子链中重复单元的平均数量（数均聚合度） |
| 化学计量比 | 共聚物中不同单体的摩尔分数 |
| 玻璃化转变温度 (Tg) | 高分子从玻璃态转变为橡胶态的温度，关键热力学参数 |

## 附件及参考资料

1. Aldeghi, Matteo and Coley, Connor W. "A graph representation of molecular ensembles for polymer property prediction." *Chemical Science* 13.35 (2022): 10486-10498. DOI: 10.1039/D2SC02839E
2. Yang, Kevin et al. "Analyzing learned molecular representations for property prediction." *Journal of Chemical Information and Modeling* 59.8 (2019): 3370-3388.
3. polymer-chemprop 源码：https://github.com/coleygroup/polymer-chemprop（MIT License）
4. 原始 Chemprop 框架：https://github.com/chemprop/chemprop
5. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 2 项）
