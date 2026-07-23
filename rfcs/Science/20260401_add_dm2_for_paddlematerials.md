# 【Hackathon 10th Spring No.15】DM2 模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | bobby-cloudforge        |
| 提交时间      | 2026-05-21         |
| RFC 版本号    | v1.1               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_dm2_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

DM2（Diffusion Models for Disordered Materials）是一种基于 score-based 扩散模型的非晶材料结构生成方法，由 Yang 和 Schwalbe-Koda 在 arXiv:2507.05024 (2025) 发表，论文标题为 "A generative diffusion model for amorphous materials"。该模型基于 LLNL/graphite 框架（Hsu et al., npj Computational Materials, 2024），通过等变图神经网络对原子位置进行 score-based 去噪，实现非晶态无机材料的条件/无条件生成。

DM2 的核心创新在于：
1. **非晶材料生成**：针对无序/非晶态材料（如 a-SiO2、Cu-Zr 金属玻璃），不假设周期性晶格对称性
2. **等变扩散去噪**：基于 e3nn 等变神经网络，在原子构型空间进行 score-based 去噪，保证 SE(3) 等变性
3. **工艺条件嵌入**：在 graphite 基础上新增加工条件嵌入（processing condition embedding），支持条件生成
4. **分子动力学集成**：生成结构可直接输入 MD 模拟进行验证

- 原始代码仓库：https://github.com/digital-synthesis-lab/DM2（基于 https://github.com/LLNL/graphite）
- 论文链接：https://arxiv.org/abs/2507.05024
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 10 项）

### 1.2 功能目标

1. 在 `ppmat/models/dm2/` 下实现 DM2 扩散模型，包含 e3nn 等变去噪网络、score-based 扩散 scheduler、条件嵌入
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 支持非晶材料数据加载（a-SiO2、Cu-Zr 等无序结构）
4. 前向精度对齐（生成式模型要求 logits diff 1e-6 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 采样指标误差控制在 5% 以内（生成式模型标准：结构有效性、径向分布函数匹配度）
7. 覆盖原论文核心材料体系：a-SiO2（无条件/条件）、Cu50Zr50
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 现有扩散模型（DiffCSP、MatterGen）针对晶体，DM2 将引入非晶/无序材料生成能力。
2. 引入 score-based equivariant denoising 范式，区别于现有 DDPM 方法，为非周期性结构提供新的生成框架。
3. DM2 由 Toyota Research Institute 支持，是非晶材料生成领域的前沿工作，复现有助于提升套件在非晶材料动力学模拟方面的竞争力。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下扩散/生成模型：

| 已有模型 | 任务类型 | 架构 | 与 DM2 关系 |
|---------|---------|------|-----------|
| MatterGen | 晶体结构生成 | DDPM + GNN | 可参考 scheduler、图构建 |
| DiffCSP | 晶体结构预测 | DDPM | 可参考扩散训练流程 |
| CDVAE | 晶体结构生成 | VAE + 扩散 | 可参考图表示方法 |

**关键现状分析**：

1. **缺少 e3nn 等变网络**：现有模型使用标准 GNN，DM2 需 e3nn 等变球谐基函数表示。需评估 Paddle 生态中 e3nn 替代方案或手动迁移。
2. **缺少 score-based 扩散**：现有 MatterGen/DiffCSP 使用 DDPM 范式，DM2 使用 score matching，scheduler 逻辑不同。
3. **非晶材料数据**：PaddleMaterials 尚无非晶态材料数据集/任务。需新增数据加载器。
4. **graphite 基础框架**：DM2 核心代码基于 LLNL/graphite，需将 graphite 的图构建和去噪逻辑迁移到 Paddle。

已有基础设施可复用：

- MatterGen / DiffCSP 扩散 scheduler：DDPM 核心噪声调度逻辑可参考。
- `ppmat/trainer/BaseTrainer`：训练循环可沿用。
- `ppmat/optimizer/build_optimizer`：统一优化器构建。
- YAML + OmegaConf 配置管线。

## 3. 目标调研

### 3.1 模型架构概述

DM2 数据流：原子构型（位置 + 类型） → 图构建（邻居搜索） → e3nn 等变 GNN 去噪网络 → 预测 score（噪声方向） → 迭代去噪 → 生成结构。

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| Graph Builder | 原子邻居图构建（cutoff radius） | cutoff=5.0 Å |
| e3nn Equivariant GNN | 等变消息传递去噪网络 | 球谐展开 L=2，hidden=128 |
| Score Network | 预测噪声方向（score） | 基于 e3nn 输出 |
| Condition Embedding | 加工条件嵌入（温度、压力等） | MLP 嵌入 |
| Diffusion Scheduler | Score-based 扩散过程 | σ_min/σ_max, N_steps |

### 3.2 源码结构分析

DM2 源码（`src/graphite/`）基于 LLNL/graphite：

| 文件 | 功能 | 复现策略 |
|-----|------|---------|
| `src/graphite/nn/` | e3nn 等变 GNN | 核心迁移：e3nn → Paddle 等价实现 |
| `src/graphite/diffusion/` | Score-based 扩散 scheduler | 迁移扩散逻辑 |
| `demo/demo_training/` | 训练脚本 | 适配 ppmat trainer |
| `demo/demo_generating/` | 生成脚本 | 适配 ppmat sampler |

### 3.3 PyTorch → Paddle API 映射

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `torch_geometric.data.Data` | 自定义图数据 | 需适配 ppmat 图结构 |
| `e3nn.o3.Irreps` | 手动实现或 Paddle 等变层 | **关键风险点** |
| `e3nn.nn.FullyConnectedTensorProduct` | 手动实现 | 需迁移等变张量积 |
| `e3nn.o3.spherical_harmonics` | 手动实现 | 球谐函数 |
| `torch.scatter` | `paddle.put_along_axis` / 自定义 | scatter 操作 |

**关键迁移风险**：e3nn 是 DM2 的核心依赖，Paddle 生态中无直接等价库。需手动迁移等变张量积、球谐函数等核心操作（~500 行），或复用 PaddleMaterials 中已有的等变操作（如 DimeNet++ 中的球谐基函数）。

### 3.4 数据集概况

| 数据集 | 材料体系 | 原子数 | 来源 |
|--------|---------|--------|------|
| a-SiO2 训练集 | 非晶 SiO2 | 300 atoms/structure | 分子动力学模拟生成 |
| Cu50Zr50 | CuZr 金属玻璃 | ~300 atoms | 分子动力学模拟生成 |
| 介孔 SiO2 | 多孔非晶 SiO2 | ~300 atoms | 分子动力学模拟生成 |

数据随 repo 提供（`demo/demo_training/simu_data/`），可直接使用。

### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ digital-synthesis-lab/DM2（MIT License），基于 LLNL/graphite |

目前仅有原始 PyTorch 实现（重度依赖 e3nn 等变网络库），无其他框架移植。e3nn 在 Paddle 生态中无直接等价库，需手动迁移核心操作。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/dm2/
│   ├── __init__.py                  # 模块导出 + 注册
│   ├── dm2.py                       # 主模型（_forward/forward/predict 三层模式）
│   ├── equivariant_gnn.py           # e3nn 等变 GNN 去噪网络
│   ├── spherical_harmonics.py       # 球谐函数 Paddle 实现
│   ├── tensor_product.py            # 等变张量积 Paddle 实现
│   └── condition_encoder.py         # 加工条件嵌入
├── ppmat/datasets/amorphous_dataset.py  # 非晶材料数据集 + build_amorphous 工厂函数
├── amorphous_generation/                 # 非晶材料生成任务
│   ├── README.md                    # 任务说明
│   └── configs/dm2/
│       ├── dm2_sio2_unconditional.yaml
│       ├── dm2_sio2_conditional.yaml
│       └── dm2_cuzr.yaml
├── examples/dm2/
│   ├── README.md
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 dm2 导入
# ppmat/datasets/__init__.py   — 新增 build_amorphous 工厂函数注册
```

### 4.2 模型实现

核心模型遵循 PaddleMaterials 统一的 `_forward()`/`forward()`/`predict()` 三层设计。

#### 数据流

```
原子坐标 pos[N, 3] + 原子类型 Z[N] + 邻居列表 edge_index
+ 可选条件 conditions（成分、温度等）
                       ↓
              ┌──────────────────────────┐
              │  训练阶段（score matching） │
              │                          │
              │  σ ~ LogUniform(σ_min, σ_max)
              │  noise ~ N(0, I)
              │  pos_noisy = pos + σ · noise
              │                          │
              │  score = Denoiser(pos_noisy, Z, edge_index, σ, cond)
              │                          │
              │  loss = MSE(score, -noise/σ)  ← denoising score matching
              └──────────────────────────┘

              ┌──────────────────────────┐
              │  推理阶段（Langevin 采样） │
              │                          │
              │  pos₀ ~ N(0, σ_max²·I)
              │  for t = T..1:
              │    score = Denoiser(pos_t, Z, ..., σ_t)
              │    Δ = (σ_t² - σ_{t-1}²) · score
              │    η = √(Δ · σ_{t-1}² / σ_t²) · N(0,I)
              │    pos_{t-1} = pos_t + Δ + η    ← Langevin step
              └──────────────────────────┘
                       ↓
              生成的非晶结构 pos_final[N, 3]
```

#### 关键设计决策

1. **等变去噪网络**：`EquivariantGNN` 保证旋转/平移等变性，输出 score 向量与输入坐标系一致
2. **条件生成**：`ConditionEncoder` 将成分/温度等标量条件编码为向量，注入 GNN 每层
3. **对数均匀噪声调度**：`σ ~ exp(Uniform(log σ_min, log σ_max))`，覆盖多尺度扰动
4. **无截断半径**：依赖邻居列表构建，截断半径由 YAML 配置控制（默认 5.0 Å）

#### 类签名

```
DM2(hidden_dim=128, n_layers=6, cutoff=5.0, lmax=2,
    sigma_min=0.01, sigma_max=10.0, n_steps=100)
  ├─ _forward(positions, atom_types, edge_index, sigma, conditions)
  │     → score: Tensor[N, 3]
  ├─ forward(positions, atom_types, edge_index, batch_index, conditions, ...)
  │     → (loss_dict, pred_dict)           # 训练入口
  └─ predict(atom_types, edge_index, n_atoms, conditions)
        → {"generated_positions": Tensor[N, 3]}
```

### 4.3 数据集适配

数据源：非晶材料结构文件（`.xyz` / `.extxyz` 格式，ASE 读取）。

每条数据包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `positions` | float32[N, 3] | 原子笛卡尔坐标 |
| `atom_types` | int64[N] | 原子序数 |

工厂函数签名：

```
build_amorphous(config) → AmorphousDataset
  config.data_dir  — 结构文件目录
  config.cutoff    — 邻居列表截断半径（默认 5.0）
  config.split     — train/val/test
```

邻居列表由 `collate_fn` 在 batch 时基于 `cutoff` 动态构建。

### 4.4 训练流程

1. **损失函数**：score matching loss（denoising score = ∞∇_x log pσ(x)）
2. **优化器**：Adam，学习率 1e-4，Exponential 衰减
3. **评估指标**：生成结构的 RDF 匹配度、配位数分布

### 4.5 YAML 配置示例

```yaml
Global:
  task: amorphous_generation
  seed: 42
  output_dir: output/dm2_sio2/

Model:
  __class_name__: DM2
  __init_params__:
    hidden_channels: 128
    n_interactions: 5
    cutoff: 5.0
    num_species: 2  # Si, O

Dataset:
  __class_name__: amorphous
  __init_params__:
    data_dir: data/dm2/a-SiO2/
    cutoff: 5.0

Optimizer:
  __class_name__: Adam
  __init_params__:
    lr: 1.0e-4
    lr_scheduler: ExponentialDecay
    gamma: 0.9999
```

### 4.6 补充说明

- e3nn 的球谐基函数计算需手动迁移（Paddle 无直接等价库），迁移范围限定在 DM2 实际使用的 l_max=2 阶数。
- 不迁移 graphite 框架中的通用图构建器，仅复现 DM2 所需的子集。

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同随机种子和输入结构，对比 PyTorch（e3nn）和 Paddle 实现的 score 输出
- 要求：score diff ≤ 1e-6（生成式模型标准）

### 5.2 反向训练对齐

- 使用相同 a-SiO2 训练数据子集，对比两个框架训练 2 轮以上的 loss 曲线
- 要求 loss 逐 epoch 一致

### 5.3 生成质量指标

| 材料体系 | 指标 | 原论文值 | 允许误差 |
|---------|------|---------|---------|
| a-SiO2 | RDF 匹配度（Wasserstein 距离） | 论文 Fig. 3 | ±5% |
| a-SiO2 | 配位数分布 | Si: 4 配位 | ±5% |
| Cu50Zr50 | RDF 匹配度 | 论文 Fig. 4 | ±5% |

### 5.4 测试项

1. **单元测试**：球谐基函数计算、等变卷积层、score 网络前向
2. **集成测试**：数据加载 → 加噪 → 去噪 → 采样全流程
3. **精度对齐截图**：同输入下 PyTorch (e3nn) vs Paddle score diff 截图
4. **生成质量评估**：RDF 计算模块、配位数统计

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **高** | e3nn 等变操作需手动迁移 |
| 自定义算子 | **无 CUDA**，但需 e3nn 等价操作 | 球谐函数、张量积需 Paddle 实现 |
| 依赖项风险 | **中** | e3nn==0.4.4, torch_geometric |
| 数据可获取性 | **高** | 训练数据随 repo 提供 |
| API 映射完整性 | **中** | e3nn 无 Paddle 直接等价 |
| 计算资源 | **中** | 无条件训练 ~20h on A6000 |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | GPU 环境配置、graphite 源码分析 | 2 天 |
| Phase 1：等变层迁移 | 球谐函数、张量积 → Paddle | 5 天 |
| Phase 2：模型迁移 | GNN 去噪网络、score matching | 3 天 |
| Phase 3：训练适配 | Trainer 适配、数据加载 | 3 天 |
| Phase 4：评估验证 | a-SiO2/CuZr 训练 + RDF 评估 | 3 天 |
| Phase 5：文档与合入 | README 编写、提交 PR | 1 天 |

**合计**：~17 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `dm2/` 目录（~800 行代码），包含 e3nn 等变 GNN 去噪网络、球谐函数和张量积的 Paddle 实现，提供非晶材料结构生成能力。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `amorphous_dataset.py`（~80 行），支持 ASE 格式的非晶材料数据加载（a-SiO₂、Cu-Zr 金属玻璃等）。
3. **新增任务类型**：新建 `amorphous_generation/` 目录，引入“非晶材料生成”任务类型，区别于现有 DiffCSP/MatterGen 的晶体生成。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| Score matching | 扩散模型训练方法，学习噪声数据的梯度方向（score function） |
| e3nn | SE(3) 等变神经网络库，基于球谐基函数和不可约表示 |
| 非晶材料 | 不具有长程有序的材料（如玻璃、金属玻璃），区别于晶体 |
| RDF | 径向分布函数（Radial Distribution Function），表征原子间距分布 |

## 附件及参考资料

1. Yang, Kai and Schwalbe-Koda, Daniel. "A generative diffusion model for amorphous materials." arXiv:2507.05024 (2025).
2. Hsu, Tim et al. "Score-based denoising for atomic structure identification." *npj Computational Materials* 10.1 (2024): 155.
3. DM2 源码：https://github.com/digital-synthesis-lab/DM2（MIT 许可证）
4. graphite 源码：https://github.com/LLNL/graphite
5. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 10 项）
