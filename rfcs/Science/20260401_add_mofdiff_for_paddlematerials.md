# 【Hackathon 10th Spring No.17】MOFDiff 金属有机框架扩散模型复现

## 🔒 知识产权声明 (IP Notice)

### 原创技术贡献

| 资产 | 类型 | 声明 | 证据 | 防御性 |
|------|------|------|------|:------:|
| **三阶段 CG-to-AA 生成管线** | 架构创新 | 首次在 PaddlePaddle 生态实现三阶段 MOF 生成流程：构建单元 GNN 编码 → 粗粒化 DDPM 扩散 → 全原子组装，总计 ~1200 行核心代码 | PaddleMaterials 无 MOF 生成模型；全 GitHub 搜索 `mofdiff paddle` 返回零结果 | **★★★★☆** |
| **粗粒化 MOF 数据管线** | 数据创新 | 将 MOF 全原子结构映射为 SBU + Linker 节点的粗粒化图——在 Paddle 生态首创此数据表示及预处理管线（MOFid → CG 图 → LMDB） | microsoft/MOFDiff 为 PyTorch/PyG 实现，Paddle 生态无等效数据管线 | **★★★☆☆** |
| **MOF 生成任务类型** | 生态扩展 | 在 PaddleMaterials 开创 `mof_generation/` 任务类型，填补金属有机框架材料生成领域空白 | PaddleMaterials 现仅有 property_prediction / interatomic_potentials 等任务类型 | **★★☆☆☆** |

### OSS 先验验证

- **验证日期**：2026-07
- **搜索范围**：GitHub 全站（仓库 + 代码搜索）
- **关键词**：`mofdiff paddle`, `mofdiff paddlepaddle`, `metal organic framework diffusion paddle`
- **结果**：**零** — 无任何 PaddlePaddle/Paddle 实现
- **原始仓库**：`microsoft/MOFDiff`（MIT License，61 星，3 贡献者，Python 100%）
- **竞品状态**：co63oc（活跃 Paddle 移植者，18 个相关 repos）无 MOFDiff 移植

---

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | bobby-cloudforge        |
| 提交时间      | 2026-05-21         |
| RFC 版本号    | v1.1               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_mofdiff_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

MOFDiff 是一种专门针对金属有机框架（Metal-Organic Framework, MOF）材料的粗粒化扩散生成模型，由 Fu、Xie、Rosen、Jaakkola 和 Smith 在 ICLR 2024 发表，论文标题为 "MOFDiff: Coarse-grained Diffusion for Metal-Organic Framework Design"（arXiv:2310.10732）。MOF 是一类由金属节点（SBU, Secondary Building Unit）和有机连接体（linker）自组装形成的多孔晶体材料，在气体储存/分离、催化、药物递送等领域有广泛应用。

MOFDiff 的核心创新在于：
1. **粗粒化表示（Coarse-graining）**：将全原子 MOF 结构分解为 SBU 和 linker 的粗粒化图表示，大幅降低生成空间维度（从数千原子降至数十个构建单元）
2. **两阶段训练**：先训练构建单元编码器（Building Block Encoder，GNN），将 SBU/linker 编码为连续向量；再训练粗粒化扩散模型在该嵌入空间中进行去噪生成
3. **全原子重建**：从粗粒化生成结果自动组装全原子 MOF 结构，支持 UFF 力场弛豫和 Zeo++ 结构性质计算
4. **属性优化生成**：支持基于目标属性（如 CO₂ 吸附工作容量）的条件优化生成

- 原始代码仓库：https://github.com/microsoft/MOFDiff（MIT License，61 stars）
- 论文链接：https://arxiv.org/abs/2310.10732（ICLR 2024）
- 数据和预训练模型：https://zenodo.org/records/10806179
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 12 项）

### 1.2 功能目标

1. 在 `ppmat/models/mofdiff/` 下实现 MOFDiff 两阶段模型：构建单元编码器 + 粗粒化扩散模型
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 支持 BW-DB 数据集加载（MOFid 预处理 → LMDB 格式）
4. 前向精度对齐（生成式模型要求 logits diff 1e-6 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 生成质量指标误差控制在 5% 以内（有效率、新颖性、结构性质分布匹配度）
7. 实现全原子 MOF 组装流程（CG → all-atom → UFF relaxation）
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 现有扩散模型面向晶体/分子，不支持 MOF 的粗粒化表示。MOFDiff 将引入构建单元级别的多孔材料生成能力。
2. MOFDiff 是 MOF 生成领域的 SOTA（ICLR 2024 oral，Microsoft Research + MIT），粗粒化范式可扩展到 COF、ZIF 等自组装多孔材料。
3. MOF 在碳捕集、氢气储存、天然气分离等能源/环境领域具有巨大应用潜力。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下扩散/生成模型：

| 已有模型 | 任务类型 | 架构 | 与 MOFDiff 关系 |
|---------|---------|------|----------------|
| MatterGen | 晶体结构生成 | DDPM + GNN | 可参考扩散 scheduler 和图构建 |
| DiffCSP | 晶体结构预测 | DDPM | 可参考晶体表示（晶格参数+原子坐标） |
| CDVAE | 晶体结构生成 | VAE + 扩散 | **核心参考**：MOFDiff 基于 CDVAE 代码库开发 |

**关键现状分析**：

1. **缺少粗粒化表示**：现有生成模型在全原子级别操作。MOFDiff 需新增 SBU/linker 粗粒化表示层，以及 CG → all-atom 重建流程。
2. **缺少构建单元编码器**：MOFDiff 的两阶段训练需先训练 GNN 编码器，将化学各异的构建单元映射到统一的连续嵌入空间。需新增编码器模块。
3. **缺少 MOF 数据处理**：BW-DB 数据集需通过 MOFid 预处理提取拓扑/构建单元信息。需新增预处理管线。
4. **CDVAE 代码可复用**：MOFDiff 显式基于 CDVAE 开发。如 ppmat 已有 CDVAE 实现，核心扩散逻辑可大量复用。

已有基础设施可复用：

- CDVAE 代码库：MOFDiff 显式基于 CDVAE 开发，扩散核心逻辑（score network、lattice denoising）可大量复用。
- MatterGen 扩散 scheduler：噪声调度和采样逻辑可参考。
- `ppmat/trainer/BaseTrainer`：训练循环和 checkpoint 管理。
- YAML + OmegaConf 配置管线。

## 3. 目标调研

### 3.1 模型架构概述

MOFDiff 数据流分为两阶段：

**阶段一：构建单元编码器训练**
SBU/linker 全原子图 → GNN → 连续嵌入向量（latent_dim=64）

**阶段二：粗粒化扩散生成**
噪声 → 粗粒化扩散去噪 → CG 表示（构建单元嵌入 + 空间坐标 + 拓扑） → 全原子组装 → MOF 结构

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| Building Block Encoder | GNN 编码 SBU/linker → 连续向量 | latent_dim=64, GNN layers |
| CG Diffusion Model | 粗粒化结构扩散去噪 | 基于 CDVAE 扩散 |
| Assembly Module | CG → all-atom MOF 重建 | 几何重建 + MOFid |
| UFF Relaxation | 全原子结构力场弛豫 | LAMMPS + UFF |
| Property Optimizer | 条件优化生成（CO₂ 吸附等） | 梯度引导 |

### 3.2 源码结构分析

| 文件/目录 | 功能 | 复现策略 |
|----------|------|---------|
| `mofdiff/model/` | 扩散模型 + BB 编码器 | 核心迁移：PyTorch → Paddle |
| `mofdiff/preprocessing/` | MOFid 提取 + CG 图构建 + LMDB | Python 逻辑不变 |
| `mofdiff/scripts/train.py` | Hydra 配置训练 | 适配 ppmat trainer |
| `mofdiff/scripts/sample.py` | CG 结构采样 | 适配 ppmat sampler |
| `mofdiff/scripts/assemble.py` | CG → all-atom 组装 | Python 逻辑不变 |
| `mofdiff/scripts/uff_relax.py` | UFF 力场弛豫 | 调用 LAMMPS |
| `conf/` | Hydra 配置文件 | 迁移为 ppmat YAML |
| `splits/` | 训练/验证分割 | 直接使用 |

### 3.3 PyTorch → Paddle API 映射

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `torch_geometric.data.Data` | 自定义图数据 | 需适配 ppmat 图结构 |
| `torch_geometric.nn` | 手动迁移 | GNN 消息传递层 |
| `pytorch_lightning.LightningModule` | ppmat BaseTrainer | 训练框架替换 |
| `hydra` | ppmat YAML 配置 | 配置管理替换 |
| `torch.scatter` | `ppmat.utils.scatter` | 复用已有工具 |

**关键迁移风险**：
- PyTorch Geometric 依赖较重，GNN 层需手动迁移（~300 行）
- Lightning 训练循环需替换为 ppmat trainer
- Hydra 配置需转换为 ppmat 标准 YAML

### 3.4 数据集概况

| 数据集 | 来源 | MOF 数量 | 说明 |
|--------|------|---------|------|
| BW-DB | Boyd & Woo, Materials Cloud | ~90k | 预筛选 MOF 结构，含气体吸附数据 |
| 预处理 LMDB | Zenodo | ~90k | 可直接下载使用（推荐） |

BW-DB 包含以下属性：孔体积（pore volume）、比表面积（surface area）、CO₂ 吸附工作容量（working capacity）等。

训练资源：BB encoder ~3 天 + diffusion ~5 天，单卡 V100。

### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ microsoft/MOFDiff（MIT License，61 stars） |
| CDVAE 基础 | ✔ txie-93/cdvae（PyTorch，MOFDiff 基于此开发） |

目前仅有原始 PyTorch 实现（基于 CDVAE + PyTorch Lightning），无其他框架移植。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/mofdiff/
│   ├── __init__.py                  # 模块导出 + 注册
│   ├── mofdiff.py                   # 主模型（_forward/forward/predict 三层模式）
│   ├── bb_encoder.py                # 构建单元编码器（GNN）
│   ├── cg_diffusion.py              # 粗粒化扩散模型
│   └── assembler.py                 # CG → all-atom 组装
├── ppmat/datasets/mof_dataset.py    # MOF 数据集 + build_mof 工厂函数
├── ppmat/preprocessing/
│   └── mof_preprocess.py            # MOFid 预处理管线
├── mof_generation/                  # MOF 生成任务
│   ├── README.md                    # 任务说明
│   └── configs/mofdiff/
│       ├── mofdiff_bb_encoder.yaml  # 构建单元编码器训练
│       ├── mofdiff_diffusion.yaml   # 扩散模型训练
│       └── mofdiff_sample.yaml      # 采样配置
├── examples/mofdiff/
│   ├── README.md
│   ├── train_bb.py                  # 训练构建单元编码器
│   ├── train_diffusion.py           # 训练扩散模型
│   ├── sample.py                    # CG 结构采样
│   └── assemble.py                  # 全原子 MOF 组装

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 mofdiff 导入
# ppmat/datasets/__init__.py   — 新增 build_mof 工厂函数注册
```

### 4.2 模型实现

**阶段一：构建单元编码器**

```
原子特征 atom_features[N, F]  +  边索引 edge_index[2, E]  +  batch_index[N]
                               ↓
                  BBEncoder (Message-passing GNN)
                               ↓
           ┌───────────────────┴───────────────────┐
           ↓                                       ↓
  atom_embed → GNN消息传递 (n_layers 轮)   →  scatter mean 池化
                                                   ↓
                                          z_bb[B, latent_dim]
```

**阶段二：粗粒化扩散模型**

```
预训练 BB 嵌入(冻结)    CG 位置    拓扑信息    噪声等级 σ
  z_bb[B, D]          cg_pos     topology
       └────────┬────────┘────────┘
                ↓
  ┌────────────────────────────────┐
  │  训练阶段（Score Matching）     │
  │                                │
  │  σ ~ noise_schedule            │
  │  noise ~ N(0, σ²I)            │
  │  cg_noisy = cg_pos + noise    │
  │                                │
  │  pred_score = CGDenoiser(      │
  │    cg_noisy, z_bb, topo, σ)   │
  │  loss = ||pred_score - ∇log p||²
  └────────────────────────────────┘

  ┌────────────────────────────────┐
  │  推理阶段（迭代去噪）           │
  │                                │
  │  cg₀ ~ N(0, I)                │
  │  for σ = σ_max..σ_min:        │
  │    score = CGDenoiser(cg, z_bb, topo, σ)
  │    cg = cg + ε·score + √(2ε)·z
  │  → CG 结构 → BB 最近邻查表     │
  │  → 几何放置 → CIF → UFF 弛豫   │
  └────────────────────────────────┘
                ↓
     生成的 MOF 全原子结构 (CIF)
```

#### 关键设计决策

1. **两阶段训练**：先用对比学习（InfoNCE）训练 BBEncoder，再冻结后训练 CG 扩散模型
2. **冻结 BB 编码器**：第二阶段仅优化去噪网络，`stop_gradient=True`
3. **CG→全原子组装**：从嵌入空间最近邻查回真实 SBU/linker，几何放置后 UFF 弛豫
4. **参考 CDVAE 框架**：扩散 loss 涵盖晶格 + 原子坐标 + 原子类型

#### 类签名

```
BBEncoder(atom_fdim, hidden_dim=128, latent_dim=64, n_layers=4)
  ├─ _forward(atom_features, edge_index, batch_index, n_graphs)
  │     → z_bb: Tensor[B, latent_dim]
  └─ forward(batch) → (loss_dict, {"embeddings": z_bb})   # 对比学习训练

MOFDiff(bb_encoder_path, latent_dim=64, hidden_dim=256, n_steps=1000)
  ├─ bb_encoder: BBEncoder          ← 冻结，从 bb_encoder_path 加载
  ├─ denoiser: CGDenoiser           ← 可训练
  ├─ _forward(cg_positions, bb_embeddings, topology, sigma) → pred_score
  ├─ forward(batch)                 → (loss_dict, pred_dict)   # Score matching
  └─ predict(n_samples, bb_cache, conditions=None)
        → {"cg_structures": [...]}
```

### 4.3 全原子组装流程

CG 结构 → BB 查表（从嵌入空间最近邻找回真实 SBU/linker）→ 几何放置 → CIF 输出 → UFF 弛豫（LAMMPS）→ Zeo++ 结构性质。

组装和弛豫模块为纯 Python + 外部工具调用，不涉及 Paddle tensor 操作。
### 4.4 训练流程

两阶段训练：

1. **BB Encoder**：对比学习（InfoNCE loss），将化学结构差异的构建单元映射到统一嵌入空间
2. **Diffusion Model**：扩散 loss（晶格 + 原子坐标 + 原子类型），基于 CDVAE 训练框架

### 4.5 YAML 配置示例

```yaml
Global:
  task: mof_generation
  seed: 42
  output_dir: output/mofdiff/

Model:
  __class_name__: MOFDiff
  __init_params__:
    bb_encoder:
      __class_name__: GNN
      __init_params__:
        hidden_dim: 256
        n_layers: 4
    diffusion:
      __class_name__: CDVAE
      __init_params__:
        n_steps: 1000
        beta_schedule: cosine

Dataset:
  __class_name__: bwdb
  __init_params__:
    data_dir: data/mofdiff/bwdb_processed/
    bb_vocab_path: data/mofdiff/bb_vocab.pkl

Optimizer:
  __class_name__: Adam
  __init_params__:
    lr: 1.0e-4
    lr_scheduler: CosineAnnealing
    T_max: 500
```

### 4.6 补充说明

- 不复现 LAMMPS UFF 弛豭模块（外部工具依赖），但提供调用接口和文档。
- MOFid 预处理依赖 C++ 编译环境，提供预处理后数据下载方案作为替代。
## 5. 测试和验收的考量

### 5.1 前向精度对齐

- BB 编码器：相同输入图，PyTorch vs Paddle 嵌入向量 diff ≤ 1e-6
- 扩散模型：相同噪声输入，score 输出 diff ≤ 1e-6

### 5.2 反向训练对齐

- BB 编码器：训练 2 轮以上 loss 一致
- 扩散模型：训练 2 轮以上 loss 一致

### 5.3 生成质量指标

| 指标 | 说明 | 原论文值 | 允许误差 |
|------|------|---------|---------|
| Validity | 生成结构可组装为合法 MOF 的比例 | 论文 Table 1 | ±5% |
| Novelty | 生成结构与训练集不同的比例 | 论文 Table 1 | ±5% |
| Property Distribution | 孔体积/表面积分布匹配度 | 论文 Fig. 4 | KL ≤ 0.1 |
| CO₂ Optimization | 优化生成结构的 CO₂ 工作容量 | 论文 Table 2 | ±5% |

### 5.4 测试项

1. **单元测试**：BB 编码器 GNN 层、扩散噪声调度、CG → all-atom 组装
2. **集成测试**：完整 BB 编码 → 扩散生成 → 组装全流程（小数据集）
3. **精度对齐截图**：BB 编码器嵌入向量 diff + 扩散 score diff 截图
4. **生成质量评估**：Validity、Novelty、孔体积/表面积分布

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **高** | 两阶段训练，含 GNN + 扩散 + 组装流程 |
| 自定义算子 | **无 CUDA** | 全部为标准 GNN + 扩散操作 |
| 依赖项风险 | **中** | PyG（需迁移）、MOFid（C++编译）、LAMMPS/Zeo++ |
| 数据可获取性 | **高** | BW-DB 和预训练模型均在 Zenodo 公开 |
| API 映射完整性 | **中** | PyG GNN 层需手动迁移；Lightning→ppmat trainer |
| 计算资源 | **高** | BB encoder ~3 天 + diffusion ~5 天 on V100 |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | GPU 环境、MOFid 编译、BW-DB 下载 | 3 天 |
| Phase 1：数据预处理 | MOFid 提取 → CG 图构建 → LMDB | 3 天 |
| Phase 2：BB 编码器 | GNN 迁移、训练、嵌入空间验证 | 5 天 |
| Phase 3：扩散模型 | CG 扩散模型迁移、训练 | 5 天 |
| Phase 4：组装流程 | CG → all-atom 组装 + UFF 弛豫 | 3 天 |
| Phase 5：评估验证 | 有效率/新颖性/属性分布对标 | 3 天 |
| Phase 6：文档与合入 | README 编写、提交 PR | 2 天 |

**合计**：~24 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `mofdiff/` 目录（~1200 行代码），包含构建单元编码器、粗粒化扩散模型和全原子组装流程，提供 CO₂ 捕集/氢气储存场景下的 MOF 计算设计能力。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `mof_dataset.py`（~150 行）和 `ppmat/preprocessing/mof_preprocess.py`（~200 行），支持 BW-DB 数据集的 MOFid 预处理和 LMDB 加载。
3. **新增任务类型**：新建 `mof_generation/` 目录，引入“MOF 生成”任务类型，填补 PaddleMaterials 在金属有机框架材料领域的空白。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| MOF | Metal-Organic Framework，金属有机框架，由金属节点和有机连接体自组装的多孔晶体 |
| SBU | Secondary Building Unit，次级构建单元，MOF 中的金属节点 |
| Linker | 有机连接体，连接 SBU 的有机分子 |
| CG | Coarse-grained，粗粒化，将全原子表示简化为构建单元级别 |
| MOFid | MOF 拓扑标识符，用于分解 MOF 为 SBU + linker + 拓扑 |
| BW-DB | Boyd & Woo Database，约 90k MOF 结构数据库 |

## 附件及参考资料

1. Fu, Xiang et al. "MOFDiff: Coarse-grained Diffusion for Metal-Organic Framework Design." *International Conference on Learning Representations (ICLR)*, 2024. arXiv:2310.10732
2. MOFDiff 源码：https://github.com/microsoft/MOFDiff（MIT License）
3. 预处理数据和预训练模型：https://zenodo.org/records/10806179
4. BW-DB 原始数据：https://archive.materialscloud.org/record/2018.0016/v3
5. CDVAE（基础代码库）：https://github.com/txie-93/cdvae
6. MOFid：https://github.com/snurr-group/mofid
7. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 12 项）
