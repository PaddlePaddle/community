# 【Hackathon 10th Spring No.11】TrinityLLM 模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | danqing-cfg        |
| 提交时间      | 2026-04-07         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260407_add_trinityllm_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

TrinityLLM 是一种面向高分子材料（polymer）性能预测的大语言模型框架，由 Liu 等人在 *Nature Computational Science* 5.3 (2025): 245-254 发表，论文标题为 "Harnessing large language models for data-scarce learning of polymer properties"。该框架基于 MoLFormer（IBM 开发的分子语言模型），通过两阶段训练策略解决材料实验数据稀缺问题。

TrinityLLM 的核心创新在于：
1. **两阶段训练策略**：第一阶段使用物理模型生成的大量合成数据进行监督预训练（supervised pretraining），第二阶段使用少量实验数据进行微调（finetuning），解决实验数据稀缺问题
2. **物理-数据融合**：将物理建模（如 FDS 火灾动力学模拟）生成的合成数据作为 LLM 的预训练信号，使模型在微调前已具备物理一致性
3. **SMILES 分子表征**：使用 SMILES 字符串表示分子结构，通过 MoLFormer 的线性注意力机制编码分子特征
4. **数据稀缺场景有效性**：在锥形量热仪（cone calorimeter）等实验数据极度稀缺的高分子阻燃性预测任务中表现优异

- 原始代码仓库：https://github.com/ningliu-iga/TrinityLLM
- 论文链接：https://www.nature.com/articles/s43588-025-00768-y
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 6 项）

### 1.2 功能目标

1. 在 `ppmat/models/trinityllm/` 下实现 TrinityLLM 模型，包含 MoLFormer Backbone（线性注意力编码器）、SMILES tokenizer、两阶段训练流程
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 支持 SMILES 分子表示的数据加载（ZINC/PubChem 预训练 + 高分子属性微调）
4. 前向精度对齐（生成式模型要求 logits diff 1e-6 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 采样指标误差控制在 5% 以内（生成式模型标准）
7. 覆盖原论文核心任务：高分子阻燃性预测（pHRR、THR、burning rate 等指标）
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 以无机晶体/小分子为主，尚无高分子性能预测模型。TrinityLLM 将填补这一空白，同时引入数据稀缺场景的两阶段训练范式。
2. TrinityLLM 发表在 *Nature Computational Science*，是材料信息学与 LLM 交叉的代表性工作，为后续材料领域 LLM 提供标准实践。
3. 与 CrystalLLM 形成互补：CrystalLLM 关注晶体结构生成（CIF token），TrinityLLM 关注分子属性预测（SMILES token）。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下模型，涵盖属性预测、势函数、结构生成等任务类型：

| 已有模型 | 任务类型 | 架构 | 与 TrinityLLM 关系 |
|---------|---------|------|-------------------|
| CrystalLLM | 晶体结构生成 | GPT-2（CIF token） | **最相近**，同为 LLM 架构，可参考 tokenizer/trainer 设计 |
| SchNet | 小分子属性预测 | GNN | 不同架构，但属性预测任务可参考评估指标 |
| DimeNet++ | 小分子属性预测 | GNN | 无直接复用 |
| MEGNet | 属性预测 | GNN | 无直接复用 |
| MatterSim | 原子间势函数 | GNN | 可参考 trainer 接口设计 |

**关键现状分析**：

1. **缺少 SMILES 表征支持**：CrystalLLM 使用 CIF（晶体信息文件）作为输入，TrinityLLM 使用 SMILES 字符串表示分子。需新增 SMILES tokenizer。
2. **缺少 MoLFormer 架构**：TrinityLLM 基于 MoLFormer（线性注意力 Transformer），而非 GPT-2 或标准 Transformer。需新增模型实现。
3. **缺少两阶段训练**：现有 trainer 支持单阶段训练。TrinityLLM 的两阶段（合成数据预训练→实验数据微调）需适配。
4. **高分子任务空白**：PaddleMaterials 尚无高分子材料相关模型/数据集。

已有基础设施可复用：

- CrystalLLM 的 tokenizer/trainer 设计模式：LLM 训练 pipeline 架构可参考。
- `ppmat/trainer/BaseTrainer`：训练循环可适配两阶段模式（先 freeze encoder 再 fine-tune）。
- `ppmat/optimizer/build_optimizer`：统一优化器和 cosine annealing 调度。
- YAML + OmegaConf 配置体系。

## 3. 目标调研

### 3.1 模型架构概述

TrinityLLM 基于 MoLFormer（IBM Research，2022），采用 Transformer Encoder + 线性注意力机制处理 SMILES 序列：

**数据流**：SMILES 字符串 → Tokenizer → MoLFormer Encoder（线性注意力） → 分子表征向量 → 属性预测头 → 标量输出

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| SMILES Tokenizer | 将 SMILES 字符串转为 token ID | 最大长度 211，vocab 从 ZINC+PubChem 构建 |
| MoLFormer Encoder | 线性注意力 Transformer 编码器 | 12 层，768 维 hidden，12 heads |
| Prediction Head | 属性回归预测 | Linear(768, 1) |

**两阶段训练流水线**：

| 阶段 | 数据来源 | 数据量 | 目标 |
|------|---------|--------|------|
| Phase 1：合成数据预训练 | FDS/物理模型生成 | ~10,000+ 样本 | 物理一致性对齐 |
| Phase 2：实验数据微调 | 锥形量热仪实验 | ~100-500 样本 | 精度提升 |

### 3.2 源码结构分析

原始 PyTorch 实现（https://github.com/ningliu-iga/TrinityLLM）基于 MoLFormer：

| 文件/目录 | 功能 | 复现策略 |
|----------|------|---------|
| `training/` | MoLFormer 预训练代码 | 迁移到 Paddle，16×V100 预训练 |
| `finetune/` | 属性预测微调代码 | 迁移 + 适配 ppmat trainer |
| `notebooks/` | Jupyter 推理/可视化 | 迁移为 examples/ |
| `data/` | 数据组织约定 | 适配 ppmat datasets |

**依赖关系**：
- MoLFormer 预训练权重：从 IBM Box 下载（https://ibm.box.com/v/MoLFormer-data）
- NVIDIA Apex：用于混合精度训练的优化器，需替换为 Paddle AMP
- PyTorch Lightning：训练框架，需替换为 ppmat trainer

### 3.3 PyTorch → Paddle API 映射

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `nn.TransformerEncoder` | `paddle.nn.TransformerEncoder` | 一致 |
| `nn.Linear` | `paddle.nn.Linear` | 一致 |
| `nn.LayerNorm` | `paddle.nn.LayerNorm` | 一致 |
| `apex.optimizers.FusedAdam` | `paddle.optimizer.Adam` | Apex 不可用，替换为标准 Adam |
| `pytorch_lightning.LightningModule` | ppmat trainer 模式 | 框架替换 |
| `torch.cuda.amp` | `paddle.amp` | 混合精度训练 |

**关键迁移风险**：
- MoLFormer 使用自定义线性注意力（`fast_transformers.feature_maps`），需确认 Paddle 等价实现或手动迁移
- Apex FusedAdam 需替换为标准 Adam，可能影响收敛速度

### 3.4 数据集概况

| 数据集 | 用途 | 规模 | 获取方式 |
|--------|------|------|---------|
| ZINC + PubChem | MoLFormer 预训练 | ~1.1B 分子 | IBM Box 下载 |
| FDS 合成数据 | Phase 1 监督预训练 | ~10,000 样本 | 论文/作者提供 |
| Cone calorimeter | Phase 2 实验微调 | ~100-500 样本 | 论文附件提供 |
| MoLFormer 预训练权重 | 迁移学习起点 | N/A | IBM Box 下载 |

**数据策略**：优先使用 MoLFormer 预训练权重（跳过 1.1B 分子预训练），仅复现两阶段微调流程。预训练权重转换：PyTorch checkpoint → Paddle state_dict。

### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ ningliu-iga/TrinityLLM（MIT License） |
| MoLFormer 基础 | ✔ IBM/molformer（PyTorch，预训练权重可下载） |

目前仅有原始 PyTorch 实现（基于 MoLFormer + PyTorch Lightning），无其他框架移植。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/trinityllm/
│   ├── __init__.py                  # 模块导出 + 注册
│   ├── molformer.py                 # MoLFormer 编码器（线性注意力 Transformer）
│   ├── trinityllm.py                # TrinityLLM 主模型（MoLFormer + 预测头）
│   ├── smiles_tokenizer.py          # SMILES tokenizer
│   └── weight_converter.py          # PyTorch → Paddle 权重转换工具
├── ppmat/datasets/polymer_dataset.py # 高分子属性数据集 + build_polymer 工厂函数
├── polymer_property_prediction/      # 高分子属性预测任务
│   ├── README.md                    # 任务说明
│   └── configs/trinityllm/
│       ├── trinityllm_pretrain.yaml # Phase 1：合成数据预训练
│       └── trinityllm_finetune.yaml # Phase 2：实验数据微调
├── examples/trinityllm/
│   ├── README.md                    # 使用说明（含结果链接）
│   ├── train.py                     # 训练入口
│   ├── predict.py                   # 预测入口
│   └── convert_weights.py           # PyTorch → Paddle 权重转换

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 trinityllm 导入
# ppmat/datasets/__init__.py   — 新增 build_polymer 工厂函数注册
```

### 4.2 模型实现

核心模型遵循 PaddleMaterials 统一的 `_forward()`/`forward()`/`predict()` 三层设计。

#### 数据流

```
SMILES 字符串
          ↓
    SmilesTokenizer
          ↓
  input_ids[B, L] + attention_mask[B, L]
          ↓
  ┌─────────────────────────────┐
  │    MoLFormerEncoder          │
  │                             │
  │  token_embed + pos_embed    │
  │          ↓                  │
  │  TransformerEncoderLayer    │ × 12 层
  │  (线性注意力, GELU FFN)      │
  │          ↓                  │
  │  LayerNorm → [CLS] 池化     │
  └─────────────────────────────┘
          ↓
    mol_repr[B, D]
          ↓
  ┌─────────────────────────────┐
  │  prediction_head (MLP)      │
  │  D → D/2 → SiLU → n_tasks  │
  └─────────────────────────────┘
          ↓
    属性预测值[B, n_tasks]
```

#### 关键设计决策

1. **两阶段训练**：Phase 1 用合成数据监督预训练，Phase 2 用少量实验数据微调——核心创新在于合成数据的利用
2. **MoLFormer 编码器**：基于 Transformer 的 SMILES 序列编码（非 GNN），复用 MoLFormer 预训练权重
3. **[CLS] 池化**：取序列第一个 token 的隐状态作为分子整体表征

#### 类签名

```
TrinityLLM(vocab_size=2362, hidden_dim=768, n_layers=12,
           n_heads=12, max_seq_len=211, n_tasks=1)
  ├─ _forward(input_ids, attention_mask) → predictions: Tensor[B, n_tasks]
  ├─ forward(input_ids, attention_mask, targets)
  │     → (loss_dict, pred_dict)       # 训练入口
  └─ predict(input_ids, attention_mask)
        → {"predictions": Tensor}

MoLFormerEncoder(vocab_size, hidden_dim, n_layers, n_heads, max_seq_len)
  └─ forward(input_ids, attention_mask) → mol_repr: Tensor[B, D]
```

### 4.3 数据集适配

数据格式：CSV，每行一条高分子记录。通过 `SmilesTokenizer` 将 SMILES 转为 token 序列。

| 字段 | 类型 | 说明 |
|------|------|------|
| `smiles` | str | 分子 SMILES 字符串 |
| `target` | float | 回归目标值 |

工厂函数签名：

```
build_polymer(config) → PolymerDataset
  config.data_path   — CSV 文件路径
  config.vocab_file  — 词表文件路径
  config.max_length  — 最大序列长度（默认 211）
```

`SmilesTokenizer` 基于自定义词表将 SMILES 编码为 `input_ids` + `attention_mask`。

### 4.4 两阶段训练适配

```yaml
# Phase 1: 合成数据监督预训练
phase1:
  Model:
    __class_name__: TrinityLLM
    __init_params__:
      pretrained_weights: data/molformer_pretrained.pdparams  # 转换后的 MoLFormer 权重
  Dataset:
    __class_name__: polymer
    __init_params__:
      data_path: data/trinityllm/synthetic_train.csv
  Trainer:
    max_epochs: 100
    batch_size: 32
    learning_rate: 1.0e-4
    save_path: checkpoints/phase1/

# Phase 2: 实验数据微调
phase2:
  Model:
    __class_name__: TrinityLLM
    __init_params__:
      pretrained_weights: checkpoints/phase1/best_model.pdparams  # Phase 1 输出
  Dataset:
    __class_name__: polymer
    __init_params__:
      data_path: data/trinityllm/experimental_train.csv
  Trainer:
    max_epochs: 200
    batch_size: 16
    learning_rate: 1.0e-5  # 低学习率微调
    save_path: checkpoints/phase2/
```

### 4.5 补充说明

- 跳过 MoLFormer 1.1B 分子预训练（资源需求 16×V100），直接使用 IBM/molformer 开源权重转换。
- Apex FusedLayerNorm 替换为 Paddle 原生 `paddle.nn.LayerNorm`，精度影响可忽略。
- 不复现 MoleculeNet 基准测试（仅复现论文中的高分子阻燃性预测任务）。

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用 MoLFormer 预训练权重（PyTorch → Paddle 转换后），对比输出 logits
- 要求：logits diff ≤ 1e-6（生成式模型标准）

### 5.2 反向训练对齐

- 使用相同合成数据子集，对比两个框架训练 2 轮以上的 loss 曲线
- 要求 loss 逐 epoch 一致

### 5.3 任务指标

在高分子阻燃性预测任务上，对比以下指标：

| 属性 | 指标 | 原论文值 | 允许误差 |
|------|------|---------|---------|
| pHRR (kW/m²) | R²/MAE | 论文 Fig. 3 | ±5% |
| THR (MJ/m²) | R²/MAE | 论文 Fig. 3 | ±5% |
| Burning Rate | R²/MAE | 论文 Fig. 3 | ±5% |

### 5.4 测试项

1. **单元测试**：SMILES tokenizer、线性注意力层、两阶段训练权重加载
2. **集成测试**：Phase 1 预训练 → Phase 2 微调全流程（小数据集）
3. **精度对齐截图**：MoLFormer 权重转换后的 logits diff 截图

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **中** | MoLFormer 为标准 Transformer Encoder，但线性注意力需手动迁移 |
| 自定义算子 | **无** | 全部为标准 NN 操作 |
| 依赖项风险 | **中** | Apex 需替换；MoLFormer 权重需格式转换 |
| 数据可获取性 | **中** | 预训练权重可下载；实验数据需从论文附件获取 |
| API 映射完整性 | **高** | PyTorch Lightning → ppmat trainer 需适配 |
| 计算资源 | **高风险** | 预训练需 16×V100（跳过预训练，直接使用权重转换缓解） |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | V100 环境配置、MoLFormer 权重下载与转换 | 3 天 |
| Phase 1：模型迁移 | MoLFormer Encoder → Paddle，SMILES tokenizer | 5 天 |
| Phase 2：训练适配 | 两阶段 trainer 适配、数据加载 | 4 天 |
| Phase 3：数据准备 | 合成数据/实验数据处理、build 工厂函数注册 | 3 天 |
| Phase 4：评估验证 | 前向对齐、两阶段训练、指标对齐 | 3 天 |
| Phase 5：文档与合入 | README 编写、提交 PR | 2 天 |

**合计**：~20 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `trinityllm/` 目录（~500 行代码），包含 MoLFormer 编码器、SMILES tokenizer 和两阶段训练流程，提供高分子阻燃性预测能力。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `polymer_dataset.py`（~80 行），支持 SMILES 分子表示和两阶段数据加载（合成数据 + 实验数据）。
3. **新增任务类型**：扩展 `polymer_property_prediction/` 目录，引入数据稀缺场景的两阶段训练范式，与 CrystalLLM（CIF token）形成互补。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| MoLFormer | IBM 开发的分子语言模型，基于线性注意力 Transformer，在 1.1B 分子上预训练 |
| SMILES | 简化分子线性输入表示法（Simplified Molecular Input Line Entry System） |
| 两阶段训练 | Phase 1：合成数据预训练获取物理一致性；Phase 2：实验数据微调提升精度 |
| pHRR | 峰值热释放率（Peak Heat Release Rate），高分子阻燃性关键指标 |
| 锥形量热仪 | 测量材料燃烧热释放的标准设备，实验数据稀缺 |

## 附件及参考资料

1. Liu, Ning et al. "Harnessing large language models for data-scarce learning of polymer properties." *Nature Computational Science* 5.3 (2025): 245-254. https://www.nature.com/articles/s43588-025-00768-y
2. TrinityLLM 源码：https://github.com/ningliu-iga/TrinityLLM（MIT 许可证）
3. MoLFormer 源码：https://github.com/IBM/molformer
4. MoLFormer 预训练权重：https://ibm.box.com/v/MoLFormer-data
5. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 6 项）
