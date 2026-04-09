# 【Hackathon 10th Spring No.12】AlloyGAN 模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | cloudforge1        |
| 提交时间      | 2026-03-23         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260323_add_alloygan_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

AlloyGAN 是一种基于大语言模型（LLM）辅助的条件生成对抗网络（Conditional GAN），用于合金成分的逆向设计。该模型由 Haoyun Xu 等人在论文 *"LLM-Assisted Conditional Generative Adversarial Network for Inverse Design of Alloys"*（arXiv:2502.18127, 2025）中提出。

传统合金设计依赖实验试错法，成本高、周期长。AlloyGAN 通过 CGAN 架构，以目标热力学性质（如熔化温度、密度、弹性模量等 26 项属性）作为条件输入，生成满足条件的合金成分（40 维成分向量），实现从"性质→成分"的逆向映射。

- 原始代码仓库：https://github.com/photon-git/AlloyGAN
- 论文链接：https://arxiv.org/abs/2502.18127
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 7 项）

### 1.2 功能目标

1. 将 AlloyGAN 中的 GAN 和 CGAN 模型从 PyTorch 迁移至 PaddlePaddle，集成到 PaddleMaterials 的 `ppmat/models/` 体系中。
2. 实现合金表格数据（Alloy_train.csv）的数据加载模块，接入 `ppmat/datasets/` 的工厂函数体系。
3. 基于 `ppmat/trainer/BaseTrainer` 实现训练流程，支持 YAML 配置驱动。
4. 复现论文中 GAN 和 CGAN 的训练结果，生成合金成分与原始论文精度基本一致。
5. 提供推理脚本，支持给定目标性质条件生成合金成分。

### 1.3 意义

1. PaddleMaterials 目前尚无生成式模型（GAN 类），AlloyGAN 将填补这一空白，为后续材料逆向设计类模型的集成（如 DM2、DiffSyn 等扩散模型）提供参考范式。
2. AlloyGAN 架构简洁（全连接网络），可作为表格数据生成任务的基线模型。
3. 扩展了 PaddleMaterials 对无机材料和合金设计领域的支持。

## 2. PaddleScience 现状

PaddleMaterials 目前已集成以下类型模型：

| 任务类型 | 已有模型 |
|---------|---------|
| 属性预测 | CHGNet, ComFormer, DimeNet++, MEGNet, InfGCN, MatENO |
| 结构生成 | DiffCSP, MatterGen |
| 谱图解析 | DiffNMR |
| 分子动力学 | MatterSim |

**尚无 GAN 类生成模型**。现有模型均围绕晶体/分子结构数据（使用图数据结构），AlloyGAN 使用**表格数据**（CSV 格式的合金成分与性质），与已有模型的数据范式不同。

已有基础设施可复用：
- `ppmat/trainer/BaseTrainer`：统一训练循环，支持 AMP、分布式训练、断点续训、VisualDL/TensorBoard 日志。
- `ppmat/losses/`：BCELoss 可直接使用 `paddle.nn.BCELoss`。
- `ppmat/optimizer/build_optimizer`：统一优化器构建。
- YAML + OmegaConf 配置体系。

## 3. 目标调研

### 3.1 论文概述

AlloyGAN 提出了一种 LLM 辅助的 CGAN 框架用于合金逆向设计：

1. **数据收集**：使用 LLM（GPT-4）从材料科学文献中自动提取合金成分和性质数据，构建结构化数据集。
2. **模型训练**：使用 CGAN 学习"性质→成分"的逆映射关系。
3. **合金生成**：给定目标性质向量，生成满足条件的合金成分。

### 3.2 模型架构

#### GAN（基线模型）

| 组件 | 架构 |
|------|------|
| Generator | Linear(100, 512) → LeakyReLU(0.2) → Linear(512, 40) → Sigmoid |
| Discriminator | Linear(40, 1024) → LeakyReLU(0.2) → Linear(1024, 1) → Sigmoid |
| 损失函数 | BCELoss |
| 优化器 | Adam(lr=0.0005, weight_decay=0.0001) |

输入噪声维度为 100，输出为 40 维合金成分向量（归一化到 [0,1]）。

#### CGAN（核心模型）

| 组件 | 架构 |
|------|------|
| Generator | Linear(5+26, 512) → LeakyReLU(0.2) → Linear(512, 40) → Sigmoid |
| Discriminator | Linear(40+26, 1024) → LeakyReLU(0.2) → Linear(1024, 1) → Sigmoid |
| 损失函数 | BCELoss |
| 优化器 | Adam(lr=0.0002, weight_decay=0.00001) |

- 生成器输入：5 维随机噪声 + 26 维条件向量（目标性质） = 31 维
- 判别器输入：40 维成分向量 + 26 维条件向量 = 66 维
- 输出：40 维合金成分向量

### 3.3 数据格式

数据集为 CSV 表格格式，每行代表一种合金样本：
- 前 40 列：合金成分（元素含量百分比）
- 后 26 列：合金性质（熔化温度、密度、弹性模量等热力学/力学性质）
- "source" 列：数据来源标记（训练时丢弃）

### 3.4 torch→Paddle 迁移分析

AlloyGAN 使用的 PyTorch API 均有 PaddlePaddle 的直接对应：

| PyTorch | PaddlePaddle | 说明 |
|---------|-------------|------|
| `nn.Linear` | `paddle.nn.Linear` | 全连接层 |
| `nn.LeakyReLU(0.2)` | `paddle.nn.LeakyReLU(0.2)` | 激活函数 |
| `nn.Sigmoid` | `paddle.nn.Sigmoid` | 输出激活 |
| `nn.Sequential` | `paddle.nn.Sequential` | 模型组合 |
| `nn.BCELoss` | `paddle.nn.BCELoss` | 损失函数 |
| `optim.Adam` | `paddle.optimizer.Adam` | 优化器 |
| `torch.randn` | `paddle.randn` | 随机噪声 |
| `torch.ones/zeros` | `paddle.ones/zeros` | 标签构造 |
| `DataLoader` | `paddle.io.DataLoader` | 数据加载 |

**迁移复杂度：低**。无自定义 CUDA 算子、无图神经网络依赖、无第三方深度学习库依赖。纯全连接网络，预计可实现逐行一一对应转换。

### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|---------|
| PaddlePaddle / PaddleMaterials | ❌ 无 |
| MindSpore | ❌ 无 |
| PyTorch（原始） | ✅ photon-git/AlloyGAN |

目前仅有原始 PyTorch 实现，无其他框架移植。

## 4. 设计思路与实现方案

### 4.1 代码结构

遵循 PaddleMaterials 既有规范，新增文件如下：

```
PaddleMaterials/
├── ppmat/
│   ├── models/
│   │   └── alloygan/
│   │       ├── __init__.py
│   │       └── alloygan.py          # GAN / CGAN 模型定义
│   └── datasets/
│       └── alloy_dataset.py         # AlloyDataset 数据集类
├── alloy_design/                    # 新任务目录（无机材料工作流）
│   ├── README.md                    # 使用说明、复现结果
│   ├── train.py                     # 训练入口
│   ├── generate.py                  # 推理/生成脚本
│   └── configs/
│       └── alloygan/
│           ├── gan_alloy_train.yaml
│           └── cgan_alloy_train.yaml
└── test/
    └── test_alloygan.py             # 单元测试
```

### 4.2 模型实现

模型注册到 `ppmat/models/__init__.py` 的工厂函数体系中。

```python
# ppmat/models/alloygan/alloygan.py
import paddle
import paddle.nn as nn

class Generator(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Layer):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class AlloyGAN(nn.Layer):
    """GAN for alloy composition generation."""
    def __init__(self, noise_dim=100, hidden_dim=512, comp_dim=40):
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim, hidden_dim, comp_dim)
        self.discriminator = Discriminator(comp_dim, 1024)

class AlloyCGAN(nn.Layer):
    """Conditional GAN for inverse alloy design."""
    def __init__(self, noise_dim=5, cond_dim=26, hidden_dim=512, comp_dim=40):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.generator = Generator(noise_dim + cond_dim, hidden_dim, comp_dim)
        self.discriminator = Discriminator(comp_dim + cond_dim, 1024)
```

### 4.3 数据集模块

AlloyGAN 使用表格数据而非图结构数据，需新增 `alloy_dataset.py`：

```python
# ppmat/datasets/alloy_dataset.py
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset

class AlloyDataset(Dataset):
    def __init__(self, csv_path, comp_dim=40):
        df = pd.read_csv(csv_path)
        if "source" in df.columns:
            df = df.drop(columns=["source"])
        data = df.values.astype("float32")
        self.compositions = data[:, :comp_dim]      # 前40列：成分
        self.properties = data[:, comp_dim:]         # 后26列：性质

    def __getitem__(self, idx):
        return {
            "composition": self.compositions[idx],
            "properties": self.properties[idx],
        }

    def __len__(self):
        return len(self.compositions)
```

将其注册到 `ppmat/datasets/__init__.py` 的 `build_dataloader` 工厂函数中。

### 4.4 训练流程

GAN 的训练需要交替更新 Generator 和 Discriminator，与 `BaseTrainer` 的标准单模型训练循环不同。有两种方案：

**方案 A**（推荐）：自定义 `AlloyGANTrainer` 继承 `BaseTrainer`，覆写 `train_epoch` 实现 GAN 对抗训练循环：
- 每步先更新 Discriminator（真实样本 + 生成样本）
- 再更新 Generator（通过 Discriminator 的梯度反传）
- 利用 BaseTrainer 已有的日志、保存、分布式等基础设施

**方案 B**：独立训练脚本，不继承 BaseTrainer，但复用 ppmat 的工具函数（logger、save_load 等）。

两种方案均支持 YAML 配置驱动，最终方案将在与维护者沟通后确定。考虑到 GAN 的交替训练范式与现有 BaseTrainer 的差异较大，且原始代码足够简洁，方案 A 可能需要较多覆写但更符合套件统一规范。

### 4.5 YAML 配置示例

```yaml
Global:
  do_train: true
  do_eval: false

Model:
  __class_name__: AlloyCGAN
  __init_params__:
    noise_dim: 5
    cond_dim: 26
    hidden_dim: 512
    comp_dim: 40

Dataset:
  train:
    __class_name__: AlloyDataset
    __init_params__:
      csv_path: ./data/Alloy_train.csv
      comp_dim: 40
    batch_size: 64
    shuffle: true

Optimizer:
  generator:
    __class_name__: Adam
    learning_rate: 0.0002
    weight_decay: 0.00001
  discriminator:
    __class_name__: Adam
    learning_rate: 0.0002
    weight_decay: 0.00001

Trainer:
  max_epochs: 50
  output_dir: ./output/alloygan_cgan
  save_freq: 10
  log_freq: 100
  eval_freq: 10
  start_eval_epoch: 1
  seed: 42
```

### 4.6 补充说明

**关于 DCGAN 模型**：原始仓库中包含 DCGAN（Deep Convolutional GAN）模型，但 DCGAN 用于图像数据生成，与合金表格数据无关，本次复现不包含 DCGAN。

**关于数据集**：原论文数据集 `Alloy_train.csv` 为通过 LLM 自动提取的合金数据表格。数据文件将通过百度网盘链接提供给工程师获取下载地址，并在 README 中注明。

## 5. 测试和验收的考量

根据任务 NO.6-NO.19 的验收标准：

### 5.1 前向精度对齐

生成式模型采用以下标准：
- 使用相同随机种子和初始权重，前向输出 logits diff 控制在 **1e-6** 量级。
- 方法：将原始 PyTorch 模型权重转换为 Paddle 格式（`.pdparams`），用相同输入对比输出。

### 5.2 反向对齐

- 使用相同数据和超参训练 2 轮以上，Generator loss 和 Discriminator loss 趋势一致。

### 5.3 训练精度对齐

- **生成式模型**：采样指标保持误差 **5%** 以内。
- 具体指标参考论文中的热力学性质预测表格（Table 3），生成合金的关键性质指标（熔化温度、密度等）与论文报告值的误差控制在 5% 以内。

### 5.4 测试项

1. **单元测试**：模型前向传播、数据加载、权重转换的正确性测试。
2. **集成测试**：完整训练流程（YAML 配置 → 数据加载 → 训练 → 保存 → 推理）。
3. **精度对齐截图**：提供 Paddle 与 PyTorch 的 loss 曲线对比截图。
4. **生成质量评估**：使用论文中的评估方法验证生成合金成分的合理性。

## 6. 可行性分析和排期规划

| 阶段 | 时间 | 内容 |
|------|------|------|
| 第 1 周 | 调研与模型迁移 | 完成 GAN/CGAN 模型的 Paddle 代码编写、权重转换脚本、前向精度对齐 |
| 第 2 周 | 数据与训练 | 完成 AlloyDataset 数据模块、YAML 配置、训练流程、反向对齐验证 |
| 第 3 周 | 验收与文档 | 完成训练精度复现、推理脚本、单元测试、README 文档、提交 PR |

**可行性分析**：
- AlloyGAN 模型架构为简单的全连接网络（2 层 MLP），无图神经网络或自定义 CUDA 算子依赖，迁移风险极低。
- PyTorch 到 Paddle 的 API 映射全部 1:1 对应，无需额外适配。
- 主要工作量在于适配 PaddleMaterials 的 BaseTrainer 体系（GAN 交替训练范式）和数据集准备。

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `alloygan/` 目录，包含 AlloyGAN 和 AlloyCGAN 两个模型。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `alloy_dataset.py`，支持 CSV 格式的表格数据加载。
3. **新增任务类型**：新建 `alloy_design/` 任务目录，引入"合金逆向设计"任务类型，填补 PaddleMaterials 在 GAN 生成模型和无机材料工作流领域的空白。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 附件及参考资料

1. Xu, H., et al. *"LLM-Assisted Conditional Generative Adversarial Network for Inverse Design of Alloys."* arXiv:2502.18127, 2025.
2. 原始代码仓库：https://github.com/photon-git/AlloyGAN
3. PaddleMaterials 模型复现共建计划：https://github.com/PaddlePaddle/PaddleMaterials/issues/194
4. WGAN-GP RFC（GAN 类模型复现参考）：https://github.com/PaddlePaddle/community/pull/1108
