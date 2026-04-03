# 基于PaddleMaterials实现CrystalLLM晶体结构生成模型复现

| 任务名称 | 基于PaddleMaterials实现CrystalLLM晶体结构生成模型复现 |
|------|------|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-03-23 |
| RFC 版本号 | v1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20260323_crystalllm.md |

## 1. 概述

### 1.1 相关背景

CrystalLLM 是一个基于 GPT-2 架构的大语言模型，用于晶体结构生成。该模型由 Antunes 等人于 2024 年发表在 *Nature Communications*（[DOI: 10.1038/s41467-024-54639-7](https://doi.org/10.1038/s41467-024-54639-7)），将晶体信息文件（CIF）作为文本序列输入，通过自回归方式生成新的晶体结构。

CrystalLLM 的核心创新在于：
1. **CIF 文本化表示**：直接将标准 CIF 格式作为模型输入，无需图表示（GNN）或几何编码，大幅简化数据预处理流程
2. **大规模预训练**：使用来自 Materials Project、OQMD、NOMAD 三大数据库共 2,285,914 条去重 CIF 记录进行预训练
3. **MCTS 引导生成**：结合蒙特卡洛树搜索（Monte Carlo Tree Search）与外部评分器实现高质量晶体结构定向生成
4. **强泛化能力**：在 Perov-5、Carbon-24、MP-20、MPTS-52 四个标准基准上均取得优异结果（有效率 94%，空间群一致性 98.9%）

参考实现：[https://github.com/lantunes/CrystaLLM](https://github.com/lantunes/CrystaLLM)（MIT License，Python 100%，156 stars）

相关 issue：[PaddleMaterials#194](https://github.com/PaddlePaddle/PaddleMaterials/issues/194)（PPMat-模型复现共建计划，模型列表编号 1）

### 1.2 功能目标

1. 在 `ppmat/models/crystalllm/` 下实现 CrystalLLM 模型，包含 GPT 模型、CIF 分词器、MCTS 采样器
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 数据集通过 build 工厂函数注册，支持 MP/OQMD/NOMAD 数据预处理与分词
4. 前向精度对齐（生成式模型要求 logits diff 1e-6 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 生成式采样指标（有效率、空间群一致性、键长合理性）误差控制在 5% 以内
7. 覆盖原论文全部基准数据集：Perov-5、Carbon-24、MP-20、MPTS-52
8. 添加对应的任务 README 文档

### 1.3 意义

- 填补 PaddleMaterials 在 **LLM 类晶体结构生成** 领域的空白（现有模型以 GNN/扩散模型为主）
- 新增任务类型 **"Crystal Structure Generation"（晶体结构生成）**，与现有 Interatomic Potentials、Structure Generation（扩散类）、Spectrum Enhancement 并列，为后续 LLM 类材料生成模型（如 MatGPT、CIF-GPT）提供标准任务框架
- 引入基于文本表示的材料生成范式，为后续多模态材料模型（CIF + 属性文本）提供基础
- CrystalLLM 是 Nature Communications 发表的高影响力工作，复现有助于提升 PaddleMaterials 的学术影响力与社区关注度

## 2. PaddleMaterials 现状

PaddleMaterials 套件目前已集成以下模型，涵盖属性预测、结构生成、扩散模型等任务类型：

| 已有模型 | 任务类型 | 架构 | 与 CrystalLLM 关系 |
|---------|---------|------|-------------------|
| CHGNet | 原子间势函数 | GNN | 无直接复用 |
| MatterGen | 结构生成 | 扩散模型 | 可参考 sampler 接口设计 |
| DiffCSP | 结构生成 | 扩散模型 | 可参考 scheduler 组件 |
| DiffNMR | 谱图预测 | 扩散模型 | 可参考 trainer 接口 |
| MatterSim | 原子间势函数 | GNN | 可参考 predictor 接口 |
| ComFormer | 属性预测 | Transformer | **架构最相近**，可参考 Transformer 层实现 |
| InfGCN | 属性预测 | GNN | 无直接复用 |

**关键现状分析**：

1. **缺少 LLM 类模型**：现有 Transformer 模型（ComFormer）用于属性预测，非自回归生成。CrystalLLM 是首个 LLM 类生成模型。
2. **sampler 模块通用性**：`ppmat/sampler/` 目前主要服务扩散模型采样。CrystalLLM 需新增自回归采样 + MCTS 采样两种模式。
3. **数据集工厂函数**：`ppmat/datasets/` 提供 `build_structure`/`build_molecule` 等工厂函数。CrystalLLM 使用 CIF 文本数据，需新增基于 CIF 文本的 dataset 构建路径。
4. **trainer 直接复用**：`ppmat/trainer/BaseTrainer` 的训练循环、日志、checkpoint 保存逻辑可直接使用，无需新增自定义训练器。CrystalLLM 模型的 `forward()` 按 ppmat 标准返回 `(loss_dict, pred_dict)`，BaseTrainer 可直接驱动训练。

## 3. 目标调研

### 3.1 论文方法概述

CrystalLLM 基于 nanoGPT（Karpathy 实现的轻量 GPT-2）架构。核心思路是将 CIF 文件视为结构化文本序列：

```
data_LiFePO4
_symmetry_space_group_name_H-M Pnma
_cell_length_a 10.3377
_cell_length_b 6.0112
_cell_length_c 4.6950
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_type_symbol
_atom_site_fract_x
...
```

模型通过自回归方式逐 token 生成 CIF 文本，使用自定义的 `CIFTokenizer` 分词器将 CIF 内容分解为原子符号、数字、CIF 关键字、空间群符号等 371 个 token。

**模型架构参数**（两种配置）：

| 参数 | Small | Large |
|-----|-------|-------|
| n_layer | 8 | 16 |
| n_head | 8 | 16 |
| n_embd | 512 | 1024 |
| block_size | 1024 | 2048 |
| vocab_size | 371 | 371 |
| dropout | 0.1 | 0.1 |
| bias | True | True |
| 参数量 | ~33M | ~250M |

**评估指标**（原论文报告，MP-20 数据集，n=10000）：

| 指标 | 值 |
|-----|-----|
| 有效率 (Validity) | 94.0% |
| 空间群一致性 (Space Group Consistency) | 98.9% |
| 原子位占据数一致性 (Atom Site Multiplicity) | 99.4% |
| 键长合理性分数 (Bond Length Reasonableness) | 0.988 |

### 3.2 源码结构分析

原始 PyTorch 实现（`lantunes/CrystaLLM`）核心模块：

| 文件 | 功能 | 行数 | 复现策略 |
|-----|-----|------|---------|
| `crystallm/_model.py` | GPT 模型（LayerNorm, Attention, MLP, Block, GPT） | ~300 | PyTorch → Paddle 逐层转换 |
| `crystallm/_tokenizer.py` | CIF 分词器（原子/数字/关键字/空间群 token） | ~120 | 纯 Python 逻辑，基本不变 |
| `crystallm/_configuration.py` | YAML 配置加载（基于 omegaconf） | ~50 | 适配 ppmat 配置系统 |
| `crystallm/_mcts.py` | MCTS 采样器（MCTSSampler, MCTSNode, Evaluator） | ~350 | 模型调用层适配 Paddle，算法逻辑不变 |
| `crystallm/_metrics.py` | 评估指标（有效性、一致性、键长合理性） | ~200 | 依赖 pymatgen，基本不变 |
| `crystallm/_scorer.py` | 外部评分器接口 | ~30 | 接口适配 |
| `crystallm/_utils.py` | CIF 解析工具函数 | ~100 | 纯 Python，不变 |
| `bin/train.py` | 训练脚本 | ~250 | 适配 ppmat/trainer |
| `bin/sample.py` | 自回归采样脚本 | ~100 | 适配 ppmat/sampler |
| `bin/evaluate_cifs.py` | 评估脚本 | ~80 | 适配 ppmat/metrics |

### 3.3 PyTorch → Paddle API 映射

CrystalLLM 仅使用标准 PyTorch NN 操作，无自定义 CUDA kernel，迁移风险极低。

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `torch.nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `nn.Linear` | `paddle.nn.Linear` | 一致 |
| `nn.Embedding` | `paddle.nn.Embedding` | 一致 |
| `nn.Dropout` | `paddle.nn.Dropout` | 一致 |
| `nn.ModuleDict` | `paddle.nn.LayerDict` | 命名不同 |
| `nn.ModuleList` | `paddle.nn.LayerList` | 命名不同 |
| `nn.Parameter(torch.ones(...))` | `paddle.create_parameter(...)` | 初始化方式不同 |
| `F.layer_norm` | `paddle.nn.functional.layer_norm` | 一致 |
| `F.scaled_dot_product_attention` | `paddle.nn.functional.scaled_dot_product_attention_` | Paddle 2.6+ 支持 |
| `F.softmax` | `paddle.nn.functional.softmax` | 一致 |
| `F.cross_entropy` | `paddle.nn.functional.cross_entropy` | 参数名略有差异（`ignore_index`） |
| `torch.multinomial` | `paddle.multinomial` | 一致 |
| `torch.topk` | `paddle.topk` | 返回值顺序一致 |
| `torch.tril` | `paddle.tril` | 一致 |
| `torch.cat` | `paddle.concat` | 命名不同 |
| `tensor.view(...)` | `paddle.reshape(...)` | Paddle 推荐使用 reshape |
| `tensor.transpose(-2, -1)` | `paddle.transpose(x, perm=[0,1,3,2])` | 需要显式指定 perm |
| `torch.optim.AdamW` | `paddle.optimizer.AdamW` | 参数传递方式不同 |
| `register_buffer` | `register_buffer` | Paddle 2.4+ 已支持 |

**无需迁移的模块**：`CIFTokenizer`（纯 Python + regex）、`_metrics.py`（依赖 pymatgen，非 PyTorch）、`_utils.py`（纯字符串操作）、`_scorer.py`（接口类）

### 3.4 数据集概况

| 数据来源 | CIF 数量 | 说明 |
|---------|---------|------|
| Materials Project (MP) | ~150K | 实验+计算晶体结构 |
| OQMD | ~1M | 高通量计算数据库 |
| NOMAD | ~2.4M | 开放材料数据档案 |
| **合计（去重后）** | **2,285,914** | CC-BY 4.0，托管于 Zenodo |

**基准数据集**：Perov-5（钙钛矿，5 元素）、Carbon-24（碳，最多 24 原子）、MP-20（通用，最多 20 原子）、MPTS-52（通用，最多 52 原子）

数据集全部可通过 Zenodo 公开获取，无许可证限制。预处理流程：下载 → 去重 → 预处理 → 分割 → 分词 → 生成 `train.bin`/`val.bin`。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
ppmat/models/crystalllm/
├── __init__.py              # 模块导出 + 注册到 ppmat/models/__init__.py
├── model.py                 # GPT 模型（GPTConfig, LayerNorm, CausalSelfAttention, MLP, Block, GPT）
├── tokenizer.py             # CIFTokenizer（分词/编码/解码）
├── mcts.py                  # MCTS 采样器（MCTSSampler, MCTSNode, MCTSEvaluator）
├── metrics.py               # 评估指标（有效性、一致性、键长合理性）
├── utils.py                 # CIF 解析工具函数
└── spacegroups.txt          # 230 种空间群符号表

ppmat/datasets/
└── crystalllm_dataset.py    # CIF 文本数据集 + build_cif_text 工厂函数

ppmat/sampler/
└── crystalllm_sampler.py    # 采样器（自回归采样 + MCTS 采样）

configs/crystalllm/
├── crystalllm_small.yaml    # small 模型配置（8层, ~33M 参数）
└── crystalllm_large.yaml    # large 模型配置（16层, ~250M 参数）

crystal_structure_generation/     # 新增任务类型目录
├── README.md                # 任务类型说明文档
└── configs/crystalllm/
    └── README.md             # 模型配置说明（含训练结果）

examples/crystalllm/
├── README.md                # 任务说明文档（含结果及链接）
├── train.py                 # 训练入口
├── sample.py                # 采样入口
└── evaluate.py              # 评估入口

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 crystalllm 导入
# ppmat/datasets/__init__.py   — 新增 build_cif_text 工厂函数注册
```

### 4.2 模型实现

核心模型类遵循 PaddleMaterials 统一的 `_forward()`/`forward()`/`predict()` 三层设计，并返回标准 `{loss_dict, pred_dict}` 字典。关键代码骨架如下：

```python
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import paddle
import paddle.nn as nn
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 371
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = True

class NewGELU(nn.Layer):
    """与原论文一致的自定义 GELU 实现（非 approximate='tanh'）"""
    def forward(self, x):
        return 0.5 * x * (1.0 + paddle.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))

class MLP(nn.Layer):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias_attr=config.bias)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias_attr=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Layer):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias_attr=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias_attr=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, axis=2)
        q = q.reshape([B, T, self.n_head, C // self.n_head]).transpose([0, 2, 1, 3])
        k = k.reshape([B, T, self.n_head, C // self.n_head]).transpose([0, 2, 1, 3])
        v = v.reshape([B, T, self.n_head, C // self.n_head]).transpose([0, 2, 1, 3])
        # Paddle 2.6+ 支持 scaled_dot_product_attention_（自动选择 Flash/Memory-Efficient）
        y = paddle.nn.functional.scaled_dot_product_attention_(q, k, v, is_causal=True)
        y = y.transpose([0, 2, 1, 3]).reshape([B, T, C])
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPT(nn.Layer):
    """CrystalLLM 主模型，遵循 ppmat 的 _forward/forward/predict 三层模式"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.LayerList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias_attr=False)
        # 权重绑定（Weight Tying）
        self.lm_head.weight = self.wte.weight

    def _forward(self, idx):
        """纯前向计算，返回 logits（不含 loss 计算）"""
        b, t = idx.shape
        pos = paddle.arange(0, t, dtype='int64').unsqueeze(0)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, idx, targets=None):
        """训练前向，返回 ppmat 标准 (loss_dict, pred_dict)"""
        logits = self._forward(idx)
        pred_dict = {"logits": logits}
        loss_dict = {}
        if targets is not None:
            loss = paddle.nn.functional.cross_entropy(
                logits.reshape([-1, logits.shape[-1]]),
                targets.reshape([-1]),
                ignore_index=-1
            )
            loss_dict["ntp_loss"] = loss
        return loss_dict, pred_dict

    def predict(self, idx):
        """推理入口，仅对最后一个位置做 lm_head 投影"""
        logits = self._forward(idx)
        logits = logits[:, [-1], :]  # 仅取最后位置
        return {"logits": logits}

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成，检测 '\n\n' 终止"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            pred = self.predict(idx_cond)
            logits = pred["logits"][:, -1, :] / temperature
            if top_k is not None:
                v, _ = paddle.topk(logits, min(top_k, logits.shape[-1]))
                logits = paddle.where(logits < v[:, [-1]], paddle.full_like(logits, float('-inf')), logits)
            probs = paddle.nn.functional.softmax(logits, axis=-1)
            idx_next = paddle.multinomial(probs, num_samples=1)
            idx = paddle.concat([idx, idx_next], axis=1)
        return idx
```

**设计要点**：
1. **ppmat 三层模式**：`_forward()` 纯计算 → `forward()` 返回 `(loss_dict, pred_dict)` → `predict()` 推理入口。与 DimeNet++/SchNet/SFIN 等模型保持一致
2. **权重绑定**（Weight Tying）：`lm_head.weight = wte.weight`，Paddle 中通过参数共享实现
3. **Flash Attention**：Paddle 2.6+ 的 `scaled_dot_product_attention_` 自动使用高效实现
4. **自定义 GELU**：`NewGELU` 实现与原论文精确一致（非 `nn.GELU(approximate=True)`），确保前向对齐
5. **推理优化**：`predict()` 仅对最后一个位置做 lm_head 投影
6. **版权声明**：Apache License 2.0，Copyright 2026 PaddlePaddle Authors

### 4.3 数据集适配

```python
# ppmat/datasets/crystalllm_dataset.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
import os
import numpy as np
import paddle
from paddle.io import Dataset

# BCS 自动下载 URL（预分词 bin 文件，百度对象存储托管）
DATA_URLS = {
    "mp20_train": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/crystalllm/mp20_train.bin",
    "mp20_val":   "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/crystalllm/mp20_val.bin",
    "perov5_train": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/crystalllm/perov5_train.bin",
    # ... 其他基准数据集
}

class CIFTokenDataset(Dataset):
    """基于预分词 bin 文件的 CIF 数据集，支持 BCS 自动下载"""
    def __init__(self, data_path, block_size=1024, auto_download=True):
        if auto_download and not os.path.exists(data_path):
            self._download(data_path)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def _download(self, data_path):
        """从 BCS 自动下载预分词数据文件"""
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        dataset_key = os.path.basename(data_path).replace('.bin', '')
        if dataset_key in DATA_URLS:
            from ppmat.utils.download import download_url
            download_url(DATA_URLS[dataset_key], os.path.dirname(data_path))

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = paddle.to_tensor(self.data[idx:idx+self.block_size].astype(np.int64))
        y = paddle.to_tensor(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y

def build_cif_text(config):
    """CIF 文本数据集工厂函数，注册到 ppmat/datasets/__init__.py
    
    与现有 build_structure/build_molecule/build_spectrum 并列，
    新增 build_cif_text 用于基于 CIF 文本的语言模型数据集。
    """
    return CIFTokenDataset(
        data_path=config.data_path,
        block_size=config.get('block_size', 1024),
        auto_download=config.get('auto_download', True)
    )
```

**数据集注册**：在 `ppmat/datasets/__init__.py` 中新增 `CIFTokenDataset` 类导入（加入 `__all__` 列表），使其可通过 ppmat 统一的 `__class_name__`/`__init_params__` 配置模式自动实例化。同时注册 `build_cif_text` 工厂函数作为便捷入口，与现有 `build_structure`、`build_molecule`、`build_spectrum` 工厂函数并列。CrystalLLM 使用 CIF 文本（非图结构、非分子坐标），因此新增独立的 `build_cif_text` 而非复用已有工厂。

**数据自动下载**：预分词 bin 文件上传至百度 BCS（Baidu Cloud Storage），首次使用时自动下载。原始 CIF 数据（2.3M 条，Zenodo CC-BY 4.0）的预处理管线保持与原实现一致（download → deduplicate → preprocess → split → tokenize），输出 `train.bin`/`val.bin` 格式。

### 4.4 Trainer 适配

直接使用 `ppmat/trainer/base_trainer.py` 中的 `BaseTrainer`，**不新增自定义训练器文件**。CrystalLLM 的 GPT 模型 `forward()` 已返回 ppmat 标准的 `(loss_dict, pred_dict)` 格式（见 4.2 节），BaseTrainer 可直接驱动训练循环，无需子类化。

关键适配点（均通过 YAML 配置传入 BaseTrainer，无需代码修改）：

1. **损失函数**：模型 `forward()` 内部计算 next-token prediction `cross_entropy`（`ignore_index=-1`），通过 `loss_dict["ntp_loss"]` 传递给 BaseTrainer
2. **优化器**：AdamW，参数分组（权重衰减 vs 不衰减）。将 LayerNorm 参数和 bias 归入 no_weight_decay 组
3. **学习率调度**：余弦退火 + warmup
4. **梯度裁剪**：`grad_clip=1.0`
5. **Checkpoint**：复用 BaseTrainer 现有 checkpoint 保存/加载逻辑
6. **训练脚本入口**：`crystal_structure_generation/train.py` 调用 `BaseTrainer(config, model, train_loader, val_loader, optimizer)`，与 `interatomic_potentials/train.py`、`spectrum_enhancement/train.py` 保持一致模式

示例配置片段（`configs/crystalllm/crystalllm_small.yaml`）：

```yaml
Model:
  __class_name__: GPT
  __init_params__:
    block_size: 1024
    vocab_size: 371
    n_layer: 8
    n_head: 8
    n_embd: 512
    dropout: 0.1
    bias: true

Dataset:
  __class_name__: CIFTokenDataset
  __init_params__:
    data_path: data/crystalllm/mp20_train.bin
    block_size: 1024
    auto_download: true

Trainer:
  max_iters: 100000
  batch_size: 32
  learning_rate: 1.0e-3
  weight_decay: 0.1
  warmup_iters: 2000
  lr_decay_iters: 100000
  min_lr: 1.0e-4
  grad_clip: 1.0
  log_interval: 10
  eval_interval: 2000
```

示例配置片段（`configs/crystalllm/crystalllm_large.yaml`）：

```yaml
Model:
  __class_name__: GPT
  __init_params__:
    block_size: 2048
    vocab_size: 371
    n_layer: 16
    n_head: 16
    n_embd: 1024
    dropout: 0.1
    bias: true

Dataset:
  __class_name__: CIFTokenDataset
  __init_params__:
    data_path: data/crystalllm/mp20_train.bin
    block_size: 2048
    auto_download: true

Trainer:
  max_iters: 48000
  batch_size: 16
  learning_rate: 1.0e-3
  weight_decay: 0.1
  warmup_iters: 2000
  lr_decay_iters: 48000
  min_lr: 1.0e-4
  grad_clip: 1.0
  log_interval: 10
  eval_interval: 2000
```

### 4.5 Sampler 适配

提供两种采样模式：

1. **自回归采样**（Autoregressive Sampling）：逐 token 生成，支持 temperature 和 top-k 参数。检测到连续两个换行符（`\n\n`）时终止生成。
2. **MCTS 引导采样**（MCTS-guided Sampling）：结合外部评分器（如 MatGL）进行定向晶体生成。集成 `MCTSSampler`、`MCTSNode`、`ContextSensitiveTreeBuilder` 等组件。

### 4.6 关键迁移注意事项

| 注意事项 | 说明 |
|---------|------|
| `block_size` 配置相关 | small 模型使用 1024，large 模型使用 2048，与原论文一致 |
| 分词器空间群消歧 | `Pm`（原子）与 `Pm`（空间群）通过添加 `_sg` 后缀消歧 |
| `register_buffer` 用于因果掩码 | 非 Flash Attention 路径需维护下三角掩码 buffer |
| 权重初始化 | 残差投影层 `c_proj` 使用缩放初始化 `std=0.02/sqrt(2*n_layer)` |
| 权重初始化验证 | 使用相同随机种子初始化 PyTorch 和 Paddle 模型，对比前向 logits 一致性 |
| 版权声明 | 所有新增文件使用 Apache License 2.0，Copyright 2026 PaddlePaddle Authors |
| 数据自动下载 | 预分词 bin 文件托管在百度 BCS，首次使用自动下载 |

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同随机种子和输入 token，对比 PyTorch 和 Paddle 实现的 logits 输出
- 生成式模型要求：logits diff ≤ 1e-6 量级
- 测试覆盖：单 token 前向、完整序列前向、不同 batch size

### 5.2 反向训练对齐

- 使用相同数据子集（如 1000 条 CIF），对比两个框架训练 2 轮以上的 loss 曲线
- 要求 loss 逐 step 一致（允许浮点误差累积导致的微小偏差）

### 5.3 生成式采样指标

在 MP-20 基准数据集上，使用训练完成的模型权重进行采样（n=10000，temperature=0.6），对比以下指标：

| 指标 | 原论文值 | 允许误差 |
|-----|---------|---------|
| Validity | 94.0% | ±5% |
| Space Group Consistency | 98.9% | ±5% |
| Bond Length Reasonableness | 0.988 | ±5% |
| Atom Site Multiplicity Consistency | 99.4% | ±5% |

### 5.4 基准数据集覆盖

全部四个基准需可运行完整的 train → sample → evaluate 流程：
- Perov-5、Carbon-24、MP-20、MPTS-52

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **低** | 标准 GPT-2，仅使用 Linear/Embedding/LayerNorm/Attention |
| 自定义算子 | **无** | 无 CUDA kernel，全部为标准 NN 操作 |
| 依赖项风险 | **低** | pymatgen（已在 ppmat 生态中使用）、omegaconf（标准 Python 库） |
| 数据可获取性 | **高** | 全部数据集公开在 Zenodo（CC-BY 4.0），~700MB/档案 |
| API 映射完整性 | **高** | 所有 PyTorch API 均有 Paddle 对应实现 |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | AI Studio V100 环境配置、pymatgen 安装、原始实现验证 | 2天 |
| Phase 1：模型迁移 | GPT 模型（small + large）+ CIFTokenizer → Paddle，前向精度对齐 | 3天 |
| Phase 2：训练对齐 | Trainer 适配、反向训练 2+ 轮，loss 曲线对齐 | 3天 |
| Phase 3：采样器 | 自回归采样 + MCTS 采样器 Paddle 实现 | 2天 |
| Phase 4：数据集 | 预处理管线适配、build 工厂函数注册 | 2天 |
| Phase 5：评估验证 | 四个基准数据集完整 train→sample→evaluate，指标对齐 | 3天 |
| Phase 6：文档与合入 | README 编写、提交 PR、代码审查 | 2天 |

## 7. 影响面

### 对 PaddleMaterials 的影响

1. **新增模型目录**：`ppmat/models/crystalllm/`（新增 ~1000 行代码，含 model.py / tokenizer.py / mcts.py / metrics.py / utils.py）
2. **新增数据集**：`ppmat/datasets/crystalllm_dataset.py`（~80 行，含 `build_cif_text` 工厂函数 + BCS 自动下载）
3. **新增采样器**：`ppmat/sampler/crystalllm_sampler.py`（~100 行），为后续 LLM 类材料模型提供基础
4. **新增任务类型目录**：`crystal_structure_generation/`（新增任务类型 README + 配置）
5. **修改已有文件**（仅新增注册行）：
   - `ppmat/models/__init__.py`：新增 `from .crystalllm import GPT, GPTConfig`
   - `ppmat/datasets/__init__.py`：新增 `CIFTokenDataset` 类注册 + `build_cif_text` 工厂函数
6. **依赖项**：pymatgen（已有）、omegaconf（可选，可使用 ppmat 现有 YAML 配置系统替代）

### 对用户的影响

- 提供开箱即用的晶体结构生成能力
- 提供完整的训练脚本，用户可自行在 AI Studio 等 GPU 环境上从零训练
- 新增 `examples/crystalllm/` 示例脚本降低使用门槛
