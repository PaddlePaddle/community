# ECFormer训练模块：独立目录设计与实践

| 所属任务 | 【Hackathon 10th Spring No.10】ECDFormer模型复现 |
| --- | --- |
| **提交作者**     | PlumBlossomMaid |
| **提交时间**     | 2026-02-13 (V1.0) → 2026-06-08 (V2.0) |
| **版本号**       | V2.0 |
| **依赖飞桨版本** | paddlepaddle-gpu 3.3.0 |
| **文件名**       | README.md |
| **计算平台**     | Windows 10 Python 3.13.1 AMD64 64bit |
| **PR状态**       | 🎉 已合并至 [PaddleMaterials](https://github.com/PaddlePaddle/PaddleMaterials) |

---

## 一、相关背景

ECDFormer是发表于**Nature Computational Science 2025**的工作，其核心创新在于 **“解耦峰属性学习”** ——将光谱预测任务分解为峰数量预测、峰位置预测、峰符号预测三个子任务，从而实现对ECD光谱的高效、可解释预测。该工作已成功泛化至IR光谱预测任务，验证了其框架的通用性。

在将ECDFormer模型家族迁移至PaddleMaterials工具库的过程中，我们面临一个**核心的架构决策问题**：

> **ECDFormer应该放在PaddleMaterials的哪个任务目录下？**

### 1.1 现有任务分类分析

PaddleMaterials当前包含三大核心任务：

| 任务类别 | 核心定义 | 输入 | 输出 | 代表模型 |
|---------|---------|------|------|---------|
| **`structure_generation`** | 从无到有创造结构 | 随机噪声/条件 | 晶体/分子结构 | CDVAE, DiffCSP |
| **`property_prediction`** | 结构→标量属性 | 分子/晶体结构 | 形成能、带隙等 | MEGNet, Comformer |
| **`spectrum_elucidation`** | **结构与谱图的相互映射** | 分子结构 **或** 谱图 | 谱图 **或** 分子结构 | **DiffNMR** |

### 1.2 ECDFormer的任务本质

- **ECDFormer的输入**：分子结构（通过原子特征、键特征、键角特征表示）
- **ECDFormer的输出**：ECD/IR谱图的峰属性（峰数量、峰位置、峰符号/强度）
- **任务本质**：**从分子结构预测谱图**

这与`spectrum_elucidation`中已存在的**DiffNMR**（从NMR谱图解析分子结构）形成**完美的任务对称**：

```
       谱图解析任务全景
   ┌─────────────────────────────────┐
   │                                 │
   │ 分子结构 ←──────────────→ 谱图   │
   │     ↑                      ↑    │
   │     │                      │    │
   │   ECDFormer            DiffNMR  │
   │   (预测谱图)          (解析结构) │
   │                                 │
   └─────────────────────────────────┘
```

因此，ECDFormer与DiffNMR是“谱图解析”这棵大树上的两个分支——一个是“从分子结构预测谱图”，一个是“从谱图解析分子结构”。两者如同**同一枚硬币的正反两面，必须归为一类，同居一室**。

---

## 二、问题识别：训练脚本的不兼容性

尽管ECDFormer在任务归属上应置于`spectrum_elucidation`，但现有`spectrum_elucidation/train.py`存在严重的**设计范式不兼容**：

### 2.1 现有训练脚本分析（以DiffNMR为例）

<pre>
# spectrum_elucidation/train.py (现有)
# 关键特征：
- 需要 `build_dataset_infos` 进行数据统计
- 需要 `ExtraFeatures` / `DomainFeatures` 进行特征工程
- 需要 `CLIP` 模块进行样本评估
- 需要 `MolecularVisualization` 进行分子可视化
- 训练流程深度耦合生成模型的特殊需求
</pre>

**为什么DiffNMR需要这些？**
- DiffNMR是**生成模型**，需要统计数据的分布（原子类型分布、键类型分布等）
- DiffNMR是**条件生成**，需要额外的特征编码器
- DiffNMR是**分子生成**，需要CLIP评估生成质量

### 2.2 ECFormer的需求分析

ECFormer是**纯监督学习模型**，需求完全不同：

| 功能模块 | DiffNMR（需要） | ECFormer（需要） | 原因 |
|---------|---------------|-----------------|------|
| 数据集统计 | ✅ 需要 | ❌ **不需要** | ECFormer是判别模型，无需先验分布 |
| 额外特征 | ✅ 需要 | ❌ **不需要** | 分子特征在数据集发布时已被充分提取 |
| CLIP评估 | ✅ 需要 | ❌ **不需要** | ECFormer输出谱图，不是生成分子 |
| 可视化 | 分子结构可视化 | **谱图可视化** | 关注点完全不同 |

**核心矛盾**：现有训练脚本是为**生成模型**设计的，而ECFormer是**监督学习模型**。强行复用会导致：
1. 执行大量无用计算，浪费资源
2. 配置复杂，用户难以理解
3. 代码耦合度高，难以维护

---

## 三、最终方案：独立目录`spectrum_prediction/`

经过与PaddleMaterials维护团队的沟通，最终采用**独立目录`spectrum_prediction/`**作为ECFormer的训练入口，而不改动原有的`spectrum_elucidation/`目录结构。这个决策的关键原因如下：

| 原方案（V1.0） | **最终方案（V2.0）** | 选型原因 |
|---------------|---------------------|---------|
| 在`spectrum_elucidation/`下新建`ECFormer/`子目录 | **新建`spectrum_prediction/`顶层目录** | ✅ 零侵入既有结构，DiffNMR完全不受影响 |
| 移动DiffNMR至子目录 | **DiffNMR保持不变** | ✅ 避免对既有贡献者的影响 |
| 共享`spectrum_elucidation/`命名空间 | **完全独立的任务目录** | ✅ 清晰区分"结构与谱图互映射"(elucidation)与"从结构预测谱图"(prediction) |

### 3.1 目录结构

```
spectrum_prediction/                # ✅ ECFormer专用训练目录（V2.0最终版）
├── README.md                       # 说明文档
├── train.py                        # 主训练脚本（argparse+OmegaConf）
├── test.py                         # 测试脚本
├── configs/
│   ├── ecd.yaml                    # ECD训练配置
│   └── ir.yaml                     # IR训练配置
└── data/                           # 数据集放置处（或使用自动下载）
```

**与V1.0设计的关键差异**：
1. **全新的顶层目录**：`spectrum_prediction/`不从属于`spectrum_elucidation/`，是两个并列的一级目录
2. **DiffNMR不变**：`spectrum_elucidation/`下的DiffNMR代码未做任何移动或修改
3. **配置驱动**：通过OmegaConf + YAML配置文件管理所有超参数

### 3.2 设计原则

| 原则 | 含义 | 体现 |
|------|------|------|
| **独立目录** | ECFormer拥有完全独立的代码空间 | 不与DiffNMR共享训练脚本、配置、工具函数 |
| **隔离运行** | 修改ECFormer不会影响DiffNMR，反之亦然 | 无交叉引用，无共享可变状态 |
| **尊重原作** | 保留DiffNMR原有实现不变 | 仅移动位置，不修改一行代码 |

### 3.3 与PaddleMaterials架构的融合

```python
# 用户使用方式完全统一
from ppmat.models import ECFormerECD, ECFormerIR    # 导入模型

# 配置文件选择ECFormer
Model:
  __class_name__: ppmat.models.ecformer.ECFormerECD
  __init_params__:
    emb_dim: 128
    num_layers: 5
```

**对用户透明**：用户通过配置文件选择模型，不感知目录结构的变化。

---

## 四、ECFormer训练脚本设计（V2.0最终版）

### 4.1 实际合入代码

```python
# spectrum_prediction/train.py（实际合入版本）

def main():
    # 1. 加载配置（argparse + OmegaConf）
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/ecd.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--predict", type=str, default=None)

    # 2. 构建数据集（通过build_dataloader工厂函数）
    dataloaders["train"] = build_dataloader(train_cfg)
    dataloaders["val"] = build_dataloader(val_cfg)

    # 3. 构建模型（通过build_model工厂函数）
    model = build_model(model_cfg)

    # 4. 构建优化器（通过build_optimizer工厂函数）
    optimizer, lr_scheduler = build_optimizer(...)

    # 5. ECFormerTrainer 统一管理训练/评估/预测
    trainer = ECFormerTrainer(
        config=config["Trainer"],
        model=model,
        train_dataloader=dataloaders.get("train"),
        val_dataloader=dataloaders.get("val"),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # 6. 执行训练/评估/测试/预测
    if do_train: trainer.train()
    if do_eval: trainer.eval(dataloaders["val"])
    if do_test: trainer.eval(dataloaders["test"])
    if do_predict: trainer.predict(dataloaders["predict"])
```

### 4.2 与V1.0设计的差异

| 维度 | V1.0设计 | **V2.0最终版** | 原因 |
|------|---------|---------------|------|
| 脚本位置 | `spectrum_elucidation/ECFormer/train.py` | **`spectrum_prediction/train.py`** | 独立顶层目录 |
| 数据加载 | 直接实例化DataLoader | **通过`build_dataloader`工厂函数** | 与PaddleMaterials框架集成 |
| 模型构建 | 直接实例化ECFormerECD | **通过`build_model`工厂函数** | 支持配置驱动 |
| 优化器 | 直接在脚本中定义 | **通过`build_optimizer`工厂函数** | 统一管理 |
| 训练循环 | 自定义训练循环 | **ECFormerTrainer类** | 复用框架基础设施 |
| 配置管理 | 简单dict | **OmegaConf + YAML** | 灵活、可嵌套、支持命令行覆盖 |
| 训练模式 | 仅训练 | **支持train/eval/test/predict四种模式** | 更完整的生命周期 |

### 4.3 配置示例

```yaml
# spectrum_prediction/configs/ecd.yaml
Global:
  do_train: True
  do_eval: True

Dataset:
  train:
    dataset:
      __class_name__: ECDDataset
      __init_params__:
        data_path: ./data/ECD
    loader:
      batch_size: 128
      collate_fn: ECDCollator

Model:
  __class_name__: ECFormerECD
  __init_params__:
    emb_dim: 256
    num_layers: 5

Optimizer:
  __class_name__: AdamW
  __init_params__:
    learning_rate: 0.001

Trainer:
  output_dir: ./output/ecd
  max_epochs: 100
```

### 4.4 纯Tensor输入（核心设计原则不变）

Paddle [Issue #77754](https://github.com/PaddlePaddle/Paddle/issues/77754) 的影响持续到最终版本。ECFormer坚持**纯Tensor输入**设计：

```python
# DataLoader返回的是Tensor字典，而非Data对象
# ECDCollator负责解包
return (
    {
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
        "batch_data": batch_data,
        "ba_edge_index": ba_edge_index, "ba_edge_attr": ba_edge_attr,
        "query_mask": query_mask,
    },
    {
        "peak_number": num_gt, "peak_position": pos_gt, "peak_height": height_gt,
    },
)
```

---

## 五、与PaddleMaterials现有框架的集成

### 5.1 目录结构的最终形态

```Text
PaddleMaterials/
├── ppmat/                          # 核心库（安装后存在）
│   ├── datasets/
│   │   ├── ecd_dataset.py          # ECDDataset
│   │   ├── ir_dataset.py           # IRDataset
│   │   ├── build_ecd.py            # ECD数据构建
│   │   ├── build_ir.py             # IR数据构建
│   │   ├── collate_fn.py           # ECDCollator, IRCollator
│   │   └── ...
│   ├── models/
│   │   ├── ecformer/               # ECFormer模型实现
│   │   │   ├── models/             # base_ecformer.py, ECD.py, IR.py
│   │   │   ├── encoders/           # GINNodeEmbedding
│   │   │   └── layers/             # AtomEncoder, BondEncoder, GINConv, RBF
│   │   └── ...
│   └── utils/
│       ├── compound_tools.py       # 特征维度定义
│       ├── graph_utils.py          # 图填充工具
│       └── PlaceEnv.py             # 设备上下文管理器
├── spectrum_prediction/            # ✅ ECFormer训练入口
│   ├── train.py
│   ├── configs/
│   │   ├── ecd.yaml
│   │   └── ir.yaml
│   └── README.md
├── spectrum_elucidation/           # 原DiffNMR目录（未改动）
│   └── ...
└── ...
```

### 5.2 集成优势

| 视角 | 优势 |
|-----|------|
| **任务视角** | 谱图解析任务下，DiffNMR与ECFormer并列，清晰展示任务全景 |
| **代码视角** | 两者完全隔离，互不干扰，修改任一不会影响另一 |
| **用户视角** | 通过配置文件选择模型，使用方式统一 |
| **维护视角** | 新增谱图任务（如质谱预测）可继续沿用此模式 |

---

## 六、测试与验证

### 6.1 测试方案

| 测试类型 | 测试内容 | 验证标准 |
|---------|---------|---------|
| **单元测试** | 各Dataset/DataLoader功能 | 正确返回预期Tensor格式 |
| **集成测试** | 完整训练流程 | 损失下降，不报错 |
| **精度对齐** | 与PyTorch原版对比 | 指标差异≤1% |
| **静态图测试** | SOT`to_static`编译 | 编译通过，推理一致 |
| **多卡测试** | 分布式训练 | 损失收敛正常 |

### 6.2 验收标准

1. **功能完整性**：ECD与IR任务均可完整训练、测试、推理
2. **精度达标**：复现论文指标（Position-RMSE ≈ 2.29, Symbol-Acc ≈ 72.7%）
3. **文档齐全**：用户可根据文档完成训练与推理

---

## 七、影响面分析

### 7.1 对现有DiffNMR用户的影响

**无影响**。DiffNMR代码未被修改或移动，原有的训练命令完全不变：

```bash
# 原命令仍然有效
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/DiffNMR.yaml

# ECFormer使用独立训练命令
python spectrum_prediction/train.py -c spectrum_prediction/configs/ecd.yaml
```

### 7.2 对ECFormer用户的影响

**正面影响**。用户获得：
- 清晰独立的训练脚本
- 完整的配置文件示例
- 开箱即用的数据集实现
- 静态图训练推理支持

### 7.3 对框架架构的影响

**无侵入**。所有修改均在`spectrum_elucidation/`目录内，不修改`ppmat/`核心库。

### 7.4 对二次开发的影响

**正面影响**。后续新增谱图任务（如拉曼光谱、质谱预测）可完全复用此模式：
1. 在`spectrum_elucidation/`下新建`Raman/`目录
2. 复制`ECFormer/train.py`作为模板
3. 实现任务特定的Dataset和Model


## 九、总结

**ECFormer训练模块的设计核心可以概括为三句话：**

> **独立目录**：在`spectrum_elucidation`下为ECFormer开辟独立空间，不与DiffNMR共享训练脚本。
>
> **隔离运行**：修改ECFormer不会影响DiffNMR，反之亦然，实现解耦维护。
>
> **尊重原作**：保留DiffNMR原有实现不变，仅移动位置，确保历史代码可用。

**最终效果**：
- 对用户：**清晰、易用、开箱即用**
- 对框架：**无侵入、可扩展**
- 对社区：**为后续谱图任务提供成熟范式**

**ECFormer与DiffNMR，一左一右，一正一反，共同照亮PaddleMaterials的谱图解析之路。**