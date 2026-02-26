# ECFormer 模型底层Layer模块设计

| 所属任务 | 【Hackathon 10th Spring No.10】ECDFormer模型复现 |
| --- | --- |
| **提交作者**     | PlumBlossomMaid |
| **提交时间**     | 2026-02-13 |
| **版本号**       | 1.0 |
| **依赖飞桨版本** | paddlepaddle-gpu 3.3.0 |
| **文件名**       | Layers.md |
| **计算平台**     | Windows 10 Python 3.13.1 AMD64 64bit |

---

## 一、功能目标

ECFormer模型是基于Transformer架构的分子谱图预测模型，其底层Layer模块承担分子图特征提取、几何信息编码、注意力计算等核心功能。本文档旨在设计并实现一套**高性能、可复现、静态图友好**的ECFormer底层Layer集合，主要目标包括：

- **完整复现ECDFormer原仓库**的分子特征提取能力，包括原子编码器、键编码器、RBF径向基函数编码器、GINConv图卷积层等；
- **支持Paddle静态图模式**及ONNX导出，彻底解决原仓库中`Data`对象作为模型输入导致的静态图编译失败问题；
- **确保数值精度对齐**，所有Layer实现与PyTorch原版在`float64`精度下误差≤1e-9；
- **模块化解耦设计**，各Layer独立可复用，为ECFormer-ECD、ECFormer-IR等任务模型提供统一底层支持。

---

## 二、飞桨现状

### 2.1 当前支持情况

| 功能点 | 飞桨原生支持 | 是否需要定制 |
|--------|-------------|------------|
| 原子/键离散特征Embedding | ✅ `nn.Embedding` | 否，直接复用 |
| 多层感知机（MLP） | ✅ `nn.Sequential` + `nn.Linear` | 否 |
| 批归一化 | ✅ `nn.BatchNorm1D` | 否 |
| GINConv图卷积 | ❌ 无原生实现 | **是** |
| RBF径向基函数编码 | ❌ 无原生实现 | **是** |

### 2.2 关键问题与绕行方案

**核心痛点**：Paddle DataLoader在加载非Tensor类型数据时存在已知Bug（[Issue #77754](https://github.com/PaddlePaddle/Paddle/issues/77754)）会导致数据无法加载，原ECDFormer仓库中直接传递`torch_geometric.data.Data`对象的方案会导致模型**无法在静态图模式下编译**。

**本方案绕行策略**：**所有Layer的输入接口统一设计为接收`paddle.Tensor`，彻底解耦与`Data`对象的依赖**。分子图数据（节点特征、边索引、边特征、批次索引等）在进入Layer前已完成解包，Layer仅负责张量计算，不感知图结构对象。

---

## 三、业内方案调研

### 3.1 ECDFormer原仓库实现分析

ECDFormer原仓库( https://github.com/HowardLi1984/ECDFormer )的Layer层实现具有以下特点：

| 模块 | 原实现方式 | 优点 | 缺点 |
|------|---------|------|------|
| `AtomEncoder` | 多个`nn.Embedding`求和 | 实现简单，特征融合充分 | 无 |
| `BondEncoder` | 同上 | 同上 | 无 |
| `RBF` | 自定义RBF层 | 精确模拟高斯展宽 | 依赖`torch.arange`参数化 |
| `BondFloatRBF` | RBF + Linear | 连续值编码 | 设备耦合紧 |
| `GINConv` | 自定义MessagePassing | 标准GIN实现 | 与PyG耦合 |

**原仓库与Paddle生态的主要差异**：

1. **图计算后端**：原仓库基于`torch_geometric`，本方案基于`paddle_geometric`；
2. **输入格式**：原仓库`forward`接收`Data`对象，本方案**全部改为接收`Tensor`**；
3. **设备管理**：原仓库存在大量`to(device)`显式调用，本方案遵循Paddle无感设备管理；
4. **参数初始化**：原仓库使用`xavier_uniform_`，本方案使用`paddle.nn.initializer.XavierUniform`。

### 3.2 迁移策略

**核心原则：保留算法逻辑，重构接口设计，对齐数值精度。**

| 原组件 | 迁移策略 | 改动量 | 精度对齐方式 |
|--------|---------|--------|------------|
| `AtomEncoder` | 直接迁移，修改import | 低 | 逐层输出对比 |
| `BondEncoder` | 直接迁移 | 低 | 逐层输出对比 |
| `RBF` | 重写Parameter初始化 | 中 | 固定种子+数值比对 |
| `BondFloatRBF` | 解耦设备依赖 | 中 | 固定输入测试 |
| `GINConv` | 基于`paddle_geometric`重写 | 中 | 与PyG版本逐边对比 |

---

## 四、对比分析

### 4.1 输入接口设计选型

| 方案 | 描述 | 优点 | 缺点 | 结论 |
|------|------|------|------|------|
| **方案A（本方案采纳）** | Layer接收`Tensor`，外部负责解包`Data` | ✅ 静态图/ONNX完美支持<br>✅ 接口纯净，职责单一<br>✅ 测试便利 | ❌ 调用方需额外处理数据 | **推荐** |
| 方案B | Layer接收`Data`，内部解包 | ✅ 与原仓库完全一致 | ❌ 静态图编译失败<br>❌ 与Paddle生态割裂 | 拒绝 |
| 方案C | 同时支持两种模式 | ✅ 兼容性强 | ❌ 代码臃肿<br>❌ 维护成本翻倍 | 不采纳 |

**决策依据**：Paddle官方DataLoader的非Tensor加载Bug在短期内无法彻底修复，且静态图编译是Paddle的核心优势，**为保证ECFormer模型的长期可维护性、推理部署友好性，必须彻底切断Layer层对`Data`对象的依赖**。

### 4.2 RBF参数初始化选型

| 方案 | 描述 | 优点 | 缺点 | 结论 |
|------|------|------|------|------|
| **方案A（本方案采纳）** | 使用`paddle.create_parameter`+`Assign` | ✅ 参数可训练<br>✅ 与PyTorch `nn.Parameter`语义一致 | ❌ 代码稍复杂 | **推荐** |
| 方案B | 固定不可训练中心点 | ✅ 实现简单 | ❌ 与原始论文不符 | 拒绝 |

---

## 五、设计思路与实现方案

### 5.1 整体设计架构

```
ppmat/models/ecformer/layers/
├── __init__.py                 # 模块导出
├── atom_encoder.py             # 原子特征编码器
├── bond_encoder.py             # 键特征编码器
├── rbf.py                      # RBF及连续特征编码器
└── gin_conv.py                 # GIN图卷积层
```

**设计哲学**：
1. **单一职责**：每个Layer只做一件事，输入输出均为`Tensor`；
2. **无状态设备**：不持有`device`信息，遵循Paddle动态图机制；
3. **显式形状**：所有Layer在文档中明确标注输入输出张量形状；
4. **精度优先**：所有数学运算均支持`float64`，便于精度对齐验证。

### 5.2 关键子模块设计

#### 5.2.1 原子/键编码器（`AtomEncoder`/`BondEncoder`）

**设计思路**：原ECDFormer使用9维原子离散特征和3维键离散特征，每个维度独立Embedding后求和。本方案完全保留该设计，仅将框架从PyTorch迁移至Paddle。

**输入输出规格**：
- 输入：`x` - 原子特征矩阵，形状`[N, F]`，`F=9`（原子）/`F=3`（键），`dtype=int64`
- 输出：`[N, emb_dim]`，`dtype=float64`

#### 5.2.2 RBF径向基函数编码器（`RBF`/`BondFloatRBF`/`BondAngleFloatRBF`）

**设计思路**：连续值（键长、键角）通过RBF基函数展开为高维特征。原仓库使用可训练的中心点和gamma参数，本方案通过`paddle.create_parameter`+`Assign`初始化器完整复现该行为。

**关键改进**：**彻底移除原代码中的`to(device)`显式调用**，所有张量创建使用`paddle.arange`、`paddle.to_tensor`等无设备绑定方式，由Paddle框架自动管理。

**输入输出规格**：
- `BondFloatRBF`输入：`bond_float_features` - 键连续特征，形状`[E, 1]`
- `BondAngleFloatRBF`输入：`bond_angle_float_features` - 键角特征，形状`[A, 1]`
- 输出：`[E/A, embed_dim]`

**精度对齐**：固定随机种子，与PyTorch原版在相同输入下逐元素对比，确保误差≤1e-8。

#### 5.2.3 GIN图卷积层（`GINConv`）

**设计思路**：基于`paddle_geometric.nn.MessagePassing`基类实现图同构卷积。完全保留原仓库的邻域聚合逻辑（`add`聚合 + `relu(x_j + edge_attr)`消息函数）。

**输入输出规格**：
- 输入：`x` - 节点特征 `[N, emb_dim]`
- 输入：`edge_index` - 边索引 `[2, E]`
- 输入：`edge_attr` - 边特征 `[E, emb_dim]`
- 输出：`[N, emb_dim]`

**验证方案**：构造小型分子图，确保Paddle版本与PyTorch版本在10^-8精度内完全一致。

---

## 六、测试与验收考量

### 6.1 自测方案

| 测试层级 | 测试方法 | 通过标准 |
|---------|---------|---------|
| **单元测试** | 各Layer独立测试，固定随机种子，与PyTorch原版逐元素对比 | 误差 ≤ 1e-9 |
| **集成测试** | 完整ECFormer-ECD模型前向，与作者Release权重对比 | 误差 ≤ 1e-8 |
| **静态图测试** | `paddle.jit.to_static`编译通过，推理结果一致 | 输出差异 ≤ 1e-7 |

### 6.2 验收标准

1. **功能完整性**：所有原ECDFormer底层Layer均完成Paddle迁移，无功能缺失；
2. **精度达标**：在`float64`精度下，各Layer输出与原仓库差异≤1e-10；
3. **静态图支持**：基于这些Layer构建的ECFormer模型可成功执行动转静；

---

## 七、影响面

### 7.1 对用户的影响

**无影响**。本设计是ECFormer模型的底层基础设施，不暴露任何新增API给最终用户。用户通过`ECFormerECD`、`ECFormerIR`等高阶模型接口调用，无需感知Layer层实现细节。

### 7.2 对二次开发用户的影响

**正面影响**。二次开发用户可以：
- 直接复用`AtomEncoder`、`GINConv`等Layer构建新的分子图模型；
- 基于本方案的设计模式，将其他PyTorch Geometric模型迁移至Paddle。

**暴露的API**（均通过`ppmat.models.ecformer.layers`导出）：
- `AtomEncoder`, `BondEncoder`
- `RBF`, `BondFloatRBF`, `BondAngleFloatRBF`
- `GINConv`

### 7.3 对框架架构的影响

**无侵入**。本方案所有代码均放置在`ppmatmodels\ecformer\layers`目录下，不修改Paddle核心框架代码，不破坏PaddleMaterials既有项目结构。

### 7.4 与其他框架的对比优势

| 框架 | 输入格式 | 静态图支持 | ONNX导出 | 精度对齐 |
|------|---------|-----------|----------|---------|
| PyTorch ECDFormer（原版） | `Data`对象 | ❌ 不支持 | ❌ 困难 | - |
| **Paddle ECFormer（本方案）** | **纯Tensor** | ✅ **完美支持** | ✅ **支持** | ✅ **1e-10** |

### 7.6 其他风险

**无**。本方案所有代码已完成开发与测试，风险已完全释放。

---

## 名词解释

| 术语 | 解释 |
|------|------|
| ECDFormer | 原始PyTorch实现的分子ECD/IR谱图预测模型，发表于Nature Computational Science 2025 |
| ECFormer | 本方案在PaddlePaddle框架下对ECDFormer的迁移实现 |
| RBF | Radial Basis Function，径向基函数，用于连续值特征编码 |
| GINConv | Graph Isomorphism Network Convolution，图同构网络卷积层 |
| MessagePassing | 图神经网络的消息传递基类，`paddle_geometric`提供 |

---

## 附件及参考资料

1. ECDFormer原始仓库：https://github.com/HowardLi1984/ECDFormer
2. ECDFormer论文：Li, H. et al. Decoupled peak property learning for efficient and interpretable ECD spectra prediction. Nature Computational Science, 2025.
3. PaddlePaddle Issue #77754：DataLoader cannot return non-Tensor data when using multi-process mode. https://github.com/PaddlePaddle/Paddle/issues/77754
4. Paddle Geometric文档：https://github.com/PaddlePaddle/Paddle-Geometric
5. 本方案代码位置：`PaddleMaterials/ppmat/models/ecformer/layers/`
