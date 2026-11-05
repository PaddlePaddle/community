# 增强 paddle.metric.Metric 基类：引入 TorchMetrics 风格的指标基础设施

|任务名称 | 增强 paddle.metric 指标系统 |
|---|---|
|提交作者 | PlumBlossomMaid |
|提交时间 | 2026-05-29 |
|版本号 | V2.0 |
|依赖飞桨版本 | develop |
|文件名 | 20260529_api_design_for_metric_enhancement.md |

# 一、概述

## 1、相关背景

飞桨的 `paddle.metric.Metric` 基类自设计以来一直保持极简风格——仅定义了 `reset()`、`update()`、`accumulate()`、`name()` 四个抽象方法和一个可选的 `compute()` 方法，基类本身不继承 `paddle.nn.Layer`，也不提供任何状态管理、分布式同步、序列化或指标组合能力。

在实际项目中（如 PaddleMaterials），开发者不得不自行实现分布式归约（`_all_reduce_sum_`）、状态管理、`__call__` 封装等基础设施，导致大量重复代码。这并非个别项目的特殊需求，而是整个飞桨指标系统的结构性缺失。

通过对 PaddlePaddle、MindSpore、TorchMetrics 三大框架的 Metric 基类进行逐维度对比，我们发现：**差距不是一两个功能点的缺失，而是设计理念上的代差**。Paddle 和 MindSpore 的基类停留在"定义接口"的层面，而 TorchMetrics 已经进化为"提供基础设施"的工业化基类。

相关 Issue：
- [Issue #78078](https://github.com/PaddlePaddle/Paddle/issues/78078)：关于 `paddle.Model` 设计问题的讨论，官方已确认 loss 处理上的设计问题
- [Issue #78079](https://github.com/PaddlePaddle/Paddle/issues/78079)：关于 `paddle.metric.Metric` 是否设计过于简洁的讨论

## 2、功能目标

将 `paddle.metric.Metric` 从一个极简 ABC 升级为功能完备的指标基础设施，具体包括：

1. **继承 `paddle.nn.Layer`**：自动获得设备管理、参数序列化能力
2. **声明式状态管理**：通过 `declare()` 方法（兼容别名 `add_state`）集中注册状态变量，基类自动处理 reset、设备迁移和分布式归约
3. **分布式训练支持**：通过 `dist_reduce_fn` 参数（兼容别名 `dist_reduce_fx`）声明式指定归约策略，内置 `sync()`/`unsync()`/`sync_context()` 机制
4. **统一接口**：`forward()` 作为一步到位的统一入口（update + compute），替代繁琐的三步调用
5. **指标组合**：通过运算符重载实现指标组合（`CompositionalMetric`），支持 `+`、`-`、`*`、`/` 等操作
6. **批量指标管理**：提供 `MetricCollection` 支持多指标统一管理和 compute group 优化
7. **丰富的指标库**：新增 40+ 分类和回归指标，均采用任务分派架构（Binary/Multiclass/Multilabel）

## 3、意义

- **消除代差**：将飞桨指标系统从"手工作坊"升级为"工业化标杆"，对齐 TorchMetrics 的设计水平
- **降低用户开发成本**：开发者不再需要手写分布式归约、状态管理、序列化等基础设施代码，只需通过 `declare()` 声明状态即可
- **与 PyTorch 生态对齐**：TorchMetrics 是 PyTorch 生态中最成熟的指标库，本次升级使飞桨用户获得同等的开发体验
- **统一代码风格**：消除各项目中重复造轮子的现象（如 PaddleMaterials 中的 `_all_reduce_sum_`）
- **提升 GPU 利用率**：旧实现中 `Accuracy` 等指标将 Tensor 转为 numpy 回 CPU 计算，新实现保持 Tensor 在 GPU 上

# 二、飞桨现状

当前 `paddle.metric.Metric`（`python/paddle/metric/metrics.py`，约 900 行）的现状：

```python
class Metric(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def compute(self, *args: Any) -> Any:
        return args
```

**主要问题：**

| 问题 | 影响 |
|------|------|
| 不继承 `nn.Layer` | 无设备管理、无 `state_dict`/`load_state_dict` 序列化 |
| 无 `add_state`/`declare` 机制 | 每个子类手动管理状态变量和 `reset`，容易遗漏 |
| 无分布式支持 | 基类完全不处理多卡场景，用户需自行实现 `all_reduce` |
| 无 `forward()` 统一入口 | 用户必须三步走：`compute` → `update` → `accumulate`，代码冗长 |
| 内置指标回 CPU 计算 | `Accuracy` 等指标将 Tensor 转为 numpy，浪费 GPU 加速 |
| 无指标组合能力 | 无法将多个指标组合为新指标 |
| 无序列化支持 | 断点续训时指标状态丢失 |

**真实案例——PaddleMaterials 的"用了如用"体验：**

在 PaddleMaterials 项目（`ppmat/metrics/diffnmr_metric.py`）中，虽然继承了 `paddle.metric.Metric`，但基类几乎什么都没提供：
- `reset` 要自己实现
- `update` 要自己实现
- `accumulate` 要自己实现
- `name` 要自己实现
- 分布式归约要自己写 `_all_reduce_sum_`
- 为简化使用甚至自己封装了 `__call__` 方法
- 每个自定义指标都要重复写 `_sum`/`_n` 模式

这完美印证了"用了如用"——用户既要承受继承的约束，又得不到任何实质帮助。

# 三、业内方案调研

## 1、TorchMetrics（PyTorch 生态）—— "工业化标杆"

[TorchMetrics](https://github.com/Lightning-AI/torchmetrics) 是 PyTorch 生态中最成熟的指标库，由 Lightning 团队维护，被广泛用于工业界和学术界。

**核心设计哲学：** 继承 `nn.Module`，将框架能力（设备管理、序列化）与指标需求深度融合。用户只需关注核心计算逻辑（`update`/`compute`），其余一切交给基类。

**关键特性：**

```python
class Metric(nn.Module):
    def __init__(self, dist_reduce_fx="sum", sync_on_compute=True, ...):
        super().__init__()
        self._defaults: dict = {}
        self._reductions: dict = {}
        self._persistent: dict = {}

    def add_state(self, name, default, dist_reduce_fx=None, persistent=True):
        """声明式状态注册：自动处理 buffer 注册、reset、设备迁移、分布式归约"""

    def forward(self, *args, **kwargs):
        """update + compute 一步到位，支持两种同步策略"""

    def reset(self):
        """自动重置所有通过 add_state 注册的状态"""

    def sync(self) / unsync() / sync_context():
        """分布式同步控制"""

    # 继承自 nn.Module：state_dict / load_state_dict / to(device) / register_buffer
```

**指标规模：** 100+ 内置指标，按领域分目录（classification、regression、image、text、audio、retrieval、detection 等 15 个子包），配套 functional 无状态函数版本。

**高级能力：**
- `MetricCollection`：批量指标管理，compute group 优化（共享 `forward` 调用）
- `CompositionalMetric`：运算符重载实现指标组合（`+`、`-`、`*`、`/`、`**`、`abs()`、`==`、`>`、`<`）
- `WrapperMetric`：ClasswiseWrapper、BootStrapper、MinMaxMetric 等包装器
- 完整的序列化支持（`state_dict`/`load_state_dict`，可控制状态持久化）
- Pickle 支持、`clone()` 方法

## 2、MindSpore（华为）—— "尽职但传统"

MindSpore 的 `mindspore.nn.Metric` 基类提供了清晰的生命周期管理，但在基础设施层面与 Paddle 一样缺失。

**核心设计：**

```python
class Metric(ABC):
    def __init__(self):
        self._indexes = None  # 支持输入参数重排

    @abstractmethod
    def clear(self): ...     # 类似 reset

    @abstractmethod
    def update(self, *inputs): ...

    @abstractmethod
    def eval(self): ...      # 类似 accumulate

    def set_indexes(self, indexes):
        """可重新排列 update 输入参数顺序"""
```

**亮点：**
- `set_indexes`：允许用户指定 `update` 输入的映射关系，增加灵活性
- 清晰的 `clear`/`update`/`eval` 三步生命周期

**不足：**
- 不继承 `nn.Cell`（MindSpore 的 `nn.Module` 等价物），无设备管理和序列化
- 无 `add_state` 机制，状态完全手动管理
- 无分布式自动同步
- 无指标组合能力
- 文档示例停留在 NumPy 层面，未展示 Tensor 操作

## 3、TensorFlow/Keras —— "务实但保守"

Keras 通过 `tf.keras.metrics.Metric` 提供基本的指标框架。

**核心设计：**

```python
class Metric:
    def __init__(self, name=None, dtype=None):
        self._variables = []

    def add_weight(self, name, shape=(), ...):
        """通过 add_weight 管理状态变量"""

    def update_state(self, y_true, y_pred, sample_weight=None): ...

    def result(self): ...
```

**亮点：**
- `add_weight` 提供了基本的状态管理（类似 `add_state`）
- 与 TensorFlow 生态深度集成

**不足：**
- 不支持分布式自动同步（需用户手动调用 `strategy.run`）
- 无指标组合能力
- 无 `MetricCollection` 批量管理
- 设备管理依赖 TensorFlow 的 placement 机制，不够显式

# 四、对比分析

## 1、三框架 Metric 基类核心功能对比

| 功能维度 | **PaddlePaddle** | **MindSpore** | **TorchMetrics (Lightning)** |
| :--- | :--- | :--- | :--- |
| **基类继承** | 独立的 `abc.ABCMeta` | 独立的 `abc.ABCMeta` | **继承自 `nn.Module`**，与框架深度集成 |
| **核心抽象方法** | `reset`, `update`, `accumulate`, `name` | `clear`, `update`, `eval` | `update`, `compute` |
| **状态管理** | **完全手动**：用户在子类定义变量，手动实现 `reset` | **半手动**：用户在子类定义变量，手动实现 `clear` | **声明式 (`add_state`)**：注册后，基类自动处理重置、设备迁移、分布式归约 |
| **分布式支持** | **无**：用户需自己实现 `_all_reduce_sum_` | **无**：基类未提供内置支持 | **完备**：通过 `dist_reduce_fx` 声明式指定归约策略，内置 `sync`/`unsync` 机制 |
| **设备管理** | **无**：用户需手动将 Tensor 移至设备 | **无**：示例中使用 `.asnumpy()` 转回 CPU | **自动**：继承自 `nn.Module`，所有状态随 `.to(device)` 自动迁移 |
| **序列化支持** | **无**：未集成 `state_dict` | **无**：未集成 `state_dict` | **完备**：继承自 `nn.Module`，支持 `state_dict`/`load_state_dict`，可控制状态持久化 |
| **组合能力** | **无** | **无** | **强大**：支持 `MetricCollection` 和运算符重载（如 `+`、`/`） |
| **易用性接口** | **仅三步走**：`update` → `accumulate` | **仅三步走**：`update` → `eval` | **双模式**：支持三步走，也支持一步到位的 `forward` |
| **输入参数灵活性** | **无** | **支持 `set_indexes`**：可重新排列 `update` 输入参数顺序 | **支持 `_filter_kwargs`**：自动过滤传递给 `update` 的关键字参数 |
| **文档与示例** | **简陋**：示例陈旧，仍在使用 NumPy | **基础**：示例清晰，但停留在 NumPy 层面 | **工业级**：完整的 API 参考，丰富的实际用例，涵盖所有参数 |

## 2、结论：差距是全方位的

**PaddlePaddle："躺平式"基类** — 定义了一个"空壳"，把所有脏活累活（状态管理、设备、分布式、序列化）都推给用户。用户只是"继承"了一个抽象类，但没有获得任何实质性帮助。这完美印证了 PaddleMaterials 项目的观察。

**MindSpore："尽职但传统"的基类** — 定义了清晰的 `clear`/`update`/`eval` 生命周期，是一个功能完整的基类。但它同样没有解决设备管理、分布式同步、序列化等现代框架的痛点，用户仍需自行处理这些复杂逻辑。其 `set_indexes` 功能虽然独特，但需要配合装饰器使用，显得不够优雅。

**TorchMetrics："工业化"的基类** — 通过继承 `nn.Module` 和引入 `add_state` 声明式状态管理，将框架本身的能力（设备、序列化）与指标需求完美结合。用户从"我必须手动管理一切"转变为"我只管核心计算（`update`/`compute`），其余交给基类"。

**这不仅是功能的堆砌，更是设计哲学的胜利。** Paddle 的问题不是一个或几个内置指标实现得不好，而是整个 `Metric` 基类的设计理念就落后了。

## 3、本方案定位

本方案的目标正是将 Paddle 的 `Metric` 基类从 **"用户必须自己动手的抽象类"** 彻底转变为 **"替用户搞定一切的实干家"**。继承 `nn.Layer` + `declare` 这套组合拳，将直接对齐业界最先进的设计，让 Paddle 的指标模块完成从"手工作坊"到"工业化标杆"的跨越。

| 特性 | 飞桨现状 | **本方案** | TorchMetrics |
|------|---------|-----------|-------------|
| 基类继承 | `ABCMeta` | **`ABC` + `nn.Layer`** | `nn.Module` |
| 状态管理 | 手动 | **`declare`（别名 `add_state`）** | `add_state` |
| 分布式同步 | 无 | **`dist_reduce_fn`（别名 `dist_reduce_fx`）** | `dist_reduce_fx` |
| 统一入口 | 三步走 | **`forward()`** | `forward()` |
| 序列化 | 无 | **继承自 `nn.Layer`** | 继承自 `nn.Module` |
| 指标组合 | 无 | **运算符重载** | 运算符重载 |
| 批量管理 | 无 | **`MetricCollection`** | `MetricCollection` |
| 内置指标数 | 4 | **40+（分类 + 回归）** | 100+ |
| 指标计算位置 | CPU（numpy） | **GPU（tensor）** | GPU（tensor） |

**命名改进（相对 TorchMetrics 的优化）：**

| TorchMetrics 原名 | 本方案主名 | 兼容别名 | 改进理由 |
|---|---|---|---|
| `add_state()` | `declare()` | `add_state` | 语义更清晰："声明"一个状态变量 |
| `dist_reduce_fx` | `dist_reduce_fn` | `dist_reduce_fx` | `fn`（function）比 `fx`（effect）更符合 Python 命名惯例 |

# 五、设计思路与实现方案

## 1、主体设计思路与折衷

### 整体全貌

```
paddle.metric/
├── __init__.py              # 公开 API + 懒加载
├── metric.py                # Metric 基类 + CompositionalMetric
├── collections.py           # MetricCollection
├── aggregation.py           # MeanMetric, SumMetric, ...
├── classification/          # 分类指标（任务分派架构）
│   ├── accuracy.py          # Accuracy -> Binary/Multiclass/Multilabel
│   ├── stat_scores.py       # StatScores 基类
│   ├── precision_recall.py
│   ├── f_beta.py
│   └── ...
├── regression/              # 回归指标
│   ├── mse.py
│   ├── mae.py
│   └── ...
├── functional/              # 无状态函数版本（镜像 class 结构）
│   ├── classification/
│   └── regression/
├── wrappers/                # 包装器
│   ├── classwise.py
│   └── ...
└── utils/                   # 工具函数
```

### 主体设计具体描述

**Metric 基类核心设计：**

```python
class Metric(ABC, paddle.nn.Layer):
    # 类属性
    full_state_update: bool | None = None
    higher_is_better: bool | None = None
    is_differentiable: bool | None = None

    def __init__(
        self,
        name: str | None = None,
        dist_reduce_fn: str | Callable = "sum",
        sync_on_compute: bool = True,
        dist_sync_on_step: bool = False,
        compute_with_cache: bool = True,
        compute_on_cpu: bool = False,
        **kwargs,  # 接受 dist_reduce_fx 作为别名
    ):
        # 别名处理
        if "dist_reduce_fx" in kwargs:
            dist_reduce_fn = kwargs.pop("dist_reduce_fx")

        # 状态注册表
        self._defaults: dict[str, list | paddle.Tensor] = {}
        self._reductions: dict[str, str | Callable | None] = {}
        self._persistent: dict[str, bool] = {}

        # 包装 update/compute
        self.update = self._wrap_update(self.update)
        self.compute = self._wrap_compute(self.compute)

        # 别名
        self.add_state = self.declare

    def declare(self, name, default, dist_reduce_fn=None, persistent=True):
        """声明式状态注册。支持 add_state 作为别名。"""

    def reset(self):
        """自动重置所有通过 declare 注册的状态。"""

    def forward(self, *args, **kwargs):
        """update + compute 一步到位。"""

    def sync(self) / unsync() / sync_context():
        """分布式同步控制。"""
```

### 主体设计选型考量

1. **继承 `ABC` + `nn.Layer`（双继承）**：
   - `ABC` 提供抽象方法机制，确保子类必须实现 `update` 和 `compute`
   - `nn.Layer` 提供设备管理和序列化
   - TorchMetrics 使用 `nn.Module` + `ABC`，本方案对应使用 `nn.Layer` + `ABC`
   - 这是对齐 TorchMetrics 的核心举措，也是弥补与 MindSpore 差距的关键一步

2. **`declare` 作为主名称**：
   - 比 `add_state` 更简洁，语义更清晰（"声明"一个状态变量）
   - 保留 `add_state` 别名确保 TorchMetrics 用户零迁移成本

3. **`dist_reduce_fn` 作为主参数名**：
   - `fn`（function）比 `fx`（effect）更符合 Python 命名惯例
   - 保留 `dist_reduce_fx` 别名通过 `**kwargs` 实现

## 2、关键技术点/子模块设计与实现方案

### 2.1 状态管理机制（`declare`）

```python
def declare(self, name: str, default, dist_reduce_fn=None, persistent=True):
    if isinstance(default, Tensor):
        self.register_buffer(name, default)
    elif isinstance(default, list):
        setattr(self, name, list(default))
    else:
        raise ValueError(f"default must be Tensor or list, got {type(default)}")

    self._defaults[name] = default
    self._reductions[name] = dist_reduce_fn or self._default_dist_reduce_fn
    self._persistent[name] = persistent
```

**选型考量：**
- Tensor 默认值通过 `register_buffer` 注册，自动获得设备迁移和序列化能力
- List 默认值用于需要动态长度的场景（如 retrieval 指标收集所有预测结果）
- `dist_reduce_fn` 支持 `"sum"`、`"mean"`、`"cat"`、`"min"`、`"max"` 或自定义 callable
- 对比 Paddle/MindSpore 的手动管理，这是一项根本性的改进

### 2.2 分布式同步机制

```python
def _sync_dist(self):
    for attr, reduction_fn in self._reductions.items():
        if reduction_fn is None:
            continue
        current_val = getattr(self, attr)
        if isinstance(current_val, Tensor):
            gathered = self._gather_all_tensors(current_val)
            setattr(self, attr, reduction_fn(gathered))
```

**选型考量：**
- 使用 `paddle.distributed.all_gather` 而非 `all_reduce`，因为 gather 后可以灵活选择归约方式
- 支持自定义 `dist_sync_fn` 以适配不同的分布式后端
- 填补了 Paddle 和 MindSpore 在分布式指标同步上的空白

### 2.3 接口统一（`forward`）

```python
def forward(self, *args, **kwargs):
    # 两种策略：
    # 1. full_state_update：先同步 -> update -> compute（安全但慢）
    # 2. reduce_state_update：update -> 同步 -> compute（快但仅适用于可归约指标）
    self.update(*args, **kwargs)
    return self.compute()
```

**选型考量：**
- `full_state_update` 是默认安全模式，适用于所有指标
- `reduce_state_update` 是优化模式，适用于状态可独立归约的指标（如 sum、mean）
- 解决了旧版三步走（`update` → `accumulate`）的冗余问题

### 2.4 指标组合（`CompositionalMetric`）

通过运算符重载，任意两个指标可以组合为新指标：

```python
class CompositionalMetric(Metric):
    def __init__(self, operator, metric_a, metric_b=None):
        # operator: add, sub, mul, div, etc.
        # metric_a, metric_b: Metric or scalar

    def update(self, *args, **kwargs):
        self.metric_a.update(*args, **kwargs)
        if self.metric_b is not None:
            self.metric_b.update(*args, **kwargs)

    def compute(self):
        return self.operator(self.metric_a.compute(), ...)
```

支持：`+`、`-`、`*`、`/`、`**`、`abs()`、`==`、`>`、`<` 等操作。

**选型考量：**
- Paddle 和 MindSpore 均无此能力，用户只能手动编写组合逻辑
- TorchMetrics 的实践证明，运算符重载是最优雅的组合方式

### 2.5 分类指标任务分派架构

```python
class Accuracy:
    def __new__(cls, task: str = "multiclass", **kwargs):
        if task == "binary":
            return BinaryAccuracy(**kwargs)
        elif task == "multiclass":
            return MulticlassAccuracy(**kwargs)
        elif task == "multilabel":
            return MultilabelAccuracy(**kwargs)
```

每个任务类型继承对应的 `StatScores` 变体，仅重写 `compute()` 方法。

### 2.6 Functional API

每个指标都有对应的无状态函数版本，位于 `paddle.metric.functional/` 目录，镜像 class 结构：

```python
# paddle.metric.functional.classification.accuracy
def accuracy(preds, target, task, num_classes, ...):
    # 纯函数，无状态
    return _accuracy_compute(stat_scores_update(...), ...)
```

## 3、主要影响的模块接口变化

### 核心接口变化

| 接口 | 旧版 | 新版 | 兼容性 |
|------|------|------|--------|
| 基类 | `Metric(ABCMeta)` | `Metric(ABC, nn.Layer)` | 子类仍需实现 `update`/`compute` |
| 状态管理 | 无 | `declare()` / `add_state()` | 新增 |
| `reset()` | 子类手动实现 | 基类自动重置 | 子类可覆盖 |
| `update()` | 抽象方法 | 抽象方法 + `_wrap_update` 包装 | 不变 |
| `compute()` | 抽象方法 | 抽象方法 + `_wrap_compute` 包装 | 不变 |
| `accumulate()` | 抽象方法 | 删除 | **BC Breaking** |
| `name()` | 抽象方法 | 属性 | **BC Breaking** |
| `forward()` | 无 | update + compute | 新增 |
| `state_dict()` | 无 | 继承自 `nn.Layer` | 新增 |
| `Accuracy` | top-k 风格 | 任务分派（Binary/Multiclass/Multilabel） | **BC Breaking** |

### 对框架各环节的影响

| 环节 | 影响 |
|------|------|
| 网络定义 | 无影响 |
| 底层数据结构 | 无影响 |
| OP | 无影响 |
| 数据 IO | 无影响 |
| 执行 | 无影响 |
| 分布式 | 新增指标自动同步能力，不影响现有分布式逻辑 |
| 模型保存 | 新增指标 `state_dict` 支持，可通过 `paddle.save`/`paddle.load` 序列化 |
| 预测部署 | 无影响 |

# 六、测试和验收的考量

1. **单元测试**：覆盖 `declare`/`add_state`、自动 `reset`、`forward`、`clone`、`pickle`、`state_dict` 序列化、`CompositionalMetric` 组合、`MetricCollection` 批量管理
2. **分布式测试**：DDP 模式下多卡同步验证
3. **指标正确性**：所有内置指标与参考实现（NumPy）对比，误差在容许范围内
4. **向后兼容**：`add_state` 和 `dist_reduce_fx` 别名可用性验证
5. **pre-commit**：通过 ruff check、ruff format、typos、copyright_checker 等所有检查

# 七、影响面

## 对用户的影响

- **BC Breaking**：旧的 `accumulate()` 和 `name()` 方法被移除，需要迁移到 `compute()` 和 `name` 属性
- **BC Breaking**：`Accuracy` 的构造方式从 `Accuracy(topk=(1,))` 变为 `Accuracy(task="multiclass", num_classes=5)`
- **正面影响**：新用户获得更强大的指标基础设施，无需重复造轮子

## 对二次开发用户的影响

新增暴露给二次开发用户的 API：
- `Metric.declare()` / `Metric.add_state()`：声明式状态管理
- `Metric.forward()`：统一入口
- `Metric.sync()` / `Metric.unsync()` / `Metric.sync_context()`：分布式控制
- `MetricCollection`：批量指标管理
- `CompositionalMetric`：指标组合
- 所有运算符重载（`+`、`-`、`*`、`/` 等）

## 对框架架构的影响

- `paddle.metric` 从单文件模块扩展为多目录模块
- 不影响 `paddle.nn`、`paddle.optimizer` 等其他模块
- `nn.Layer` 继承使指标与 Paddle 生态更紧密集成

## 对性能的影响

- 正面：指标计算保持在 GPU 上（旧版回 CPU）
- 正面：`MetricCollection` 的 compute group 优化减少重复 `forward` 调用
- 中性：`sync_on_compute` 可配置关闭以避免不必要的同步

## 对比业内深度学习框架的差距与优势的影响

- 消除与 TorchMetrics 的代差，从"躺平式基类"跃升至"工业化基类"
- 命名改进（`declare`/`dist_reduce_fn`）在保持兼容的同时提供更清晰的 API
- 相比 MindSpore 的指标系统，飞桨将获得领先优势

## 其他风险

- 大量新代码引入可能带来潜在 bug，需充分测试
- 依赖 `scipy` 等第三方库的指标（如 AUROC、AUC）需确认依赖可用性

# 八、排期规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| 第一阶段 | Metric 基类 + CompositionalMetric + MetricCollection + aggregation | ✅ 已完成 |
| 第一阶段 | 分类指标全套（28 个）+ functional 镜像 | ✅ 已完成 |
| 第一阶段 | 回归指标（21 个）+ functional 镜像 | ✅ 已完成 |
| 第一阶段 | wrappers（10 个）+ utils | ✅ 已完成 |
| 第一阶段 | 测试 + pre-commit 通过 | ✅ 已完成 |
| 第二阶段 | audio / text / image / video 等领域指标 | 后续 |
| 第二阶段 | DDP 分布式集成测试 | 后续 |

# 名词解释

| 术语 | 说明 |
|------|------|
| `declare` | 声明式状态注册方法（兼容别名 `add_state`） |
| `dist_reduce_fn` | 分布式归约函数参数（兼容别名 `dist_reduce_fx`） |
| `forward` | 统一入口方法，等价于 update + compute |
| `CompositionalMetric` | 通过运算符组合多个指标得到的复合指标 |
| `MetricCollection` | 批量管理多个指标的容器 |
| `compute group` | MetricCollection 的优化机制，共享 update 调用 |
| functional API | 无状态的函数版本指标，与 class 版本一一对应 |
| 任务分派 | Accuracy 等指标通过 `task` 参数自动选择 Binary/Multiclass/Multilabel 实现 |

# 附件及参考资料

## 参考资料

- [TorchMetrics 源码](https://github.com/Lightning-AI/torchmetrics)
- [PaddleMetrics 第三方移植](https://github.com/PlumBlossomMaid/paddlemetrics)
- [MindSpore nn.Metric](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#metric)
- [TensorFlow Keras Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric)
- [Issue #78078: paddle.Model 设计问题讨论](https://github.com/PaddlePaddle/Paddle/issues/78078)
- [Issue #78079: paddle.metric.Metric 设计讨论](https://github.com/PaddlePaddle/Paddle/issues/78079)
- [PR: Enhance paddle.metric with TorchMetrics-style infrastructure](https://github.com/PaddlePaddle/Paddle/pull/79191)
