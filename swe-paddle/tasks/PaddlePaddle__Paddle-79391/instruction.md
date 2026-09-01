# 让 `paddle.enable_compat()` 可以把 PyTorch 对齐的 compat API 接管到 `paddle.*`

## 详细描述

Paddle 已经提供了一批与 PyTorch 严格对齐的 API，放在 `paddle.compat.*` 下（例如 `paddle.compat.sort`、
`paddle.compat.nn.Linear`、`paddle.compat.nn.functional.softmax`）。它们和同名的原生 `paddle.*` API
在**相同位置上的参数类型相同**（第一个位置参数都是 Tensor，`axis` / `dim` 都是 int），因此无法在单次调用中
按参数类型判断调用方想要哪一套语义。

目前 `paddle.enable_compat()` 只做一件事：往 `sys.meta_path` 插入一个 finder，让 `import torch` 解析到
Paddle。它**不会**改变 `paddle.*` 本身的语义。这带来两个问题：

1. 由 PyTorch 代码「只改前缀」迁移过来的用户代码（`torch.sort(x, dim=-1)` → `paddle.sort(x, dim=-1)`）
   仍然会落到原生 `paddle.sort`，因为原生签名里没有 `dim=`，直接报错；`x.max(dim=1)` 这类 Tensor 方法同理。
2. `paddle.compat` 根包里的顶层函数（`sort`/`split`/`min`/`max`/`unique`/`median`/`nanmedian`/
   `allclose`/`equal`/`seed`/`slogdet`）在 torch import proxy 下也拿不到 compat 版本，`torch.sort` 会落到
   原生 `paddle.sort`。

需要给 `paddle.enable_compat()` 增加一个显式的兼容级别开关，让用户可以在进程级别选择「`paddle.*` 就是
torch 语义」。同时必须保证：**Paddle 框架自己的库代码仍然按原生语义运行**。Paddle 内部有大量组合实现会以
原生关键字调用这些同名符号（例如 `vsplit`/`hsplit`/`dsplit`/`tensor_split`/`chunk` 内部调用 `split`，
`quantile` 内部调用 `paddle.sort(x, axis)`，`nan_to_num` 内部调用 `paddle.equal`，
`histogram_bin_edges` 内部调用 `paddle.min`/`paddle.max`，`F.nll_loss`、`scaled_dot_product_attention`
的 math backend 内部调用 `F.softmax`）。如果这些内部调用被换成 compat 语义，框架自身会大面积出错。

## 验收说明

### 1. 级别开关

* `paddle.enable_compat()` 新增一个名为 `level` 的参数，**放在参数列表最后**，默认值为 `1`。
* `level=1` 与现有行为完全一致：只安装 torch import proxy，不改动 `paddle.*`。
* `level=2` 在 torch import proxy 之外，额外让 `paddle.*` 与 `paddle.Tensor` 的方法解析到 torch 对齐实现。
* `level` 取 `1`、`2` 以外的值时抛 `ValueError`，消息形如 `Unsupported level: 3. It should be 1 or 2.`，
  并且不能产生任何副作用（finder 不入 `sys.meta_path`，`paddle.*` 不被改动）。
* `paddle.use_compat_guard()` 的签名**不新增** `level` 参数；在 `level=2` 已启用的进程里，
  `use_compat_guard()` 进出前后当前生效的级别保持不变。

### 2. `level=2` 覆盖哪些符号

* 遍历 `paddle.compat` 及其子模块中声明了 `__all__` 的模块（**包含 `paddle.compat` 根包本身**），
  对每个公开符号 `X`，在对应的 `paddle` 侧模块（`paddle.compat.nn` → `paddle.nn`，
  `paddle.compat.nn.functional` → `paddle.nn.functional`，`paddle.compat.distributions` →
  `paddle.distributions`，根包 → `paddle`）上接管同名属性。
* **只接管目标模块中已经存在的同名属性。** compat 独有的符号不得被新增到 Paddle 命名空间：启用后
  `paddle.slogdet`、`paddle.nn.AvgPool1d/2d/3d`、`paddle.nn.BatchNorm1d/2d/3d`、
  `paddle.nn.MultiheadAttention` 仍然必须 `hasattr(...) == False`。
* `paddle.compat` 根包里那些同时也是 Tensor 方法的 API（`max`/`min`/`sort`/`split`/`unique`/…），
  对应的 `paddle.Tensor.<name>` 也要一起接管，使 `x.max(dim=1)` 得到 torch 语义。
* torch import proxy 侧：`level=1` 和 `level=2` 下都必须能拿到根包的 compat 实现，即
  `torch.sort is paddle.compat.sort`、`torch.min is paddle.compat.min`、
  `torch.unique is paddle.compat.unique`、`torch.slogdet is paddle.compat.slogdet`。

### 3. 调用方感知

* **外部调用**（非 `paddle` 包内的模块）拿到 compat 实现：`paddle.sort(t, dim=-1)` 返回带 `values` /
  `indices` 的具名元组；`paddle.split(t, 1, dim=0)` 返回 tuple；`paddle.allclose` / `paddle.equal`
  返回 Python `bool`；`paddle.seed()` 不接收参数并返回 int；`paddle.max(t, axis=1)` 因 torch 契约而抛
  `TypeError`；`paddle.nn.functional.softmax(x, axis=-1)` 同理抛 `TypeError`。
* **Paddle 内部调用**（调用栈上一层所在模块的 `__name__` 为 `paddle` 或以 `paddle.` 开头）拿到原生实现。
  `level=2` 下 `paddle.vsplit`/`hsplit`/`dsplit`/`tensor_split`/`chunk`/`quantile`/`nan_to_num`/
  `histogram_bin_edges`、`F.nll_loss(..., ignore_index=..., reduction="mean")`、
  `paddle.nn.TransformerEncoderLayer`/`MultiHeadAttention`/`LayerNorm`、以及 fp32 下
  `F.scaled_dot_product_attention` 的数值结果都必须与不开兼容模式时一致。
* 若干 compat 实现自身会回调同名的原生 API（`allclose`、`min`、`max`、`median`、`nanmedian`、`unique`、
  `seed`、`nn.functional.unfold`、`scaled_dot_product_attention`）。接管之后这些回调不能变成无限递归，
  也不能被 torch 契约的关键字校验拒绝。

### 4. 类的接管

* 已存在同名原生类的 compat 类（`nn.Unfold`、`nn.Linear`、`nn.Softmax`、`nn.AvgPool1D/2D/3D`、
  `nn.BatchNorm1D/2D/3D`、`nn.SmoothL1Loss`、`nn.MultiheadAttention`、`distributions.Categorical`）
  接管后同样要区分调用方：外部 `paddle.nn.Linear(2, 2)` 构造出的实例类型是 `paddle.compat.nn.Linear`，
  且传原生关键字 `weight_attr=False` 抛 `TypeError`；Paddle 内部调用得到原生类，可以正常接收 `weight_attr=`。
* 用户从接管后的类派生子类（`class MyLinear(paddle.nn.Linear)`）时走 torch 风格构造，
  `MyLinear(3, 4, bias=False)` 必须成立，且 `isinstance(m, paddle.nn.Linear)` 与
  `issubclass(MyLinear, paddle.nn.Linear)` 均为 True。
* **无论是否启用兼容模式**，每个 compat 类都要与其原生对应类保持单向类型关系：
  `isinstance(compat_cls(...), native_cls)` 为 True、`issubclass(compat_cls, native_cls)` 为 True，
  而 `isinstance(native_cls(...), compat_cls)` 为 False。注意 `paddle.compat.nn.MultiheadAttention`
  对应的原生类名是 `paddle.nn.MultiHeadAttention`（大小写不同）。

### 5. 需要暴露的接口约定（测试直接断言）

* 新增模块 `paddle.compat.api_dispatch`，其中的 `_PADDLE_NAMESPACE_SAVED` 是本次接管的注册表：
  未启用时 `len(...) == 0`，`enable_compat(level=2)` 后 `len(...) > 0`，`disable_compat()` 后回到 `0`。
* 被接管的函数对象暴露 `__compat_fn__` 属性指向对应的 compat 函数；被接管的类对象暴露 `__compat_cls__`
  属性指向对应的 compat 类。`level=1` 下 `paddle.sort` 不得带有 `__compat_fn__`。
* 被接管后的对象签名与 compat 侧一致：`inspect.signature(paddle.sort) == inspect.signature(paddle.compat.sort)`，
  `inspect.signature(paddle.nn.Linear) == inspect.signature(paddle.compat.nn.Linear)`。

### 6. 生命周期

* `disable_compat()` 必须把每一个被接管的模块属性和 Tensor 方法**按对象身份**还原
  （`getattr(paddle, "sort") is` 原来的函数对象）。
* `enable_compat(scope=..., level=2)` 下 `scope` 只约束 torch import proxy 的作用范围，
  `paddle.*` 的接管照常安装。
* 重复调用 `enable_compat()`（level 1）时 finder 会重复入栈，每次 `disable_compat()` 弹出一个，
  这一既有行为不能改变；接管注册表在完全关闭后必须清空。

## 技术要求

* 熟悉 Python 的 `sys.meta_path` import hook、模块属性运行时改写与全局状态的保存 / 还原。
* 熟悉调用栈帧（`sys._getframe`）以及 `functools.wraps` / `inspect.signature` 对可调用对象的影响。
* 熟悉 metaclass、`__instancecheck__` / `__subclasscheck__`、继承链改写与 `paddle.Tensor` 方法绑定。
* 了解 Paddle 中哪些组合算子是由同名的公开 API 拼装出来的。
