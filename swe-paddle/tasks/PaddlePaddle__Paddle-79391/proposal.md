# 任务提案：PaddlePaddle__Paddle-79391

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79391`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79391
- PR 标题：`[API Compatibility] enhance paddle.enable_compat -part`
- `base_commit`：`ad1d2d4df4731d62fe41263e276bb7d7f30e16e7`（squash 合入 commit `718431cf276c9bd32c089ee2daf6cc7d54af2aa8` 的第一父提交）
- merged 时间：2026-07-22 13:08:19 +0800
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

Paddle 原有 `paddle.enable_compat()` 主要通过 Torch import proxy 把 `import torch` 映射到 Paddle，默认不会改变 `paddle.*` 的原生语义。该 PR 增加显式的 `level=2`，让已经存在的 `paddle.compat.*` PyTorch 对齐 API 可以通过 `paddle.*` 和相应的 Tensor 方法使用，同时让 Paddle 内部调用继续走原生实现，避免兼容参数和返回值影响框架自身。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自 Paddle 已合入的 API Compatibility 工作，目标是支持代码转换和用户代码直接使用 PyTorch 风格的 Paddle API，同时保持既有 `enable_compat()` 调用的行为不变。
- **代表性**：样本覆盖 Python API 兼容、`sys.meta_path` import hook、运行时命名空间修改、调用方感知分发、类代理与 metaclass、Tensor method 绑定、全局状态恢复以及 Paddle 内部组合算子回归。
- **边界清楚**：`level=1` 仍只启用原有 Torch proxy；只有 `level=2` 才安装 Paddle 命名空间分发。分发只处理 `paddle.compat` 模块 `__all__` 中、且在目标 `paddle` 模块中已经存在同名属性的公开 API；兼容包独有的符号不会被直接添加到 Paddle 命名空间。外部调用走 compat 实现，Paddle 内部调用走 native 实现。
- **非平凡性**：任务不是简单的 `setattr` 别名。实现需要保存并恢复原始函数、类和 Tensor 方法，处理函数签名、类构造、继承关系和类型判断，还要兼顾重复启用、scope、Torch 根 API 注册以及 Paddle 内部组合算子的原生参数语义。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, import_system, metaprogramming, testing]`

## 5. 验证思路

- 目标测试文件 / 命令：
  - `test/compat/test_compat_namespace_aliased.py`
  - `test/compat/test_compat_level2_internal_composites.py`
  - 直接运行：`python test/compat/test_compat_namespace_aliased.py` 和 `python test/compat/test_compat_level2_internal_composites.py`
  - 可用现有 `test/compat/test_torch_proxy.py`、`test/compat/test_torch_proxy_mixed.py` 作为 Torch proxy 回归测试。
- 修复前预期：在 base 上应用测试补丁后，新增测试无法导入 `paddle.compat.api_dispatch`，或因 `enable_compat(level=2)`、Paddle 命名空间 alias、Tensor 方法分发和内部组合行为尚未实现而失败。
- 修复后预期：应用 PR 的真实代码改动后，level 2 下外部调用得到 PyTorch 对齐的函数、类和 Tensor 方法行为；level 1 保持原生 Paddle 命名空间；关闭兼容模式后原始对象恢复；内部组合算子继续使用 native 实现；新增测试通过。
- F2P 候选：PR 新增的 `test_compat_namespace_aliased.py` 和 `test_compat_level2_internal_composites.py`，覆盖 level 参数、公开 API alias、函数分发、类构造、Tensor 方法、scope、生命周期和内部组合算子。
- P2P 候选：`test_torch_proxy.py`、`test_torch_proxy_mixed.py` 及同目录已有 Torch proxy 测试，检查 level 1 的既有 import proxy 和 scope 行为没有回归。

## 6. 环境与资源

- 是否能提供 Docker：原 PR 在当前场内开发机的容器环境中完成修改、构建和测试；当前 proposal 未提供可复用的 Docker 镜像。
- Dockerfile 或镜像地址：无公开地址。
- Paddle 来源：Paddle source build。该 PR 的代码改动为纯 Python；目标 verifier 可以复用与 base 兼容的 native 构建并替换 Python 源码，不要求为本任务重新编译 C++、CUDA 或其他 native 扩展。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不适用。
- 已验证构建环境：当前场内开发容器为 Linux x86_64，使用 Python 3.10.13 和 CMake 3.18.0，CUDA 12.6，cudnn 8.9.0
- 硬件：目标 verifier 使用 CPU 即可；当前厂内环境可见 NVIDIA Tesla V100-SXM2-32GB GPU，但本任务的目标测试不依赖 GPU。
- patch 类型：纯 Python。

## 7. 风险与注意事项

- `level=2` 会修改进程内的 `paddle.*`、`paddle.Tensor` 和 Torch import proxy 状态。测试必须成对管理 `enable_compat` / `disable_compat`，避免状态泄漏影响后续用例。
- 调用方判断依赖调用栈中的模块名。新增或调整兼容 API 时，需要同时确认外部调用的 PyTorch 语义和 Paddle 内部调用的原生参数、默认值没有互相污染。
- 类代理涉及 metaclass、继承和 `isinstance` / `issubclass`，普通函数测试不足以覆盖这部分行为，必须保留类构造和用户子类测试。
- scope 只限制 Torch proxy 的导入范围；最终实现中 `level=2` 的 Paddle 命名空间分发仍会安装，因此 scope 与 namespace alias 的边界需要单独验证。
- 可用运行时如果与精确 base 相差较大，可能在导入、生成模块或 Python/native 接口上产生无关失败。应优先使用 base 对应的构建，或在隔离环境中确认实际加载的 Paddle 包和测试文件路径。
- 该 PR 不包含 C++、CUDA、kernel、infermeta 或 native binding 修改；验证重点是 Python API 语义和全局兼容状态管理。
