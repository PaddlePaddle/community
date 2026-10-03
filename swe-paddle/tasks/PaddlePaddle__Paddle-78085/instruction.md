# 让 Paddle 更安全地加载模型文件

## 详细描述

使用 `paddle.load` 加载模型参数或训练状态时，文件中的 Pickle 数据可能包含会在反序列化阶段执行的 Python 调用。如果模型文件来自不可信来源，加载过程可能在用户不知情的情况下执行系统命令或访问本地文件。

请改进 Paddle 的模型反序列化行为，使加载过程只接受模型和 checkpoint 所需的常见数据类型，并在遇到危险或未知对象时立即拒绝加载。正常的参数文件、NumPy 数组、嵌套容器和已有 state dict 应继续兼容，Paddle 中使用 Pickle 读取模型信息的主要入口也应保持一致的安全行为。

## 验收说明

- 包含系统命令、进程调用、动态代码执行或文件操作的恶意 Pickle 数据应在产生副作用前被拒绝。
- NumPy 数组、基础容器、`OrderedDict` 和常见模型 state dict 应能正常保存与加载。
- 现有 `paddle.save` / `paddle.load` 使用方式、模型参数恢复和大文件兼容路径应保持可用。

## 技术要求

- 熟悉 Python Pickle 的序列化、反序列化流程及 `__reduce__` 带来的安全风险。
- 了解 Paddle 模型参数、state dict 和 checkpoint 的保存与加载流程。
- 熟悉 Python 文件 I/O、异常处理和向后兼容性测试。
