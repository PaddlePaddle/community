# 为 Paddle 框架 API 添加类型提示

| 领域         | 为 Paddle 框架 API 添加类型提示    |
| ------------ | ---------------------------------- |
| 提交作者     | megemini (柳顺)                    |
| 提交时间     | 2024-04-07                         |
| 版本号       | V1.0                               |
| 依赖飞桨版本 | develop 分支                       |
| 文件名       | 为 Paddle 框架 API 添加类型提示.md |

# 一、概述

## 1、相关背景

Python 在 3.5 版本通过 [PEP 484 – Type Hints](https://peps.python.org/pep-0484) 正式规范了 `类型提示` 功能，以帮助开发者提高代码质量，Python 目前 (`3.12` 版本) 已经完成的相关 `PEP` 有 `11` 个，具体可以参考 [Typing PEPs](https://peps.python.org/topic/typing/) 。经过前期的几个版本迭代，Python 的 `类型提示` 功能已经受到开发者的广泛认可。Paddle 目前支持的 Python 版本 `3.8` 已经可以较好的支持 `类型提示`，本文旨在阐述 Paddle 引入 `类型提示` 的可行性与具体方案。

本文档为 [【Hackathon 6th】Fundable Projects](https://github.com/PaddlePaddle/Paddle/issues/62908) 中 [为 Paddle 框架 API 添加类型提示（Type Hints）](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E4%B8%80%E4%B8%BA-paddle-%E6%A1%86%E6%9E%B6-api-%E6%B7%BB%E5%8A%A0%E7%B1%BB%E5%9E%8B%E6%8F%90%E7%A4%BAtype-hints) 的实现方案。

## 2、功能目标

正确完成 Paddle 开放 API 的类型标注，但不做 Paddle 类型标注的完备性说明。

> **说明**： 类型标注是个循序渐进的过程，且存在较多私有 API 与 c++ 接口，此次任务无法保证完成以上所有接口的类型标注，故此，不做 Paddle 类型标注的完备性说明。

### 2.1 typing 模块

为 Paddle 添加 `typing` 模块，并作为开放 API 的一部分。

功能特性类似：

- PyTorch 中的 `torch/types.py`
- Numpy 中的 [Typing (numpy.typing)](https://numpy.org/devdocs/reference/typing.html)

`typing` 模块包含 Paddle 中用到的特殊类型，如 `dtype`，`device` 等，具体实现可参考 @SigureMo 的 [paddle-stubs/_typing](https://github.com/cattidea/paddlepaddle-stubs/tree/main/paddle-stubs/_typing)。

具体需要实现：

- `typing` 模块代码，实现路径 `Paddle/python/paddle/typing/*`
- `typing` 模块文档
- `typing` 模块测试用例，实现路径 `Paddle/test/typing/*`

### 2.2 开放 API 的类型标注

比目标需要：

- 对现存 Paddle 代码的开放 API 进行类型标注。

    Paddle 目前开放 API 有 1500 个左右，需要分批次对其进行类型标注。可以参考 [paddlepaddle-stubs Roadmap](https://github.com/orgs/cattidea/projects/3/views/1) ，将 API 分为 `P1 ～ P5` 多个等级分批完成。

- 修改或对齐 API 中 docstring 的类型说明，与实际的标注类型。

此目标的完成，应保证主流 IDE，如 vscode，的类型提示功能可以正常使用。

此目标 **涉及**：

- Paddle 的开放 API
- API 中 docstring 的类型说明

此目标 **不涉及** ：

- 添加类型检查测试用例(`typing` 模块除外)

### 2.3 CI 流水线

更新 CI 流水线，对以下代码进行类型检查：

- `typing` 模块的测试用例
- 旧/新 API 的示例代码
- 旧/新 API 的测试用例

此目标 **涉及**：

- 静态类型检查 (`staic type checking`)。使用工具 [mypy](https://mypy.readthedocs.io/en/stable/)

此目标 **不涉及** ：

- 运行时类型检查 (`runtime type checking`)，如工具 [beartype](https://beartype.readthedocs.io/en/latest/)
- 整个 Paddle 包的类型检查。

### 2.4 文档建设

在 Paddle 的 `docs` 中添加文档 `《Paddle 中的类型提示》`，用以辅助：

- 开发 Paddle 的开发者，正确标注 API 中的类型
- 使用 Paddle 的开发者，正确使用 `typing` 模块

### 2.5 后续任务

后续可以完成

- `_C_ops` 的标注
- 私有 API 的类型标注
- Paddle 整个包的类型检查
- 添加运行时类型检查

## 3、意义

`Type Hints` 的引入，可以提升开发者的使用体验，并提升 Paddle 的代码质量。

# 二、飞桨现状

Paddle 目前没有 `Type Hints` 功能，但是存在第三方的尝试，如：

- [paddlepaddle-stubs](https://github.com/cattidea/paddlepaddle-stubs)
- [types-paddle](https://github.com/wj-Mcat/types-paddle/tree/master)

之前也有过对于添加类型提示功能的讨论，如:

- [Type Hinting for Tensor of Paddle](https://github.com/jiamingkong/community/blob/4bde11a8a861c8aae4bdb1284579e00d4799f7b9/rfcs/type-hinting/type_hinting_for_paddle_tensor.md)

以上均可作为本次方案的实现参考。

# 三、设计思路与实现方案

首先，明确几个概念：

- `type hint`
  
  `类型提示`，泛指 Python 中的数据类型标注与提示功能。

- `type annotation`

  `类型标注`，常与 `type hint` 混用，也可作为 `动词` 专门指代标注类型这个动作。

- `type checking`

  `类型检查`，检查类型标注的正确性，或数据在接口中传递时的类型正确性。

- `static type checking`

  `静态类型检查`，使用工具如 `mypy/pyright` 等，对特定代码或包进行类型标注的静态检查，此动作不依赖源代码的运行时状态。

  是目前广泛使用的类型检查方式，常与 `type checking` 混用。

- `runtime type checking`

  `动态类型检查`，使用工具如 `beartype`，在代码运行时保证数据类型的正确性。

  借助额外的类型支持包，如：
  
  - [nptyping](https://github.com/ramonhagenaars/nptyping) 之于 `numpy`
  - [jaxtyping](https://github.com/patrick-kidger/jaxtyping) 之于 `PyTorch`
  
  此类检查可以提供代码运行时的额外类型检查，如 `shape`，`dtype` 等信息。

其次，Paddle 或 PyTorch 对于类型检查的支持，与 `pure python` 的包实现方式不尽相同：

- 实现方式的差别

  一个包的开发逻辑，大体可以分为：

  - 底层支撑接口
  - 上层开放接口
  - 开发者的使用

  对于 `pure python` 的包，以上三者通常都是使用 Pyhton 完成开发的，以此，可以方便的使用工具进行类型检查。

  而对于 Paddle 和 PyTorch 此类 Python 混合 C++ 或其他语言的开发包，底层逻辑使用其他语言实现，上层接口通常只是使用 Python 将其暴露出来，接口间的调用较少，也就缺乏了检查开发包本身数据类型的动力。如：

  ``` python
  def abs(x):
    return _C_ops.abs(x)
  ```
  API 中大量采用此类范式，也就是说，Python 的类型检查的完备性需要依赖 C++ 的接口类型完备性，而 C++ 的接口通常不是开放 API ，由此，即使添加了静态类型检查工具，实际作用也很小。

  PyTorch 并不对外声明类型检查的完备性，为了规避此类问题，使用 `mypy` 检查 API 中 docstring 的 `示例代码`，而不是检查整个 PyTorch 包，其中原由便可略知一二。

- 关注点不同

  Paddle 与 PyTorch 等此类深度学习框架，由于底层逻辑大多由 C++ 实现，而 C++ 中有自身的类型约束，上层 Python 接口也就缺少对于类型检查的关注。

  进而，深度学习框架对于数据的 `shape`，`dtype` 关注要远多与其他类型的软件，而 Python 对于 `shape`，`dtype` 的类型检查支持又不够完善，如 [PEP 646 – Variadic Generics](https://peps.python.org/pep-0646/) 仍然处于 `accepted; may not be implemented yet` 的状态，也就催生了诸如 `nptyping`, `jaxtyping` 此类第三方工具的使用。
  


