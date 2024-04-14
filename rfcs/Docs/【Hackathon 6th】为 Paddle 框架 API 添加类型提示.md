# 为 Paddle 框架 API 添加类型提示

| 领域         | 为 Paddle 框架 API 添加类型提示    |
| ------------ | ---------------------------------- |
| 提交作者     | megemini (柳顺)                    |
| 提交时间     | 2024-04-07                         |
| 版本号       | V1.1                               |
| 依赖飞桨版本 | develop 分支                       |
| 文件名       | 为 Paddle 框架 API 添加类型提示.md |

## 修订记录

### v1.1

- `typing` 模块改为私有模块 `_typing`，不创建 API 及中英文文档
- 删除代理文件的对比
- 《Paddle 中的类型提示》的内容目录
- 《Paddle 类型提示 Q&A》需要文档或工具，跟踪类型标注的最佳实践

# 一、概述

## 1、相关背景

Python 在 3.5 版本通过 [PEP 484 – Type Hints](https://peps.python.org/pep-0484) 正式规范了 `类型提示` 功能，以帮助开发者提高代码质量，Python 目前 (`3.12` 版本) 已经完成的相关 `PEP` 有 `21` 个，具体可以参考 [Typing PEPs](https://peps.python.org/topic/typing/) 。经过前期的几个版本迭代，Python 的 `类型提示` 功能已经受到开发者的广泛认可。Paddle 目前支持的 Python 版本 `3.8` 已经可以较好的支持 `类型提示`，本文旨在阐述 Paddle 引入 `类型提示` 的可行性与具体方案。

本文档为 [【Hackathon 6th】Fundable Projects](https://github.com/PaddlePaddle/Paddle/issues/62908) 中 [为 Paddle 框架 API 添加类型提示（Type Hints）](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E4%B8%80%E4%B8%BA-paddle-%E6%A1%86%E6%9E%B6-api-%E6%B7%BB%E5%8A%A0%E7%B1%BB%E5%9E%8B%E6%8F%90%E7%A4%BAtype-hints) 的实现方案。

## 2、功能目标

正确完成 Paddle 公开 API 的类型标注，但不声明 Paddle 类型标注的完备性。

> **说明**： 类型标注是个循序渐进的过程，且存在较多私有 API 与 c++ 接口，此次任务无法保证完成以上所有接口的类型标注，故此，不做 Paddle 类型标注的完备性说明。

### 2.1 _typing 模块

此目标为 Paddle 添加 `_typing` 模块，并作为私有 API 的一部分。

功能特性类似：

- PyTorch 中的 `torch/types.py`
- Numpy 中的 [Typing (numpy.typing)](https://numpy.org/devdocs/reference/typing.html)

`_typing` 模块包含 Paddle 中用到的特殊类型，如 `dtype`，`device` 等，具体实现可参考 @SigureMo 的 [paddle-stubs/_typing](https://github.com/cattidea/paddlepaddle-stubs/tree/main/paddle-stubs/_typing)。

具体需要实现：

- `_typing` 模块代码
- `_typing` 模块测试用例

### 2.2 开放 API 的类型标注

此目标需要：

- 对现存 Paddle 代码的开放 API 进行类型标注。

    Paddle 目前开放 API 有 1500 个左右，需要分批次对其进行类型标注。可以参考 [paddlepaddle-stubs Roadmap](https://github.com/orgs/cattidea/projects/3/views/1) ，将 API 分为 `P1 ～ P5` 多个等级分批完成。

- 修改或对齐 API 中 docstring 的参数类型说明，与实际的标注类型一致。

此目标的完成，应保证主流 IDE，如 VS Code、PyCharm 的类型提示功能可以正常使用。

此目标 **涉及**：

- Paddle 的开放 API
- API 中 docstring 的参数类型说明（输入和输出）

此目标 **不涉及** ：

- 添加 API 的类型检查测试用例 (`_typing` 模块除外)

### 2.3 CI 流水线

此目标需要更新 CI 流水线，对以下代码进行类型检查：

- `_typing` 模块的测试用例
- 旧/新 API 的示例代码
- 旧/新 API 的测试用例（推荐）

> **说明:** `推荐` 表示此任务如果条件允许则可以进行，如果条件不允许，则，不进行，或者采取其他方式代替。

此目标 **涉及**：

- 静态类型检查 (`static type checking`)。使用工具 [mypy](https://mypy.readthedocs.io/en/stable/) 。

此目标 **不涉及** ：

- 运行时类型检查 (`runtime type checking`)，如工具 [beartype](https://beartype.readthedocs.io/en/latest/) 。
- 整个 Paddle 包的类型检查。

### 2.4 文档建设

此目标需要：

- 在 Paddle 的 `docs` 中添加文档 `《Paddle 中的类型提示》` 。

  用以帮助 Paddle 的开发者，正确标注 API 中的类型；使用 Paddle 的开发者，正确使用 `_typing` 模块。

- 在 Paddle 的 `docs` 中添加文档 `《Paddle 类型提示 Q&A》` ，并不断更新。

  类型标注在具体操作过程中难免产生各类问题 (参考 [Static types in Python, oh my(py)!](https://blog.zulip.com/2016/10/13/static-types-in-python-oh-mypy/)，mypy 的各类 issue 促成了其早期发展)，《Q&A》可以方便的帮助开发者定位问题。

### 2.5 后续任务

如工期允许，后续可以完成：

- `_C_ops` 的标注
- 私有 API 的类型标注
- Paddle 整个包的类型检查
- 添加运行时类型检查

## 3、意义

`Type Hints` 的引入：

- 提升开发者使用体验

  极大提高用户 API 补全体验，为每个参数提供准确的类型信息，结合静态类型检查工具，可以提供一流的开发体验。

  并且，可以为下游生态，如 `PaddleScience` 等套件，类型提示集成提供基础，带来更好的下游套件开发体验。

- 提升 Paddle 代码质量

  在开发阶段可以杜绝低级的传参错误，规范参数类型，规范并统一文档中参数类型的标注等内容。

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

  `类型标注`，常与 `type hint` 混用，也可作为 `动词` 专门指代 `标注类型` 这个动作。

- `type checking`

  `类型检查`，检查类型标注的正确性，或数据在接口中传递时的类型正确性。多指静态类型检查 (`static type checking`) 。

- `static type checking`

  `静态类型检查`，使用工具如 `mypy/pyright` 等，对特定代码或包进行类型标注的静态检查，此动作不依赖源代码的运行时状态。

  是目前广泛使用的类型检查方式，常与 `type checking` 混用。

- `runtime type checking`

  `动态类型检查`，使用工具如 `beartype`，在代码运行时保证数据类型的正确性。

  借助额外的类型支持包，如：

  - [nptyping](https://github.com/ramonhagenaars/nptyping) 之于 `numpy`
  - [jaxtyping](https://github.com/patrick-kidger/jaxtyping) 之于 `PyTorch`

  此类工具可以提供代码运行时的额外信息检查，如 `shape`、`dtype` 等信息。

其次，Paddle 或 PyTorch 对于类型检查的支持，与 `pure python` 的包不尽相同：

- 实现方式的差别

  一个包的开发逻辑，大体可以分为：

  - 底层支撑接口
  - 上层开放接口
  - 开发者使用接口

  对于 `pure python` 的包，以上三者通常都是使用 Python 完成开发的，因此，可以方便的使用工具进行类型检查。

  而对于 Paddle 和 PyTorch 此类 Python 混合 C++ 或其他语言的开发包，底层逻辑使用其他语言实现，上层接口通常只是使用 Python 将其暴露出来，接口间的调用较少，也就缺乏了检查开发包本身数据类型的动力。如：

  ``` python
  def abs(x):
    return _C_ops.abs(x)
  ```

  API 中大量采用此类范式，也就是说，Python 的类型检查的完备性需要依赖 C++ 的接口类型完备性，而 C++ 的接口通常不是开放 API ，由此，即使添加了静态类型检查工具，实际作用也很小。

  PyTorch 并不对外声明类型检查的完备性，为了规避此类问题，PyTorch 使用 `mypy` 检查 API 中 docstring 的 `示例代码`，而不是检查整个 PyTorch 包。

- 关注点不同

  Paddle 与 PyTorch 等此类深度学习框架，由于底层逻辑大多由 C++ 实现，而 C++ 中有自身的类型约束，上层 Python 接口也就缺少对于类型检查的关注。

  进而，深度学习框架对于数据的 `shape`、`dtype` 关注要远多与其他类型的软件，并且部分信息需要在代码运行过程中获取，而 Python 对于 `shape`、`dtype` 等的类型检查支持不够完善，如 [PEP 646 – Variadic Generics](https://peps.python.org/pep-0646/) 仍然处于 `accepted; may not be implemented yet` 的状态，也就催生了诸如 `nptyping`、`jaxtyping` 此类第三方工具的开发与使用。

因此，此次任务不涉及整个 Paddle 包的类型检查，不涉及动态类型检查，不声明 Paddle 具有类型检查的完备性。

最后，本方案采用 `Inline type annotation + Stub files in package` 的方式，类似 PyTorch 的实现方案。基本原则为：

- 非 Python 接口，提供 `stub` 标注文件
- Python 接口，使用 `inline` 方式标注

标注的基本原则为：**在不违背 Paddle 最低支持版本 `3.8` 语法的基础上，尽可能使用新版本 typing 特性**，如有必要，使用 `typing_extension` 模块。

如，标注中使用 `Union` 的情况，在参数中使用 `|` 代替，同时需要引入 `from __future__ import annotations`:

``` python
from __future__ import annotations
def test(a: int | str): ...
```

而在 `TypeAlias` 中仍使用 `Union` ：

``` python
from typing_extensions import TypeAlias
from typing import Union
t: TypeAlias = Union[int, str]
```

> **说明** 竞品分析与实现方案等内容，[Type Hinting for Tensor of Paddle](https://github.com/jiamingkong/community/blob/4bde11a8a861c8aae4bdb1284579e00d4799f7b9/rfcs/type-hinting/type_hinting_for_paddle_tensor.md) 已有较详细的阐述，本文不再赘述。

## 1、总览

Paddle 的类型标注体系，由底层而上，可以划分为：

- `_typing` 模块

  如 `dtype`、`shape`、`device` 等，参考 [paddle-stubs/_typing](https://github.com/cattidea/paddlepaddle-stubs/tree/main/paddle-stubs/_typing)

- `Tensor`

  各个接口依赖的输入和输出类型，且本身具有 `Tensor.xxx` 等方法。

- 基础类

  如 `paddle.nn.Layer`、`paddle.optimizer.Optimizer`、`paddle.ParamAttr` 等基础类。

- 其他开放接口

  如 `paddle.abs`、`paddle.nn.Linear` 等。

> **说明** 此处不涉及 `_C_ops` 等底层接口。

结合上文 [2、功能目标](#2功能目标) ，将整个工作拆分为以下几个阶段：

- **第一阶段**，基础能力的引入

  此阶段需要完成基础类型的引入和标注，主要完成以下工作及输出件：

  - Paddle 中引入 `_typing` 模块，添加测试用例。
  - Paddle 中引入 `Paddle/python/paddle/__init__.pyi` 和 `Paddle/python/paddle/py.typed` 文件，作为 `paddle.Tensor` 的类型 `stub` 。
    此阶段可以只做文件的引入，后续完善接口和文档。
  - Paddle docs 中添加文档 `《Paddle 中的类型提示》` 。
  - Paddle 的 CI 中引入 `mypy` 对于 API 中 docstring 的 `示例代码` 的类型检查。

- **第二阶段**，开放接口的类型标注

  此阶段承担了 Paddle 类型标注的主要工作，需要开源社区的广泛参与共同完成。

  此阶段的工作及输出件：

  - 完成开放 API 的类型标注。
  - Paddle docs 中添加并不断完善文档 `《Paddle 类型提示 Q&A》` 。

    > **说明：** 此任务需要额外的文档或工具，在实施过程中不断跟踪标注任务的最佳实践，并统一和规范标注任务。

- **第三阶段**，CI 中引入对于单元测试文件的类型检查 (推荐)

  在上述第二阶段中，并未引入类型标注的 `单元测试`，这是出于以下考虑：

  - 类型标注工作量太大，如果再引入 `单元测试`，不具备开源社区协作与短期完成的条件。
  - 目前 Paddle 的接口都有单元测试文件，单元测试本身应对于接口的类型做完备测试，因此，通过静态检查这些单元测试文件，即可达到类型检测的目的。

  但是，由于第二阶段中的类型检查只针对 `示例代码`，难免在此出现毗漏，因此，此阶段在 CI 中引入单元测试文件的类型检查后，需要跟踪并修改其中错误。

  另外，`示例代码` 的类型检查，与 `单元测试` 的类型检查，需要制定不同的 `mypy` 配置文件。

  此阶段的输出件：

  - Paddle 的 CI 中引入 `mypy` 对于 API `单元测试` 文件的类型检查。
  - 修改并完善开放接口的标注。

  > **说明：** 单元测试涉及较多内部接口，如果对类型检查影响较大，可以省略此阶段，或通过补充类型检查的测试用例进行代替。

- **第四阶段**，收尾阶段

  此阶段主要完善以上阶段中的遗漏工作，如有需要，可进行私有接口的标注工作。

## 2、第一阶段

此阶段需要完成基础类型的引入和标注。

### 2.1 Paddle 中引入 `_typing` 模块

Python 软件包在进行类型标注的过程中，通常会使用到一些相同或相似的数据类型，因此，有必要引入 `_typing` 模块统一管理与维护。并且，`_typing` 模块可以作为私有 API 的一部分，以方便使用 Paddle 的开发者在自己的项目中使用类型提示。

本方案参考 [paddle-stubs/_typing](https://github.com/cattidea/paddlepaddle-stubs/tree/main/paddle-stubs/_typing) 引入 `_typing` 模块。

#### 2.1.1 `_typing` 模块代码

实现路径 `Paddle/python/paddle/_typing/*`，具体可包括：

- `basic.py`，数字、嵌套序列等
- `device.py`，设备，`CPU`、`GPU` 等
- `dtype.py`，数据类型，`uint8`、`float32` 等
- `layout.py`，数据布局，`NCHW`、`NHCW` 等
- `shape.py`，数据形状，`ShapeLike` 等

#### 2.1.2 `_typing` 模块测试用例

- 测试脚本

  实现路径 `Paddle/test/test_typing.py`。脚本可在 CI 中调用，通过 `mypy` 的 API 进行测试。

  第一阶段只需要实现 `_typing` 模块的检查即可，后续如需检查其他内容，通过调用不同的配置，可更新此文件。

- 测试内容

  实现路径 `Paddle/test/_typing/*`。参考 PyTorch 与 Numpy 的测试方式，这里可以包括：

  - `mypy.ini` 文件，`mypy` 的配置文件。
  - `pass` 目录，正常的类型检查，如 `a: ArrayLike = [1, 2, 3]`
  - `fail` 目录，错误的类型参数，如 `a: ArrayLike = (i for i in range(10))`
  - `reveal` 目录，检查返回值的类型，如 `ar_iter = np.lib.Arrayterator(AR_i8);assert_type(ar_iter.var, npt.NDArray[np.int64])`

### 2.2 Paddle 中引入 `Paddle/python/paddle/__init__.pyi` 文件和 `Paddle/python/paddle/py.typed` 文件

#### 2.2.1 `Paddle/python/paddle/__init__.pyi` 文件

`Paddle/python/paddle/__init__.pyi` 文件主要为 `Tensor` 的 `stub` 标注文件。

本方案采用 `Inline type annotation + Stub files in package` 的方式，不同于 [Type Hinting for Tensor of Paddle](https://github.com/jiamingkong/community/blob/4bde11a8a861c8aae4bdb1284579e00d4799f7b9/rfcs/type-hinting/type_hinting_for_paddle_tensor.md) 中讨论的 `代理文件` 方式，直接在 `Paddle/python/paddle/__init__.pyi` 中直接生成 `Tensor` 的类型标注。基本形式如：

``` python

from __future__ import annotations
from typing_extensions import TypeAlias

class dtype:
    def __init__(self, arg0: int) -> None: ...

DTypeLike: TypeAlias = dtype | str

class Tensor:
    def cast(self, dtype: DTypeLike) -> Tensor:
        """ cast docstring ... """ # 通过脚本生成，从代码中抽取 docstring 插入此处
    ...

from .tensor.creation import (
    to_tensor,
    ...
)

__all__ = [
    "to_tensor",
    ...
]

```

另外，`paddlepaddle-stubs/paddle-stubs/__init__.pyi` 中，通过:

``` python
from ._typing import Tensor as Tensor
```

将 `_typing` 的 `Tensor` 作为 `paddle.Tensor` 的类型标注，与本方案的作用相同。

此方式不影响目前已有的类型标注代码，此文件可以使用模板的方式生成，参考 `pytorch/torch/_C/__init__.pyi.in` 。

对于一些 C++ 或 patch 的接口，如上述 `cast` 方法，需要同时提取接口的 docstring 并插入，以提供完整的类型提示与文档提示功能。

对于 Python 接口，则可以通过 `明确` 引入的方式直接导入此处，如使用 `from a import b as b` ，或上述的 `__all__` 的方式。

#### 2.2.2 `Paddle/python/paddle/py.typed` 文件

此文件作为类型提示的标识文件，直接创建即可。

另外，Paddle 在打包时，需要将上述几个文件打包进去，由此需要修改打包脚本 `setup.py` 等文件。

### 2.3 Paddle docs 中添加文档 `《Paddle 中的类型提示》`

`《Paddle 中的类型提示》` 此文档辅助开发者进行 Paddle 的类型标注，为后续工作提供参考基础。

实现路径为 `文档 > 贡献指南 > 规范和参考信息 > 《Paddle 中的类型提示》` 。

`《Paddle 中的类型提示》` 文档中可以包含以下内容：

- `_typing` 模块的使用方法
- docstring 中的参数与 annotation 中的标注对应关系
- 类型标注最佳实践
- Q&A

### 2.4 Paddle 的 CI 中引入 `mypy` 对于 API 中 docstring 的 `示例代码` 的类型检查

此文件与 [2.1.2 `_typing` 模块测试用例](#212-_typing-模块测试用例) 中的 `Paddle/test/test_typing.py` 可以是同一个文件。

脚本需要：

- 抽取 `git diff` 的 API 的 docstring
- 抽取 docstring 中的 `示例代码`
- 对示例代码进行静态类型检查

此任务完成后，可进行后续的 Paddle 代码标注工作。

## 3、第二阶段

此阶段分批次完成开放接口的类型标注，主要划分依据为 [paddlepaddle-stubs Roadmap](https://github.com/orgs/cattidea/projects/3/views/1) ，将 API 分为 `P1 ～ P5` ：

- `P1`
  - `paddle.nn.layer.*`
  - `paddle.vision.transforms.transforms.*`
  - `paddle.nn.initializer.*`
  - `paddle.optimizer.*`
  - `paddle.Model`
  - `paddle.vision.models.*`
  - `paddle.nn.Layer`

- `P2`
  - `paddle.tensor.*`
  - `paddle.vision.datasets.*`
  - `paddle.metric.*`
  - `paddle.Tensor`
  - `paddle.regularizer.*`
  - `paddle.vision.transforms.functional*`

- `P3`
  - `paddle.hub.*`
  - `paddle.linalg.*`
  - `paddle.signal.*`
  - `paddle.callbacks.*`
  - `paddle.onnx.*`
  - `paddle.nn.functional.*`
  - `paddle.io.*`
  - `paddle.distribution.*`
  - `paddle.device.*`
  - `paddle.autograd.*`
  - `paddle.amp.*`
  - `paddle.fft.*`
  - `paddle.jit.*`

- `P4`
  - `paddle.sysconfig.*`
  - `paddle.utils.*`
  - `paddle.text.*`
  - `paddle.sparse.*`
  - `paddle.profiler.*`
  - `paddle.nn.quant.*`
  - `paddle.nn.utils.*`
  - `paddle.distributed.*`

- `P5`
  - `paddle.inference.*`
  - `paddle.proto.*`
  - `paddle.common_ops_import`
  - `paddle.check_import_scipy.*`
  - `paddle.batch.*`
  - `paddle.reader.*`
  - `paddle.hapi.*`
  - `paddle.framework.*`
  - `paddle.cost_model.*`
  - `paddle.static.*`
  - `paddle.incubate.*`
  - `paddle._C_ops.*`

以上可作为批次任务的依据，但并非需要此次任务全部完成，如 `paddle._C_ops.*`，可作为后续补充任务。

各接口需要完成以下工作：

- 完成类型标注
- 通过 [2.4](#24-paddle-的-ci-中引入-mypy-对于-api-中-docstring-的-示例代码-的类型检查) CI 对于接口中示例代码的类型检查
- 修改并对齐 docstring 中的类型说明，包括输入与输出

标注过程中需要引入并不断完善 Paddle 的 docs 中文档《Paddle 类型提示 Q&A》 (或者是 《Paddle 中的类型提示》 的 `Q&A` 部分)。

目前存在较多的自动化标注工具，如 ([Automate annotation of legacy code](https://mypy.readthedocs.io/en/stable/existing_code.html))：

- [MonkeyType](https://monkeytype.readthedocs.io/en/latest/index.html)
- [autotyping](https://github.com/JelleZijlstra/autotyping)
- [PyAnnotate](https://github.com/dropbox/pyannotate)

此阶段开始前，可以先挑取部分 API ，使用上述工具，或手动，进行标注。经过流程验证并形成最佳实践后，指导开源社区进行协同完成。

如有必要，可以通过额外的文档或者工具进行统一管理或规范标注行为。

## 4、第三阶段 (推荐)

此阶段在 Paddle 的 CI 中引入 `mypy` 对于 API `单元测试` 文件的类型检查。

完成 `单元测试` 文件的类型检查，可以进一步验证并加强 Paddle 中类型标注的正确性。

- 实现路径 `Paddle/test/test_typing.py`，同 [2.1.2](#212-_typing-模块测试用例)。

- `mypy` 配置文件 `Paddle/test/mypy.ini`。

类型标注是个循序渐进的过程，如发现引入 API `单元测试` 文件的类型检查出现较多问题，有可能需要再启动一次开源社区的协同修改工作，因此，此阶段为 `推荐` 完成，并存在 `不确定风险` 。

如此任务不具备完成的条件，可以省略，或者采取补充测试用例等手段代替。

## 5、第四阶段 (推荐)

此阶段主要完善以上阶段中的遗漏工作，如有需要，可进行私有接口的标注工作，主要如 `_C_ops` 等接口的标注。此类接口通过 `stub` 方式实现，如 `Paddle/python/paddle/_C_ops.pyi`。

此阶段为 `推荐` 完成，需要视前面工作的完成进度而定。

# 四、测试和验收的考量

- Paddle 中引入 `_typing` 模块

  需要添加测试用例，CI 中通过测试。

- Paddle 中引入 `Paddle/python/paddle/__init__.pyi` 和 `Paddle/python/paddle/py.typed` 文件，作为 `paddle.Tensor` 的类型 `stub`

  需要保证 CI 能够正确执行对于类型的检查。

- Paddle 的 docs 中添加文档 `《Paddle 中的类型提示》`

  需要能够指导开发者使用 `_typing` 模块，并进行类型标注。

- Paddle 的 CI 中引入 `mypy` 对于 API 中 docstring 的 `示例代码` 的类型检查

  需要 CI 能够正确检查 API 中的示例代码。

- 完成开放 API 的类型标注

  需要能够通过 CI 中对于示例代码的类型检查。

- Paddle 的 CI 中引入 `mypy` 对于 API `单元测试` 文件的类型检查

  需要 CI 能够正确检查单元测试文件中的代码。

# 五、排期规划

1. Paddle 中引入 `typing` 模块
2. Paddle 中引入 `Paddle/python/paddle/__init__.pyi` 文件和 `Paddle/python/paddle/py.typed` 文件
3. Paddle docs 中添加文档 `《Paddle 中的类型提示》`，并新建第三方文档或工具用以管理下阶段标注任务
4. Paddle 的 CI 中引入 `mypy` 对于 API 中 docstring 的 `示例代码` 的类型检查
5. 分批次完成开放接口的类型标注 (*)
6. Paddle 的 CI 中引入 `mypy` 对于 API `单元测试` 文件的类型检查 (**)
7. 分批次修复开放接口的类型标注 (**)
8. 私有 API 的类型标注 (*)

以上任务依次进行，其中：

- 任务 `5, 8` 需要开源社区协作完成。
- 任务 `6` 需要视任务 `5` 的完成情况而定是否进行
- 任务 `7` 需要视任务 `6` 的检查结果而定是否进行，如需进行，评估是否通过开源社区完成。
- 任务 `1, 2, 3, 4, 6` 为单独任务。

# 六、影响面

1. Paddle 中引入 `_typing` 模块

    **对用户的影响：** 用户可以使用 `_typing` 模块提供的接口

    **对开发者的影响：** 开发者可以使用 `_typing` 模块标注 Paddle 内部模块

    **对框架架构的影响：** Paddle 框架中增加 `_typing` 模块与相应接口

    **其他风险：** `_typing` 模块需要在后续工作中不断完善

2. Paddle 中引入 `Paddle/python/paddle/__init__.pyi` 文件和 `Paddle/python/paddle/py.typed` 文件

    **对用户的影响：** 用户可以使用 `Tensor` 进行类型标注

    **对开发者的影响：** 开发者可以使用 `Tensor` 对 Paddle 内部模块进行类型标注

    **对框架架构的影响：** Paddle 增加 `stub` 文件

    **其他风险：** `Tensor` 以及此 `stub` 文件需要在后续工作中不断完善

3. Paddle docs 中添加文档 `《Paddle 中的类型提示》`

    **对用户的影响：** 用户可以参考文档进行类型标注

    **对开发者的影响：** 开发者可以参考文档对 Paddle 内部模块进行类型标注

    **对框架架构的影响：** Paddle 的 docs 中增加相应文档

4. Paddle 的 CI 中引入 `mypy` 对于 API 中 docstring 的 `示例代码` 的类型检查

    **对用户的影响：** 不涉及

    **对开发者的影响：** 开发者需要能够通过此 CI 检查

    **对框架架构的影响：** Paddle 增加新的 CI 流水线任务

    **其他风险：** `示例代码` 中可能出现接口依赖问题，需要提前规划，或另行规避

5. 分批次完成开放接口的类型标注

    **对用户的影响：** 用户可以使用完整的开放 API 的类型标注

    **对开发者的影响：** 开发者的新增 API 需要进行类型标注，并通过 CI 检查

    **对框架架构的影响：** Paddle 中开放 API 需完成类型标注

    **其他风险：** 有可能出现接口依赖问题，需要提前规划，或另行规避

6. Paddle 的 CI 中引入 `mypy` 对于 API `单元测试` 文件的类型检查

    **对用户的影响：** 不涉及

    **对开发者的影响：** 开发者需要能够通过此 CI 检查

    **对框架架构的影响：** Paddle 增加新的 CI 流水线任务

    **其他风险：** 此任务依赖任务 `5` 的完成情况，`单元测试` 中可能出现接口依赖问题，需要提前规划，或另行规避

7. 分批次修复开放接口的类型标注

    **对用户的影响：** 不涉及

    **对开发者的影响：** 不涉及

    **对框架架构的影响：** 修改并完善 `typing` 模块；修改错误类型标注或用例

    **其他风险：** 此任务依赖任务 `6` 的检查结果

8. 私有 API 的类型标注

    **对用户的影响：** 用户可以使用私有 API 的类型标注

    **对开发者的影响：** 开发者的新增 API 需要进行类型标注，并通过 CI 检查

    **对框架架构的影响：** Paddle 中私有 API 完成类型标注

    **其他风险：** 有可能出现接口依赖问题，需要提前规划，或另行规避
