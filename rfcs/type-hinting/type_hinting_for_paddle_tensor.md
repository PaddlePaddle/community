# Type Hinting for Tensor of Paddle

| 任务名称 | Type Hinting for Tensor of Paddle |
|---|---|
| 提交作者 | wj-Mcat、jiamingkong、zrr1999、SigureMo |
| 提交时间 | 2022-12-10 |
| 版本号 | v0.3 |
| 依赖飞桨版本 | develop |
| 文件名 | type_hinting_for_paddle_tensor.md |

> **Warning**
>
> - 所有 Warning 会在终稿前删除
> - 部分 suggestion 还未修改

## 一、概述

### 1、相关背景

Python 是一门动态语言，静态类型分析工具很难直接从代码中获得完善的类型信息，这也使得 IDE / Editor 也很难通过它来提供准确的智能提示功能，导致开发体验普遍较差。为了改善这一问题，Python 在 PEP 484 中提出了 Type Hints[^1]，允许以一种规范的语法来为 Python 代码提供类型注释。这也催生了很多静态类型检查工具，使得开发者可以在静态代码检查阶段发现一些潜在的类型错误。这一语法在 Python 3.5 中正式引入，并在之后的几个版本中快速迭代与完善，目前已经普遍应用于众多的 Python 代码库中。

Tensor 是深度学习中的最基础的概念之一，开发者在编写深度学习代码时不可避免地会频繁使用 Tensor 并调用其相关方法与属性。然而目前 Paddle 的 Tensor 情况较为复杂，不仅动态图和静态图下的表示不一致，而且它们都是利用 Python C API 在 C++ 端实现的，这就导致静态类型分析工具无法通过分析 Python 源码的方式来获取类型信息，使得在使用 Tensor 时无法从 IDE / Editor 中获得准确的类型提示，影响开发者的开发效率及体验。

关于 Paddle 的 Tensor 类型提示问题，社区中也有些前置讨论[^2]，为了解决这一问题，本 RFC 旨在通过为 Tensor 类添加类型注解来为开发者提供更好的开发体验。

### 2、功能目标

由于 Paddle 中 Tensor 相关数学函数（如 `paddle.randn`），以及常用的 `paddle.Tensor` 类的方法和属性均没有类型提示信息，为了提高在 IDE 中的开发体验，此提案旨在为 Tensor 提供类型提示信息以解决 Tensor 的智能提示的问题。

需要满足 PEP 484 中所述语法和 PEP 561 中所述分发方式，这样才可以保证包括但不限于 VS Code、PyCharm 等的常规 IDE 均可支持本方案。

IDE 类型提示示例效果如下：

| | Before | After |
| - | - | - |
| 截图 | <img width="500" alt="image" src="https://user-images.githubusercontent.com/38436475/207819538-89f902af-e212-4856-9490-5d4bf153e9f2.png"> |<img width="500" alt="image" src="https://user-images.githubusercontent.com/38436475/207818934-3706a1e3-46ee-4303-92a2-d4d16e6b6bf0.png"> |
| 可连续推导 | ❌ | ✅ |
| 可智能提示 | ❌ | ✅ |
| 效果说明 | 即便明确说明返回值是 Tensor，仍无法智能提示 | 变量类型可自动推导，智能提示信息全面（含类型提示信息、文档等） |

> **Note**
>
> 图中蓝底字为自动推导出的类型提示信息，依托于 IDE / Editor 的 [Inlay Hints](https://code.visualstudio.com/docs/editor/editingevolved#_inlay-hints) 特性显示在源码中，但并不是源码的一部分。

### 3、意义

在 PaddlePaddle 内添加类型提示的主要意义有：

- 添加类型提示有助于优化 IDE 的智能提示，提高开发体验与开发效率，无论是框架开发者还是框架用户都能从中受益；
- 类型提示信息相比于 Docstring 中的类型信息更加直观，而且有着严格的语法，可以提高代码的可读性。

## 二、业内方案调研

### 类型提示信息分发方式调研

Python 在 PEP 561[^3] 中提出了类型提示信息的分发与打包方式，主要包含以下三种方式：

| 方式 | 方式介绍 | 方式优势 | 主要应用项目 |
| - | - | - | - |
| Inline type annotation | 直接在源码中添加类型提示信息，内联于 `.py` 代码中 | 有着较高的可读性和可维护性 | PyTorch、FastAPI、Typer |
| Stub files in package   | 即在包内添加额外的 stub files（`.pyi` 文件），为包中的模块提供类型提示信息| 不需要修改现有源码 | PyTorch、NumPy |
| Distributed stub files  | 不将类型提示信息打包到包中，而是将类型提示信息以第三方库的形式单独发布 | 无任何运行时影响，单独维护 | TensorFlow、django-stubs |

第一种方式是最为推荐的方式，因为与 Python 代码结合紧密，有着较高的可读性和可维护性，第二种方式常常用于一些 C/C++ 扩展模块，在一些大型 Python + C/C++ 混合代码库中，往往是使用第一、第二种方式混合的实现。比如 [PyTorch](https://github.com/pytorch/pytorch) 在大多数代码 Python 代码中直接使用内联的方式添加了类型提示，而通过解析 [YAML](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml) 的方式来自动生成 C++ 扩展 API 的 stub file，生成脚本见 [tools/pyi/gen_pyi.py](https://github.com/pytorch/pytorch/blob/master/tools/pyi/gen_pyi.py)。

第三种方式主要是由社区维护一份 stub-only 的包并发布到 PyPI 中，用户可以按需安装以获取类型提示效果，目前有一些由于历史原因难以迁移的大型代码库使用该方式，如 [django-stubs](https://github.com/typeddjango/django-stubs) 是由社区维护的 [Django](https://github.com/django/django) stub-only 包。Python 社区也维护了一些常用包的 stub-only 的类型提示信息存储在 [typeshed](https://github.com/python/typeshed)。

此外也有一些曾经使用第三种方式先做一些探索，后来将相关经验应用于主代码库的成功案例，比如 [NumPy](https://github.com/numpy/numpy) 首先在 [numpy-stubs](https://github.com/numpy/numpy-stubs) 进行了尝试，目前已经将其吸纳入了主代码库中。

### 深度学习框架 Tensor / Array 类型信息分发方式调研

#### PyTorch

PyTorch 在大多数原生 `.py` 文件直接使用了内联类型提示，部分 C++ 扩展 API 则是使用自动生成 `.pyi` stub file 的方式提供类型提示信息。PyTorch 的 Tensor 主体（`torch._C.TensorBase`）是在 C++ 端实现的，通过继承的方式实现了一个新的 [`Tensor`](https://github.com/pytorch/pytorch/blob/master/torch/_tensor.py#L81)（`torch.Tensor`），因为最终暴露的 `torch.Tensor` 是在 Python 端实现的，其包含了完整的类型提示信息，且继承自 `torch._C.TensorBase` 的属性方法也通过 [pyi 文件](https://github.com/pytorch/pytorch/blob/master/torch/_C/__init__.pyi.in#L1146)提供了完整的类型提示信息。

#### NumPy

NumPy 为大多数 `.py` 文件都额外维护了一份 `.pyi` 的 stub file，以提供类型提示信息。此外，NumPy 专门提供了一个类型模块 [numpy.typing](https://numpy.org/doc/stable/reference/typing.html)，用以提供复杂的类型提示用法。

NumPy 的 Array（`ndarray`）本身是在 C 中实现的，并[利用 Cython 暴露到 Python 端](https://github.com/numpy/numpy/blob/main/numpy/__init__.pxd#L238)，其类型信息同样是[通过 stub file 提供的](https://github.com/numpy/numpy/blob/main/numpy/__init__.pyi#L1483)。`np.ndarray` 本身是一个泛型，提供了两个类型参数 shape 和 dtype，可以为用户提供更加详细和精准的类型提示信息。

此外，NumPy 还在 `numpy.typing` 中提供了更为易用的 `numpy.typing.NDArray` 类型方便用户使用。

#### TensorFlow

TensorFlow 尚未提供包内的类型提示信息（即第一、第二种），但有一些由社区维护和发布的 stub-only 的包，如 [tensorflow-stubs](https://github.com/deepmind/tensor_annotations/tree/master/tensorflow-stubs/tensorflow-stubs)，typeshed 也有一个相关的讨论 [python/typeshed#7144](https://github.com/python/typeshed/issues/7144)。

## 三、飞桨现状

Paddle 目前的 Tensor 是动态图 `eager.Tensor` 和静态图 `Variable` 概念的统一，`eager.Tensor` 是在在 C++ 端实现并通过 Python C API 暴露到 Python 端，并在 Python 端通过 monkey patch 注入了一些额外的方法与属性。Paddle 在 2.0 API 设计之初重新组织了代码库结构（[PaddlePaddle/Paddle#23151](https://github.com/PaddlePaddle/Paddle/pull/23151)），其中包含了 [`python/paddle/tensor/tensor.py`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/tensor.py) 文件，根据注释该文件原准备定义 Tensor 类，但到现在也没有实现，目前只是一个空文件。

Paddle 代码库内目前尚未提供类型提示信息，但有由社区维护的 stub-only 的包，如 [@SigureMo](https://github.com/SigureMo) 发布的 [paddlepaddle-stubs](https://github.com/cattidea/paddlepaddle-stubs)。该包首先通过自动生成的方式来为 Paddle Python 端代码自动生成了 `.pyi` stub file，并添加了了一些[常用类型集合](https://github.com/cattidea/paddlepaddle-stubs/tree/main/paddle-stubs/_typing)，之后通过手工维护的方式为部分函数、类添加详细的类型信息，不过由于作者的时间与精力有限，因此尚未为 Tensor 类及相关函数提供类型提示信息。

此外 [@wj-Mcat](https://github.com/wj-Mcat) 发布了 [types-paddle](https://github.com/wj-Mcat/types-paddle) 利用 inspect 模块在运行时分析 Tensor 的成员，并通过自动生成的方式在用户已经安装的 PaddlePaddle 包里注入 Tensor 类型信息。

整体来说，这两种社区方案都是权宜之计，前者需要社区来额外维护一个包，需要额外的维护成本，且与上游代码库很容易出现不一致；后者则是在用户已经安装的包里注入新的信息，如果 Paddle 本身支持 Tensor 类型提示的话，就不再需要这样的操作了，这也是本 RFC 的初衷，即对后者方案进行优化并集成到 Paddle 主代码库中。

## 四、对比分析

经过「二、业内方案调研」和「三、飞桨现状」的调研，我们已经了解到了目前 Python 所支持的三种类型提示信息分发方式以及各大框架的现状，和 PaddlePaddle 社区中所做出的一些尝试。下面针对这些方案在 Paddle 中可能的实现方法进行对比：

| 方案 | 类似 PyTorch 在 Python 端实现一个 Tensor | 类似 NumPy 在 Python 端仅仅添加 stub   file | 类似 TensorFlow 完全由社区维护 stub-only   包 | 增加类型信息专用代理文件（自动生成） |
|---|---|---|---|---|
| Tensor 类型提示信息分发方式 | Inline type annotation + Stub files in package | Stub files in package | Distributed stub files | Inline type annotation |
| 不需要额外安装包 | ✅ | ✅ | ❌ | ✅ |
| 对主代码库影响 | 较大，甚至可能造成一定的性能影响 | 无任何运行时影响 | 无任何运行时影响 | 无任何运行时影响 |
| 支持 Docstring | ✅ | ❌ | ❌ | ✅ |
| 维护成本 | 高，任何在 C++ 中的参数修改的同时都应该及时 stub file 进行修改 | 高，原因同左 | 高，需要与主代码库保持一致 | 低，所有的 tensor 信息都采用解析Tensor元信息并生成格式化数据的方法，此时可将此信息自动同步至代理文件和官网文档之中。 |

> **Note**
>
> `元信息` 在此 RFC 中主要包含函数列表和属性列表等信息，其中函数列表信息包含函数名称、函数参数列表、参数列表类型信息、返回值类型以及 docstring 等信息，属性列表信息包含属性名称、返回值类型以及 docstring 等信息。

这里第四和第五种方案是指利用 Python typing 模块的特殊常量 `TYPE_CHECKING` 来区别运行时和静态检查阶段，在静态检查阶段为 Tensor 提供一个代理类，以提供完整的 Docstring 和类型提示信息，并且保证在运行时不会有任何性能影响。该方案也是本 RFC 着重介绍的具体实施方案。

## 五、设计思路与实现方案

在「四、对比分析」中我们已经介绍了五种可行方案，第五种方案在各方面明显优于其他方案，本章节将会展开说明该方案的实现步骤以及细节。

### 1、主体设计思路与折衷

#### 整体全貌

本方案的核心设计为增加一个代理 Tensor 类（于 `.py` 文件），该类并不会包含任何代码实现的细节，在运行时也不会访问该类，避免造成任何性能影响，且能够提供完整的 Docstring 以及类型提示信息。

本方案对 Paddle 主代码库主要修改有以下三处：

1. 为了在静态分析阶段暴露该代理 Tensor 类，需要修改 [`python/paddle/__init__.py`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py#L58-L61)，利用特殊常量 `typing.TYPE_CHECKING` 在仅静态分析阶段暴露该代理 Tensor 类。

    ```diff
      if fluid.framework._in_eager_mode_:
          Tensor = framework.core.eager.Tensor
      else:
          from .framework import VarBase as Tensor  # noqa: F401
    + if typing.TYPE_CHECKING:
    +     from .tensor.tensor_proxy import Tensor
    ```

    > **Note**
    >
    > `typing.TYPE_CHECKING` 在静态分析阶段值为 `True` 而在运行时为 `False`，这就可以保证运行时没有任何改变，而在静态分析阶段，类型检查工具将会访问代理的 Tensor 类，得到更好的提示效果。

2. 该代理 Tensor 类将会在 `python/paddle/tensor/tensor_proxy.py` 中基于自动生成的方式实现。该文件最终效果如下：

    ```python
    from __future__ import annotations

    class Tensor:
        def abs(self, name: str | None = None) -> Tensor:
            """
            This operator is used to perform elementwise abs for Tensor
            Args:
                name (str, optional) – Name for the operation (optional, default is None)
            Returns:
                Tensor, The output tensor of abs op.
            """
            pass

        # ... 其他属性及方法
    ```

3. 此外还会对 Tensor 相关数学函数（`python/paddle/tensor/` 目录下的函数）进行修改，在其中内联类型信息，如：

    ```diff
    - def lerp(x, y, weight, name=None):
    + def lerp(x: Tensor, y: Tensor, weight: float | Tensor, name: str | None = None) -> Tensor:
    ```

4. 由于当前Python C API的实现中缺乏 docstring 的相关实现，导致在 python 端无法获取到[`eager_method`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/eager_method.cc)和[`eager_properties`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/eager_properties.cc)的相关元信息，此时只需要完善[PyMethodDef](https://github.com/PaddlePaddle/Paddle/blob/67fc8e9384188d49c3e5d2829e145b737b633e53/paddle/fluid/pybind/eager_method.cc#L1946-L1950)中的 `doc` 实参即可，修改如下所示：

    ```diff
    PyMethodDef variable_methods[] = {
        {"numpy",
        (PyCFunction)(void (*)(void))tensor_method_numpy,
        METH_VARARGS | METH_KEYWORDS,
    -   NULL},
    +   "Returns a numpy array shows the value of current tensor
    +   ""
    +   "Returns:"
    +   "    ndarray: The numpy value of current tensor object"
    +   ""
    +   "Examples:"
    +   "    .. code-block:: python"
    +   "
    +   "        import paddle"
    +   "
    +   "        tensor = paddle.randn([2,3])"
    +   "        # Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,"
    +   "        #        [[-0.43208161,  0.61182195,  1.04432261],"
    +   "        #         [ 0.92133152, -0.91787916,  1.00561225]])"
    +   "        numpy_value = tensor.numpy()"
    +   "        #    [[-0.43208161,  0.61182195,  1.04432261],"
    +   "        #    [ 0.92133152, -0.91787916,  1.00561225]])"
    ```

5. 解析 C++ eager 端的元信息，并与 Python 端的元信息融合成为整体 Tensor 结构化信息， 基于此信息可生成官网 [Tensor 页面](https://www.paddlepaddle.org.cn/documentation/docs/en/2.2/api/paddle/Tensor_en.html#tensor)以及代理 Tensor 文件（tensor_proxy.py）。

此解决方案从类型信息源头开始解决：完善 Python 端 inline-annotation 信息以及 Python C API 端 docstring 注释信息，从而可使用解析工具分别抽取出 Tensor 对象下的所有有效元信息，最后基于解析后的结构化信息生成代理 Tensor 文件。

#### 主体设计选型考量

此解决方案主要分为两部分：在代码中完善类型注释信息以及解析代码生成代理 Tensor 文件。

针对于第一部分，由于 Tensor 函数和属性分为 Python 端注释和 Python C API Bind 端注释，前者可直接通过[`inspect`](https://docs.python.org/3/library/inspect.html)来获取 Python 端相关元信息，后者可通过获取[PyMethodDef](https://docs.python.org/3/c-api/structures.html#c.PyMethodDef)这种的 `ml_doc` 信息从而获取eager 模式下的元信息。

有了 Tensor 对象所有函数和属性的结构化元信息，可基于此实现第二部分的功能：生成代理 Tensor 文件，并在打包时将此代理文件编译到paddlepaddle 包中。

在实现以上解决方案过程中，还考虑过利用解析文档、解析算子 YAML 文件等方式来自动标注，之后对剩余无法自动标注部分进行手动修正并生成代理 Tensor 类的方法，方案如下：

1. 方案一是同 Tensor 相关数学函数的标注方案一致，通过解析文档、算子 YAML 文件等方式先自动生成后，将其存储在代码库，并手动对错误的部分进行修正；
2. 方案二是完全基于解析文档、解析算子 YAML 文件加之以解析 C++ 源码、运行时获取签名等方式以尽可能地自动生成正确的代理 Tensor 类，该类完全依赖于自动生成；
3. 方案三是基于已经修正好的 Tensor 相关数学函数来生成代理 Tensor 类，该代理 Tensor 类同样完全依赖于自动生成，可是 eager 模式下的函数和属性是需要手动维护。
4. 方案四是直接从类型注释源头出发，完善Python 端函数和 Python C API Bind 端的类型注释信息，直接生成代理 Tensor 类，此外官网Tensor 文档也会同步自动更新。

方案对比如下：

| 方案 | 方案一 | 方案二 | 方案三  | 方案四 ✅ |
|---|---|---|---|---|
| 实现成本 | 高，两者各自需要拟定一套生成方案且方案之间相互割裂 | 高，原因同左 | 低，标注后的 Tensor   相关数学函数可以以较低成本直接生成代理 Tensor 类，需要额外处理的只有少数 Tensor 类专有属性和方法 | 低，标注后的 Tensor 相关数学函数可以以较低成本直接生成代理 Tensor 类，没有额外处理的 Tensor 类属性和方法 |
| 准确性 | 高，由于存储在代码库中，其准确性在维护后是有保障的 | 低，基于各种方法的自动生成方案准确率没有保障 | 高，标注后的   Tensor 相关数学函数会直接存储在源码中，其准确性有着保障 | 高，标注后的   Tensor 相关数学函数会直接存储在源码中，其准确性有着保障 |
| 维护成本 | 高，Tensor   相关数学函数修改后需要同步修改，两处需要时刻保持一致 | 高，自动生成方案里的各种边界情况很难处理 | 低，Tensor   相关数学函数由于在源码里很容易维护 | 低，Tensor   相关数学函数由于在源码里很容易维护 |
| 是否包含全量方法和属性 | 否 | 否 | 否 | 是 |

以上四种方案中，由于方案一需要额外存储大量自动生成的重复代码而最先被否决，而后提出的方案三虽然其实现成本、准确性、维护成本都明显优于方案二，可是并不能包含全量方法和属性，方案四和方案三相比，唯一的区别在于[eager_method]()和[eager_properties]()的处理上，方案三是手工维护，可是方案四是直接通过Python C API反射方法自动抽取eager相关元信息，此外此方案完成后还可自动同步官网中的[Tensor页面](https://www.paddlepaddle.org.cn/documentation/docs/en/2.2/api/paddle/Tensor_en.html#tensor)信息，这样不仅实现起来更加简单，而且维护成本也较低（不需要任何手动维护的 Tensor 专有属性、方法），故最后选用此方案作为最优解决方案。

#### 主体设计具体描述

基于上述讨论，我们决定使用如下顺序来进行推进：

1. 首先定义一个空的 Tensor 类，这个 Tensor 类不会作为公开 API 暴露，只是作为临时的方便标注的占位符；
2. 为 Tensor 相关数学函数（`python/paddle/tensor/` 目录下的函数）添加类型注解，在需要 Tensor 类时引用步骤一定义的 Tensor 类，这一步可以通过多人协作、自动或手动的方式逐步推进；
3. 完善`eager_method.cc`和`eager_properties.cc`中公开方法中的docstring，此部分可直接从其他模块中引用即可。
4. 基于完全标注的 Tensor 相关数学函数来拟定代理 Tensor 类的自动生成方式，由于 Tensor 相关数学函数占据了 Tensor 中方法的绝大一部分（约在 90% 以上），这一部分方法可以通过非常简单的方式就可以生成大部分的函数属性元信息，剩余极少量的 eager 方法和属性可直接通过Python C API反射获取元信息，最后融以上两者元信息即可生成包含所有函数和属性的代理 Tensor 类。
5. 代理 Tensor 类完成时，可以移除 Tensor 相关数学函数引用的步骤一中临时的占位 Tensor，改为直接引用代理 Tensor 类。

### 2、关键技术点/子模块设计与实现方案

本小节将会详细介绍 Tensor 相关数学函数的标注方案、代理 Tensor 类的自动生成方案以及对类型信息打包的方案。

#### Tensor 相关数学函数的标注方案

由于 Tensor 相关数学函数众多（根据 [PaddlePaddle/Paddle#48632](https://github.com/PaddlePaddle/Paddle/pull/48632) 统计有 246 个），完全由手动标注将会非常耗时，因此我们采用了自动生成 + 手动修改维护的方式来进行标注。下面罗列一些可能的自动标注方案：

> **Warning** TODO
>
> 需要确认一下这里的 246，因为 https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L293 只有 234 个
>
> 另外需要分别统计下下面的每个自动标注方案能大概解决多少个函数，以及剩余需要手动修改的函数还有多少

1. 从文档（Docstring）中提取类型信息

    即根据文档中已有的类型信息来自动生成标注信息，如 lerp 的文档中已有的类型信息如下：

    ```python
    def lerp(x, y, weight, name=None):
    r"""
    # 略去
    Args:
        x (Tensor): An N-D Tensor with starting points, the data type is float32, float64.
        y (Tensor): An N-D Tensor with ending points, the data type is float32, float64.
        weight (float|Tensor): The weight for the interpolation formula. When weight is Tensor, the data type is float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.
    # 略去
    """
    ```

    因此可以根据此推断出 lerp 的标注信息为：

    ```python
    def lerp(x: Tensor, y: Tensor, weight: float | Tensor, name: str | None = None) -> Tensor:
    ```

    得益于 Paddle 文档为每个函数参数都添加了类型信息，因此大多数函数都可以通过本方式自动标注，但这也要求文档中的类型信息是可靠的，否则则会造成标注信息的错误，对于错误的案例手动修正即可。

2. 从 YAML 的 op 定义中提取类型信息

    即从 API 对应算子本身签名来自动生成标注信息，根据定义在 `ops.yaml` / `legacy_ops.yaml` 中的算子（包括相应的 `parsed.yaml`，是通过飞桨相应脚本进行了预处理的 YAML 文件）签名来解析生成，如 `trace`：

    算子描述如下：

    ```yaml
    - op : trace
      args : (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1)
      output : Tensor
      infer_meta :
        func : TraceInferMeta
      kernel :
        func : trace
      backward : trace_grad
    ```

    我们可以通过一定的解析和拼接操作将其签名写出：

    ```python
    class Tensor:
        def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1, name: str | None = None) -> None:
            pass
    ```

    不过需要注意的是，这既要求算子与 Python API 参数高度一致，又需要 Python 端函数没有额外的参数处理操作，因此有一些自动生成的标注信息需要手动修改。

    关于本方案，[@zrr1999](https://github.com/zrr1999) 已经有了一个示例 PR [PaddlePaddle/Paddle#48632](https://github.com/PaddlePaddle/Paddle/pull/48632)

3. 根据函数名推断返回值类型

    这是一个非常简单的策略，仅仅针对基于前两种方案无法得到返回值的情形，如根据命名规范，以 `is_`、`has_` 为前缀的函数可以推断其返回值类型为 `bool`。

> **Warning** 临时注释
>
> 我们无法将 inplace 函数推断为返回 None，目前已知的 inplace 明显都是有返回值的，如 [reshape_](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py#L3705)、[unsqueeze_](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py#L2691)、[zero_](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/manipulation.py#L857)

对于这些自动标注方案无法涵盖的类型信息，只需要根据文档和源码进行手动标注即可。

#### 代理 Tensor 类的自动生成方案

在上一步我们已经为 Tensor 相关数学函数进行了完整的标注，本方案将是基于标注好的类型信息完备且准确的 Tensor 相关数学函数进行自动生成代理 Tensor 类。

Paddle 的 Tensor 类的成员来源非常复杂，既包含来自于 C++ 端通过 Python C API 暴露的 API（以新动态图为例，如 [eager_math_op_patch.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/eager_math_op_patch.cc#L1841)、[eager_method.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/eager_method.cc#L1945)、[eager_properties.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/eager_properties.cc#L282)），又包含了在 Python 端通过 monkey patch 注入的一些属性和方法（同样以动态图为例，如 [math_op_patch.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dygraph/math_op_patch.py) 和 [varbase_patch_methods.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dygraph/varbase_patch_methods.py)），其中 Tensor 相关数学函数也是[通过 monkey patch 的方式](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dygraph/math_op_patch.py#L519)注入到 Tensor 类中的。

我们最终实现的代理 Tensor 类需要覆盖全部的类型提示信息，也就是包含 Tensor 下的全部成员。由于 Tensor 相关数学函数的类型提示信息已经在上一步标注好了，因此这一部分可以直接通过对源码裁剪掉具体实现的方式得到。这一部分 API 占比较高，且实现较为简单，大大降低了整个方案的实现难度。

对于剩余的一些 Tensor 类下特有的属性和方法，可以在生成脚本中维护一份列表，不过应当在相应的源码中做出提示，在修改源码时应当及时修改这部分手动维护的提示信息。也可以在 CI 中添加相应的检查条件，在更新相关方法签名时提示需要同时修改这一部分。这部分属性和方法较少，维护成本也较低。

关于该自动生成脚本，可以存放在 [tools](https://github.com/PaddlePaddle/Paddle/tree/develop/tools) 目录中，[@SigureMo](https://github.com/SigureMo) 提供了一个示例 PR [PaddlePaddle/Paddle#49053](https://github.com/PaddlePaddle/Paddle/pull/49053)。

#### 类型信息打包方案

由于代理 Tensor 类完全自动生成，因此不需要存储到代码库中，只需要在代码打包成 wheel 包时自动生成并打包进去即可。这样可以避免同时维护两套内容一致的代码，且不会影响最终的效果。

由于我们的类型提示信息是完全基于 PEP 561[^3] 中第一种方案 Inline type annotation 的，因此需要在 wheel 包中包含一个空白的 `py.typed` 文件，以表明我们的包支持类型提示。

关于 wheel 打包可参考 [setup.py](https://github.com/PaddlePaddle/Paddle/blob/develop/setup.py)。

> **Warning** 临时注释
>
> setup.py 应该正在取代 [python/setup.py.in](https://github.com/PaddlePaddle/Paddle/blob/develop/python/setup.py.in)，但目前还不是全部 CI 流水线都替换掉了，只是一部分

#### API 签名更新方案
1. 存量
2. 增量

### 3、主要影响的模块接口变化

#### 请列出核心设计对应的直接接口变化

对 API 无任何运行时影响，只是会添加类型提示信息，会对 API 签名造成影响。添加的类型提示信息将会被用于 IDE / Editor 的智能类型提示。

#### 请逐一排查对对框架各环节的影响

无影响。

## 六、测试和验收的考量

确保对运行时无影响，且在 IDE / Editor 中能够正常使用类型提示。

## 七、影响面

### 对用户的影响

极大地提升 IDE / Editor 的智能提示能力，提升用户体验。

### 对二次开发用户的影响

为二次开发用户提供类型提示信息，提升二次开发用户的开发体验。

### 对框架架构的影响

无影响。

### 对性能的影响

无影响。

### 对比业内深度学习框架的差距与优势的影响

PyTorch 在 Python 端重新实现 Tensor，新添加的方法能够直接在 Python 代码中集成内联类型提示信息，维护成本较小，而 Paddle 新添加的方法都是通过 monkey patch 注入的，因此无法通过此种方式来直接实现，根据本方案中的实现方法，具体实现与代码类型提示信息在不同文件，维护成本会高一些。

### 其他风险

无。

## 八、排期规划

本方案由社区开发者来执行，具体完成时间由社区开发者自行安排。

## 九、待讨论

通过集成静态类型检查工具可以进一步保证类型提示信息的正确性，且可以发现一些代码中的潜在问题，但由于静态类型检查工具的引入有很多需要考虑的问题，因此暂不在本 RFC 讨论范围内，相关讨论以及尝试将会在之后进一步进行。

## 名词解释

- Tensor 相关数学函数：指定义在 [python/paddle/tensor](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/tensor) 目录中的用于操作 Tensor 的数学函数，其中[部分函数](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py#L293)通过 monkey patch 注入到了 Tensor 类上，这里指注入的这些函数

## 附件及参考资料

[^1]: [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
[^2]: [社区讨论 - support Tensor type hinting](https://github.com/PaddlePaddle/Paddle/issues/45979)
[^3]: [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
