# Type Hinting for Paddle

## 2.4., 孔嘉明 & 吴京京，2022.11.19

# 一、概要

## 1、相关背景

在深度学习代码中充斥着各种`Tensor`，可是Python是一门动态编程语言，如果不给变量添加类型，IDE将无法进行智能提示，故Type Annotation（类型注释）对于提升编程体验至关重要。然而，在Paddle框架当中的Tensor对象直接是通过`pybind11`将C++`Tensor`类型暴露给外部开发者，并没有对应的类型信息，故无法进行静态类型推断或动态stub file等操作。

为了给Paddle的Tensor添加类型注释的功能，可使用stub-file相关技术来实现此功能。

参考链接：

* [Introduction of Stub File](https://mypy.readthedocs.io/en/stable/getting_started.html#stubs-intro)
* [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
* [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)

如有对应重要相关issue、icafe task的也请说明。

## 2、功能目标

由于 Paddle 中内置的函数（如`paddle.randn()`），以及常用的`paddle.Tensor`类的方法均没有返回类型的提示，在IDE中写代码开发体验较差，此提案旨在解决支持 Tensor 的智能提示的问题。

## 3、方案要点

# 二、意义

`Type Hint`作为编程中最为基础体验，Python 3.5 已然官方支持 [typing](https://docs.python.org/3/library/typing.html)，从[google/pytype](https://github.com/google/pytype#requirements)对于 python 版本的要求可发现在python3.7 以后兼容情况已然稳定。同时也伴随有众多的代码分析和类型检查工具应运而生，比如[PyCQA/pylint](https://github.com/PyCQA/pylint)、[psf/Black](https://github.com/psf/black)、[python/mypy](https://github.com/python/mypy)、[google/pytype](https://github.com/google/pytype)等等，这些工具已然支持对于类型注释的检查，而且项目的 star 数量都非常高，从侧面看出此类工具的应用面之广，甚至 已然成为 python 开源项目的基础工具。

给 Paddle 添加`Type Hinting`的功能，主要有以下作用：

* 由于有函数的智能提示，可提升编程体验。
* 提升代码的可阅读性。
* 基于开源类型审查工具，可进一步检查代码潜在 bug，从而提升开源项目的质量。

本次为吴京京联合外部开发者一起商量讨论出的解决方案，同时由孔嘉明实践落地。

# 三、竞品对照

`Tensor`的类型注释属于基础编程体验，在 TF 和 PyTorch 中很早就已支持，故在 Paddle 框架中也应该支持此基础体验。以相似度较高的PyTorch框架为例，其源代码已经带有了type hinting的信息，在IDE中编写torch代码时能方便地得到提示。

# 四、设计思路与实现方案

## 1、主体设计思路与折衷

### 整体全貌

在 Paddle 框架当中，`Tensor`是通过Pybind11/PyObject将 C++的 Tensor 类暴露给Python端，所以在paddle安装好的框架内，paddle/tensor/tensor.py 是一个空文件，这也是IDE无法完成类型推理和提示的原因。此方案设计了一个新的tensor类，并配合了typing.TYPE_CHECKING的环境变量，仅在代码开发阶段对tensor类进行替换，替换后的tensor类是一个所有属性和方法都带有type hinting信息的类，改善代码开发阶段。

在实际代码执行的时候，参与计算的仍然是C++的Tensor类。

### 主体设计具体描述

对paddle主库代码的改动分为两大部分：

- `paddle/tensor/tensor_proxy.py` 的实现

  在tensor_proxy.py，我们构建了一个专门服务于类型提示的空类。它拥有所有Tensor类的方法签名，并且带有类型提示信息，以下面的代码片段为例：

  ```python
  class Tensor:
      def abs(self, name: Optional[str] = None) -> Tensor:
          """
          This operator is used to perform elementwise abs for Tensor
          Args:
  	        name (str, optional) – Name for the operation (optional, default is None)
      	Returns:
          	Tensor, The output tensor of abs op.
          """
          pass
  ```

  

- `paddle/__init__.py`的改动

  在`paddle/__init__.py`最后，我们加入了判断TYPE_CHECKING并自动选择Tensor版本的代码：

  ```python
  from typing import TYPE_CHECKING
  
  if TYPE_CHECKING:
      from .tensor.tensor_proxy import Tensor as Tensor
  ```

其中TYPE_CHECKING是一个环境变量，在第三方静态类型检查工具、IDE中被设定为True，而在python运行时中设定为False。所以在IDE写代码到实际运行的过程中，系统发生的实际行为是：

- 在IDE写代码的过程中，paddle会导入`tensor_proxy.py`中定义的类，这个类的所有方法签名都具备了类型提示的信息和内嵌文档。IDE能自动识别这类信息并加以应用。
- 在实际运行的时候，paddle会导入正常的tensor类，从而不影响运行时的安全和效率。



### 主体设计选型考量

在实现类型提示的过程中，我们讨论了提升开发体验的三个最大的诉求：

- 不改变paddle主库的运行时效率和行为
- 能实现类型提示
- 最好能同步提供内嵌文档

所以我们总共考虑了以下四种可能的实现方法：

1. inline type annotation

使用带有类型提示的python代码重新实现整个`paddle/tensor/tensor.py`，

* 优点：
    * 使用最佳推荐方法
    * 代码的可阅读性也最好
    * IDE 友好型，天然支持类型提示
* 缺点：
    * 工作量比较大。
    * 相当于调整了`Tensor`对象的实现方式，存在一定的潜在风险。

推荐Paddle主库中的新模块代码都添加类型注释。

2. stub file in package

这种方式只需要在需要添加类型注释的module同目录下添加对应的stub file（`.pyi`文件）即可。

* 优点：
    * 工作量比较小
    * 可最小程度上支持`Tensor`的类型提示
* 缺点：
    * 临时解决方案
    * pyi文件不支持docstring，无法展示函数内嵌文档
    * 增加代码文件数量（如果只是支持Tensor智能提示的话，只需要增加一个文件即可）

针对于此方式，我已经实现了一个解决方案：[types-paddle](https://github.com/wj-Mcat/types-paddle)，也与Paddle有一个[讨论](https://github.com/PaddlePaddle/Paddle/issues/45979)，感兴趣的小伙伴可以看看。

3. distributed stub file

此方法即使用第三方库的形式来给Paddle添加类型注释，并IDE中所有模块的代码智能提示都从该第三方package当中走，不会扫描和加载原始Paddle库。

* 优点：
    * 代码智能提示的速度会非常快
    * 对主库不存在任何改动
* 缺点：
    * 需要安装第三方库才会生效
    * 不支持函数内嵌的Docstring
    * 需要与主库当中的API保持严格一致，不然IDE会识别不到新特性，此时产生的工作量将是：同时维护两个package。

其中 [@SigureMo](https://github.com/SigureMo)给出了[paddlepaddle-stubs](https://github.com/cattidea/paddlepaddle-stubs)解决方案。

4. 利用TYPE_CHECKING变量让Paddle库加载不同的Tensor定义

此方法即本文提出的方法，通过不同的Tensor类型定义，我们可以实现IDE和运行时的不同行为，实现类型提示。

- 优点：
  - 可以实现Docstring
- 缺点：
  - 会改动主库

***

综合考虑以上三种解决方案，推荐使用**第四种。**


## 2、关键技术点/子模块设计与实现方案

本实现中有一个关键因素，即函数签名和返回值类型的自动获取。以Tensor类为例，Tensor类中的方法可以通过inspect获得

### 传入参数类型的获得：

传入参数的获得是通过inspect 模块加上一定的规则匹配完成的。

TODO

### 返回值类型的获得：

我们总结了Tensor类中方法的来源，主要有如下三种情况：

- 定义在ops.yaml / legacy_ops.yaml中的算子函数

  - 每个函数都有对应的yaml定义，我们可以通过自动化的方法将yaml定义改写成函数签名：

    例如`tensor.trace()`函数：

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

    我们可以通过一定的字符串操作将其签名写出：

    ```python
    class Tensor:
    	def trace(self, offset:int=0, axis1:int=0, axis2:int=1, name: Optional[str] = None) -> None:
            pass
    ```
    

- 定义在.cc 中的函数

  - 这些函数可以通过人工阅读源代码维护一个返回值的列表，例如`tensor.topk()` 返回的类型是Tuple[Tensor, Tensor]，考虑到这类函数的变动通常不大，我们可以定期更新这个人工维护的列表。

- 一定的规则补充

  - 有一些函数名称以"\_"结束，表明其是inplace操作的函数，例如`tensor.zero_()`，我们默认返回值是None
  - 有一些函数会以"is\_"开始，表明其是判断函数，我们默认其返回值是bool。




## 3、主要影响的模块接口变化

### 请列出核心设计对应的直接接口变化

此方案只在IDE开发阶段影响paddle库加载的行为，在运行时没有影响。没有直接的接口变化。

### 请逐一排查对对框架各环节的影响

包括不限于网络定义、底层数据结构、OP、数据IO、执行、分布式、模型保存、预测部署各环节的影响。如没有影响也要注明。

## 4、测试和验收的考量

自测方案，CE，目标达成验收的度量方式。

# 五、示例

完成之后的效果截图如下所示：

![](https://user-images.githubusercontent.com/10242208/202877685-a722a168-1155-4705-91eb-d5f3e1279b21.jpg)


# 六、影响和风险总结

## 对用户的影响

## 对二次开发用户的影响

用户将在IDE开发的过程中体验到类型智能提升

## 对框架架构的影响

## 对性能的影响

##对与竞品差距与优势的影响

## 其他风险


# 七、规划排期

时间和人力规划，主要milestone

# 八、待讨论

需要进一步讨论的问题，开放性问题，有争议问题

# 九、评审意见

|问题 | 提出人 | 处理说明 | 状态 | 
|---|---|---|---|
| |  |  |  | 
| |  |  |  | 
| |  |  |  | 

# 名词解释

#附件及参考资料
