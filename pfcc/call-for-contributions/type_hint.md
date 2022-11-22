# Type Hint for Tensor

> this project will be mentored by [@wj-Mcat](http://github.com/wj-Mcat)

## 背景

在深度学习代码中充斥这各种`Tensor`，可是Python是一门动态编程语言，如果不给变量添加类型，IDE将无法进行智能提示，故Type Annotation（类型注释）对于提升编程体验至关重要。然而，在Paddle框架当中的Tensor对象直接是通过`pybind11`将C++`Tensor`类型暴露给外部开发者，并没有对应的类型信息，故无法进行静态类型推断或动态stub file等操作。

为了给Paddle的Tensor添加类型注释的功能，可使用stub-file相关技术来实现此功能。

## Type Hint技术调研

实现Type Hint的场景大概有三种：

1. 直接在代码中添加类型注释（inline type annotation）。
2. 保证原始代码不变，通过stub file的形式添加类型提示（stub file in package）。
3. 对pacakge不做任何改动，通过第三方package的形式来给Paddle添加类型注释（distributed stub file）。

### 分析

1. inline type annotation

这种方式需要实现`paddle/tensor/tensor.py`中的TODO，具体解决方案目前看来工作量较大，难以实现。

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
    * 增加代码文件数量（如果只是支持Tensor智能提示的话，只需要增加一个文件即可）

针对于此方式，我已经实现了一个解决方案：[types-paddle](https://github.com/wj-Mcat/types-paddle)，也与Paddle有一个[讨论](https://github.com/PaddlePaddle/Paddle/issues/45979)，感兴趣的小伙伴可以看看。

3. distributed stub file

此方法即使用第三方库的形式来给Paddle添加类型注释，并IDE中所有模块的代码智能提示都从该第三方package当中走，不会扫描和加载原始Paddle库。

* 优点：
    * 代码智能提示的速度会非常快
    * 对主库不存在任何改动
* 缺点：
    * 需要安装第三方库才会生效
    * 需要与主库当中的API保持严格一致，不然IDE会识别不到新特性，此时产生的工作量将是：同时维护两个package。

其中 [@SigureMo](https://github.com/SigureMo)给出了[paddlepaddle-stubs](https://github.com/cattidea/paddlepaddle-stubs)解决方案。

***

综合考虑以上三种解决方案，推荐使用第二种。

## 可行性分析和规划排期

1. 需要python3.7+版本支持，而Paddle也正在放弃对于python3.6的支持。
2. 从技术的角度而言，通过stub file的方式添加类型注释是最简单的。
3. 从改动存在的潜在风险而言，stub file的方式对于代码运行不存在任何影响。
4. 从工作量而言，stub file可自动生成，也可以人为编辑，工作量适中。

## 参考链接

* [Introduction of Stub File](https://mypy.readthedocs.io/en/stable/getting_started.html#stubs-intro)
* [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
* [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
