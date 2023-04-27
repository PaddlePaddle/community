# 飞桨框架 C++ 文档抽取与展示

|领域 | 飞桨框架 C++ 文档抽取与展示                       | 
|---|--------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | Liyulingyue、gouzil             | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-04-27                     | 
|版本号 | V1.0                           | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle>2.4               | 
|文件名 | 飞桨框架 C++ 文档抽取与展示.md<br> | 


# 一、概述
## 1、相关背景

自 paddle 2.3 版本开始，飞桨深度学习框架提供定义与用法与相应 Python API 类似的 C++ API，其 API 命名、参数顺序及类型均和相应的 paddle Python API 对齐，可以通过查找相应 Python API 的官方文档了解其用法，并在自定义算子开发时使用。通过调用这些接口，可以省去封装基础运算的时间，从而提高开发效率。

[中国软件开源创新大赛：飞桨框架任务挑战赛 赛题6](https://github.com/PaddlePaddle/Paddle/issues/53172#paddlepaddle06)要求为飞桨框架自动抽取和展示 C++ 文档，并上线至飞桨官网。

## 2、功能目标

在[飞桨API文档页面](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)引入新的章节`PADDLE_API`，用于展示飞桨当前的类 Python 的 C++运算 API。

展示内容应与Python API文档对其，包括：API名称、对应的Python API名称、API介绍、参数、返回值、示例代码。

## 3、意义

提升c++开发用户的开发体验。

# 二、飞桨现状

## 1、文档生成与更新
飞桨当前的英文文档信息保存在源代码的注释中，中文文档在`paddle/docs`目录下。每天，后台拉取develop分支，抽取英文文档和中文文档，转换为html后展示在官网上。

其中，英文文档的抽取代码可以开源。

## 2、 C++ API
飞桨的 C++ API 体系还在建设中，最终暴露给用户的API信息通过在安装根目录`site-packages/paddle/include/paddle`中搜索`PADDLE_API`获取。当前有12个class，450个API以及2个宏定义。

相比于Python API，无法在C++ API的源码中获取对应的API说明和示例代码。

# 三、设计思路与实现方案

## 1、 总述
综合考虑对当前的框架体系，拟通过人工构造与自动化脚本相结合的方式构造C++ API 文档。

其中，能够通过自动化脚本获取的信息有：
- C++ API的文件路径、接口注释、命名空间、返回值信息。这些信息可以通过扫描源代码文件自动提取。
- C++ API对应的Python API，由于两种语言的API命名几乎保持一直，可以通过搜索匹配的方式获取对应的Python API名称。
- API对应的介绍文本。C++ API和Python API的介绍文本可以保持一直。

无法确定能够通过自动化脚本获取的信息有：
- C++ API的参数信息和返回值信息，如果C++ API的参数信息完全与Python对齐，则C++文档直接抽取Python文档的参数信息即可，否则，则需要人工补足。
- C++ API的示例代码，C++API的示例代码目前需要人工补足。

此外，可以进一步在docs下构造C++的示例代码运行测试脚本，以提高文档质量。

## 2、 C++ API抽取

C++ API抽取可以通过Python脚本解析`site-packages/paddle/include/paddle`文件实现。

## 3、 C++ API 与 Python API 对齐

C++ API 与 Python API 有两种对齐方案，其一是通过文件记录映射表，将C++ API名称映射到指定的Python API上；其二，根据C++ API文件名直接匹配对应的Python API。

本次优先使用后者，对于特殊的API，使用第一种方案进行铆定。

## 4、 C++ API文档

C++ API文档能够在一定程度上自动更新，但由于其中包含大量不可更新元素，故 C++ API文档以类似于Paddle Python中文文档的形式，存放在Docs目录下。

以`PADDLE_API Tensor abs(const Tensor& x);`为例，其中文文档内容应为：

```python
.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. cpp:function:: PADDLE_API Tensor abs(const Tensor& x)

绝对值函数。

.. math::
    out = |x|

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同。

代码示例
:::::::::

#include "paddle/extension.h"

auto x = paddle::full({3, 4}, -1.0);
auto grad_x = paddle::abs(x);

```

该API对应的Python文档为：

```python
.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.abs(x, name=None)

绝对值函数。

.. math::
    out = |x|

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.abs

```

## 5、 更新与维护

每日更新时，拉取最新paddle源码，并编译对应的whl包，用于自动提取PADDLE_API。对于不同的提取结果，采用不同的策略：
- 对于没有提取到的API，可以被视为退场，docs目录下虽然保存对应的rst文件，但不在官网页面展示
- 对于新增的API，通过脚本自动匹配已有的Python文档，填充Python文档信息，记录对应C++ API 到指定文档，用于后期审查。
- 对于Python API描述发生改变，复制对应描述到C++ 的文档中。
- 对于Python API函数参数、示例代码发生改变，记录对应C++ API 到指定文档，用于后期审查。

# 四、测试和验收的考量

C++文档上线官网的develop分支。

# 五、排期规划
整个任务的规划实时步骤如下 
1. 构建API抽取脚本，实现API抽取
2. 构建API对齐脚本，用于根据抽取的API匹配当前已有的Python文档，生成对应rst。
3. 开放Tracking Issue，募集开发者人工审查文档信息，并补充C++ 示例代码
4. C++ 文档接入官网文档页面develop分支
5. C++ 文档自动化更新脚本接入官网文档页面develop分支
6. (可选) 在Docs仓库下构建C++ 代码审查CI

# 六、影响面

仅对文档展示页面存在影响。
