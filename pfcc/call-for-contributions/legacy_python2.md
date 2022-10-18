# Python 2.7 相关代码退场
> This project will be mentored by [@luotao1](http://github.com/luotao1)
>
> Tracking issue: [PaddlePaddle/Paddle#46837](https://github.com/PaddlePaddle/Paddle/issues/46837)

## 一、概要
### 1、相关背景
[Python 2.7在2020年1月1日终止支持](https://www.python.org/doc/sunset-python-2/)，Paddle从2.1版本（2021年）开始不再维护Python 2.7。
Python 2 和 Python 3 在整型、字符串、比较运算符、算数运算符、`round()`、输入/输出函数、文件操作函数、`exec/execfile()`、`xrange()/ range()`、字典、`zip()`、`map()`、`filter()`、迭代器、缩进方式、异常处理、`object`、模块导入方式、列表推导式等都有差异，详细见 [差异与兼容一览表](https://zhuanlan.zhihu.com/p/385023142)。

对 Paddle 中 Python 2.7相关代码进行退场可以提高源码整洁性，提升开发者阅读的便利性。同时，开发者可以直接使用python 3 新特性，无需考虑和 python 2 的兼容性，更专注于编写代码逻辑。

### 2、功能目标

对 Paddle 中 Python 2.7相关代码进行退场，提升开发者阅读和开发源码的便利性。

### 3、方案要点

Python2.7 相关代码退场，主要涉及到针对 Python 2 subpackage（子包）、module（模块） 与 requirement（运行环境依赖）等几个方面的处理：

- 删除Python 2 子包
- 删除没有其它功能的 Python 2 模块
- 删除非必要的环境依赖
- 清理 Python2 相关逻辑分支
- 清理文档中涉及到 Python 2 的内容

## 二、意义
Paddle从2.1版本（2021年）开始不再维护Python 2.7，因此，可以对相关代码进行退场，开发者可以直接使用python 3 新特性，无需考虑和 python 2 的兼容性，提升开发源码的便利性，更专注于编写代码逻辑。

举例：

1. 整型的处理，不需要写两个分支。[PR#46696](https://github.com/PaddlePaddle/Paddle/pull/46696/files)

<img width="937" alt="image" src="https://user-images.githubusercontent.com/6836917/196356294-ead46c28-f81a-460f-8b04-ca3075b96875.png">


2. 对于`range()`的处理，不需要引入`six`子包来兼容。[tensorflow/commit/cfc45e1](https://github.com/tensorflow/tensorflow/commit/cfc45e1027d43cf54b0358d7d9e2fc01f58938dd)

<img width="954" alt="image" src="https://user-images.githubusercontent.com/6836917/196356599-28a9eb1b-1b72-4bb1-bf93-da175508ce74.png">

3. 在开头加上`from __future__ import print_function` 这句之后，即使在 python2.X ，使用 print 就得像 python3.X 那样加括号使用。python2.X 中 print不需要括号，而在 python3.X 中则需要。去掉`__future__`后更简单。[PR#46686](https://github.com/PaddlePaddle/Paddle/pull/46686/files)
```python
#python2.7
print "hello world"
#python3
print("hello world")
```
<img width="945" alt="image" src="https://user-images.githubusercontent.com/6836917/196360783-515444ee-17e7-4e7b-b77a-64196e3742e4.png">

## 三、业内调研

### Python 2 涉及到的特性

#### Python 2 子包
1. [six](https://pypi.org/project/six/)：six 的设计目的是为了解决Python2和3的不兼容问题，名字的来源就是 2×3=6, SIX = Python2 Times Python3。不再维护 Python 2.7 后，整个 six 库都可以删除。
2. `__future__`：使用 3.0 及更高版本的内置特性后，均无需从 `__future__` 模块 import。 因此可以移除全部 `from __future__ import xxx` 结构，见 [future 官方说明](https://docs.python.org/3/library/__future__.html)。

#### Python 2 模块

**1. 删除类中不必要的显式 `object` 继承**

Python 2.x 中默认都是经典类，只有显式继承了 `object` 才是新式类函数；Python 3.x 中默认都是新式类，经典类被移除，没必要显式继承 `object` 。见 [why-do-python-classes-inherit-object](https://stackoverflow.com/questions/4015417/why-do-python-classes-inherit-object) ：

<img width="660" alt="image" src="https://user-images.githubusercontent.com/6836917/196360960-56ce8c1c-c081-4f64-a542-b99ae378b764.png">
<img width="665" alt="image" src="https://user-images.githubusercontent.com/6836917/196361119-6bc76af3-1358-42be-9ad3-70a1a6c81d6f.png">

结论：之前为了兼容性的考虑，保留了显式 `object` 继承，如果 python2.7 退场，可以删除这些显式继承。

**2. 删除 `super()` 函数中不必要的参数**

见 [When do you need to pass arguments to python super()?](https://stackoverflow.com/questions/59538746/when-do-you-need-to-pass-arguments-to-python-super) 和 https://peps.python.org/pep-3135/ 中的解释：

<img width="770" alt="image" src="https://user-images.githubusercontent.com/6836917/196361507-b224bb5a-5f8e-41a0-b788-efdf4f3969eb.png">

### Tensorflow
TF 从今年1月份开始逐步清理 python2.7 代码，见[Cleanup legacy Python2 PR 列表](https://github.com/tensorflow/tensorflow/search?p=2&q=python2%20legacy&type=commits)，包含以下内容：

|清理内容 | tf commit 示例 | 
|---|---|
|Remove usage of the `six` package（删除six子包） | [tensorflow/commit/cfc45e1](https://github.com/tensorflow/tensorflow/commit/cfc45e1027d43cf54b0358d7d9e2fc01f58938dd) |
|Remove unecessary `__future__` imports（删除`__future`子包） |[tensorflow/commit/cfc45e1](https://github.com/tensorflow/tensorflow/commit/cfc45e1027d43cf54b0358d7d9e2fc01f58938dd)|
|Remove unnecessary explicit `object` inheritance for classes（删除显式`object`继承）| [tensorflow/commit/cfc45e1](https://github.com/tensorflow/tensorflow/commit/cfc45e1027d43cf54b0358d7d9e2fc01f58938dd)|
|Remove unnecessary arguments to `super()` calls（删除`super()`函数中不必要的参数）|[tensorflow/commit/fb5a58c](https://github.com/tensorflow/tensorflow/commit/fb5a58c02a4638570b3a803e65655f06b91f24f9)|

### Pytorch
Pytorch 从2020年8月开始逐步清理 Python2.7 代码，见 [Legacy Python2 and early Python3 leftovers](https://github.com/pytorch/pytorch/issues/42919)，包括：

- Remove unused six code for Python 2/3 compatibility
- Clean up future imports for Python 2 

Pytorch 的清理力度没有 Tensorflow 全面。

## 四、设计思路与实现方案
### 1、主体设计思路与折衷
Paddle Python2.7 相关代码退场涉及如下内容：

- 删除Python 2 子包：`six`和`__future__`
- 删除没有其它功能的 Python 2 模块：类中不必要的显式`object`继承和`super()`函数中不必要的参数，和`python/paddle/compat.py`文件
- 删除非必要的环境依赖
- 清理 Python2 相关逻辑分支
- 清理文档中涉及到 Python 2 的内容

由于 Paddle 不再维护 Python 2.7，因此，代码退场的 PR 只要 CI 能通过，就可以合入，无其他风险。

### 2、 关键技术点/子模块设计与实现方案
#### 删除 Python 2 子包
-  `Six`
   - 有250多处import six：其中有些import是冗余的，可以直接删除；其余需要进行更改
   - 有68处from six：涉及使用了six的string_types, zip, range, xrange, cStringIO, cPickle等
   - [python/requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)：移除six库
-  `__future__`【已完成 by [SigureMo](https://github.com/SigureMo) 和 [Yulv-git](https://github.com/Yulv-git) 】   
   - [Paddle#46411](https://github.com/PaddlePaddle/Paddle/pull/46411) [Paddle#46463](https://github.com/PaddlePaddle/Paddle/pull/46463) 移除存量，[Paddle#46466](https://github.com/PaddlePaddle/Paddle/pull/46466) 控制增量

#### 删除没有其它功能的 Python 2 模块
- 删除类中不必要的显式`object`继承：约530处
- 删除`super()`函数中不必要的参数：约1600处
- 删除 [python/paddle/compat.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/compat.py) 文件：该文件是为了兼容设计的API，框架中使用`compat.xxx`部分可以直接用Python 3的API代替。 

#### 删除非必要的环境依赖
Paddle 镜像 [tools/dockerfile/Dockerfile.ubuntu](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/dockerfile/Dockerfile.ubuntu#L83) 安装了Python 2.7.15，可以进行删除来减少镜像体积大小。同时可以删除其中的 `pip --no-cache-dir install xxx` 内容。此项工作完成后，需统计一下包体积变化。

#### 清理 Python2 相关逻辑分支

部分代码中使用 `sys.version_info` （共62处）来区分不同 Python 版本，并对不同版本做不同处理， Python2 逻辑分支可以删除。
```python
# https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/param_attr.py#L87
if sys.version_info.major == 2:
    check_type(name, "name", (str, type(None), unicode), "ParamAttr")
else:
    check_type(name, "name", (str, type(None)), "ParamAttr")

# 其中 Python2 分支可以直接删除
check_type(name, "name", (str, type(None)), "ParamAttr")

# 其他可全局搜索 `sys.version_info` 根据具体情况来处理
```

#### 清理文档中涉及到 Python 2 的内容
在 [docs](https://github.com/PaddlePaddle/docs) 仓库下用`grep -irn python2 . | wc -l`， 可以看到有53条结果。如 [飞桨框架昆仑XPU版安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/hardware_support/xpu_docs/paddle_install_cn.html)：

<img width="543" alt="image" src="https://user-images.githubusercontent.com/6836917/196361719-1fd6b624-2608-4153-8a20-491ae40d52db.png">

# 五、影响和风险总结
## 对用户的影响
对开发者的影响：

- 提高源码整洁性，提升开发者阅读的便利性；
- 开发者可以直接使用python 3 新特性，无需考虑和 python 2 的兼容性，提升开发源码的便利性，更专注于编写代码逻辑。

对用户使用模型影响（无）：

- Paddle 从 2.1 版本（2021年4月）开始不再发 Python 2.7 的包，目前套件 repo 都已支持 Python3 ，因此用户使用模型无影响。

## 风险

由于 Paddle 不再维护 Python 2.7，因此，代码退场的 PR 只要 CI 能通过，就可以合入，无其他风险。
