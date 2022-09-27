# Python 2.7 相关代码退场
## 背景
Python 2.7在2020年1月1日终止支持，Paddle从2.1版本（2021年）开始不再维护Python 2.7，并准备在2.5版本（2023年）完成Python 2.7相关代码的退场。
主要涉及到针对 Python 2 subpackage（子包）、module（模块） 与 requirement（运行环境依赖）等几个方面的处理：
* 删除Python 2 子包
* 删除没有其它功能的 Python 2 模块
* 删除非必要的环境依赖
* 清理文档中涉及到 Python 2 的内容

## 具体内容
如有遗漏，欢迎大家补充！
### 删除 Python 2 子包
#### `Six`
[six](https://pypi.org/project/six/) 的设计目的是为了解决Python2和3的不兼容问题，名字的来源就是 2×3=6, SIX = Python2 Times Python3。由于Paddle不再维护Python 2.7，整个six库都可以删除。
* 有250多处import six：其中有些import是冗余的，可以直接删除；其余需要进行更改
* 有68处from six：涉及使用了six的string_types, zip, range, xrange, cStringIO, cPickle等
* [python/requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)：移除six库
#### `__future__`【已完成 by [SigureMo](https://github.com/SigureMo) 】
Paddle 目前代码里使用的全部是 3.0 及更高版本的内置特性（`generator_stop` 和 `annotations` 未使用），均无需从 `__future__ impor`t。
因此可以移除全部 `from __future__ import xxx` 结构。
* future 详细说明见：https://docs.python.org/3/library/__future__.html 
* [Paddle#46411](https://github.com/PaddlePaddle/Paddle/pull/46411) [Paddle#46463](https://github.com/PaddlePaddle/Paddle/pull/46463) 
移除存量，[Paddle#46466](https://github.com/PaddlePaddle/Paddle/pull/46466) 控制增量

### 删除没有其它功能的 Python 2 模块
[python/paddle/compat.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/compat.py) 文件内为了Python3和2的兼容设计的API，
整个文件可以删除。框架中使用`compat.xxx`部分可以直接用Python 3的API代替。

### 删除非必要的环境依赖
Paddle 镜像 [tools/dockerfile/Dockerfile.ubuntu](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/dockerfile/Dockerfile.ubuntu#L83) 
安装了Python 2.7.15，可以进行删除来减少镜像体积大小。
同时可以删除其中的`pip --no-cache-dir`内容。

### 清理文档中涉及到 Python 2 的内容
在 [docs](https://github.com/PaddlePaddle/docs) 仓库下用`grep -irn python2 . | wc -l`， 可以看到有53条结果。
一个具体的例子是：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/hardware_support/xpu_docs/paddle_install_cn.html

## 可行性分析和规划排期
由于 Paddle 不再维护 Python 2.7，因此，代码退场的PR只要CI能通过，就可以合入，无其他风险。
