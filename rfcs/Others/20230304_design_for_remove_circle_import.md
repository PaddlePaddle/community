# Circle import 消除设计文档

|任务名称 | 清理动态 import语句，解决circle import 问题                      | 
|---|-------------------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | 张一乔                                                   | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-04                                            | 
|版本号 | 1.0                                                   | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发                                | 
|文件名 | 提交的markdown设计文档文件名称，如：20230304_design_for_remove_circle_import.md<br> | 

# 一、概述
## 1、相关背景
目前飞桨框架 python/paddle/jit 目录下部分模块函数存在动态 import 语句，即 import 语句并未放到文件的最开始地方。这种动态 import 语句存在两个问题：
- 可能因为存在 circle import 而临时添加的，但不合规范
- 每次调用函数都会解释执行 import，影响性能

参考页面：
- https://github.com/PaddlePaddle/Paddle/issues/50663#task89
- https://github.com/PaddlePaddle/Paddle/discussions/50711

## 2、功能目标
在保持 API 不变的情况下，适当调整函数或文件位置，以实现「分层管理」。

# 二、设计思路与实现方案

## 1、整体全貌
目前jit文件夹下共下述文件存在函数中import的情况：
- python\paddle\jit\api.py
- python\paddle\jit\dy2static\call_transformer.py
- python\paddle\jit\dy2static\convert_call_func.py
- python\paddle\jit\dy2static\convert_operators.py
- python\paddle\jit\dy2static\partial_program.py
- python\paddle\jit\dy2static\program_translator.py
- python\paddle\jit\dy2static\utils.py

这些文件中的import行为及修改策略可以分为以下几类：
- 可以直接移动到文件头。此类import语句虽然在函数中执行，但是直接移动到头部不影响编译。
- 无需处理。此类import仅在本文件中声明，也仅由本函数使用，用于进行环境检查。如：import six
- 被import对象较为简单。此类import仅从其他文件中引入函数，这个函数较为单纯不依赖于其他函数，可以直接将该函数移动到对应文件中实现。
- 需要拆分。例如utils.py应当为基础模块，但是引用了static_analysis；并且static_analysis依赖于fluid，因此需要对两个文件的函数调用信息进行分层，拆解新的文件。

## 2、实施计划
考虑到import行为的复杂性，以及对外部（如fluid等非jit目录下模块）的引入，实施过程中，应当按照逐个文件，逐个功能点，先内部再外部的方式进行解耦。

不同文件的更新内容如下

| 文件名                                               | 函数中import语句                                                                                                                                                                                                                                                                                   | 说明                                                                                |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| python\paddle\jit\api.py                          | from paddle.fluid.io import save_inference_model                                                                                                                                                                                                                                              | 参考https://github.com/PaddlePaddle/Paddle/pull/50677 直接解决                          |
| python\paddle\jit\dy2static\call_transformer.py   | from paddle.jit.dy2static.convert_call_func import is_builtin                                                                                                                                                                                                                                 | is_builtin作为公用函数移动到utils                                                          |
| python\paddle\jit\dy2static\convert_call_func.py  | import six</br>from paddle.nn import Sequential</br>paddle.jit.dy2static.program_translator import()                                                                                                                                                                                          | 第一个import无需修改；第二个import暂时挂起；将program_translator对本文件的import内容移动到program_translator |
| python\paddle\jit\dy2static\convert_operators.py  | from paddle.fluid.dygraph.base import in_declarative_mode</br>from paddle.static.nn.control_flow import Assert                                                                                                                                                                                | 暂时挂起                                                                              |
| python\paddle\jit\dy2static\partial_program.py    | from paddle.incubate.autograd.primapi import to_prim</br>from paddle.amp.auto_cast import _in_amp_guard, _in_pure_fp16_guard                                                                                                                                                                  | 暂时挂起                                                                              |
| python\paddle\jit\dy2static\program_translator.py | from paddle.static import InputSpec</br>from paddle.incubate.autograd.primapi import to_prim</br>from paddle.fluid.dygraph.base import _switch_declarative_mode_guard_</br>from paddle.jit.dy2static.program_translator import ProgramTranslator                                              | 前两条暂时挂起，最后一条直接删除import即可                                                          |
| python\paddle\jit\dy2static\utils.py              | from paddle.jit.dy2static.return_transformer import</br> import paddle, fluid ...</br>import numpy as np</br>from .static_analysis import StaticAnalysisVisitor</br>from paddle.jit.dy2static.static_analysis import NodeVarType</br>from paddle.jit.dy2static.ifelse_transformer import</br>from paddle.jit.dy2static.loop_transformer import |jit内的circle import统一将实现移动到utils内|

处理完毕后按照引用情况查找对应的fluid文件调整circle import。

# 三、测试和验收的考量
PR通过CI即可

# 四、影响面
无

# 五、排期规划
排期时间规划暂不定，主要按照以下顺序进行：
- 优先处理上层文件中的引用，如jit\api.py等
- 优先处理jit内部的文件中循环引用
- 最后处理jit与fluid等外部文件的循环引用
