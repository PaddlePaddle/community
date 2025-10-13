# 动转静 SOT Python 3.14 支持

|任务名称 | 动转静 SOT Python 3.14 支持 | 
|---|---|
|提交作者 | gouzil | 
|提交时间 | 2025-10-11 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20251011_sot_py314_support.md | 

# 一、概述
## 1、相关背景

动转静 SOT 模块是基于 Python 字节码的 JIT 编译模块，旨在在运行时将 PaddlePaddle 动态图组网代码转换为静态图组网代码，具体设计参见：[PaddleSOT 项目介绍](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/symbolic_opcode_translator)
由于 SOT 模块需要针对每条字节码进行模拟执行，并在 CodeGen 时生成合法的 Python 字节码，因此对于 Python 版本非常敏感。我们现在对 Python 3.9-3.13 已经有了较为全面的支持，但新发布的 Python 3.14 目前还是不支持的，因此需要专项对 Python 3.14 进行支持

## 2、功能目标

1. PR-CI-SOT 流水线上线 Python 3.14 监控，确保已有单测不会回归
2. 适配 Eval Frame 模块，适配模拟执行、CodeGen 等流程


## 3、意义

动转静 SOT 模块是 PaddlePaddle 框架中非常重要的模块，能够使得用户在使用动态图组网代码的情况下通过添加一行装饰器低成本编译优化，从而提升模型的执行效率和性能。支持 Python 3.14 能够让更多用户在最新的 Python 版本下使用 PaddlePaddle 框架，提升用户体验。

# 二、飞桨现状

当前 PaddlePaddle 对 Python 3.14 的支持总体处于建设阶段：主框架层面尚未在官方 CI 建立完整的 3.14 编译与单测链路、也未开展规模化的 3.14 单元/集成/模型用例验证，开发/调试镜像与预置环境仍需补齐。

SOT 模块尚未适配 3.14 的字节码与 `eval_frame`，在 3.14 环境下将自动回退至 AST 路径，功能正确但相较基于模拟执行模块的字节码路径在捕获/优化能力与覆盖范围上受限。本 RFC 旨在补齐上述缺口，给出完整的 3.14 适配方案与验收标准。


# 三、业内方案调研

目前主流的深度学习框架 PyTorch 已经支持 Python 3.14 版本，并且在 dynamo 模块方面也有类似的实现和设计。我们可以参考这些框架的实现方式和设计思路，来进行 PaddlePaddle 框架中 SOT 模块的修改和适配。

Python 3.14 引入了一些新的特性和变化，主要包括以下几个方面：

（1）`eval_frame` 与解释器相关变化（与 SOT 直接相关）

- 自由线程模式完善（PEP 703，3.14 官方支持）：3.14 在 free-threaded 构建下启用更多优化（包含专用解释器的自适应特化）。
- 所有对 `frameobjects` 中对象的引用都使用 `_PyStackRef` 而不是 `PyObject *`, 以减少引用计数的修改次数，从而提升性能（影响的是 `cpython_internals` 部分, 也会影响到现有对 func_closure 的引用）。

参考资料：
- https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html 其中与解释器相关的章节：一种新型的解释器、自由线程模式的改进、实验性 JIT
- `_PyStackRef` 相关改动说明：https://github.com/python/cpython/issues/127705

（2）CPython 字节码的变化（与 Python 虚拟机/字节码生成强相关）

重点与 SOT 相关的变更摘录（更完整表格参考 Paddle 议题）：
- `BINARY_SUBSCR` 被替换为 `BINARY_OP` 搭配 `oparg: NB_SUBSCR`。
- 移除 `BUILD_CONST_KEY_MAP`（使用 `BUILD_MAP` 实现常量键字典的构造）。
- `with`/`async with` 编码路径变化：移除 `BEFORE_WITH` 和 `BEFORE_ASYNC_WITH`，新引入 `LOAD_SPECIAL` 以支撑 `with`/`async with` 进入协议（需要调整上下文管理器进入/退出的栈效果保持一致）。目前 SOT [尝试过适配](https://github.com/PaddlePaddle/Paddle/pull/72736) with 相关字节码，但因为一些其他原因[被回滚了](https://github.com/PaddlePaddle/Paddle/pull/73816)。本次任务不包含 with 相关支持。
- 新增 `LOAD_COMMON_CONSTANT`（替代 `LOAD_ASSERTION_ERROR`），并扩展对常见异常常量的载入（例如 `NotImplementedError`）。
- 新增/增强的指令（节选）：
	- `LOAD_SMALL_INT`、`LOAD_CONST_IMMORTAL`（常量加载优化）。
	- `LOAD_FAST_BORROW`、`LOAD_FAST_BORROW_LOAD_FAST_BORROW`（借用引用的快速加载；对 Python 语义无可见差异，但虚拟机执行需要保持栈行为一致）。
	- `POP_ITER`（配合“虚拟”迭代器生命周期）。
	- `NOT_TAKEN`（供 `sys.monitoring` 上报分支事件）。
	- `BUILD_TEMPLATE`、`BUILD_INTERPOLATION`（PEP 750 模板字符串字面值支持）。
	- `JUMP_IF_TRUE`、`JUMP_IF_FALSE`（伪指令，栈不变的条件跳转，替代 `COPY 1`、`TO_BOOL`、`POP_JUMP_*` 的组合）。
- 调用协议更新：
	- 3.13 起 `CALL` 与 `CALL_KW` 分离；3.14 进一步引入 `CALL_KW` 的特化伪指令（`CALL_KW_PY`/`CALL_KW_BOUND_METHOD`/`CALL_KW_NON_PY`），以利于解释器特化与内联缓存。
	- `KW_NAMES` 在 3.14 路径中不再使用；SOT 生成含关键字参数的调用需遵循 `CALL_KW` 新路径。
- 比较与跳转：
	- `POP_JUMP_IF_TRUE`/`POP_JUMP_IF_FALSE` 自 3.13 起要求栈顶为“精确 `bool`”；3.14 新增的 `JUMP_IF_TRUE`/`JUMP_IF_FALSE` 伪指令不改变栈。

参考资料：
- Python 3.14 字节码变化：https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html#cpython-bytecode-changes
- Paddle 字节码差异表（含 3.14）：https://github.com/PaddlePaddle/Paddle/issues/69134#issue-2631286178

（3）SOT 的适配要点与建议

- 模拟执行层：
	- 覆盖新/变更指令：`BINARY_OP:NB_SUBSCR`、`LOAD_SPECIAL`、`LOAD_COMMON_CONSTANT`、`LOAD_SMALL_INT`、`LOAD_FAST_BORROW*`、`POP_ITER`、`JUMP_IF_TRUE/FALSE`、`BUILD_TEMPLATE/BUILD_INTERPOLATION` 等；
	- 调整 `with`/`async with` 的状态机与异常路径（`END_FOR`/`END_SEND`/`CLEANUP_THROW` 等已在 3.12+ 引入，需与 3.14 组合验证）。
- CodeGen 模块：
	- 插入合法 3.14 字节码：用 `BINARY_OP:NB_SUBSCR` 替换 `BINARY_SUBSCR`；用 `BUILD_MAP` 替换 `BUILD_CONST_KEY_MAP`；用 `LOAD_SPECIAL` 编码 `with`/`async with`；避免发出 `KW_NAMES`；


PyTorch 适配 Python 3.14 的相关 PR 如下：
- 正式开始适配 Python 3.14 [pytorch/pytorch#158184](https://github.com/pytorch/pytorch/pull/158184)
- eval_frame 适配 [pytorch/pytorch#161555](https://github.com/pytorch/pytorch/pull/161555)
- PyTorch Python 3.14 适配规划 [pytorch/pytorch#156856](https://github.com/pytorch/pytorch/pull/156856)


# 四、设计思路与实现方案

## 1、主体设计思路与折衷
### 整体全貌
本方案延续过往 `3.12`/`3.13` 版本适配的分层策略，将 SOT 分为三层：
1) `eval_frame`（C/C++，`paddle/fluid/pybind/eval_frame.c` 及相关绑定）：基于 PEP 523 的 `frame_eval` 钩子在解释器维度安装与清理，负责将 Python 运行时的 `PyFrameObject` 交给 SOT。
2) 模拟执行（Python，`python/paddle/jit/sot/opcode_translator/executor/`）：以模拟执行的方式解释执行新老指令集，维护执行栈/块栈/符号栈等状态，收集 `FunctionGraph` 与 `Guard`。
3) CodeGen（Python，`python/paddle/jit/sot/opcode_translator/executor/pycode_generator.py` 等）：面向 `3.14` 的合法字节码插入，产出可回放的 `code object` 与可执行函数。
4) 兼容优先：以“最小改动覆盖新语义”为原则，不改动既有 3.9–3.13 路径；


### 主体设计具体描述
核心数据流：
`PyFrameObject` → `eval_frame` 钩子 → 模拟执行（按 `3.14` 指令集执行，应用守卫与黑白名单）→ 收束为静态子图 → CodeGen 生成 `code object` → 回写/替换执行。

主要改动点与文件位置（以 Paddle 主仓路径为例）：
- `paddle/fluid/pybind/sot/eval_frame.c`：
	- 适配 `3.14` 的 `PyFrameObject` 与 `code` 访问细节。
- `python/paddle/jit/sot/opcode_translator/executor/opcode_executor.py`（模拟执行模块）：
	- 新增/更新 `opcode` 分发表以覆盖 `3.14`：`BINARY_OP:NB_SUBSCR`、`LOAD_COMMON_CONSTANT`、`LOAD_SMALL_INT`、`LOAD_FAST_BORROW*`、`POP_ITER`、`JUMP_IF_TRUE/FALSE`等。
- `python/paddle/jit/sot/opcode_translator/executor/pycode_generator.py`（CodeGen 模块）：
	- 插入合法 `3.14` 字节码：以 `BINARY_OP:NB_SUBSCR` 替换 `BINARY_SUBSCR`；用 `BUILD_MAP` 替换 `BUILD_CONST_KEY_MAP`；遵循 `CALL`/`CALL_KW` 新协议（伪指令）。


### 主体设计选型考量

选型因素与依据：
- 大部分逻辑复用既有 3.9–3.13 路径，降低适配成本（参考 `#61173`、`#69245` 的做法）。
- CodeGen 模块保持原有设计，新增部分需要的 `3.14` 字节码，确保一致性。

## 2、关键技术点/子模块设计与实现方案

模拟执行指令支持
- `NOT_TAKEN`: 不会有任何操作，但是需要支持，防止打断。
- `POP_ITER`: 从栈顶移除迭代器。与 `POP_TOP` 保持一致的栈行为。
- `LOAD_COMMON_CONSTANT`: 将一个普通常量推入栈顶。 有一个列表在 [cpython](https://github.com/python/cpython/blob/7519ac294fc5c4fd7fb9cb8dc0edc960688cf887/Python/pylifecycle.c#L814) 中
- `LOAD_SMALL_INT`: 将整数 i 推入栈顶。 i 必须在 range(256) 范围内。应该创建一个 Variable 把他 push 上去就行了。
- `BUILD_TEMPLATE`、`BUILD_INTERPOLATION`: 可能需要创建一个新的 Variable 来表示 template string。
- `LOAD_FAST_BORROW`: 从描述上看是跟 LOAD_FAST 一样的。它还有一个特化 `LOAD_FAST_BORROW_LOAD_FAST_BORROW`
- `LOAD_SPECIAL`: with 相关的，这期不做支持。


## 3、主要影响的模块接口变化

无对外接口变化，均为内部实现改动。

# 五、测试和验收的考量

1. CI 流水线能够监控 Python 3.14 SOT 单测
2. SOT 在 Python 3.14 下功能完备，全部 SOT 单测能够在 Python 3.14 下验证通过

# 六、影响面

## 对用户的影响

- 用户在 Python 3.14 环境下使用 PaddlePaddle 框架时，可以直接使用 SOT 功能，无需进行额外的配置或修改代码。

## 对二次开发用户的影响

无

## 对框架架构的影响

无

## 对性能的影响

无

## 其他风险

无

# 七、排期规划

1. PR-CI-SOT 流水线上线 Python 3.14 监控，确保已有单测不会回归 - 7天
2. Eval Frame 适配 - 10天
3. 模拟执行模块与 CodeGen 模块适配 - 14 天
4. SOT 单测验证推全 - 14 天
5. 模型和各组件库适配 - 待定

# 附件及参考资料

1. [Python 3.11 支持规划](https://github.com/PaddlePaddle/PaddleSOT/issues/357)
2. [SOT Python3.12 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/61173)
3. [SOT Python 3.13 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/69245)
4. [Python 3.14 新变化-一种新型的解释器](https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html#tail-call-interpreter)
5. [🗓️ Python 各版本字节码差异表格](https://github.com/PaddlePaddle/Paddle/issues/69134)
6. [Hackathon 9th FundableProject 任务合集](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_9th/%E3%80%90Hackathon_9th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md)
