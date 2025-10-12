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

- 语言/运行时支持
	- 在 CPython 3.14 上默认启用 SOT，功能与 3.9–3.13 等价，无需用户修改业务代码。
	- 基于 PEP 523 的 `eval_frame` 在主解释器与子解释器均可正确安装/卸载，支持 free-threaded 构建，不共享可变全局状态。

- 字节码适配与版本路由
	- 虚拟机（VM）与生成层（CodeGen/Assembler）完整覆盖 3.14 关键变更。
	- 与 3.13 及更早版本共存：按 `sys.version_info` 路由不同 opcode 表与生成规则；移除/替代过时指令。

- 性能目标
	- 与 3.13 相比无显著回退：端到端基准回归 ≤ 3%（统计方差内），关键模型用例持平或更优；内存占用无异常增长与泄漏。

## 3、意义

动转静 SOT 模块是 PaddlePaddle 框架中非常重要的模块，能够帮助用户将动态图代码转换为静态图代码，从而提升模型的执行效率和性能。支持 Python 3.14 能够让更多用户在最新的 Python 版本下使用 PaddlePaddle 框架，提升用户体验，能够使得用户在使用动态图组网代码的情况下通过添加一行装饰器低成本编译优化

# 二、飞桨现状

当前 PaddlePaddle 对 Python 3.14 的支持总体处于建设阶段：主框架层面尚未在官方 CI 建立完整的 3.14 编译与单测链路、也未开展规模化的 3.14 单元/集成/模型用例验证，开发/调试镜像与预置环境仍需补齐；SOT 模块尚未适配 3.14 的字节码与 `eval_frame`，在 3.14 环境下将自动回退至 AST 路径，功能正确但相较基于虚拟机（VM）的字节码路径在捕获/优化能力与覆盖范围上受限。本 RFC 旨在补齐上述缺口，给出完整的 3.14 适配方案与验收标准（含 opcode/CodeGen 更新、`eval_frame` 安装策略、CI 增维与镜像产出）。


# 三、业内方案调研

目前主流的深度学习框架 PyTorch 都已经支持 Python 3.14 版本，并且在 dynamo 模块方面也有类似的实现和设计。我们可以参考这些框架的实现方式和设计思路，来进行 PaddlePaddle 框架中 SOT 模块的修改和适配。

torch 适配 python 3.14 的相关 pr :
  - 正式开始适配 python 3.14 [pytorch/pytorch#158184](https://github.com/pytorch/pytorch/pull/158184)
  - eval_frame 适配 [pytorch/pytorch#161555](https://github.com/pytorch/pytorch/pull/161555)
  - torch python 3.14 适配规划 [pytorch/pytorch#156856](https://github.com/pytorch/pytorch/pull/156856)

# 四、对比分析
## 1、Python 3.14 主要变化（字节码和 eval_frame 方面的）
Python 3.14 引入了一些新的特性和变化，主要包括以下几个方面：

（1）`eval_frame` 与解释器相关变化（与 SOT 直接相关）

- 自由线程模式完善（PEP 703，3.14 官方支持）：3.14 在 free-threaded 构建下启用更多优化（包含专用解释器的自适应特化）。对基于 `eval_frame`/`JIT` 的桥接代码有两点注意：
	- 引用计数策略变化：解释器在可能的情况下借用引用以减少 refcount 修改，这会导致通过 `Py_REFCNT(op)==1` 判断“唯一引用”的逻辑不再可靠；（影响的是 `cpython_internals` 部分）
	- 并发与隔离：free-threaded 下必须保证 faster guard 与 `JIT` 工件是线程安全且按解释器实例隔离（每 Interpreter 独立安装 `eval_frame` hook，不共享可变全局状态）。


参考资料：
- https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html 其中与解释器相关的章节：一种新型的解释器、自由线程模式的改进、实验性 JIT

（2）CPython 字节码的变化（与 python 虚拟机/字节码生成强相关）

重点与 SOT 相关的变更摘录（更完整表格参考 Paddle 议题）：
- `BINARY_SUBSCR` 被替换为 `BINARY_OP` 搭配 `oparg: NB_SUBSCR`（必须更新虚拟机（VM）与 CodeGen 的索引路径）。
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
- 生成层（CodeGen/Assembler）：
	- 插入合法 3.14 字节码：用 `BINARY_OP:NB_SUBSCR` 替换 `BINARY_SUBSCR`；用 `BUILD_MAP` 替换 `BUILD_CONST_KEY_MAP`；用 `LOAD_SPECIAL` 编码 `with`/`async with`；避免发出 `KW_NAMES`；
- `eval_frame` 安装策略：
	- free-threaded 下的锁粒度与缓存隔离（按 Interpreter/Thread 维度划分）；避免使用 `Py_REFCNT==1` 的假设，改用 3.14 提供的唯一引用检测 API。
	- 与 `sys.monitoring`/`JIT` 共存：支持配置项选择优先级；在官方 `JIT` 开启时打印提示或降级范围。
- 测试与验收：
    - CI 流水线能够监控 Python 3.14 SOT 单测
    - SOT 在 Python 3.14 下功能完备，全部 SOT 单测能够在 Python 3.14 下验证通过


# 五、设计思路与实现方案

## 1、主体设计思路与折衷
### 整体全貌
本方案延续过往 `3.12`/`3.13` 版本适配的分层策略，将 SOT 分为三层：
1) `eval_frame`（C/C++，`paddle/fluid/pybind/eval_frame.c` 及相关绑定）：基于 PEP 523 的 `frame_eval` 钩子在解释器维度安装与清理，负责将 Python 运行时的 `PyFrameObject` 交给 SOT。
2) 模拟执行（Python，`python/paddle/jit/sot/opcode_translator/executor/`）：以 VM 的方式解释执行新老指令集，维护执行栈/块栈/符号栈等状态，收集图与守卫（guards）。
3) CodeGen（Python，`python/paddle/jit/sot/opcode_translator/executor/pycode_generator.py` 等）：面向 `3.14` 的合法字节码插入，产出可回放的 `code object` 与可执行函数。
4) 兼容优先：以“最小改动覆盖新语义”为原则，不改动既有 3.9–3.13 路径；


### 主体设计具体描述
核心数据流：
`PyFrameObject` → `eval_frame` 钩子 → VM（按 `3.14` 指令集执行，应用守卫与黑白名单）→ 收束为静态子图 → CodeGen 生成 `code object` → 回写/替换执行。

主要改动点与文件位置（以 Paddle 主仓路径为例）：
- `paddle/fluid/pybind/sot/eval_frame.c`：
	- 适配 `3.14` 的 `PyFrameObject` 与 `code` 访问细节（如 `GetLocals` 路径、`f_lasti` 使用差异参照 `#69245` 实践）。
	- 为子解释器提供 `eval_frame` 安装/卸载 API，暴露按解释器作用域的上下文键（Interpreter id）。
- `python/paddle/jit/sot/opcode_translator/executor/opcode_executor.py`（VM）：
	- 新增/更新 `opcode` 分发表以覆盖 `3.14`：`BINARY_OP:NB_SUBSCR`、`LOAD_SPECIAL`、`LOAD_COMMON_CONSTANT`、`LOAD_SMALL_INT`、`LOAD_FAST_BORROW*`、`POP_ITER`、`JUMP_IF_TRUE/FALSE`、模板相关等。
	- `with`/`async with` 状态机改造：用 `LOAD_SPECIAL` 替换 `BEFORE_WITH/BEFORE_ASYNC_WITH` 路径，确保进入/退出协议的栈效果一致。
	- 严格布尔路径：在 VM 中对 `POP_JUMP_*`、`COMPARE_OP` 的布尔严格性进行规范化，必要时插入显式 `TO_BOOL` 等等（参考 `3.13` 的 `TO_BOOL` 处理经验）。
- `python/paddle/jit/sot/opcode_translator/executor/pycode_generator.py`（生成层）：
	- 插入合法 `3.14` 字节码：以 `BINARY_OP:NB_SUBSCR` 替换 `BINARY_SUBSCR`；用 `BUILD_MAP` 替换 `BUILD_CONST_KEY_MAP`；用 `LOAD_SPECIAL` 编码 `with`/`async with`；遵循 `CALL`/`CALL_KW` 新协议；`COMPARE_OP` 的 `oparg` 编解码适配。
	- 维持 `MAKE_FUNCTION` 附加属性通过 `SET_FUNCTION_ATTRIBUTE` 的处理方式（承接 `3.13`）。
- 配置与路由：在 SOT 初始化处增加 `3.14` 能力位，运行时按 `sys.version_info` 选择表与策略；提供 `SOT_PY314_STRICT_BOOL`、与官方 `JIT` 共存的开关等。
### 主体设计选型考量
选型因素与依据：
- 复用既有 VM/CodeGen 基础，仅增量扩展 `3.14` 指令与语义，避免引入新后端，降低风险与维护成本（参考 `#61173`、`#69245` 的做法）。
- 字节码生成坚持“就地合法化”策略：在 `3.14` 下插入 CPython 认可的最小集合指令，减少跨版本行为差异。

## 2、关键技术点/子模块设计与实现方案

1) `eval_frame` 适配与子解释器隔离
- 接口与语义：提供 `sot_enable_eval_frame(interpreter_id)`/`sot_disable_eval_frame(interpreter_id)`；为每个解释器维持独立上下文与缓存。

1) VM 指令支持与状态机
- 指令分派：为 `3.14` 新/变更指令补齐 VM 处理函数；为伪指令（如 `JUMP_IF_TRUE/FALSE`）实现“不改栈”的跳转行为；为 `LOAD_FAST_BORROW*` 保持栈一致性。
- `with`/`async with`：围绕 `LOAD_SPECIAL` 重写进入流程，校验 `__enter__`/`__aenter__`、`__exit__`/`__aexit__` 的取值与栈布局；补齐 `END_FOR`/`END_SEND` 组合路径的异常清理（参照 `3.12` 经验）。
- 布尔严格化：对 `POP_JUMP_*`、`COMPARE_OP` 加统一布尔规范层，必要时在 VM 内进行 `bool` 化以对齐 CPython 要求。

1) 生成层（CodeGen/Assembler）
- 指令插入：统一使用 `BINARY_OP:NB_SUBSCR`、`BUILD_MAP`、`LOAD_SPECIAL`、`CALL`/`CALL_KW` 等 `3.14` 合法指令；按需插入 `LOAD_COMMON_CONSTANT`、`LOAD_SMALL_INT`，并保证回放一致性。
- 操作数编码：实现 `COMPARE_OP` 的 `oparg` 高位/强制布尔位编码；与 `3.13` 保持差异化路径。
- 函数构造：延续 `SET_FUNCTION_ATTRIBUTE` 方案；

## 3、主要影响的模块接口变化

无对外接口变化，均为内部实现改动。

# 六、测试和验收的考量

1. CI 流水线能够监控 Python 3.14 SOT 单测
2. SOT 在 Python 3.14 下功能完备，全部 SOT 单测能够在 Python 3.14 下验证通过

# 七、影响面

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

# 八、排期规划

1. PR-CI-SOT 流水线上线 Python 3.14 监控，确保已有单测不会回归 - 3天
2. Eval Frame 适配 - 10天
3. 字节码适配 - 30天
4. 模型和各组件库适配 - 待定

# 名词解释

# 附件及参考资料

1. [Python 3.11 支持规划](https://github.com/PaddlePaddle/PaddleSOT/issues/357)
2. [SOT Python3.12 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/61173)
3. [SOT Python 3.13 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/69245)
4. [Python 3.14 新变化-一种新型的解释器](https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html#tail-call-interpreter)
5. [Hackathon 9th FundableProject 任务合集](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_9th/%E3%80%90Hackathon_9th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md)
