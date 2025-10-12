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

## 3、意义

动转静 SOT 模块是 PaddlePaddle 框架中非常重要的模块，能够帮助用户将动态图代码转换为静态图代码，从而提升模型的执行效率和性能。支持 Python 3.14 能够让更多用户在最新的 Python 版本下使用 PaddlePaddle 框架，提升用户体验和满意度。

# 二、飞桨现状

目前 PaddlePaddle 框架已经支持 Python 3.9-3.13 版本，但对于 Python 3.14 还不支持。由于 Python 3.14 引入了一些新的特性和变化，因此需要对 SOT 模块进行相应的修改和适配，才能够在 Python 3.14 下正常工作。


# 三、业内方案调研

目前主流的深度学习框架 PyTorch 都已经支持 Python 3.14 版本，并且在 SOT 模块方面也有类似的实现和设计。我们可以参考这些框架的实现方式和设计思路，来进行 PaddlePaddle 框架中 SOT 模块的修改和适配。

# 四、对比分析
## 1、Python 3.14 主要变化（字节码和 eval_frame 方面的）
Python 3.14 引入了一些新的特性和变化，主要包括以下几个方面：

（1）`eval_frame` 与解释器相关变化（与 SOT 直接相关）

- 新型解释器实现（tail-call interpreter，配置开关 `--with-tail-call-interp`）：属于 CPython 内部实现细节，不改变 Python 语义，但可能影响基于 `eval_frame` 的性能特征与热点分布，建议在 3.14 基线与启用该解释器时分别做一轮基准评估。[参考：Python 3.14 新变化-一种新型的解释器]
- 自由线程模式完善（PEP 703，3.14 官方支持）：3.14 在 free-threaded 构建下启用更多优化（包含专用解释器的自适应特化）。对基于 `eval_frame`/`JIT` 的桥接代码有两点注意：
	- 引用计数策略变化：解释器在可能的情况下借用引用以减少 refcount 修改，这会导致通过 `Py_REFCNT(op)==1` 判断“唯一引用”的逻辑不再可靠；若 SOT 在 C/C++ 扩展层使用此类优化，需要改用 3.14 提供的替代检测方法（如 `PyUnstable_Object_IsUniqueReferencedTemporary()` 等）。
	- 并发与隔离：free-threaded 下必须保证 SOT 的缓存、守卫（guards）与 `JIT` 工件是线程安全且按解释器实例隔离（每 Interpreter 独立安装 `eval_frame` hook，不共享可变全局状态）。
- 标准库子解释器（PEP 734）：`concurrent.interpreters` 向 Python 层暴露了创建子解释器的能力。`eval_frame` 钩子是“按解释器生效”的，SOT 需要在新建子解释器时再次安装/初始化自身（包括环境、缓存与黑名单/白名单策略）。跨解释器的共享缓存需显式隔离（建议以 `PyInterpreterState`/`PyThreadState` 或 Interpreter id 作为 key 作用域）。
- 安全外部调试器接口（PEP 768）与 `sys.monitoring`：3.14 增强了监控/调试基础设施，并新增 `NOT_TAKEN` 伪指令用于分支事件记录。若 SOT 利用 `sys.monitoring` 做热点探测或分支覆盖，需要适配这些事件，同时避免与 `eval_frame` 重入冲突（必要时设置优先级或禁用重叠功能）。
- 实验性 `JIT`（PEP 744）与 `sys._jit`：官方 macOS/Windows 包可选择开启 `JIT`。开启后 `eval_frame` 仍会工作，但热点可能被官方 `JIT` 提前接管，影响 SOT 的捕获时机与收益。建议：
	- 启动时探测 `sys._jit` 可用性与是否启用；
	- 给出告警或自动降级策略（例如仅在 JIT 关闭时启用 SOT，或在二者同时开启时限制 SOT 的转换范围）。

参考资料：
- https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html
- 其中与解释器相关的章节：一种新型的解释器、自由线程模式的改进、实验性 JIT、sys.monitoring/调试接口

（2）CPython 字节码的变化（与 python 虚拟机/字节码生成强相关）

重点与 SOT 相关的变更摘录（更完整表格参考 Paddle 议题）：
- `BINARY_SUBSCR` 被替换为 `BINARY_OP` 搭配 `oparg: NB_SUBSCR`（必须更新虚拟机（VM）与 CodeGen 的索引路径）。
- 移除 `BUILD_CONST_KEY_MAP`（使用 `BUILD_MAP` 实现常量键字典的构造）。
- `with`/`async with` 编码路径变化：移除 `BEFORE_WITH` 和 `BEFORE_ASYNC_WITH`，新引入 `LOAD_SPECIAL` 以支撑 `with`/`async with` 进入协议（需要调整上下文管理器进入/退出的栈效果保持一致）。
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
	- `COMPARE_OP` 的 `oparg` 编码延续 3.13 规则（高位存放 `cmp_op` 索引，增加强制布尔转换位）。
	- `POP_JUMP_IF_TRUE`/`POP_JUMP_IF_FALSE` 自 3.13 起要求栈顶为“精确 `bool`”；3.14 新增的 `JUMP_IF_TRUE`/`JUMP_IF_FALSE` 伪指令不改变栈。

参考资料：
- Python 3.14 字节码变化：https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html#cpython-bytecode-changes
- Paddle 字节码差异表（含 3.14）：https://github.com/PaddlePaddle/Paddle/issues/69134#issue-2631286178

（3）SOT 的适配要点与建议

- 版本路由与 opcode 表：为 3.14 新增独立的 opcode 映射与特性开关；在运行时根据 `sys.version_info` 选择 3.9–3.13 与 3.14 的不同路径。
- 虚拟机层（VM/Tracer）：
	- 覆盖新/变更指令：`BINARY_OP:NB_SUBSCR`、`LOAD_SPECIAL`、`LOAD_COMMON_CONSTANT`、`LOAD_SMALL_INT`、`LOAD_FAST_BORROW*`、`POP_ITER`、`JUMP_IF_TRUE/FALSE`、`BUILD_TEMPLATE/BUILD_INTERPOLATION` 等；
	- 调整 `with`/`async with` 的状态机与异常路径（`END_FOR`/`END_SEND`/`CLEANUP_THROW` 等已在 3.12+ 引入，需与 3.14 组合验证）。
	- 严格布尔语义：对 `POP_JUMP_*`、`COMPARE_OP` 强制布尔位等在虚拟机执行阶段显式规范化，避免隐式 truthy/falsy 导致的分支偏差。
- 生成层（CodeGen/Assembler）：
	- 插入合法 3.14 字节码：用 `BINARY_OP:NB_SUBSCR` 替换 `BINARY_SUBSCR`；用 `BUILD_MAP` 替换 `BUILD_CONST_KEY_MAP`；用 `LOAD_SPECIAL` 编码 `with`/`async with`；避免发出 `KW_NAMES`；`MAKE_FUNCTION` 附加属性使用 `SET_FUNCTION_ATTRIBUTE` 路径（延续 3.13 行为）。
	- `COMPARE_OP` 的 `oparg` 编码/解码适配 3.14；必要时提供 3.13/3.14 双分支。
- `eval_frame` 安装策略：
	- 每子解释器安装 hook；在 `concurrent.interpreters` 创建新解释器时自动初始化 SOT。
	- free-threaded 下的锁粒度与缓存隔离（按 Interpreter/Thread 维度划分）；避免使用 `Py_REFCNT==1` 的假设，改用 3.14 提供的唯一引用检测 API。
	- 与 `sys.monitoring`/`JIT` 共存：支持配置项选择优先级；在官方 `JIT` 开启时打印提示或降级范围。
- 测试与验收：
    - CI 流水线能够监控 Python 3.14 SOT 单测
    - SOT 在 Python 3.14 下功能完备，全部 SOT 单测能够在 Python 3.14 下验证通过


# 五、设计思路与实现方案

## 1、主体设计思路与折衷
### 整体全貌
本方案延续过往 `3.12`/`3.13` 版本适配的分层策略，将 SOT 分为三层：
1) `eval_frame` 注入层（C/C++，`paddle/fluid/pybind/eval_frame.c` 及相关绑定）：基于 PEP 523 的 `frame_eval` 钩子在解释器维度安装与清理，负责将 Python 运行时的 `PyFrameObject` 交给 SOT。
2) 虚拟机层（VM/Tracer，Python，`python/paddle/jit/sot/opcode_translator/executor/`）：以 VM 的方式解释执行新老指令集，维护执行栈/块栈/符号栈等状态，收集图与守卫（guards）。
3) 生成层（CodeGen/Assembler，Python，`python/paddle/jit/sot/opcode_translator/executor/pycode_generator.py` 等）：面向 `3.14` 的合法字节码插入，产出可回放的 `code object` 与可执行函数。

4) 兼容优先：以“最小改动覆盖新语义”为原则，不回退既有 3.9–3.13 路径；采用“版本路由 + 能力位（feature flag）”方式分流。


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
- 函数构造：延续 `SET_FUNCTION_ATTRIBUTE` 方案；清理 `KW_NAMES` 旧路径。

## 3、主要影响的模块接口变化

无对外接口变化，均为内部实现改动。

# 六、测试和验收的考量

1. CI 流水线能够监控 Python 3.14 SOT 单测
2. SOT 在 Python 3.14 下功能完备，全部 SOT 单测能够在 Python 3.14 下验证通过

# 七、影响面

## 对用户的影响

无

## 对二次开发用户的影响

无

## 对框架架构的影响

- 官方 JIT 的共存策略，以避免两者在热点探测与转换范围上的冲突，确保用户在启用官方 JIT 时仍能获得稳定的 SOT 行为。
- free-threaded 模式下的线程安全与缓存隔离，避免多线程环境中状态污染或竞态。
- multiple interpreter 的支持与隔离，确保子解释器间不共享可变状态。

## 对性能的影响

无

## 其他风险

无

# 八、排期规划

1. Eval Frame 适配 - 2025-10-11 ~ 2025-10-30
2. 字节码适配 - 2025-10-31 ~ 2025-11-30
3. 模型和各组件库适配 - 待定

# 名词解释

# 附件及参考资料

1. [Python 3.11 支持规划](https://github.com/PaddlePaddle/PaddleSOT/issues/357)
2. [SOT Python3.12 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/61173)
3. [SOT Python 3.13 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/69245)
4. [Python 3.14 新变化-一种新型的解释器](https://docs.python.org/zh-cn/3.14/whatsnew/3.14.html#tail-call-interpreter)
5. [Hackathon 9th FundableProject 任务合集](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_9th/%E3%80%90Hackathon_9th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md)
