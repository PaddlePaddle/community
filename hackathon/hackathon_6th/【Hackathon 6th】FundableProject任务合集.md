此文档展示 **PaddlePaddle Hackathon 第六期活动——Fundable Projects** 任务详细介绍。Fundable Projects 赛道定位硬核任务，要求高水平的开发者独立进行任务拆解和完成，更多详见 [PaddlePaddle Hackathon 说明]()。

## 产出要求

- 任务拆解 tracking issue
- 答辩 PPT
- 书面的技术报告
- 代码运行无误，通过社区 maintainers 的评审并合入代码仓库。

## 任务详情

### 一、为 Paddle 框架 API 添加类型提示（Type Hints）

**任务背景：**

Python 在 [3.0 版本](https://peps.python.org/pep-3107/)引入了类型提示功能，并在 [PEP 484](https://peps.python.org/pep-0484/) 中将其规范化。之后随着相关规范和工具的逐渐完善，类型提示逐渐成为 Python 代码中的标准实践，如今较新的主流库也基本都提供了类型提示。由于 Python 本身类型高度动态化，类型提示的存在可以帮助开发者更快地了解代码的类型信息，提高代码的可读性和可维护性，结合工具还可以提供静态类型检查，在开发阶段就能发现一些潜在的类型错误。Paddle 框架由于历史原因尚未提供类型提示，本任务希望引入尽可能多的对 Paddle 有利的类型提示。

**详细描述：**

- 添加通用类型提示模块，为用户提供便捷的类型提示 aliases
- 为 Paddle 全部公开 API 添加类型提示
- 在 CI 中添加公开 API 类型提示的检查，确保公开 API 存在类型提示
- （可选）为 Paddle 内部函数添加类型提示

**验收说明：**

- Paddle 框架公开 API 类型提示覆盖率达到 100%
- CI 中添加公开 API 类型提示的检查，确保公开 API 存在类型提示
- 通过打包后的 wheel 包安装，可以在主流编辑器中显示正确的类型提示信息

**技术要求：**

- 熟练掌握 Python 语言
- 有 Mypy、Pyright 等类型检查工具使用经验，及项目中使用类型提示的经验
- 熟悉 [PEP 561](https://peps.python.org/pep-0561/)、[PEP 563](https://peps.python.org/pep-0563/) 等重要 Python 类型提示相关规范

**参考资料：**

- [paddlepaddle-stubs](https://github.com/cattidea/paddlepaddle-stubs)
- [Static Typing with Python](https://typing.readthedocs.io/en/latest/)
- [typing — Support for type hints](https://docs.python.org/3/library/typing.html)
- [Paddle 集成类型提示的早期 RFC（注意和本任务目标不同，因此实现路径也不同，参考价值不多）](https://github.com/PaddlePaddle/community/pull/346)

### 二、引入 clang-tidy

**任务背景：**

飞桨是集深度学习核心训练和推理部署、基础模型库、端到端开发套件和丰富的工具组件于一体的深度学习框架。飞桨在追求高性能的同时，也非常关注框架自身的安全隐患以及健壮性，例如：我们严格且严谨地将一切 warning 视为 error，引入 clang-tidy 有助于增加飞桨的健壮性。

**详细描述：**

- 希望引入尽可能多的对 paddle 有利的 clang-tidy 的功能。
- 作用于项目全局，即对飞桨所有在 clang-tidy 功能覆盖范围之内的所有文件都应起作用。
- 利用 clang-tidy 使得 paddle 代码更加健壮，例如消除 paddle 尽可能多的编译 warning 等，并可顺利编译使用 paddle。
- 可以在之前开发者 [未开发完的](https://github.com/PaddlePaddle/Paddle/issues/54073) 基础上进行开发。

**验收说明：**

- 存量修复：修复由 clang-tidy 能够检测出的错误类型的存量问题。
- 增量拦截：将 clang-tidy 同步到 CI，将相应存量修复完成的错误类型进行增量拦截。

### 三、Paddle 框架旧动态图机制退场

**任务背景：**

飞桨 Paddle 资 2.0 版本以来，进行了多个重大机制改造。包括：高可复用算子库 PHI、全新的动态图体系。随着新机制的发布使用，旧机制和功能代码需要进行退场和移除，保持架构清晰和代码库的条理性，为内外部开发者提供更好的二次开发环境。这就包括了 Operators 算子库的清理、旧动态图机制代码的清理。

**详细描述：**

- Operators 算子库的清理（前置依赖项，占约 70%工作）
  - 梳理 Operators 算子库显存算子清单
  - 删除不再使用算子
  - 仍在使用算子迁移至 PHI 算子库
- 清理新动态图兼容 Operators 算子库的代码，包括但不限于：
  - Python C API 自动代码生成模块
  - 新动态图中间态 AutoGrad 模块
- 算子单测基类及算子单测改造，包括但不限于：
  - 探索新动态图下算子基类的技术路线
  - 清理依赖老动态图 append_op 的单测基类逻辑
  - 如有必要，联动改造算子单测
- 清理旧动态图 C++代码，包括但不限于：
  - 移除 C+++ imperative 不再使用的逻辑

**验收说明：**

- Operators 算子库不再存在于 Paddle 框架中
- 新动态图中间态不再存在于 Paddle 框架中
- 算子单测使用新动态图机制
- 旧动态图 C++代码不再存在于 Paddle 框架中

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 C++和 Pybind 的使用
- 了解动态图执行原理

**参考资料：**

- [飞桨 PHI 算子库介绍](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2994)
- [PHI 算子库 kernel 注册全流程](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/PHI_kernel_registration/PHI_kernel_registration.md)
- [Kernel 选择分发体系梳理与优化](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/kernel_selection/20221130_kernel_selection.md)
- [飞桨动态图技术设计](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/Dygraph)
- [飞桨动态图官方介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/quick_start/dynamic_graph.html#dongtaitu)

### 四、Paddle 框架旧执行器功能退场

**任务背景：**

飞桨 Paddle 自 2.0 版本以来，进行了多个重大机制改造。包括：高可复用算子库 PHI、全新的动态图体系、全新的静态图执行引擎等。随着新机制的发布使用，旧机制和功能代码需要进行退场和移除，保持架构清晰和代码库的条理性，为内外部开发者提供更好的二次开发环境。这就包括了 Operators 算子库的清理、旧动态图机制代码的清理、旧静态图执行引擎的清理。

2.0 版本后静态图虽然是非默认形态，但以 Interpreter 为中心的内核执行器正式取代了旧的 ParallelExecutor 执行器，提供了更优调度策略的新执行器。因此飞桨考虑虑将此系列旧执行器进行退场处理。

**详细描述：**

- 移除 ParallelExecutor、SSAGraphExecutor、以及派生出的相关执行器
- 移除与上述执行器相关联的模块，如 OpHandle 等组件
- 移除与执行器相关的 Python 端类、单测

**验收说明：**

- Paddle 框架无旧执行器关类、函数和单测
- 上下游关联模块同步删除，CMakeLists.txt 删除对应编译依赖

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 C++和 Pybind 的使用
- 了解执行器模块代码

**参考资料：**

- [飞桨静态图执行流程](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/static_graph_execution)
- [飞桨全新执行器升级](https://www.paddlepaddle.org.cn/documentation/docs/zh/release_note_cn.html#jingtaituxinzhixingqiquanmianshangxian)

### 五、全套件模型接入动转静训练功能

**任务背景：**

目前飞桨的开源套件如 PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR 等，都支持了动转静训练功能，但是并非所有的模型都接入了`--to_static`策略，随着 PaddleSOT 功能的完善和上线，动转静训练成功率大幅度提升，故此任务旨在对开源套件中所有模型进行动转静训练策略推全。

**详细描述：**

需要对现有的 PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR、PaddleRec、PaddleGAN、PaddleVideo、PaddleYOLO 套件中的所有模型依次添加 to static 策略，支持开启动转静进行训练，且保证对套件模型尽可能少的代码侵入。

**验收说明：**

- 明确全套件列表，包含：PaddleClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR、PaddleRec、PaddleGAN、PaddleVideo、PaddleYOLO
- 在每个套件中，同学需要调研搜集套件的所有模型列表，并对所有模型的动转静支持情况进行调研，**产出《待支持动转静模型列表文档》**
- 针对每个待支持动转静的模型，对套件代码进行修改，以支持动转静训练。同时提供开启动转静训练前后前 50 个 step 的 loss 一致性截图作为 PR 描述，[样例 PR](https://github.com/PaddlePaddle/PaddleNLP/pull/1290/files)
- 让飞桨验收 RD review，同意并合入 PR 后，此模型视为接入动转静。

**技术要求：**

- 熟练掌握 Python 语言、了解飞桨套件模型

**参考资料：**

- [飞桨动态图转静态图官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/index_cn.html)
- [飞桨动转静 SOT 训练技术设计](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/symbolic_opcode_translator)

### 六、解决 PaddleOCR 中的长期存在的 issue

**任务背景：**

PaddleOCR 项目有非常多的使用者，在 issue 区的讨论也很多。甚至有不少 issue 已经是长期存在的 issue。这些 issue 缺少诊断，复现，以及修复。因此，期望能够挑选部分长期存在的，讨论较多的，issue，能够对齐进行分析，复现，以及解决。

**详细描述：**

- 从 PaddleOCR 积累的 issue 中挑选 10 个左右长期存在的 issue，对其进行深入分析。
- 对这 10 个 issue，进行复现，找到其问题的 root cause，并提出解决方案。
- 提交 PR 到 PaddleOCR 仓库，对这些问题进行修复。

**验收说明：**

- 给出所挑选的 issue 的列表和分析报告及解决方案。
- 给出修复的 PR list，或者 修复的建议。

**技术要求：**

- 了解开源开发及 PaddleOCR 项目

**参考资料：**

- [PaddleOCR 中按照 most commented 排序的 issue 列表。](https://github.com/PaddlePaddle/PaddleOCR/issues?q=is%3Aissue+is%3Aopen+sort%3Acomments-desc)
