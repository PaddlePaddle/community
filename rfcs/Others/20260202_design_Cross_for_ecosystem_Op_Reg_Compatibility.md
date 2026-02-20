# 跨生态自定义算子注册层兼容能力增强

|任务名称 | 跨生态自定义算子注册层兼容能力增强 |
|---|---|
| 提交作者 | gouzil |
| 提交时间 | 2026-02-02 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
|文件名 | 20260202_design_Cross_for_ecosystem_Op_Reg_Compatibility.md<br> |

# 一、概述
## 1、相关背景
飞桨在跨生态自定义算子接入上，逐步形成了自底向上的多层兼容架构：
- **C++ API 兼容层**：提供与其他框架相近的 C++ API 表达；
- **算子注册兼容层**：适配对方框架的注册/派发语义；
- **Python 接口兼容层**：兼容 Python 侧调用方式与对象语义；
- **Python API 代理层**：将对方框架的 API 代理到飞桨实现。

该方案已在 PyTorch、TVM FFI 生态的部分算子库完成验证，但在实际使用中暴露出注册兼容层的若干短板：
1) `TORCH_LIBRARY` 注册缺少 schema 支持，类型推导与参数绑定能力不足；
2) 不支持默认参数、keyword 参数等复杂传参场景；
3) 仅支持单 backend 注册，导致多 backend 场景下的实现需要额外改造。

这些问题在适配 [paddlecodec](https://github.com/PFCCLab/paddlecodec) 过程中尤为突出，导致需要对原始库做较多改动。本 RFC 聚焦“算子注册兼容层”的能力增强，使跨生态自定义算子在飞桨侧尽可能保留原生注册与调用方式。

## 2、功能目标
1) **为 `TORCH_LIBRARY` 注册机制添加 schema 支持**，用于类型推导与参数绑定；
2) **基于 schema 支持默认参数与 keyword 参数**，兼容复杂调用方式；
3) **支持多 backend 注册与派发**，实现不同设备/后端的实现切换；
4) **尽量减少生态库适配改动**，以 paddlecodec 为代表的生态库可保留原生注册方式。


## 3、意义
- **显著降低生态库接入成本**：保留原生 `TORCH_LIBRARY` 注册风格，减少适配代码与维护成本。
- **提升跨生态兼容完整性**：支持默认参数/keyword 参数/多 backend 后，调用语义与 PyTorch 更一致。
- **可扩展基础能力**：schema 与派发逻辑完善后，可复用到更多生态库或其他框架兼容层。

# 二、飞桨现状
从现有 `test/cpp/compat/torch_library_test.cc` 可见，兼容层已经覆盖了部分核心能力：
- **基础注册与调用**：支持 `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL` 注册，能通过 `OperatorRegistry` 查找并以 `FunctionArgs/IValue` 触发调用；当前测试均为 `DispatchKey::CPU`。
- **增量注册**：支持 `TORCH_LIBRARY_FRAGMENT` 分片注册多个算子。
- **自定义类**：支持 `ClassRegistry` 注册类、构造函数、成员方法与静态方法，并进行调用。
- **参数类型覆盖**（已有测试涉及）：Tensor/ScalarType/int/double/string；int const/const& 入参；`torch::optional<T>`、`c10::optional<c10::ArrayRef<int64_t>>`；`c10::ArrayRef<int64_t>`；`optional<Tensor> const&`。
- **返回类型覆盖**（已有测试涉及）：`Tensor`、`List<Tensor>`（`std::vector`）、`Tuple<Tensor,...>`（`std::tuple`）。
- **IValue 基础类型**：None/bool/int/double/string/tensor/list/tuple 的构造与类型检查。

尚存在的主要不足：
- **缺乏 schema 解析/注册与绑定能力**：测试使用显式 schema 字符串与纯位置参数调用，尚未覆盖默认参数、keyword 参数、keyword-only 绑定；
- **参数绑定/类型推导不足**：未形成统一的 schema 驱动绑定逻辑，复杂调用仍需手工处理。

# 三、业内方案调研
## PyTorch
- 通过 `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` 完成注册；
- Schema 采用 `c10::FunctionSchema`，支持类型描述、默认值、keyword-only 等；
- 使用 Dispatcher 按 `DispatchKey`（backend）进行派发；
- Python 调用层基于 schema 做参数绑定与类型检查。

## 结论
PyTorch 的注册与 schema 机制与本任务高度一致，是最匹配的参考对象。

# 四、对比分析
- **PyTorch**：
  - 优点：schema 完整、参数绑定与类型推导一体化、backend dispatch 成熟；
  - 缺点：实现体系较重，完全复刻成本高。

**结论**：采用“**PyTorch schema 语义的最小子集** + **飞桨兼容层轻量实现**”的折衷路线：满足参数绑定、默认值、keyword 与 backend 选择的核心需求，同时避免引入过重的 dispatcher 体系。

# 五、设计思路与实现方案

## 1、主体设计思路与折衷
### 整体全貌（示意）
```mermaid
flowchart LR
  A[第三方库: TORCH_LIBRARY] --> B[兼容层注册: Schema 解析/注册]
  B --> C[Compat Op Registry]
  C --> D[参数绑定/类型推导]
  D --> E[Backend Dispatch]
  E --> F[Backend Kernel Impl]
  F --> G[Paddle Kernel/Op]
```

### 主体设计具体描述
1) `TORCH_LIBRARY` 注册阶段：
   - 解析 schema 字符串（如 `op(Tensor x, int k=1, *, bool flag=False) -> Tensor`）；
   - 生成 `CompatSchema` 并存入 registry；
   - 基于 schema 生成或补充飞桨侧 `KernelSignature`/参数描述。

2) `TORCH_LIBRARY_IMPL` 注册阶段：
   - 以 `backend key` 维度保存实现函数指针；
   - 同一算子可注册多个 backend 实现。

3) 调用阶段：
   - 参数绑定器基于 schema 完成 **位置参数 + keyword 参数 + 默认值** 绑定；
   - 按注册的 `DispatchKey` 选择实现并调用对应 backend kernel。

### 主体设计选型考量
- **严格校验 vs 宽松绑定**：默认仅做参数绑定与必要类型转换，不强制全量 schema 校验；可通过环境变量启用严格校验用于调试。

## 2、关键技术点/子模块设计与实现方案

### 2.1 Schema 表达与解析
**核心能力**：解析 `TORCH_LIBRARY` schema 并建立兼容层的 `CompatSchema`。

**实现要点**：
- **现状对齐**：`OperatorRegistration` 中的 schema 目前以 `std::string` 形式保存，`Library::def` 在遇到 `schema` 字符串时通过 `OperatorRegistry::register_schema` 直接注册原始字符串（未做解析）。
- **目标改造（对齐 PyTorch）**：注册阶段将 schema string **解析为结构化 `FunctionSchema` 并保存**，后续参数绑定/类型推导均以结构化 schema 为准；原始字符串仅作为输入，不作为持久“数据源”（需要展示时通过 `operator<<` 输出）。
- 实现 `c10::FunctionSchema`：
  - 覆盖当前实现的核心类型（标量/字符串/Tensor/Optional/Tuple/固定长度列表 `T[N]`）；
  - 支持 alias/mutation 标注（如 `Tensor(a!)`）并保存在 schema 元数据中；
  - 支持 `default value`、`*` keyword-only 分隔符、`...`（vararg/varret）；
  - 保留参数名与默认值字符串，用于参数绑定。
- Registry 以 `namespace::op` 为 key：保存 `FunctionSchema` 与 `implementations`，必要时可缓存派发所需的 `signature`/`metadata`；schema 字符串仅用于解析输入或调试展示。

**类型映射**：
| PyTorch schema 类型 | 典型 C++ 类型（PyTorch） | Paddle 兼容层映射建议 |
|---|---|---|
| float | double | double |
| int | int64_t | int64_t |
| bool | bool | bool |
| str | std::string | std::string |
| Tensor | at::Tensor | paddle::Tensor |
| (T1, T2, ...) | std::tuple<T1, T2, ...> | std::tuple<...> |
| (Tensor, Tensor, Tensor)? | std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> | std::optional<std::tuple<paddle::Tensor, paddle::Tensor, paddle::Tensor>> |
| Tensor(a!) / Tensor(a) | at::Tensor（带别名/写标注） | Tensor + alias/mutation 标注（建议保留在 schema 元数据中） |
| float?/int?/bool?/str?/Tensor? | std::optional<T> | std::optional<T> |
| T[N] / T[N]? | 固定长度列表（支持可选） | 固定长度列表（支持可选） |


补充说明：
- 重点覆盖 torchcodec 所需的 Optional/Tuple/alias/kw-only/default/variadic 语法；
- `Tensor(a!)` 中 `(a)` 表示 alias 集合，`!` 表示该 alias 会被写入；属于 schema 的别名/可变性标注，类型仍是 `Tensor`。
- `(Tensor, Tensor, Tensor)?` 表示可选的 Tuple（`None` 或三元组 Tensor），常见于需要一次性传入多路映射/元信息的场景。

**FunctionSchema 相关数据结构（当前实现）**：
- **c10::FunctionSchema**：算子 schema 的核心结构，包含参数列表、返回列表、可变参数标识等。
- **c10::FunctionSchema 已实现方法**：`arguments()`、`returns()`、`is_vararg()`、`is_varret()`、`checkSchema()`。
- **c10::Argument**：单个参数/返回的结构化描述，通常包含名称、类型（TypePtr）、默认值（IValue）、是否 keyword-only、alias 信息等。
- **c10::Type / c10::TypePtr**：类型系统节点指针，当前落地重点为 TensorType、OptionalType、TupleType、StringType、NumberType、DeviceObjType 等。
- **c10::IValue**：默认值与运行时参数的通用载体（schema default value 及调用时实参承载）。
- **c10::AliasInfo**：别名/可变性标注的数据结构（支持 `Tensor(a!)` 语义）。
- **Schema 解析器**：`torch::jit::parseSchema` / `parseSchemaOrName` 与 `SchemaTypeParser` 负责从字符串构造 `FunctionSchema` 与类型节点。


### 2.2 参数绑定（默认参数 + keyword 参数）
**核心能力**：在调用时依据 schema 完成参数绑定与默认值注入。

**实现要点**：
- 注册表与函数对象均绑定结构化 schema：
  - `OperatorRegistration` 的 schema 存储由 `std::string` 升级为 `std::optional<std::variant<std::string, FunctionSchema>>`；
  - `register_schema`/`register_implementation` 会将 `FunctionSchema` 绑定到 `CppFunction`；
  - `CppFunction::call_with_args` 在调用前执行 schema 驱动的参数归一化。
- 输入：`positional args`、`keyword args`、`schema`；
- 输出：`bound args`（按 schema 顺序排列），用于后续调用；
- 处理规则：
  - 位置参数从左到右绑定；
  - `*` 之后参数仅允许通过 keyword；
  - 缺省参数按 schema 默认值填充；
- 位置参数与 keyword 参数重复绑定时报错；
- 传入不存在的 keyword 报错；
- 对 schema 不覆盖的类型，保持“透传”并延后到 kernel 内部处理。
- `FunctionArgs` 支持 `named_args_` 与 `add_arg(torch::arg(...)=...)` 形式的 keyword 传参，并对重复 keyword/空 value 报错。

**正确示例（默认值/keyword/kw-only/列表与可选）**：
```cpp
// 0) keyword + 默认值（含 kw-only）
// schema: nms(Tensor boxes, Tensor scores, float iou=0.5, int topk=-1, *, bool normalized=False) -> Tensor
// 调用: nms(boxes, scores, topk=200)
// 绑定结果: boxes, scores, iou=0.5, topk=200, normalized=False

// 1) 纯默认值补全
// schema: roi_align(Tensor x, Tensor rois, int pooled_h=7, int pooled_w=7) -> Tensor
// 调用: roi_align(x, rois)
// 绑定结果: x, rois, pooled_h=7, pooled_w=7

// 2) 位置参数 + keyword 混合
// schema: topk(Tensor x, int k=1, int axis=-1, bool largest=True, bool sorted=True) -> (Tensor, Tensor)
// 调用: topk(x, 5, sorted=False)
// 绑定结果: x, k=5, axis=-1, largest=True, sorted=False

// 3) keyword-only 参数
// schema: softmax(Tensor x, int axis=-1, *, bool use_cudnn=True) -> Tensor
// 调用: softmax(x, axis=1, use_cudnn=False)
// 绑定结果: x, axis=1, use_cudnn=False
// 非法: softmax(x, 1, False)  // kw-only 不能用位置参数

// 4) keyword 覆盖默认值
// schema: clamp(Tensor x, float? min=None, float? max=None) -> Tensor
// 调用: clamp(x, max=0.0)
// 绑定结果: x, min=None, max=0.0

// 5) Tuple 参数（作为单一入参）
// schema: blend((Tensor, Tensor) inputs, float alpha=0.5) -> Tensor
// 调用: blend((x, y), alpha=0.3)
// 绑定结果: inputs=(x,y), alpha=0.3

// 6) Optional Tuple 参数（None 或三元组）
// schema: add_video_stream(Tensor(a!) decoder, *, (Tensor, Tensor, Tensor)? custom_frame_mappings=None) -> ()
// 调用: add_video_stream(decoder, custom_frame_mappings=None)
// 绑定结果: decoder, custom_frame_mappings=None

// 7) alias/mutation 标注（in-place 语义）
// schema: normalize_(Tensor(a!) x, float eps=1e-5) -> Tensor(a!)
// 调用: normalize_(x, eps=1e-6)
// 绑定结果: x, eps=1e-6  // x 被原地修改
```

**错误示例**：
```cpp
// 1) 未知 keyword
// schema: add(Tensor a, Tensor b) -> Tensor
// 调用: add(a, b, axis=1)
// 错误: unexpected keyword 'axis'

// 2) 重复绑定
// schema: topk(Tensor x, int k=1, int axis=-1) -> (Tensor, Tensor)
// 调用: topk(x, 5, k=3)
// 错误: argument 'k' specified twice

// 3) 缺少必需参数
// schema: matmul(Tensor a, Tensor b) -> Tensor
// 调用: matmul(a)
// 错误: missing required argument 'b'

// 4) kw-only 以位置传参
// schema: dropout(Tensor x, float p=0.5, *, bool training=True) -> Tensor
// 调用: dropout(x, 0.2, False)
// 错误: keyword-only argument 'training' passed as positional
```

### 2.3 多 backend 注册与派发
**核心能力**：支持不同设备后端的算子实现注册，并按 `DispatchKey` 选择对应实现。

**实现要点**：
- 引入 `BackendKey`（示例）：CPU、CUDA 等；
- `TORCH_LIBRARY_IMPL(ns, Backend, m)` 绑定 `backend_impls[Backend] = fn`；
- 运行时通过注册表中的 `implementations[DispatchKey]` 命中具体实现；当前实现未新增基于 `Place` 的自动推导/fallback 派发器。

### 2.4 与现有兼容层的关系与衔接
- **优先复用**（来自 `paddle/phi/api/include/compat/torch` 体系）：`IValue`、`FunctionArgs/FunctionResult`、`Library/OperatorRegistry/CppFunction`、`DispatchKey`、`ClassRegistry` 等基础设施与调用/注册路径。
- **在 compat/torch 内部实现或轻量移植**：`FunctionSchema`/`Argument`/`TypePtr` 等结构，以及 schema 解析器、参数绑定器、schema registry 与 backend 派发策略。
- **不直接引入 PyTorch 解析器/dispatcher**：避免引入大体量依赖、ABI/编译链复杂度与运行时耦合；保持 Paddle 兼容层轻量可控，便于与现有构建/发布体系对齐。
- **工程集成**：
  - 新增 `ATen/core/function_schema.*`、`ATen/core/alias_info.h`、`ATen/core/jit_type*.h`、`ATen/core/type_ptr.h`；
  - 新增 `torch/csrc/jit/schema_parser_defs.h`、`schema_type_parser.*`、`function_schema_parser.*`；
  - `paddle/phi/api/include/compat/CMakeLists.txt` 增加对应源码编译项。

### 2.5 Schema 覆盖范围与边界
**明确支持**（保证解析 + 绑定 + 调用通过）：  
- 基础标量与字符串：`int/float/bool/str`  
- `Tensor` 及 alias/mutation 标注（如 `Tensor(a!)`）  
- `Optional[T]`  
- `Tuple(T1, T2, ...)` 及可选形式 `Tuple(...)?`  
- argument 级固定长度列表 `T[N]`（含 `T[N]?`）  
- keyword / default / kw-only 绑定语义
- vararg / varret 语义

**后续支持**：  
- 更完整容器类型（如 `List[T]` 无固定长度、`Dict(K, V)` 及复杂嵌套）  
- 更完整设备/布局相关类型与默认值解析分支  
- `allow_typevars`、overload 名合法性校验等上游 parser 兼容细节  

## 3、主要影响的模块接口变化
### 核心接口变化
- `TORCH_LIBRARY`：新增/补充 schema 解析与注册逻辑；
- `TORCH_LIBRARY_IMPL`：支持 backend key 维度注册；
- compat 调用路径：支持 keyword 参数与默认参数绑定。

### 对框架各环节的影响
- 底层数据结构：新增 schema/dispatch registry 数据结构；

# 六、测试和验收的考量
**验收标准**：
- 完成 `TORCH_LIBRARY` 注册机制的 schema 支持，能够正确处理算子参数的类型，以及默认参数、keyword 参数传递功能，单测添加到 `test/cpp/compat/torch_library_test.cc`（合入 Paddle repo）；
- 完成多 backend 支持的注册兼容层实现，并新增相关单测（合入 Paddle repo）；
- 基于上述功能，后续继续验证 [paddlecodec](https://github.com/PFCCLab/paddlecodec) 适配改动可否进一步收敛。

# 七、影响面
## 对用户的影响
- 用户可以保持原生 PyTorch 注册方式，降低迁移与维护成本。

## 对二次开发用户的影响
- 新增能力主要体现在兼容层；对飞桨原生自定义算子 API 影响较小。

## 对框架架构的影响
- 兼容层新增 schema registry 与 dispatch 逻辑；核心框架结构基本不变。


## 对比业内深度学习框架的差距与优势的影响
- 在 PyTorch 生态的注册与调用语义方面显著对齐，降低生态差距；

## 其他风险
- Schema 覆盖范围不足导致个别算子仍需手工适配；

# 八、排期规划
- **第 1 周**：设计评审与方案确认；
- **第 2-3 周**：实现 schema 解析 + 参数绑定 + 单测；
- **第 4 周**：实现多 backend 注册与派发 + 单测；
- **第 5 周**：生态库验证（paddlecodec）与问题修复。

# 名词解释
- **TORCH_LIBRARY**：PyTorch C++ 自定义算子注册宏。
- **Schema**：算子函数签名描述，包括类型、默认值、keyword-only 等信息。
- **Backend/DispatchKey**：用于区分设备/后端的派发维度（如 CPU、CUDA）。

# 附件及参考资料
- 跨生态自定义算子接入 - 原理和迁移方式：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/custom_op/cross_ecosystem_custom_op/design_and_migration_cn.html
- 兼容性生态库 paddlecodec：https://github.com/PFCCLab/paddlecodec
- PyTorch C++ 扩展与自定义算子（TORCH_LIBRARY 相关）：https://pytorch.org/docs/stable/cpp_extension.html
- Paddle 兼容层 TORCH_LIBRARY/OperatorRegistry 等实现：https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/include/compat/torch/library.h
- Paddle 兼容层单测（torch_library_test）：https://github.com/PaddlePaddle/Paddle/blob/develop/test/cpp/compat/torch_library_test.cc
- PyTorch FunctionSchema 定义： https://github.com/pytorch/pytorch/blob/main/c10/core/FunctionSchema.h
- PyTorch schema 类型解析/映射（SchemaTypeParser）：
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/schema_type_parser.h
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/schema_type_parser.cpp
- PyTorch schema 字符串解析（parseSchema/parseSchemaOrName）：
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/function_schema_parser.h
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/function_schema_parser.cpp
- PyTorch schema 类型覆盖测试（op_registration_test.cpp）：https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/op_registration/op_registration_test.cpp
