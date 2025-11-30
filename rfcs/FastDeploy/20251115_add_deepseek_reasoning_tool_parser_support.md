# FastDeploy 新增支持 DeepSeek 模型的 Reasoning Parser & Tool Parser

| 任务名称 | 为 FastDeploy 新增支持 DeepSeek 模型的 Reasoning Parser & Tool Parser |
|------|------|
| 提交作者 | fgeygfe |
| 提交时间 | 2025-11-15 |
| 版本号 | V1.0 |
| 文件名 | 20251115_add_deepseek_reasoning_tool_parser_support.md |

# 一、概述

## 1、相关背景

FastDeploy 作为高性能推理引擎，目前已经支持多种大语言模型的推理和服务。随着 DeepSeek 系列模型的推出，这些模型在推理能力和工具调用方面表现出色。本 RFC 专注于支持以下三种模型：
- `unsloth/DeepSeek-V3.1-BF16`
- `unsloth/DeepSeek-V3-0324-BF16`
- `unsloth/DeepSeek-R1-BF16`

为了更好地支持这些模型的特性，需要为 FastDeploy 新增专门针对 DeepSeek 模型的 Reasoning Parser 和 Tool Parser。

## 2、功能目标

* 实现 DeepSeek Reasoning Parser：解析模型输出中的思考内容（reasoning_content）和回复内容（content）
* 实现 DeepSeek Tool Parser：解析模型输出中的工具调用（tool_calls）信息
* 支持流式和非流式两种场景
* 支持并行工具调用
* 提供完整的单元测试和文档

## 3、意义

完善 FastDeploy 对 DeepSeek 系列模型的支持，为用户提供更强大的推理和工具调用能力，提升模型部署的便利性和性能。

# 二、DeepSeek 模型输出协议分析

## 1、Reasoning 输出格式

DeepSeek 模型使用 `<think>...</think>` 标记包裹思考内容：

```xml
<think>
这里是模型的思考过程...
用户询问了关于天气的问题，我需要调用天气工具
</think>

这是最终的回复内容
```

**特点**：
- 使用 `<think>...</think>` 标记包裹思考内容
- 思考内容在前，回复内容在后
- 需要容错处理不完整的标记

### 1.1 不同模型版本的思考输出方式差异

通过调研不同 DeepSeek 模型版本的实现和文档，发现**不同模型版本在思考输出的行为上存在显著差异**，主要体现在是否支持思考开关（reasoning toggle）以及默认行为上：

**支持思考开关的模型版本**（DeepSeek-V3.1）：
- 支持通过参数控制是否开启思考模式
- **默认情况下思考模式关闭**：模型直接生成答案，**不会输出 `<think>` 标签**
- **开启思考模式后**：模型会在输出中包含 `<think>...</think>` 标签，展示思考过程
- 解析器需要能够处理**两种输出模式**：有思考标签和无思考标签的情况

**不支持思考开关的模型版本**（如 DeepSeek-V3-0324、DeepSeek-R1）：
- **默认情况下总是输出思考内容**：模型会在输出中包含 `<think>...</think>` 标签
- DeepSeek-R1 作为推理优化模型，默认开启思考模式，总是展示思考过程
- 解析器主要需要处理包含思考标签的输出格式

**设计考虑**：
1. **版本检测**：需要根据模型版本或请求参数判断是否可能包含思考标签
2. **兼容性处理**：解析器应该能够同时处理有思考标签和无思考标签两种情况
3. **参数感知**：对于支持思考开关的模型，需要检查请求中是否开启了思考模式（如 `reasoning` 参数）
4. **容错机制**：即使模型支持思考开关但未开启，解析器也应该能够正确处理纯文本输出，将整个输出作为 `content` 返回

## 2、Tool Calling 输出格式

### 2.1 协议差异调研

通过调研 vLLM 项目中的实现（参考：`vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`）和 Hugging Face 的 tokenizer 配置（参考：`huggingface.co/unsloth/DeepSeek-V3.1-BF16`），确认了 **不同 DeepSeek 模型版本使用了不同的工具调用协议格式**：

#### DeepSeek-V3.1 协议格式（已确认）

DeepSeek-V3.1 使用特殊的标记序列来包裹工具调用，格式已通过 tokenizer 配置确认：

```xml
<think>
需要查询天气信息
</think>

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
get_weather<｜tool▁sep｜>{"location": "北京", "unit": "c"}
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

**关键标记及其 Token ID**（来自 tokenizer_config.json，已确认）：
- `<｜tool▁calls▁begin｜>`：工具调用块开始标记（Token ID: 128806）
- `<｜tool▁calls▁end｜>`：工具调用块结束标记（Token ID: 128807）
- `<｜tool▁call▁begin｜>`：单个工具调用开始标记（Token ID: 128808）
- `<｜tool▁call▁end｜>`：单个工具调用结束标记（Token ID: 128809）
- `<｜tool▁sep｜>`：分隔函数名和参数的标记（Token ID: 128814）

**注意**：这些标记在 tokenizer 的 `added_tokens_decoder` 中定义，实际使用时通过 Token ID 进行识别和解析。

**格式特点**：
- 函数名和参数通过 `<｜tool▁sep｜>` 分隔，而非 JSON 格式
- 参数部分直接是 JSON 字符串，不需要 `{"name": ..., "arguments": ...}` 的包装
- 支持并行工具调用，多个 `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` 块可以连续出现
- 根据 chat_template，工具调用格式为：`<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>tool_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>`

#### DeepSeek-V3-0324 协议格式（已确认）

根据 Hugging Face tokenizer 配置（参考：`huggingface.co/unsloth/DeepSeek-V3-0324-BF16`），DeepSeek-V3-0324 使用与 DeepSeek-V3.1 相同的工具调用协议格式，但实际输出格式略有差异：

```xml
<think>
需要查询天气信息
</think>

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
function<｜tool▁sep｜>get_weather
```json
{"location": "北京", "unit": "c"}
```
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

**关键标记及其 Token ID**（来自 tokenizer_config.json，已确认）：
- `<｜tool▁calls▁begin｜>`：工具调用块开始标记（Token ID: 128806）
- `<｜tool▁calls▁end｜>`：工具调用块结束标记（Token ID: 128807）
- `<｜tool▁call▁begin｜>`：单个工具调用开始标记（Token ID: 128808）
- `<｜tool▁call▁end｜>`：单个工具调用结束标记（Token ID: 128809）
- `<｜tool▁sep｜>`：分隔工具类型和函数名的标记（Token ID: 128814）

**格式特点**（基于 chat_template 分析）：
- 工具类型（通常是 "function"）和函数名通过 `<｜tool▁sep｜>` 分隔
- 函数名和参数之间有换行符
- 参数部分被包裹在 ```json ... ``` 代码块中
- 参数为 JSON 字符串格式，直接是函数参数，不需要 `{"name": ..., "arguments": ...}` 的包装
- 支持并行工具调用，多个 `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` 块可以连续出现
- 根据 chat_template，完整格式为：`<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{type}<｜tool▁sep｜>{name}\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`

**注意**：DeepSeek-V3-0324 与 DeepSeek-V3.1 使用相同的特殊标记 Token ID，但实际输出格式中参数部分被包裹在代码块中，解析时需要处理这个差异。

#### DeepSeek-R1 协议格式（已确认）

根据 Hugging Face tokenizer 配置（参考：`huggingface.co/unsloth/DeepSeek-R1-BF16`），DeepSeek-R1 使用与 DeepSeek-V3-0324 完全相同的工具调用协议格式：

```xml
<think>
需要查询天气信息
</think>

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
function<｜tool▁sep｜>get_weather
```json
{"location": "北京", "unit": "c"}
```
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

**关键标记及其 Token ID**（来自 tokenizer_config.json，已确认）：
- `<｜tool▁calls▁begin｜>`：工具调用块开始标记（Token ID: 128806）
- `<｜tool▁calls▁end｜>`：工具调用块结束标记（Token ID: 128807）
- `<｜tool▁call▁begin｜>`：单个工具调用开始标记（Token ID: 128808）
- `<｜tool▁call▁end｜>`：单个工具调用结束标记（Token ID: 128809）
- `<｜tool▁sep｜>`：分隔工具类型和函数名的标记（Token ID: 128814）

**格式特点**（基于 chat_template 分析）：
- 工具类型（通常是 "function"）和函数名通过 `<｜tool▁sep｜>` 分隔
- 函数名和参数之间有换行符
- 参数部分被包裹在 ```json ... ``` 代码块中
- 参数为 JSON 字符串格式，直接是函数参数，不需要 `{"name": ..., "arguments": ...}` 的包装
- 支持并行工具调用，多个 `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` 块可以连续出现
- 根据 chat_template，完整格式为：`<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{type}<｜tool▁sep｜>{name}\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`

**注意**：DeepSeek-R1 与 DeepSeek-V3-0324 使用完全相同的工具调用协议格式和 Token ID，解析逻辑可以复用。

### 2.2 协议差异对比

| 特性 | DeepSeek-V3.1 | DeepSeek-V3-0324 | DeepSeek-R1 |
|------|---------------|-----------------|------------|
| 外层包裹标记 | `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>` | `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>` | `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>` |
| 单个工具调用标记 | `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` | `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` | `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` |
| 工具类型与函数名分隔 | `<｜tool▁sep｜>` | `<｜tool▁sep｜>` | `<｜tool▁sep｜>` |
| 参数格式 | 直接 JSON 字符串 | ```json ... ``` 代码块包裹的 JSON | ```json ... ``` 代码块包裹的 JSON |
| 参数内容 | 函数参数 JSON | 函数参数 JSON | 函数参数 JSON |
| 解析复杂度 | 较高（需要处理多个特殊标记） | 较高（需要处理代码块和特殊标记） | 较高（需要处理代码块和特殊标记） |

### 2.3 解析方案设计考虑

基于上述协议差异，需要设计**版本自适应的解析方案**：

1. **版本检测**：根据模型名称或 tokenizer 中的特殊标记存在性，自动识别模型版本
2. **多协议支持**：
   - **DeepSeek-V3.1**：使用正则表达式解析 `<｜tool▁call▁begin｜>...<｜tool▁sep｜>...<｜tool▁call▁end｜>` 格式，参数为直接 JSON 字符串
   - **DeepSeek-V3-0324 / DeepSeek-R1**：使用正则表达式解析相同格式，但需要额外处理参数部分的 ```json ... ``` 代码块包裹（提取代码块内的 JSON）
3. **统一接口**：不同协议解析后，统一转换为 OpenAI 兼容的 `ToolCall` 格式
4. **流式解析适配**：针对不同协议的标记特点，实现相应的流式解析逻辑

**参考实现**：
- vLLM DeepSeek-V3.1 Parser：使用正则表达式 `r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>"` 进行解析
- Hugging Face tokenizer 配置已确认：
  - DeepSeek-V3.1 的协议格式和 Token ID（参考：`huggingface.co/unsloth/DeepSeek-V3.1-BF16`）
  - DeepSeek-V3-0324 的协议格式和 Token ID（参考：`huggingface.co/unsloth/DeepSeek-V3-0324-BF16`）
  - DeepSeek-R1 的协议格式和 Token ID（参考：`huggingface.co/unsloth/DeepSeek-R1-BF16`），与 V3-0324 完全相同

# 三、设计思路与实现方案

## 0、思考机制架构调整说明

### 架构调整背景

FastDeploy 正在对思考机制框架进行优化（参考 PR: [Feature]Optimization of Thinking Pattern Framework FastDeploy#4302），主要解决以下问题：

1. **参数定义不统一**：不同模型对思考开关的参数定义不统一（如 `reasoning`、`enable_reasoning` 等）
2. **默认参数不统一**：不同模型的思考开关默认状态不一致
3. **Prompt 灵活性**：用户输入的或 chattemplate 拼接后的 prompt 灵活多样
4. **阶段感知需求**：架构上需要感知模型的输出是处于思考阶段还是回复阶段
5. **解析逻辑统一**：reasoning_parser 和 tool_call_parser 从严格解析逻辑上来看，也需要感知模型输出所处的阶段

### 新架构设计要点

**1. 阶段检测方法**
- `reasoning_parser` 需要提供一个方法：根据进入模型的 `prompt_token_ids`，判断接下来模型输出在哪个阶段
- 不同模型的识别方式可能会不一样（如通过特殊 token、prompt 结构等）

**2. 阶段参数传递**
- 解析方法中会增加一个**模型输出所处阶段的参数**（不是实时的，可以辅助对首 token 进行判断）
- 该参数用于帮助解析器理解当前输出应该处于哪个阶段

**3. 统一调用机制**
- **无论思考开关是打开还是关闭**，输出后处理过程都会统一调用 parser
- 由 parser 根据实际情况（prompt 结构、阶段参数、实际输出内容等）进行判断要返回的内容
- 这样可以统一处理不同模型的差异，简化上层调用逻辑

### 对 DeepSeek Parser 实现的影响

在实现 DeepSeek Reasoning Parser 和 Tool Parser 时，需要：

1. **实现阶段检测方法**：根据 DeepSeek 模型的 prompt 结构，判断输出阶段
2. **支持阶段参数**：在解析方法中接收并使用阶段参数
3. **统一处理逻辑**：无论思考开关状态如何，都通过统一的解析逻辑处理
4. **阶段感知解析**：根据检测到的阶段，正确解析思考内容、回复内容或工具调用

## 1、Reasoning Parser 设计

### 类设计
```python
@ReasoningParserManager.register_module(["deepseek", "deepseek-r1", "deepseek-v3.1", "deepseek-v3-0324"])
class DeepSeekReasoningParser(ReasoningParser):
    """DeepSeek 系列模型的推理解析器（支持 V3.1、V3-0324、R1）"""
```

### 核心方法

#### 初始化方法
- `__init__(tokenizer, model_name=None)`: 
  - 初始化，获取特殊标记的 token ID：
    - `<think>` (Token ID: 128798)
    - `</think>` (Token ID: 128799)
  - 根据模型名称检测是否支持思考开关（V3.1）
  - 初始化阶段检测所需的特殊标记和规则

#### 阶段检测方法（架构调整新增）
- `detect_output_stage(prompt_token_ids)`: 
  - **根据进入模型的 `prompt_token_ids`，判断接下来模型输出在哪个阶段**
  - 返回阶段标识（如 `REASONING_STAGE`、`CONTENT_STAGE` 等）
  - DeepSeek 模型的检测方式：
    - 检查 prompt 中是否包含 `<think>` 开始标记
    - 检查 prompt 结构，判断是否处于思考阶段
    - 不同模型版本可能有不同的检测逻辑

#### 内容提取方法
- `is_reasoning_end(input_ids)`: 检查推理内容是否结束（检查 `</think>` 标记）
- `extract_content_ids(input_ids)`: 提取 `</think>` 之后的内容 token IDs
- `extract_reasoning_content(model_output, request, output_stage=None)`: 
  - 非流式场景提取推理内容和回复内容
  - **新增 `output_stage` 参数**：模型输出所处阶段（由阶段检测方法提供，辅助对首 token 进行判断）
  - **统一处理逻辑**：无论思考开关是打开还是关闭，都通过此方法处理
  - 根据 `output_stage` 和实际输出内容判断：
    - 如果检测到思考阶段且有思考标签：解析 `<think>...</think>` 内容
    - 如果检测到思考阶段但无思考标签（思考开关关闭）：将整个输出作为 content
    - 如果检测到回复阶段：直接作为 content 处理
- `extract_reasoning_content_streaming(..., output_stage=None)`: 
  - 流式场景逐步解析推理内容
  - **新增 `output_stage` 参数**：辅助判断流式输出的阶段
  - 需要处理思考模式关闭时的情况（无思考标签）

### 思考开关支持（架构调整后的统一处理）

**架构调整说明**：
- **统一调用机制**：无论思考开关是打开还是关闭，输出后处理过程都会统一调用 parser
- **由 parser 自主判断**：parser 根据实际情况（`output_stage` 参数、prompt 结构、实际输出内容等）进行判断要返回的内容
- **不再依赖请求参数**：不再需要检查请求参数中的 `reasoning` 等参数，由阶段检测和实际输出内容决定

**输出模式处理**：
- **有思考标签模式**：正常解析 `<think>...</think>` 内容
- **无思考标签模式**（思考开关关闭时）：
  - 将整个输出作为 `content`
  - `reasoning_content` 为 `None` 或空字符串
  - 确保与有思考标签模式的输出格式保持一致

**阶段感知处理**：
- 通过 `detect_output_stage()` 方法检测输出阶段
- 结合 `output_stage` 参数和实际输出内容，智能判断应该返回什么内容
- 支持不同模型版本的差异（V3.1 支持思考开关，V3-0324/R1 默认开启思考）

### 边界情况处理
| 情况 | 处理策略 |
|-----|---------|
| 完全没有推理标记 | 将整个输出作为 content（可能是思考开关关闭的情况） |
| 只有 `</think>` 没有 `<think>` | 将结束标记前的内容作为 reasoning |
| `</think>` 后立即是 `<tool_call>` | content 为空，由 Tool Parser 处理 |
| 多个 `</think>` 标记 | 只识别第一个 |
| 思考开关关闭（V3.1）且无思考标签 | 将整个输出作为 content，reasoning_content 为空 |

## 2、Tool Parser 设计

### 类设计
```python
@ToolParserManager.register_module(["deepseek", "deepseek-r1", "deepseek-v3.1", "deepseek-v3-0324"])
class DeepSeekToolParser(ToolParser):
    """DeepSeek 系列模型的工具调用解析器（支持 V3.1、V3-0324、R1 三种模型）"""
```

### 版本检测与协议适配

**初始化时的版本检测**：
- 检查 tokenizer 中是否存在以下特殊标记的 Token ID：
  - V3.1/V3-0324/R1 标记：128806 (`<｜tool▁calls▁begin｜>`)、128807 (`<｜tool▁calls▁end｜>`)、128808 (`<｜tool▁call▁begin｜>`)、128809 (`<｜tool▁call▁end｜>`)、128814 (`<｜tool▁sep｜>`)
- 根据标记存在性自动识别模型版本（V3.1 vs V3-0324/R1）
- 或根据模型名称（如包含 "v3.1"、"v3-0324"、"r1"）进行版本判断

**多协议支持**：
- **DeepSeek-V3.1 协议**：使用正则表达式解析 `<｜tool▁calls▁begin｜>...<｜tool▁sep｜>...<｜tool▁call▁end｜>` 格式，参数为直接 JSON 字符串
- **DeepSeek-V3-0324 / DeepSeek-R1 协议**：使用正则表达式解析相同格式，但需要额外处理参数部分的 ```json ... ``` 代码块包裹（提取代码块内的 JSON）

### 核心方法

#### 初始化方法
- `__init__(tokenizer)`: 
  - 检测模型版本
  - 根据版本获取对应的特殊标记 token ID：
    - V3.1/V3-0324/R1: 
      - `<｜tool▁calls▁begin｜>` (Token ID: 128806)
      - `<｜tool▁calls▁end｜>` (Token ID: 128807)
      - `<｜tool▁call▁begin｜>` (Token ID: 128808)
      - `<｜tool▁call▁end｜>` (Token ID: 128809)
      - `<｜tool▁sep｜>` (Token ID: 128814)

#### 阶段检测方法（架构调整新增）
- `detect_output_stage(prompt_token_ids)`: 
  - **根据进入模型的 `prompt_token_ids`，判断接下来模型输出是否处于工具调用阶段**
  - 返回阶段标识（如 `TOOL_CALL_STAGE`、`CONTENT_STAGE` 等）
  - DeepSeek 模型的检测方式：
    - 检查 prompt 中是否包含工具定义或工具调用相关的特殊标记
    - 检查 prompt 结构，判断是否处于工具调用阶段
    - 不同模型版本可能有不同的检测逻辑

#### 非流式解析
- `extract_tool_calls(model_output, request, output_stage=None)`: 
  - 根据检测到的版本选择对应的解析策略
  - **新增 `output_stage` 参数**：模型输出所处阶段（辅助判断是否应该解析工具调用）
  - **V3.1**: 使用正则 `r"<｜tool▁call▁begin｜>(?P<tool_type>.*?)<｜tool▁sep｜>(?P<function_name>.*?)(?P<function_arguments>.*?)<｜tool▁call▁end｜>"` 提取，参数为直接 JSON 字符串
  - **V3-0324 / R1**: 使用正则 `r"<｜tool▁call▁begin｜>(?P<tool_type>.*?)<｜tool▁sep｜>(?P<function_name>.*?)\n```json\n(?P<function_arguments>.*?)\n```<｜tool▁call▁end｜>"` 提取，需要从代码块中提取 JSON
  - **统一处理逻辑**：无论思考开关状态如何，都通过此方法处理

#### 流式解析
- `extract_tool_calls_streaming(..., output_stage=None)`: 
  - 维护版本特定的状态（如 V3.1 需要跟踪 `current_tool_id`、`streamed_args_for_tool` 等）
  - **新增 `output_stage` 参数**：辅助判断流式输出的阶段
  - 根据版本使用不同的流式解析逻辑
  - 参考 vLLM 实现中的状态管理机制

#### 辅助方法
- `_detect_model_version(tokenizer)`: 检测模型版本（V3.1 vs V3-0324/R1）
- `_parse_v31_tool_call(text)`: 解析 V3.1 格式的工具调用（参数为直接 JSON 字符串）
- `_parse_v30324_r1_tool_call(text)`: 解析 V3-0324/R1 格式的工具调用（参数被包裹在代码块中，两者格式完全相同）
- `_parse_tool_json(tool_json)`: 解析工具调用的 JSON（支持 partial JSON）
- `_extract_json_from_codeblock(text)`: 从 ```json ... ``` 代码块中提取 JSON 字符串（用于 V3-0324/R1）

### 解析策略

#### DeepSeek-V3.1 解析策略
1. 使用正则表达式匹配 `<｜tool▁calls▁begin｜>...<｜tool▁call▁begin｜>...<｜tool▁sep｜>...<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>` 模式
2. 提取工具类型、函数名和参数（参数为直接 JSON 字符串）
3. 直接解析参数 JSON，无需额外包装

#### DeepSeek-V3-0324 解析策略
1. 使用正则表达式匹配与 V3.1 相同的模式
2. 提取工具类型、函数名和参数（参数被包裹在 ```json ... ``` 代码块中）
3. 从代码块中提取 JSON 字符串（去除 ```json 和 ``` 标记）
4. 解析提取出的 JSON 字符串

#### DeepSeek-R1 解析策略
1. 使用正则表达式匹配与 V3-0324 相同的模式
2. 提取工具类型、函数名和参数（参数被包裹在 ```json ... ``` 代码块中）
3. 从代码块中提取 JSON 字符串（去除 ```json 和 ``` 标记）
4. 解析提取出的 JSON 字符串

**注意**：DeepSeek-R1 与 DeepSeek-V3-0324 使用完全相同的协议格式，解析逻辑可以完全复用。

#### 通用 JSON 解析策略（用于参数部分）
1. 首先尝试标准 `json.loads`
2. 如果失败，使用 `partial_json_parser` 处理不完整 JSON
3. 如果仍然失败，使用正则提取关键字段

### 并行工具调用支持

**DeepSeek-V3.1**：
- 识别多个 `<｜tool▁calls▁begin｜>...<｜tool▁call▁begin｜>...<｜tool▁sep｜>...<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>` 块
- 使用 `current_tool_id` 索引维护多个工具的状态
- 在流式场景下跟踪每个工具的 `streamed_args_for_tool`

**DeepSeek-V3-0324**：
- 识别多个与 V3.1 相同格式的工具调用块
- 使用 `current_tool_id` 索引维护多个工具的状态
- 在流式场景下跟踪每个工具的 `streamed_args_for_tool`，需要额外处理代码块的开始和结束标记

**DeepSeek-R1**：
- 识别多个与 V3-0324 相同格式的工具调用块
- 使用 `current_tool_id` 索引维护多个工具的状态
- 在流式场景下跟踪每个工具的 `streamed_args_for_tool`，需要额外处理代码块的开始和结束标记

### 流式解析状态管理（参考 vLLM 实现）

**V3.1 需要的状态变量**：
- `current_tool_name_sent`: 是否已发送当前工具名称
- `prev_tool_call_arr`: 之前解析的工具调用数组（用于 diff）
- `current_tool_id`: 当前工具调用的索引
- `streamed_args_for_tool`: 每个工具已流式发送的参数部分

**流式解析关键逻辑**：
- 通过统计 `tool_call_start_token_id` 和 `tool_call_end_token_id` 的数量判断解析状态
- 区分"生成文本"、"开始新工具"、"更新工具"、"关闭工具"等不同阶段
- 使用 diff 机制只发送新增的参数部分，避免重复发送

## 3、执行方案设计（按协议与输出场景拆分）

### 3.1 按模型输出协议梳理“正常返回”形态

以支持思考 + 工具调用的模型为例（示意协议，非 DeepSeek 专用）：

- **有工具调用意图时（存在 tool_calls）**

  ```text
  reasoning_content</think>\n\n<tool_call>XXX</tool_call><tool_call>YYY</tool_call></s>
  ```

  - `reasoning_content`：思考内容  
  - `</think>` 之后不再输出自然语言回复，直接进入工具调用阶段  
  - 一个响应中可以包含多个 `<tool_call>...</tool_call>`，表示并行工具调用

- **无工具调用意图时（不存在 tool_calls）**

  ```text
  reasoning_content</think>content</s>
  ```

  - `reasoning_content`：思考内容  
  - `content`：最终给用户的自然语言回复  
  - 整个输出中不包含 `<tool_call>` 块

实现时，建议为各模型族（DeepSeek / Qwen / ERNIE 等）各自整理一份“正常协议输出示例表”，用于解析逻辑与单测对齐。

### 3.2 按不同输出场景制定解析策略

以下规则适用于非流式和流式两种场景。流式场景在 token/delta 级别做增量决策，但语义保持一致。阶段信息由上层通过 `output_stage` 传给 parser（例如：`REASONING_STAGE` / `CONTENT_STAGE` / `TOOL_CALL_STAGE`）。

#### 3.2.1 未出现 `</think>` 的场景

- 示例：`ABCD`

- 若 `output_stage = REASONING_STAGE`（模型应处于思考阶段）：
  - 视为模型未正确输出关闭标签
  - 将 `ABCD` 全部归入 `reasoning_content`
  - 作为协议异常/降级情况记录日志

- 若 `output_stage = CONTENT_STAGE`（模型应处于回复阶段）：
  - 将 `ABCD` 全部归入 `content`
  - 不再尝试解析 reasoning

流式场景下，首批 delta 未看到 `</think>` 时，依赖 `output_stage` 决定写入 `reasoning_content` 还是 `content`。

#### 3.2.2 已出现 `</think>`，但未出现 `<tool_call>`

- 示例：`ABCD </think> EFG`

- 解析结果：
  - `</think>` 之前：全部归入 `reasoning_content`
  - `</think>` 之后至结尾：全部归入 `content`
  - 不触发 Tool Parser，本轮认为无工具调用意图

#### 3.2.3 出现 `<tool_call>`，但 `</think>` 与 `<tool_call>` 之间存在非 `\n` 字符

- 示例：`ABCD </think>\nEFG\n<tool_call>XXX</tool_call>`

说明：思考结束后，先输出了一段自然语言 `EFG`，然后才输出 `<tool_call>`，协议不规范。

- 非流式策略：
  - 检测到 `</think>` 与第一段 `<tool_call>` 之间存在非 `\n` / 非空白字符（如 `EFG`）
  - 整个响应按“已进入自然语言回复阶段”处理：
    - `reasoning_content`：`</think>` 之前内容
    - `content`：`</think>` 之后全部（包括 `EFG` 和后续 `<tool_call>` 文本）
    - `tool_calls`：不再解析，返回空列表
  - Tool Parser 内部需显式做该规则判断并跳过解析，同时记录协议异常日志

- 流式策略：
  - 当流式遇到 `EFG` 时，这些 delta 已经被当作 `content` 返回给用户
  - 后续再遇到 `<tool_call>`，为避免“先回答再工具”的体验混乱，沿用非流式策略：
    - 之后所有内容均当作 `content` 追加
    - Tool Parser 对本次响应不再输出任何 `tool_calls`

#### 3.2.4 出现 `<tool_call>`，且 `</think>` 与 `<tool_call>` 之间仅有 `\n`

- 示例：`ABCD </think>\n\n<tool_call>XXX</tool_call>`

- 解析策略：
  - `</think>` 之前：`reasoning_content`
  - `</think>` 与 `<tool_call>` 之间：只包含 `\n`/空白，解析时直接忽略
  - `<tool_call>...</tool_call>`：按协议正常解析为 `tool_calls`
  - 本轮响应中：`tool_calls` 非空，`content` 为空（与工具调用内容互斥）

流式下，`</think>` 后的若干 `\n` 可以在 Reasoning Parser / Tool Parser 中统一过滤，不向用户输出。

#### 3.2.5 多工具调用场景与跨工具间内容

- 示例：  
  `ABCD </think>\n\n<tool_call>XXX</tool_call>\nXYZ\n<tool_call>YYY</tool_call>`

- 预期理想协议：多个 `<tool_call>` 连续出现，中间仅允许 `\n`/空白。

- 解析建议：
  - 第一段 `<tool_call>XXX</tool_call>`：正常解析为一个工具调用
  - `</tool_call>` 与下一次 `<tool_call>` 之间：
    - 仅有 `\n`/空白：忽略这些空白，继续解析下一个 `<tool_call>`
    - 若出现非空白字符（如 `XYZ`）：
      - 已完成解析的工具调用保持不变
      - 后续内容（包括 `XYZ` 和其后的 `<tool_call>YYY</tool_call>`）全部并入 `content`
      - Tool Parser 不再解析新的工具调用块

当前版本采取“遇到跨工具非空白字符则停止后续工具解析”的保守策略，后续可视实践反馈再做迭代。

#### 3.2.6 `<tool_call>` 内容不合法时的处理

- 情况：`<tool_call>...</tool_call>` 内部 JSON/参数不完整或不合法

处理策略：

1. 优先尝试标准 `json.loads`
2. 失败时使用 `partial_json_parser` 做容错解析
3. 若仍失败：
   - 将该 `<tool_call>` 标记为失败调用，可以：
     - 不返回该条工具调用，仅打日志；或
     - 以原始字符串作为 `arguments` 透传给上层，由上层决定重试/降级策略
   - 不影响同一响应中其他合法工具调用的解析

### 3.3 非思考模式下的解析方案

非思考模式下，常见两类输出：

- 仅自然语言：`content</s>`
- 直接输出工具调用：`<tool_call>XXX</tool_call></s>`

**Reasoning Parser**：

- 无 `<think>` 时：
  - 将全量输出视为 `content`
  - `reasoning_content` 为空
- 与“思考开关关闭”的行为对齐

**Tool Parser**：

- 若输出中检测到 `<tool_call>`：
  - 仍按上述工具协议规则解析
  - 与 `content` 互斥约束保持一致：
    - 若 `<tool_call>` 前已输出自然语言（并被视作回复），则参照 3.2.3 的异常策略：优先 `content`，跳过工具解析
    - 若响应从头即 `<tool_call>` 开始，则视为“纯工具调用”场景，正常解析

## 4、文件结构

| 文件路径 | 说明 |
|---------|------|
| `fastdeploy/reasoning/deepseek_reasoning_parser.py` | Reasoning Parser 实现 |
| `fastdeploy/entrypoints/openai/tool_parsers/deepseek_tool_parser.py` | Tool Parser 实现 |
| `tests/reasoning/test_deepseek_reasoning_parser.py` | Reasoning Parser 单元测试 |
| `tests/entrypoints/openai/tool_parsers/test_deepseek_tool_parser.py` | Tool Parser 单元测试 |

# 四、测试和验收的考量

## 1、Reasoning Parser 测试

- **非流式测试**：标准格式、缺少开始标记、无推理内容、仅推理内容、推理+工具调用等场景
- **流式测试**：逐字符推理、推理结束、跨 delta 结束等场景
- 测试覆盖率 > 90%

## 2、Tool Parser 测试

- **非流式测试**：标准单工具、并行工具、不完整 JSON、缺少字段、无工具调用等场景
- **流式测试**：逐步解析 name、参数流式输出、完整工具调用等场景
- 测试覆盖率 > 90%

## 3、集成测试

- 启动 FastDeploy 服务，配置 `--reasoning-parser deepseek` 和 `--tool-call-parser deepseek`
- 发送包含工具的请求，验证返回的 reasoning_content 和 tool_calls
- 测试以下三种模型：
  - `unsloth/DeepSeek-V3.1-BF16`
  - `unsloth/DeepSeek-V3-0324-BF16`
  - `unsloth/DeepSeek-R1-BF16`
- 性能测试：测试大量并发请求下的解析性能

# 五、可行性分析和排期规划

## 1、参考实现

可参考现有的 Qwen、ERNIE 系列的 Reasoning Parser 和 Tool Parser 实现：
- `fastdeploy/reasoning/qwen3_reasoning_parsers.py`
- `fastdeploy/reasoning/ernie_x1_reasoning_parsers.py`
- `fastdeploy/entrypoints/openai/tool_parsers/ernie_x1_tool_parser.py`

## 2、排期规划

| 阶段 | 任务 | 预计时间 |
|-----|------|---------|
| Week 1 | Reasoning Parser 实现 | 2天 |
| Week 1 | Tool Parser 实现 | 2天 |
| Week 1 | 单元测试开发 | 2天 |
| Week 2 | 集成测试 | 1天 |
| Week 2 | 文档更新 | 1天 |
| Week 2 | Code Review & 修复 | 2天 |

**总计**：约 10 个工作日

# 六、影响面

## 1、新增模块

在 `fastdeploy/reasoning` 和 `fastdeploy/entrypoints/openai/tool_parsers` 中新增类，不影响 FastDeploy 已有功能。

## 2、文档更新

需要更新以下文档：
- `docs/features/reasoning_output.md`：添加 DeepSeek 模型支持说明
- `docs/zh/features/reasoning_output.md`：中文版本
- `docs/features/tool_calling.md`：添加 DeepSeek 工具调用说明
- `docs/zh/features/tool_calling.md`：中文版本

## 3、使用示例

启动服务（以 DeepSeek-V3-0324 为例）：
```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model unsloth/DeepSeek-V3-0324-BF16 \
    --port 8192 \
    --reasoning-parser deepseek \
    --tool-call-parser deepseek
```

其他支持的模型示例：
- `unsloth/DeepSeek-R1-BF16`
- `unsloth/DeepSeek-V3.1-BF16`
- `unsloth/DeepSeek-V3-0324-BF16`

调用 API：
```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8192/v1")

response = client.chat.completions.create(
    model="deepseek",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                }
            }
        }
    }]
)

print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.tool_calls)
```

# 七、参考资料

1. FastDeploy Reasoning Output: `docs/features/reasoning_output.md`
2. FastDeploy Tool Calling: `docs/features/tool_calling.md`
3. OpenAI Tool Calling 规范
4. `partial_json_parser` 库文档
5. FastDeploy 思考机制架构调整 PR：
   - [Feature]Optimization of Thinking Pattern Framework FastDeploy#4302
   - 该 PR 引入了阶段检测方法和统一调用机制，影响本 RFC 的实现方案
6. vLLM DeepSeek-V3.1 Tool Parser 实现参考：
   - GitHub: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py
   - 该实现展示了 DeepSeek-V3.1 使用的特殊标记格式（`<｜tool▁calls▁begin｜>`、`<｜tool▁call▁end｜>`、`<｜tool▁sep｜>` 等）和流式解析的完整逻辑
7. Hugging Face DeepSeek-V3.1 Tokenizer 配置（官方确认）：
   - 模型页面：https://huggingface.co/unsloth/DeepSeek-V3.1-BF16
   - tokenizer_config.json 中确认了所有特殊标记的 Token ID：
     - 工具调用标记：128806-128809, 128814
     - 思考标记：128798-128799
   - chat_template 中展示了实际的工具调用格式和思考模式的使用方式
8. Hugging Face DeepSeek-V3-0324 Tokenizer 配置（官方确认）：
   - 模型页面：https://huggingface.co/unsloth/DeepSeek-V3-0324-BF16
   - tokenizer_config.json 中确认了所有特殊标记的 Token ID：
     - 工具调用标记：128806 (`<｜tool▁calls▁begin｜>`)、128807 (`<｜tool▁calls▁end｜>`)、128808 (`<｜tool▁call▁begin｜>`)、128809 (`<｜tool▁call▁end｜>`)、128814 (`<｜tool▁sep｜>`)
     - 思考标记：128798 (`<think>`)、128799 (`</think>`)
   - chat_template 中展示了实际的工具调用格式，参数部分被包裹在 ```json ... ``` 代码块中
9. Hugging Face DeepSeek-R1 Tokenizer 配置（官方确认）：
   - 模型页面：https://huggingface.co/unsloth/DeepSeek-R1-BF16
   - tokenizer_config.json 中确认了所有特殊标记的 Token ID：
     - 工具调用标记：128806 (`<｜tool▁calls▁begin｜>`)、128807 (`<｜tool▁calls▁end｜>`)、128808 (`<｜tool▁call▁begin｜>`)、128809 (`<｜tool▁call▁end｜>`)、128814 (`<｜tool▁sep｜>`)
     - 思考标记：128798 (`<think>`)、128799 (`</think>`)
   - chat_template 中展示了实际的工具调用格式，与 V3-0324 完全相同，参数部分被包裹在 ```json ... ``` 代码块中
