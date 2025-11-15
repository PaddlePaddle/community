# FastDeploy 新增支持 DeepSeek 模型的 Reasoning Parser & Tool Parser

| 任务名称 | 为 FastDeploy 新增支持 DeepSeek 模型的 Reasoning Parser & Tool Parser |
|------|------|
| 提交作者 | fgeygfe |
| 提交时间 | 2025-11-15 |
| 版本号 | V1.0 |
| 文件名 | 20251115_add_deepseek_reasoning_tool_parser_support.md |

# 一、概述

## 1、相关背景

FastDeploy 作为高性能推理引擎，目前已经支持多种大语言模型的推理和服务。随着 DeepSeek 系列模型（包括 DeepSeek-V3、DeepSeek-V3.1、DeepSeek-R1 等）的推出，这些模型在推理能力和工具调用方面表现出色。为了更好地支持这些模型的特性，需要为 FastDeploy 新增专门针对 DeepSeek 模型的 Reasoning Parser 和 Tool Parser。

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

## 2、Tool Calling 输出格式

DeepSeek 模型的工具调用格式遵循类似 OpenAI 的 JSON 协议：

```xml
<think>
需要查询天气信息
</think>

<tool_call>
{"name": "get_weather", "arguments": {"location": "北京", "unit": "c"}}
</tool_call>
```

**支持并行工具调用**：

```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "北京"}}
</tool_call>

<tool_call>
{"name": "get_weather", "arguments": {"location": "上海"}}
</tool_call>
```

# 三、设计思路与实现方案

## 1、Reasoning Parser 设计

### 类设计
```python
@ReasoningParserManager.register_module(["deepseek", "deepseek-r1", "deepseek-v3"])
class DeepSeekReasoningParser(ReasoningParser):
    """DeepSeek 系列模型的推理解析器"""
```

### 核心方法
- `__init__(tokenizer)`: 初始化，获取特殊标记 `<think>`、`</think>` 的 token ID
- `is_reasoning_end(input_ids)`: 检查推理内容是否结束（检查 `</think>` 标记）
- `extract_content_ids(input_ids)`: 提取 `</think>` 之后的内容 token IDs
- `extract_reasoning_content(model_output, request)`: 非流式场景提取推理内容和回复内容
- `extract_reasoning_content_streaming(...)`: 流式场景逐步解析推理内容

### 边界情况处理
| 情况 | 处理策略 |
|-----|---------|
| 完全没有推理标记 | 将整个输出作为 content |
| 只有 `</think>` 没有 `<think>` | 将结束标记前的内容作为 reasoning |
| `</think>` 后立即是 `<tool_call>` | content 为空，由 Tool Parser 处理 |
| 多个 `</think>` 标记 | 只识别第一个 |

## 2、Tool Parser 设计

### 类设计
```python
@ToolParserManager.register_module(["deepseek", "deepseek-r1", "deepseek-v3"])
class DeepSeekToolParser(ToolParser):
    """DeepSeek 系列模型的工具调用解析器"""
```

### 核心方法
- `__init__(tokenizer)`: 初始化，获取 `<tool_call>`、`</tool_call>` 的 token ID
- `extract_tool_calls(model_output, request)`: 非流式场景提取所有工具调用
- `extract_tool_calls_streaming(...)`: 流式场景逐步解析工具调用
- `_parse_tool_json(tool_json)`: 解析工具调用的 JSON（支持 partial JSON）

### JSON 解析策略
1. 首先尝试标准 `json.loads`
2. 如果失败，使用 `partial_json_parser` 处理不完整 JSON
3. 如果仍然失败，使用正则提取 name 和 arguments 字段

### 并行工具调用支持
- 正确识别多个 `<tool_call>` 块
- 为每个工具调用分配唯一的 ID
- 在流式场景下使用 `current_tool_id` 索引维护多个工具的状态

## 3、文件结构

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
- 测试 DeepSeek-R1, V3, V3.1 等不同模型
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

启动服务：
```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model unsloth/DeepSeek-R1-BF16 \
    --port 8192 \
    --reasoning-parser deepseek \
    --tool-call-parser deepseek
```

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
