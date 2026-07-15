# 修复 Windows 下 Paddle Inference 的 UTF-8 路径识别问题

## 详细描述

Paddle Inference 的配置接口会接收模型文件、参数文件和配置文件路径。这些路径以字符串形式传入，并可能包含 UTF-8 编码的非 ASCII 字符，例如中文目录名。

在 Windows 上，存在这样一种问题：文件或目录实际存在，但当路径中包含非 ASCII 字符时，路径校验逻辑可能错误地认为该文件或目录不存在。结果是用户无法在包含中文等非 ASCII 字符的目录下正常加载 Inference 模型或配置。

## 复现方式

在 Windows 环境中创建一个包含非 ASCII 字符的目录，并在其中创建一个普通文件。将该目录路径和文件路径以 UTF-8 字符串形式传给 Paddle Inference 的路径校验流程。

修复前，ASCII 路径应继续保持原有行为，但 UTF-8 非 ASCII 路径可能无法被正确识别。

## 验收说明

- 已存在的 ASCII 文件和目录仍应被正确识别。
- 缺失的 ASCII 文件仍应被正确识别为不存在。
- Windows 上包含 UTF-8 非 ASCII 字符的已存在文件和目录应被正确识别。
- 非 Windows 平台的既有行为不应被改变。
- 修复应限定在路径处理语义内，不能通过删除校验、弱化断言、硬编码测试路径或把所有路径都视为存在来通过测试。

## 技术要求

- 熟悉 C++。
- 了解 Windows 文件系统路径编码。
- 了解 Paddle Inference 基础配置和路径校验流程。

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Missing paths should still be rejected.
- The solution should not bypass validation broadly or weaken the target tests.
