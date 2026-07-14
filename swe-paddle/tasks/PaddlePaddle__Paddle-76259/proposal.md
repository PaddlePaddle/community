# Task Proposal: PaddlePaddle__Paddle-76259

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-76259`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/76259
- PR 标题：`[Bug fixes] Fix Windows UTF-8 path support for Paddle Inference model/json files`
- `base_commit`：`fcf3b100085b10efed4c1fb8880b1df1fd5241d6`
- gold commit：`59670139385c14790ecd3c764d819eff63821978`
- merged 时间：`2025-11-06`
- 你的身份：原 PR 作者 / 熟悉该模块的 contributor
- 后续联系人：Github @Echo-Nie

## 2. 问题一句话

修复 Windows 下 Paddle Inference 对 UTF-8 非 ASCII 模型路径、参数路径和配置路径的文件/目录识别问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Inference bug-fix PR，不是合成任务。
- **代表性**：它覆盖 Paddle C++ Inference API 与 Windows 文件系统编码边界。
- **边界清楚**：目标行为集中在路径存在性和目录判断，不涉及模型计算语义。
- **回归护栏明确**：ASCII 路径必须继续工作，缺失路径必须继续被判定为不存在。
- **区分度信号**：正确修复需要理解 UTF-8 `std::string` 与 Windows wide-character filesystem API 的边界；简单放宽校验或跳过检查会被 P2P 护栏拦住。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行平台：`windows`
- 验证资源：`windows_cpu`
- 设备范围：`cpu_runner_sufficient`
- 模块标签：`[inference, windows, utf8_path, filesystem, cxx_api, path_encoding]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/cpp/inference/api/utf8_path_test.cc`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，ASCII 路径 P2P 用例通过，UTF-8 非 ASCII 路径 F2P 用例失败。
- 修复后预期：继续应用 `solution/code.patch` 后，P2P 和 F2P 用例均通过。
- P2P 护栏：ASCII 文件、ASCII 目录、ASCII 缺失文件的既有行为不能被破坏。

## 6. 环境与资源

- 资源需求：Windows CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 编译器：MSVC x64
- 推荐生成器：Ninja
- 是否需要 GPU：否
- 是否依赖 Python wheel：否
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述目标行为和验收标准，不直接指出具体文件或实现方式。
- 平台风险：该样本必须在 Windows 上验证；Linux 不能暴露 Windows 文件系统编码问题。
- 环境风险：该样本涉及 C++ Inference API，需要 source build 或等价已编译环境。
- flaky 风险：测试只创建临时文件和目录，不依赖网络、模型下载或 GPU。
- 拆分风险：该 PR 的目标集中在 Windows UTF-8 路径处理，适合作为一个独立样本。
