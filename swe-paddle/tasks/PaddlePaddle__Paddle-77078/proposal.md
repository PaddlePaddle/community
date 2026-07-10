# Task Proposal: PaddlePaddle__Paddle-77078

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-77078`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/77078
- PR 标题：`[API Compatibility] Improve Cpp sink mechanism and sink paddle.inverse to cpp -part`
- `base_commit`：`f2de7486a07cbdbb6586771b5943df4bccc6d35c`（squash 合入 commit `78499bd` 的父提交）
- merged 时间：`2026-01-30`（UTC `05:05:34`，merge commit `78499bd`）
- 你的身份：GitHub @Manfredss
- 关联 issue：#76301「PaddlePaddle API 兼容性增强」（启航计划），关联 PaConvert #796（`torch.inverse`）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

将 `paddle.inverse` 下沉（sink）到 C++，为其补齐参数别名 `input`（对齐 PyTorch）与新的 `out` 参数，并重构 Cpp sink 代码生成机制，使其支持任意模块路径（如 `paddle.linalg.inv`），从而修复下沉后 `paddle.linalg.inv` 未生效导致的 API 兼容性问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的「启航计划 / API 兼容性增强」框架开发 PR，且下沉后触发了真实的 CE-Framework 测试失败（`test_inv.py` 中 `paddle.linalg.inv` undefined），作者据此定位并修复，属于真实研发闭环，非合成任务。
- **代表性**：覆盖 Paddle 一类典型的框架级能力——API 下沉到 C++、eager 层 monkey-patch 代码生成器（`monkey_patch_gen.py`）、`python_api_info.yaml` 配置驱动、Python/C++ 双端 API 语义对齐（参数别名 + `out` 参数），是 benchmark 中稀缺的「代码生成 + API 兼容」组合。
- **边界清楚**：目标行为明确——下沉后 `paddle.inverse`/`paddle.Tensor.inverse`/`paddle.linalg.inv` 三条路径行为一致，支持 `x`/`input` 别名、支持 `out` 参数、动态图与静态图结果一致；非目标行为（inverse 数值算法本身、其他算子）不受影响。
- **非平凡性**：不是简单加一个 YAML 条目就能过。Reviewer（zhwesky2010）明确要求「整体合并成一套逻辑」，把原本只硬编码支持 `paddle.` / `paddle.Tensor.` / `paddle.nn.functional.` 三个前缀的 `ClassifyAPIByPrefix` 重构为支持任意模块路径的统一映射（unified map），否则遇到 `paddle.linalg.inv` 会直接抛异常。模型需要同时理解代码生成器机制、YAML 配置、以及 `paddle/__init__.py` 中 `monkey_patch_generated_methods_for_tensor` 的执行时机，才能正确修复。
- **区分度潜力**：涉及构建期代码生成 + 多路径 API 语义一致性，属于框架内部机制类任务，预期对模型的框架级理解有较强区分度。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, cpp_sink, code_generator, eager, yaml_config, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_inverse_op.py`
- 关键新增/修改用例：
  - 新增 `TestInverseAPICompatibility`：校验 `paddle.inverse(x)` / `paddle.inverse(x=x)` / `paddle.inverse(input=x)` / `paddle.inverse(x, out=out)` / `x.inverse()` 在动态图与静态图下结果一致。
  - 新增 `test_dygraph_with_name` / `test_static_with_name`：校验 `name` 参数。
  - 新增 `TestInverseOpBatchedComplex`（`[2,3,5,5]`，complex64）及 `TestInverseOpComplex128.test_grad`。
  - 存量用例统一将 `self.python_api` 从 `paddle.tensor.math.inverse` 切换到 `paddle.inverse`。
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，兼容性用例应 fail/error——`input=` 关键字与 `out=` 参数在下沉前不被支持（触发 `TypeError`/未预期关键字），`paddle.linalg.inv` 路径未对齐。
- 修复后预期：继续应用 `solution/code.patch` 并重新构建后，全部目标用例 pass。
- P2P 候选：`test/legacy_test/test_inverse_op.py` 中原有的 `TestInverseOp`、`TestInverseOpBatched`、`TestInverseOpFP32`、`TestInverseAPI` 等存量用例可作为回归护栏，由 verifier 自动抽取稳定 `nodeid`，避免 agent 只改目标测试却破坏原有 inverse 语义。

## 6. 环境与资源

- 资源需求：CPU（inverse 在 CPU place 即可运行）
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`（需 source build）
- 是否能提供 Docker：暂无，建议后续补 source-build Dockerfile
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试（Win11 Home，9800X3D + RTX 5070Ti，CUDA 12.9 + cuDNN 9.12.0，Python 3.12，CMake 3.18.6，VS 2022）
- patch 类型：**含构建期代码生成改动**。`paddle/fluid/eager/auto_code_generator/generator/monkey_patch_gen.py` 与 `paddle/phi/ops/yaml/python_api_info.yaml` 在编译期被消费以生成 eager 层 C++/pybind 绑定，`python/paddle/_paddle_docs.py`、`python/paddle/tensor/math.py` 为纯 Python 改动。因此**不能只依赖已有 wheel**，需 source build 或等价的已编译环境重新触发代码生成。
- 变更文件（gold patch）：
  - `paddle/fluid/eager/auto_code_generator/generator/monkey_patch_gen.py`（+59/-63，重构为统一 map）
  - `paddle/phi/ops/yaml/python_api_info.yaml`（+5，新增 `inverse` 三路径配置）
  - `python/paddle/_paddle_docs.py`（+43，下沉后文档/签名）
  - `python/paddle/tensor/math.py`（+1/-57，新增从生成模块 import `inverse`，删除原 Python 端 `def inverse` 实现，文档迁移到 `_paddle_docs.py`）
- 测试文件（test patch）：`test/legacy_test/test_inverse_op.py`（+152/-10）
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为与验收标准（下沉、别名 `input`、`out` 参数、`paddle.linalg.inv` 一致性），不直接指出 `monkey_patch_gen.py` 的具体重构行或 YAML 条目。
- 环境风险：涉及构建期代码生成，历史 commit 复现需 source build；`_paddle_docs.py` 易冲突（reviewer 已提示），维护组构建时需注意补丁落点。
- flaky 风险：complex 梯度用例与随机矩阵可逆性依赖 seed（测试已 `np.random.seed(123)` 并做可逆性检查），需 verifier 重复运行确认稳定，抽取稳定 F2P/P2P `nodeid`。
- 拆分风险：低。该 PR 目标聚焦「inverse 下沉 + Cpp sink 机制通用化」，最终合入 diff 仅涉及 5 个文件且均围绕该主线（`math.py` 只是删除旧 Python 实现并从生成模块 import），无混入的独立目标，适合作为单一样本。
- 其他不确定点：Linux CPU source build 的最小硬件与编译时长尚未实测，需在 `environment/README.md` 中以「已验证环境 + 未确认最小配置」方式标注。
