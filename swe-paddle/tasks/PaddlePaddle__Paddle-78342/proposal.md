# 任务提案：PaddlePaddle__Paddle-78342

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78342`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78342
- PR 标题：`[API Compatibility] add new api paddle._assert -part`
- `base_commit`：`fa323f323bb35359c9d4ba77763834fee82a87b4`（squash 合入 commit `f92a35feea4acf62b2df2259ae491b992851f854` 的第一父提交）
- merged 时间：2026-03-26 12:47:30 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

Paddle 缺少与 `torch._assert(condition, message)` 对齐、同时可用于动态图和静态图的公开断言 API。该 PR 新增 `paddle._assert`：在动态图中立即检查 Python 或 Tensor 条件并抛出 `AssertionError`，在静态图中为 Tensor 条件构建可执行的断言节点。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自已合入的 Paddle API Compatibility 工作，解决模型代码从 PyTorch 迁移时缺少可符号追踪断言入口的真实兼容需求。
- **代表性**：样本覆盖 Python 顶层 API 暴露、Python truthiness、Tensor / PIR / 静态 Variable 分派、动态图立即执行、静态图控制流节点以及位置参数和关键字参数兼容，是动态图与静态图统一 API 设计的典型任务。
- **边界清楚**：动态图下，truthy 的 bool、int、表达式和单元素 Tensor 应正常返回，falsy 条件应抛出 `AssertionError` 并保留显式 message；未提供 message 时错误文本为空。静态图下，Tensor 条件应加入图并可在 CPU Executor 中执行。接口应支持 `(condition, message)` 位置参数、两个关键字参数和混合调用。任务不扩展到多元素 Tensor 的真值定义，也不额外规定上游测试未覆盖的静态失败 message 文本。
- **非平凡性**：该功能不能简单调用 Python `assert`，因为静态 Tensor 在建图阶段不能直接求值。实现需要识别动态图 Tensor、PIR Value、静态 Variable 与普通 Python 对象，在 eager 路径立即报错，在 static 路径生成运行时断言，同时保持公开导入路径和现有 testing comparison API 不受影响。
- **范围单一**：squash commit 修改 3 个 Python 导出 / 实现文件和 1 个兼容性测试文件，全部 118 行新增与 1 行调整均服务于 `paddle._assert`，没有需要剔除的独立功能或清理 hunk。

## 4. 任务类型和标签

- 任务类型：`feature_implementation`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, dynamic_graph, static_graph, pir, control_flow]`

## 5. 验证思路

- 目标测试文件 / 命令：

  ```bash
  python -m pytest \
    test/legacy_test/test_api_compatibility_part2.py::TestAssertAPI \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 上，顶层 `paddle._assert` 不存在；目标类的七个用例在首次调用时因 `AttributeError` 失败，包括 Python 条件、Tensor 条件、默认 message、PyTorch 风格调用形式和静态图 Tensor 条件。
- 修复后预期：在 `base_commit + test_patch + code_patch` 上，目标类全部通过。动态图 truthy 条件不报错，falsy Python / Tensor 条件抛出 `AssertionError`，显式与默认 message 符合预期；位置、关键字和混合调用均可用；静态图中 true Tensor 条件可成功建图并由 CPU Executor 执行。
- P2P 候选：`test/legacy_test/test_assert_close.py` 可保护新增导出所在 testing comparison 模块的既有断言 API；`test/legacy_test/test_assert_op.py` 可保护底层静态 Assert 算子行为；`test_api_compatibility_part2.py` 中其他存量兼容类也可按单用例结果作为导入与模式切换回归护栏。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无与该历史 commit 精确匹配的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，配合与该 revision 兼容的本地 Paddle 包；patch 为纯 Python，可优先使用精确 base-compatible wheel 加源码 overlay，否则使用 base revision 的本地构建产物
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：暂未固定精确 wheel URL；完整任务阶段优先查找对应日期的 Linux x86_64 CPU nightly wheel，不使用无法确认 commit / ABI 兼容性的任意新版 wheel
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti
- patch 类型：纯 Python，不含 C++、CUDA、kernel 或 infermeta 编译改动
- 最小测试命令：`python -m pytest test/legacy_test/test_api_compatibility_part2.py::TestAssertAPI -q`
- 是否有 oracle 日志：有；合入 PR 的 CI 提供修复后测试记录，完整任务阶段可补充精确 base 环境的 fail-before / pass-after 本地日志

## 7. 风险自查

- **泄露风险**：后续 `instruction.md` 只描述 `paddle._assert` 的公开调用形式、动态图错误和静态图可执行行为，不应点名内部实现模块、类型分派代码、静态控制流 helper 或具体修改位置。
- **环境风险**：patch 本身为纯 Python，但静态测试依赖历史 revision 已编译的 Assert 算子和匹配的 Paddle core。完整任务必须记录 `paddle.__file__`、版本和 commit，并固定 wheel 或本地构建来源，避免新版安装包提前包含该 API 或改变模式行为。
- **测试风险**：仅检查符号存在会使任务失去判别力。目标测试必须保留 truthy / falsy、Python / Tensor、message、三种参数调用形式和静态 Executor 路径；同时避免把未覆盖的多元素 Tensor 或静态失败 message 语义写成额外验收要求。
- **范围风险**：任务只新增 `paddle._assert` 并复用已有静态 Assert 能力，不修改 Python `assert`、`paddle.static.nn.control_flow.Assert` 或其他 `paddle.testing.assert_*` API 的契约，也不要求新增 kernel 或控制流算子。
- **版本风险**：动态图、旧静态图与 PIR 对 Tensor 类型和 truthiness 的处理可能随版本演进；验收必须固定在该 base/gold commit 对，并以 PR 明确覆盖的单元素条件和调用形式为准。
