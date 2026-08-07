# 任务提案：PaddlePaddle__Paddle-78220

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78220`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78220
- PR 标题：`[API Compatibility] add paddle.compat.nn.functional.log_softmax -part`
- `base_commit`：`56be465924264e1251cf127dbff56d17a7554d01`（squash 合入 commit `bfe91230d558176d2d932b50953cb7b4391065d1` 的第一父提交）
- merged 时间：2026-03-16 15:20:08 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

Paddle 原有 `paddle.nn.functional.log_softmax` 缺少 PyTorch 风格的 `input`、`dim` 和 `out` 调用形式，也未在 `paddle`、Tensor、`paddle.special` 与 `paddle.compat.nn.functional` 等兼容入口统一暴露该能力。该 PR 在保持原有 `x`、`axis`、数值、梯度和静态图行为的同时，补齐这些公开调用入口及参数兼容语义。

需要注意，各入口在省略维度时保留不同的默认行为：标准 `paddle.nn.functional.log_softmax` 使用 Paddle 原有的 `axis=-1`；`paddle.log_softmax`、`Tensor.log_softmax`、`paddle.special.log_softmax` 和 `paddle.compat.nn.functional.log_softmax` 使用兼容实现的 `dim=None` 规则，其中输入 rank 为 0、1 或 3 时取 `dim=0`，其他 rank 取 `dim=1`。因此，五个入口只在显式指定等价 `axis`/`dim` 时要求数值一致，不能把省略维度时的结果一致作为要求。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自已合入的 Paddle API Compatibility 工作，目标是让真实用户可以按 PyTorch 常见签名调用 `log_softmax`，并使 Paddle 的多条公开访问路径在显式指定等价维度时保持一致。
- **代表性**：样本覆盖 Python API 参数别名、顶层与子模块导出、Tensor method 注册、严格 compat wrapper、动态图 / 静态图 / PIR 分支、dtype 转换以及 `out` 写入语义，是 Paddle Python API 兼容改造的典型任务。
- **边界清楚**：`paddle.nn.functional.log_softmax` 需要保留位置参数及 `x`、`axis`、`name` 调用，同时新增 `input`、`dim`，并在动态图 / PIR 路径支持仅关键字 `out`；兼容入口采用 PyTorch 风格参数，并应拒绝不属于该入口的 `x`、`axis`、`name`。`dim=None` 的默认维度、dtype 转换、别名冲突报错和输出 Tensor 写入均有明确测试。算子数学定义、支持 dtype、梯度和已有静态图行为不应改变。
- **非平凡性**：修复不是单一签名重命名；它需要统一五条公开 API 路径，正确处理成对别名及冲突、Tensor method 绑定、默认维度规则、dtype 和 `out`，同时避免破坏原有 Paddle 参数名和底层 `log_softmax` 算子行为。
- **范围单一**：squash commit 修改的 8 个 Python / 测试文件均服务于同一个 `log_softmax` API 兼容目标，未发现需要从任务中剔除的独立功能或清理提交。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, tensor, dynamic_graph, static_graph, pir]`

## 5. 验证思路

- 目标测试文件 / 命令：

  ```bash
  python -m pytest test/legacy_test/test_compat_log_softmax.py -q
  python -m pytest \
    test/legacy_test/test_log_softmax.py::TestLogSoftmaxParamAlias \
    test/legacy_test/test_log_softmax.py::TestLogSoftmaxOutParam \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 上，`paddle.nn.functional.log_softmax(input=..., dim=...)` 和 `out=` 调用不受支持；`paddle.log_softmax`、`Tensor.log_softmax`、`paddle.special.log_softmax`、`paddle.compat.nn.functional.log_softmax` 等新增入口缺失；相关 F2P 用例因 `TypeError` 或 `AttributeError` 失败。原有位置参数及 `x` / `axis` 调用仍应有效。
- 修复后预期：在 `base_commit + test_patch + code_patch` 上，上述目标测试通过。五条公开访问路径的结果与 SciPy reference 一致；`input` / `dim` 别名、别名冲突、`out`、dtype 转换、`dim=None` 的 0D 到 4D 默认规则以及 compat 入口的非法关键字检查均符合预期。`paddle.nn.functional.log_softmax` 的静态图别名调用也应通过。
- P2P 候选：`test/legacy_test/test_log_softmax.py` 中原有的 `TestLogSoftmaxOp`、`TestNNLogSoftmaxAPI`、`TestNNFunctionalLogSoftmaxAPI`、`TestLogSoftmaxOp_ZeroSize` 等测试，可继续保护 CPU 数值、shape、dtype、梯度、静态图和 PIR 行为；完整模块可按单用例结果区分新增 F2P 与原有 P2P。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无与该历史 commit 精确匹配的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，配合与该 revision 兼容的本地构建 Paddle 包；由于 patch 为纯 Python，也可在确认 ABI / API 匹配的 base-compatible wheel 上使用源码 overlay
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：暂未固定精确 wheel URL；完整任务阶段优先查找对应日期的 Linux x86_64 CPU nightly wheel，否则使用 base revision 本地构建产物，不使用无法确认 commit 的任意新版 wheel
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti
- patch 类型：纯 Python，不含 C++、CUDA、kernel 或 infermeta 编译改动
- 最小测试命令：`python -m pytest test/legacy_test/test_compat_log_softmax.py test/legacy_test/test_log_softmax.py::TestLogSoftmaxParamAlias test/legacy_test/test_log_softmax.py::TestLogSoftmaxOutParam -q`
- 是否有 oracle 日志：有；合入 PR 的 CI 提供修复后测试记录，完整任务阶段可补充精确 base 环境的 fail-before / pass-after 本地日志

## 7. 风险自查

- **泄露风险**：后续 `instruction.md` 只描述五条公开 API 的可观察签名、结果和错误行为，不应点名内部装饰器、注册列表、实现文件或具体修改位置；gold patch 与题面保持分离。
- **环境风险**：主要风险是历史 commit 与已安装 Paddle wheel 不匹配。patch 本身为纯 Python，但测试仍依赖匹配的 Paddle compiled core；完整任务必须记录 `paddle.__file__`、版本和 commit，并固定 wheel 或本地构建来源。SciPy 仅作为本地数值 oracle，不依赖外部服务。
- **测试风险**：上游新增兼容测试覆盖面较广，后续拆分 `test_patch` 时需保证 base 上确实暴露目标失败，而不是因导入或环境问题整体中断；同时保留原有 operator 测试作为 P2P 护栏，避免只验证签名而漏掉数值和梯度回归。
- **范围风险**：多个公开访问路径看似涉及多个模块，但它们共同表达同一个 `log_softmax` 兼容契约，不应拆成互相独立的任务。任务不扩展到其他 softmax API，也不要求改变底层算子、kernel、精度或设备支持。
- **版本风险**：`dim=None` 的隐式维度选择和 compat 入口的严格关键字限制具有版本语义，验收应以该 PR 合入时明确覆盖的行为为准，避免使用未来版本行为反向定义本任务。
