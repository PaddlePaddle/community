# 任务提案：PaddlePaddle__Paddle-78441

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78441`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78441
- PR 标题：`[API Compatibility] add aminmax op-part`
- `base_commit`：`35b36cca24a780061268d20d6abe512e758837e6`（squash 合入 commit `156159726b64d8f85747de864fb3ce41ea1f3f2f` 的第一父提交）
- merged 时间：2026-04-27 08:27:19 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

Paddle 缺少与 `torch.aminmax` 对齐的 API，用户无法通过一次调用同时获得张量沿指定轴的最小值和最大值。该 PR 新增 `paddle.aminmax` 和对应 Tensor 方法，并补齐双输出形状推导、CPU/GPU kernel、反向传播、静态/动态图及兼容参数语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自真实的 Paddle API Compatibility 需求，目标是补齐 PyTorch 已有的 `aminmax` 能力，且 PR 已在 Paddle 主仓库合并并通过 CI。
- **代表性**：样本覆盖一个新算子从 Python API、YAML/codegen、infermeta、CPU/GPU kernel、autograd 到 PIR symbolic shape 和动态形状测试的完整实现链路，能代表 Paddle 新算子开发的核心能力。
- **边界清楚**：两个输出必须分别等价于相同参数下的 `amin` 和 `amax`；`axis=None`、单轴、多轴和负轴、`keepdim`、零维及空张量需要得到正确形状；兼容 `input`/`x`、`dim`/`axis` 别名和 `out=(min_out, max_out)`；动态图、静态图和 Tensor 方法的公开语义应一致。存在重复最小值或最大值时，每个输出的梯度必须在对应的重复极值之间均匀分配。
- **非平凡性**：任务不是简单导出一个 Python 名称。正确实现需要定义双输出 op 和 backward op，保持两路 infermeta/symbolic shape 一致，注册不同后端和 dtype 的 kernel，并正确合并最小值、最大值两路梯度；仅拼接两次 Python reduction 或只实现前向无法通过完整测试。

## 4. 任务类型和标签

- 任务类型：`feature_implementation`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, operator_kernel, infermeta, autograd, symbolic_shape, pir, codegen]`

## 5. 验证思路

- 目标测试文件：
  - `test/legacy_test/test_aminmax_op.py`
  - `test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py`
- 目标测试命令：

  ```bash
  python -m pytest test/legacy_test/test_aminmax_op.py -q
  python -m pytest \
    test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 下，`paddle.aminmax`、Tensor 方法和底层 op 尚不存在，legacy 测试会在 API 查找、op 创建或执行阶段失败；新增 PIR symbolic-shape 用例也无法构建对应运算。
- 修复后预期：在 `base_commit + test_patch + code_patch` 下，CPU runner 上的 forward、gradient、API compatibility、静态/动态图、out 参数、动态形状和 symbolic shape 用例全部通过。最小值和最大值结果与 NumPy reference 一致，两个输出形状一致且符合 reduction 语义，重复极值梯度按预期均匀分配。
- P2P 候选：`test/legacy_test/test_max_min_amax_amin_op.py` 中现有 `min`、`max`、`amin`、`amax` 输出和梯度测试，以及 `test/prim/pir_prim/test_prim_amax_amin_op.py` 中既有 reduction 测试，可用于确认共享 reduction kernel、梯度和 PIR 行为未被破坏。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无与该历史 commit 精确匹配的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，必须使用源码构建后应用 patch；发布版或 nightly wheel 无法承载新增 C++ op/kernel 和 YAML codegen 变更
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用；当前没有确认与 `base_commit` 精确匹配且包含待修改源码的 wheel
- OS / Python / CUDA / cuDNN / 其他关键依赖：建议 Linux x86_64、该 commit 支持的 Python 3、CMake、GCC/Clang、NumPy 和 `pytest`；核心 legacy 测试不依赖 CUDA/cuDNN，symbolic-shape 用例需要构建中具备对应 PIR 测试能力，但目标用例不启用 CINN backend
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti，GPU 不是本任务要求
- patch 类型：含 C++、CPU/GPU kernel、infermeta、PIR symbolic shape、YAML/codegen 和 Python API 导出，需要重新编译 Paddle
- 最小测试命令：`python -m pytest test/legacy_test/test_aminmax_op.py -q`
- 是否有 oracle 日志：无固定 oracle 日志；以 NumPy reference、梯度断言、shape 断言和目标测试结果作为验收依据

## 7. 风险自查

- 泄露风险：后续 `instruction.md` 只描述新增公开 API 的可观察行为和验收标准，不包含 source PR、具体文件、内部 op/kernel/infermeta 函数名、diff 结构或实现顺序；proposal 中的实现范围仅供维护组评估样本价值
- 环境风险：任务必须在历史 `base_commit` 上完成 Paddle 源码构建，并触发 YAML/codegen 和 C++ kernel 编译，无法使用普通已安装 wheel 验证；symbolic-shape 用例依赖 PIR 测试能力，但目标用例不启用 CINN backend，维护组 verifier 仍需固定完整构建配方
- flaky 风险：min/max 在极值切换点的数值梯度较敏感，source tests 已避免对不稳定的 float32 场景做通用有限差分，并用显式重复值测试梯度分配；完整任务包应保留这些输入、dtype 和容差，不扩大为随机低精度梯度检查。测试不依赖外部数据、多卡同步或网络服务
- 拆分风险：虽然 PR 跨越多个子系统，但所有改动共同组成一个可调用、可求导且可进行形状推导的 `aminmax` 算子；拆分 kernel、API、autograd 或 symbolic shape 会得到无法独立验收的残缺任务，因此保留为一个样本
- 其他不确定点：当前环境未完成该历史 commit 的 Paddle 源码构建，测试命令和 PIR/CINN 构建选项需由维护组 verifier 最终确认；CPU 是主验收后端，GPU 注册不作为本样本通过条件
