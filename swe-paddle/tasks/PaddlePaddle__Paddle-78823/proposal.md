# 任务提案：PaddlePaddle__Paddle-78823

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78823`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78823
- PR 标题：`[API Compatibility] add pin_memory for randint`
- `base_commit`：`c3a5e799eb9e390f830c9e6c3fbea2e9370afa7f`（squash 合入 commit `ddb483237539d4d23c7dbbd44e3a360439c780ed` 的第一父提交）
- merged 时间：2026-05-12 09:32:06 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

Paddle 的多个张量创建和随机生成 API 虽然提供了 `pin_memory=True`，但对 CPU、CUDA、XPU 设备的 pinned place 处理分散且不一致，导致 `randint` 等 API 无法按 PyTorch 兼容语义工作。该 PR 统一 pinned place 映射逻辑，使 CPU 请求能够在可用的 CUDA/XPU 构建中映射到对应的 pinned allocator，同时保留纯 CPU 构建不支持 pinned memory 时的明确错误。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自 Paddle 的真实 API Compatibility 需求，目标是对齐 PyTorch 的 `pin_memory` 行为，且 PR 已在 Paddle 主仓库合并。
- **代表性**：样本覆盖 Python API 语义、设备 place 转换、CUDA/XPU pinned memory，以及 tensor creation 和 random API 的动态执行路径，代表深度学习框架中的跨设备 API 兼容能力。
- **边界清楚**：修复需要同时满足以下边界：已经是 `CUDAPinnedPlace` 或 `XPUPinnedPlace` 的 place 应保持不变；CUDA/XPU place 应转换为对应的 pinned place；CPU place 应根据编译环境选择可用的 pinned allocator；纯 CPU 构建没有 pinned allocator 时仍应抛出原有的显式错误。未设置 `pin_memory` 的普通设备创建行为不应被改变；本样本的 F2P 重点是动态执行路径，静态图行为作为既有回归范围而不是新的主验收目标。
- **非平凡性**：变更不只是给 `randint` 增加参数，而是抽取共享的 place 转换语义，并同步替换 tensor、random、audio window 等多个 API 中重复且不一致的逻辑；同时还要兼顾动态模式、已 pinned place、CUDA/XPU 编译分支和无加速器环境。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cuda`（XPU 分支可选验证）
- 设备范围：`special_hardware`
- 模块标签：`[python_api, api_compatibility, pin_memory, place, cuda, xpu, tensor_creation, random]`

## 5. 验证思路

- 目标测试文件：
  - `test/legacy_test/test_to_pinned_place.py`
  - `test/legacy_test/test_eager_tensor.py`
  - `test/legacy_test/test_rand.py`
  - `test/legacy_test/test_randint_op.py`
  - `test/legacy_test/test_randperm_op.py`
- 目标测试命令：

  ```bash
  python -m pytest \
    test/legacy_test/test_to_pinned_place.py \
    test/legacy_test/test_eager_tensor.py \
    test/legacy_test/test_rand.py \
    test/legacy_test/test_randint_op.py \
    test/legacy_test/test_randperm_op.py \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 下，新增 pinned-place 分支测试无法导入统一的转换能力；在具备 CUDA/XPU 的构建中，`device="cpu", pin_memory=True` 的 tensor/random/randint/randperm 调用仍会进入旧的 unsupported-device 错误路径，相关回归测试失败。
- 修复后预期：在 `base_commit + test_patch + code_patch` 下，统一转换逻辑可被所有目标 API 使用；已 pinned place 保持不变，CUDA/XPU/CPU place 按编译环境映射到正确的 pinned place，支持的设备测试通过，纯 CPU 构建的 unsupported case 仍返回明确错误，全部目标测试通过。
- P2P 候选：同模块的 `test_eager_tensor.py`、`test_rand.py`、`test_randint_op.py` 和 `test_randperm_op.py` 中现有 tensor/random creation、device 和 shape/dtype 测试可作为回归护栏；`test/legacy_test/test_audio_functions.py` 中现有 `get_window` 测试可作为 audio window 的基础回归护栏。需要明确的是，source PR 没有新增专门验证 audio `pin_memory` 的 F2P 用例，因此 audio window 只作为共享逻辑的 P2P 范围，不宣称有独立的端到端覆盖；这些测试用于确认普通设备创建、已有随机 API 和音频 window 行为未因 pinned place 逻辑统一而改变。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无可复用的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，建议使用源码构建以保证 Python API 与该 PR 的基线一致
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用；当前没有确认与 `base_commit` 精确匹配的固定 wheel URL
- OS / Python / CUDA / cuDNN / 其他关键依赖：建议 Linux x86_64 或 Windows、该 commit 支持的 Python 3 版本和 `pytest`；完整验收需要 CUDA 或 XPU 构建，CUDA runner 还需要匹配的 CUDA/cuDNN 版本
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：完整 verifier 需要 CUDA 或 XPU 设备，单卡即可，不要求多卡或多节点；原 PR 验证机器为 9800X3D + RTX 5070Ti。CPU-only runner 只能覆盖纯 CPU 错误分支和部分 mock 分支，不能替代完整验收
- patch 类型：纯 Python；不修改 C++、CUDA、kernel 或 infermeta 编译产物
- 最小测试命令：`python -m pytest test/legacy_test/test_to_pinned_place.py test/legacy_test/test_eager_tensor.py test/legacy_test/test_rand.py test/legacy_test/test_randint_op.py test/legacy_test/test_randperm_op.py -q`
- 是否有 oracle 日志：无；以目标测试结果和 place/异常断言作为验收依据

## 7. 风险自查

- 泄露风险：后续生成的 `instruction.md` 只描述可观察的 `pin_memory` 行为和验收标准，不泄露 source PR、具体修改行、统一 helper 名称或 gold patch 结构；本 proposal 允许保留来源和实现范围信息供维护组审核
- 环境风险：任务依赖 Paddle 在 `base_commit` 上的源码构建，没有已确认的历史 wheel 或 Docker 镜像；完整验收依赖 CUDA 或 XPU 构建，CPU-only runner 只能覆盖纯 CPU 错误分支和部分 mock 分支，不能把被条件跳过的 accelerator 测试视为完整通过
- flaky 风险：测试主要断言 place、shape 和预期异常，不依赖随机数具体取值、固定多卡拓扑或外部服务；随机 API 的生成结果不作为数值 oracle，因此预期无明显 flaky 风险
- 拆分风险：PR 的共享 place 转换、多个 tensor/random/audio API 调用点和测试共同实现同一个 `pin_memory` 兼容性契约，拆分后无法独立表达完整行为，保留为一个样本
- 其他不确定点：当前环境无法完成 Paddle 源码构建；需由维护组 verifier 确认历史 commit 的 Python/pytest 配置和测试启动方式，并在可用的 CUDA/XPU runner 上补充真实设备分支验证
