# 任务提案：PaddlePaddle__Paddle-77064

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-77064`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/77064
- PR 标题：`[API Compatibility] Sink paddle.allclose to cpp -part`
- `base_commit`：`1a6a9ab02e12fd792d036dc78b94f46a1371e6fa`（squash 合入 commit `407e3b6931a282a78653d559e675a598153ae977` 的第一父提交）
- merged 时间：2025-12-25 11:41:33 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

`paddle.allclose` 原有 Python wrapper 未接入统一的 C++-sink API 生成与参数别名机制，不能完整支持 `input`/`other` 兼容参数；同时 Paddle 缺少返回 Python `bool`、签名与 PyTorch 对齐的兼容入口。该 PR 将 `allclose` 下沉到 C++ 绑定，补充函数和 Tensor 方法的参数别名，并新增 `paddle.compat.allclose`。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自真实的 Paddle API Compatibility 工作，目标是统一 Python/C++ API 行为并提供 PyTorch 风格兼容入口，且 PR 已在 Paddle 主仓库合并。
- **代表性**：样本覆盖 Python API C++ sink、YAML/codegen 参数别名、PIR 静态参数预处理、Tensor method、compat 适配层及返回类型差异，代表 Paddle API 兼容改造的典型链路。
- **边界清楚**：`paddle.allclose` 应继续返回单元素 Tensor，并同时接受 `x`/`y`、`input`/`other` 及测试中的混合关键字组合；Tensor 方法应接受 `other`。`paddle.compat.allclose` 应使用 `input`、`other` 参数并返回 Python `bool`。两者都需要保留 `rtol`、`atol`、`equal_nan`、CPU/可选 CUDA 和原有 dtype 校验语义，普通位置参数调用不得回归。
- **非平凡性**：任务不是简单增加一个 Python 别名。正确修复需要把现有 wrapper 切换到生成的 C++ 绑定，在 API 元数据中建立函数与 Tensor method 的别名映射，在静态路径补齐输入和容差 dtype 预处理，并保证 compat wrapper 的 Python `bool` 契约不改变底层 `paddle.allclose` 的 Tensor 返回契约。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, cpp_binding, pybind, codegen, tensor_method, static_graph, dtype]`

## 5. 验证思路

- 目标测试文件：
  - `test/legacy_test/test_allclose_op.py`
  - `test/legacy_test/test_compat_allclose.py`
- 目标测试命令：

  ```bash
  python -m pytest \
    test/legacy_test/test_allclose_op.py::TestAllcloseAlias \
    test/legacy_test/test_compat_allclose.py \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 下，`paddle.allclose(input=..., other=...)` 及混合别名调用会因旧 Python 签名不接受这些关键字而报错，Tensor method 的 `other` 别名也不可用；`paddle.compat` 尚未导出 `allclose`，新增 compat 测试在导入或调用阶段失败。
- 修复后预期：在 `base_commit + test_patch + code_patch` 下，函数 API 的四种别名/原名组合结果一致，Tensor method 可使用 `other`，`equal_nan=True` 保持正确行为；compat API 在 CPU 上对接近和不接近的输入分别返回 Python `True`/`False`，而底层 `paddle.allclose` 继续返回 Tensor，全部目标测试通过。
- P2P 候选：`test/legacy_test/test_allclose_op.py` 中现有 allclose operator、静态/动态图、dtype、FP16、容差和 NaN 测试可作为主要回归护栏，确认 C++ sink 后数值与类型检查没有变化；同模块既有 `isclose` 测试可辅助保护共享 close-family 预处理逻辑。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无与该历史 commit 精确匹配的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，必须使用源码构建后应用 patch；普通 wheel 无法承载 pybind C++ 和 YAML/codegen 变更
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用；当前没有确认与 `base_commit` 精确匹配且包含待修改源码的 wheel
- OS / Python / CUDA / cuDNN / 其他关键依赖：建议 Linux x86_64、该 commit 支持的 Python 3、CMake、GCC/Clang、NumPy 和 `pytest`；目标测试不依赖 CUDA/cuDNN
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti，GPU 不是本任务要求
- patch 类型：含 C++ pybind 参数预处理、YAML/codegen 和 Python API/compat wrapper，需要重新编译 Paddle
- 最小测试命令：`python -m pytest test/legacy_test/test_allclose_op.py::TestAllcloseAlias test/legacy_test/test_compat_allclose.py -q`
- 是否有 oracle 日志：无固定 oracle 日志；以既有 allclose 结果、Tensor/Python bool 类型断言、别名一致性和 NaN 行为作为验收依据

## 7. 风险自查

- 泄露风险：后续 `instruction.md` 只描述公开 API 的可观察签名、返回类型和兼容行为，不包含 source PR、具体文件、内部预处理函数、YAML 映射、diff 结构或实现顺序；proposal 中的实现范围仅供维护组审核
- 环境风险：任务必须在历史 `base_commit` 上完成 Paddle 源码构建，并触发 YAML/codegen 和 pybind C++ 编译，无法通过修改普通已安装 wheel 完成；维护组 verifier 需要固定源码构建配方
- flaky 风险：新增测试使用固定小张量，只断言确定性的 allclose 结果、别名一致性、返回类型和 `equal_nan` 行为，不依赖随机 seed、外部数据、网络服务或多卡同步，预期无明显 flaky 风险
- 拆分风险：PR 同时包含 C++ sink/参数别名和 `paddle.compat.allclose`，但两者共同定义同一 allclose 兼容边界，且 compat 层直接复用下沉后的公开 API。拆分会使一个子任务只验证内部迁移、另一个子任务依赖未纳入范围的别名与底层契约，因此保留为一个样本
- 其他不确定点：当前环境未完成该历史 commit 的 Paddle 源码构建，最终 build flags、生成代码步骤和测试启动方式需由维护组 verifier 确认；CPU 是主验收后端，CUDA 条件分支不作为通过条件
