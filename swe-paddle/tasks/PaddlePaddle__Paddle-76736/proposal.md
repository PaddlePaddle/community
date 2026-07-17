# 任务提案：PaddlePaddle__Paddle-76736

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-76736`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/76736
- PR 标题：`[API Compatibility No.33] Cpp sink for atan2 -part`
- `base_commit`：`7743e779aff3e35b8bd748b2c69b9332f5d8dfd7`（squash 合入 commit `3e4695db8fc3a19e9e055709941ad2b99f7f6c5f` 的父提交）
- merged 时间：`2026-02-05T06:21:01Z`
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

原有 `paddle.atan2` 由 Python 包装层负责广播，尚未完整下沉到 C++，也不支持 PyTorch 风格的 `input`/`other` 参数别名、`out` 参数和 Tensor 方法调用。该 PR 将 API 下沉到 C++ 生成机制，并在 infermeta、前向 kernel 和反向 kernel 中补齐广播及梯度归约语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 Paddle API Compatibility 真实研发 PR。Review 过程中明确要求只处理 `atan2`，将 Python 前置广播逻辑下沉到 infermeta/kernel，并补测 `out` 参数；最终改动经过 Paddle CI、coverage 和两位 reviewer 审批。
- **代表性**：覆盖 Paddle 框架中典型的端到端 API 下沉能力，包括配置驱动的 C++ API 暴露、参数别名、Tensor 方法、`out` 参数、infermeta 广播推导、CPU/GPU kernel、广播反向梯度归约、整数输入 dtype 规则和空 Tensor。
- **边界清楚**：目标行为限定为 `atan2`。位置参数和原有 `x`/`y` 关键字调用保持可用，同时新增 `input`/`other`、`out` 和 `Tensor.atan2`；不同形状输入遵循广播规则，梯度恢复为各输入原始形状，整数输入输出为 `float64`。数值公式、其他激活函数及其他二元算子不在任务范围内。
- **非平凡性**：不能只修改 Python 签名或 YAML。删除 Python 包装后，原先由 Python 执行的广播必须在 infermeta 和 kernel 中完整实现；反向传播还需识别广播轴并归约梯度，同时避免为未请求的梯度分配内存，并兼顾同形状快速路径、空 Tensor 和多维广播。
- **区分度潜力**：修复横跨 API 生成、shape/dtype 推导、前向 kernel 和 autograd。仅让别名测试通过而忽略广播梯度，或仅保留 Python 包装而未完成下沉，都无法通过完整验收。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, cpp_sink, infermeta, operator_kernel, autograd, broadcasting, dtype, yaml_config]`

## 5. 验证思路

- 目标测试文件：
  - `test/legacy_test/test_api_compatibility.py`
  - `test/legacy_test/test_atan2_op.py`
- 关键 F2P 场景：
  - 动态图和静态图支持 `paddle.atan2(x, y)`、`paddle.atan2(x=x, y=y)` 与 `paddle.atan2(input=x, other=y)`，结果和 NumPy 一致。
  - 动态图支持 `out` 参数，返回结果写入指定 Tensor。
  - 补充 `x.atan2(y)` / `x.atan2(other=y)` 的 Tensor 方法覆盖，验证 C++ sink 暴露的两条公开调用路径一致。
- 修复前预期：在 `base_commit` 上应用测试补丁后，原有位置参数及 `x`/`y` 调用仍通过，但 `input`/`other`、`out` 和 Tensor 方法调用因旧 Python API 不支持而出现 `TypeError` 或 `AttributeError`。
- 修复后预期：继续应用来自 #76736 的非测试改动并重新构建后，新增兼容性用例全部通过；广播前向结果、广播梯度、整数 dtype、空 Tensor 等既有行为保持正确。
- P2P 候选：
  - `test/legacy_test/test_atan2_op.py::TestAtan2API`：原有静态图和动态图公开 API 数值回归。
  - `TestAtan2Broadcasting`：不同 rank/shape 的前向广播及输入梯度形状回归；完整任务包应补充一个小规模广播梯度数值用例，分别校验两个输入的梯度及其原始 shape。
  - `TestAtan2EmptyTensorInput`：广播空 Tensor 回归。
  - `TestAtan2`、`TestAtan2_float`：同 shape 算子前向与梯度数值回归。
  - `TestAtan2_int32`、`TestAtan2_int64`：整数输入输出 `float64` 的 dtype 与数值回归。
- 候选测试命令：

  ```bash
  export PYTHONPATH="$(pwd)/test/legacy_test${PYTHONPATH:+:$PYTHONPATH}"
  python -m pytest \
    test/legacy_test/test_api_compatibility.py::TestAtan2API_Compatibility \
    test/legacy_test/test_atan2_op.py::TestAtan2API \
    test/legacy_test/test_atan2_op.py::TestAtan2Broadcasting \
    test/legacy_test/test_atan2_op.py::TestAtan2EmptyTensorInput \
    test/legacy_test/test_atan2_op.py::TestAtan2_int32 \
    test/legacy_test/test_atan2_op.py::TestAtan2_int64 \
    -q
  ```

  完整任务包阶段将命令收敛为 `bash tests/test.sh`，并根据 base/fixed 实测结果保留稳定的 CPU F2P/P2P 节点。
- 原 merge diff 在 `test_activation_op.py` 中保留了两行 `asinh`/`atan` 兼容性改动，但 reviewer 已明确本 PR 只处理 `atan2`；完整任务包的 `tests/test.patch` 将排除这两行以及 `test_atan2_op.py` 中仅增加空行的无语义改动。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 通过后依据维护组 verifier 基础环境补充 source-build 配方。
- Dockerfile 或镜像地址：暂无。
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，必须重新构建。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用；该任务修改构建期 API 生成配置、infermeta 及 C++ kernel，历史 wheel 无法承载 gold patch。
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti，GPU 不是本任务要求
- patch 类型：含 C++、CUDA、infermeta、kernel、构建期 YAML 代码生成和 Python 文档/导出改动，需要 source build。
- gold patch 预计包含：
  - `paddle/phi/infermeta/binary.cc`
  - `paddle/phi/kernels/funcs/eigen/broadcast.cc`
  - `paddle/phi/kernels/funcs/eigen/broadcast.cu`
  - `paddle/phi/kernels/impl/atan2_grad_kernel_impl.h`
  - `paddle/phi/kernels/impl/atan2_kernel_impl.h`
  - `paddle/phi/kernels/impl/broadcast_tensors_kernel_impl.h`
  - `paddle/phi/ops/yaml/python_api_info.yaml`
  - `python/paddle/_paddle_docs.py`
  - `python/paddle/tensor/math.py`
- 最小测试命令：`bash tests/test.sh`（完整任务包阶段提供）。
- 是否有 oracle 日志：无；由 SWE-Paddle Run/Test/Fix verifier 记录。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述 `atan2` 的公开调用形式、广播、梯度、dtype 和空 Tensor 等可观察行为，不应给出具体文件、内部 helper、归约轴算法或 YAML 配置方式。
- 环境风险：较高。任务包含 C++/CUDA kernel 与构建期代码生成，不能通过 Python overlay 或普通 wheel 验证；历史 base commit 必须能够完成 source build。
- flaky 风险：低。目标测试使用固定 seed，并与 NumPy 结果比较；梯度检查和设备枚举需在 verifier 中选择稳定 CPU node，避免把可选 GPU/custom-device 路径作为最低验收。
- 拆分风险：中等但可控。最终 diff 横跨 12 个文件，但核心改动均服务于 `atan2` 完整下沉和广播语义；完整任务包会排除与 `asinh`/`atan` 有关的两行测试及纯空白改动，不把其他 API 纳入题意。
- 编译风险：广播模板实例化扩展可能增加编译成本；这是 merge snapshot 的组成部分，不能在 gold patch 中自行省略。验证时应同时检查 CPU 构建和目标测试，不以未重建的旧 wheel 代替。
- 其他不确定点：PR 的新增兼容性测试覆盖别名和 `out`，但没有直接覆盖 Tensor 方法；完整任务包阶段应补充一个窄范围 F2P 用例，并在 base build 上确认其确实失败、修复后通过。
