# 任务提案：PaddlePaddle__Paddle-78138

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78138`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78138
- PR 标题：`[API Compatibility] cpp sink paddle.nn.functional.pixel_shuffle -part`
- `base_commit`：`555b4a95615a35b301f348e081e56435a6d75da6`（squash 合入 commit `01b7cdd95813a88bca9569f55328c4f6f0e675cb` 的第一父提交）
- merged 时间：2026-03-11 02:13:30 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

`paddle.nn.functional.pixel_shuffle` 原有 Python wrapper 未接入统一的 C++-sink API 生成与参数别名机制，无法使用 PyTorch 风格的 `input` 关键字调用。该 PR 将公开 API 下沉到生成的 C++ binding，补充 `input` 别名、参数预处理和英文 API 文档，同时保持原有 pixel-shuffle 数值语义不变。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自真实的 Paddle API Compatibility 工作，PR #78138 已合入主仓库，目标是让现有公开 API 支持兼容参数名并统一 Python/C++ API 暴露方式。
- **代表性**：样本覆盖 Python API C++ sink、YAML/codegen 元数据、pybind 参数预处理、公开签名和文档、动态图与静态图兼容调用，代表 Paddle API 兼容改造的典型链路。
- **边界清楚**：`paddle.nn.functional.pixel_shuffle` 应继续支持位置参数和 Paddle 参数名 `x`，并为输入 Tensor 支持 `input` 别名；`x`/`input` 调用应与普通位置参数调用得到相同结果。`upscale_factor`、`data_format` 和 `name` 的现有接口保持不变，`data_format` 只接受 `NCHW` 或 `NHWC`。原有四维输入、通道重排、输出 shape、dtype、动态图、静态图和梯度语义不得回归。任务不要求修改 pixel-shuffle kernel 或增加新的设备实现。
- **非平凡性**：任务不是单纯修改 Python 函数签名。正确修复需要将旧 wrapper 切换到生成绑定，在 API 元数据中声明公开入口和参数映射，在 C++ 参数预处理阶段保留 `data_format` 校验，并保证动态图、静态图和原有 operator 测试继续通过。只在 Python 层增加一个转发别名无法完整覆盖生成签名和 C++ sink 行为。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, cpp_binding, pybind, codegen, vision, static_graph, dynamic_graph, dtype]`

## 5. 验证思路

- 目标测试文件：
  - `test/legacy_test/test_api_compatibility.py`
  - `test/legacy_test/test_pixel_shuffle_op.py`
- 目标测试命令：

  ```bash
  python -m pytest \
    test/legacy_test/test_api_compatibility.py::TestPixelShuffleAPI_Compatibility \
    test/legacy_test/test_pixel_shuffle_op.py \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 下，普通位置参数及 `x=` 调用仍可工作，但 `paddle.nn.functional.pixel_shuffle(input=..., upscale_factor=...)` 在动态图和静态图中因旧 Python 签名不接受 `input` 而失败；既有 pixel-shuffle operator 测试应保持通过。
- 修复后预期：在 `base_commit + test_patch + code_patch` 下，位置参数、`x=` 和 `input=` 三种调用在动态图及静态图中的输出完全一致；既有 operator 数值、shape、布局、dtype、梯度和异常测试继续通过。
- P2P 候选：`test/legacy_test/test_pixel_shuffle_op.py` 中已有的 CPU operator、NCHW/NHWC、不同 dtype、前向、梯度和输入约束测试可作为主要回归护栏，确认 C++ sink 后只改变公开绑定与参数兼容性，不改变底层计算结果。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无与该历史 commit 精确匹配的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，必须在源码树应用 patch 并重新构建；普通已安装 wheel 无法承载 pybind C++ 和 YAML/codegen 变更
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用预构建外部 wheel；可安装从该源码 revision 构建出的本地 wheel进行验证
- OS / Python / CUDA / cuDNN / 其他关键依赖：建议 Linux x86_64、该 commit 支持的 Python 3、CMake、兼容的 C++ 编译器、NumPy 和 pytest；目标验收使用 CPU，不依赖 CUDA/cuDNN
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti
- patch 类型：包含 C++ pybind 参数预处理、YAML/codegen、Python API 导出与文档，需要重新编译 Paddle 才能验证生成绑定
- 最小测试命令：`python -m pytest test/legacy_test/test_api_compatibility.py::TestPixelShuffleAPI_Compatibility test/legacy_test/test_pixel_shuffle_op.py -q`
- 是否有 oracle 日志：暂无固定 oracle 日志；后续按 Run/Test/Fix 顺序记录 `base_commit + test_patch` 的别名失败，以及应用 code patch、重编后目标与回归测试全部通过的结果

## 7. 风险自查

- 泄露风险：后续 `instruction.md` 只描述公开 API 的可观察参数兼容性、返回结果和不回归要求，不包含具体实现文件、内部预处理函数、YAML 映射或 diff 结构；proposal 中的实现范围仅供维护组审核
- 环境风险：任务修改生成式 C++ binding，solution patch 后必须重新运行 codegen 并编译安装 Paddle；若继续使用旧 wheel 或未更新的构建产物，`input` 别名仍可能失败
- flaky 风险：新增兼容测试使用固定 seed 和固定小 Tensor，只比较同一 API 不同调用形式的确定性输出；既有 operator 测试不依赖网络或多卡同步，预期无明显 flaky 风险
- 拆分风险：PR 主体围绕同一个 `pixel_shuffle` 公开 API 的 C++ sink、参数别名、校验和文档，行为目标不可再合理拆分。PR 中附带的无行为文档清理不属于验收目标，可在后续任务包整理时与 pixel-shuffle 行为范围明确区分
- 其他不确定点：CPU 是主验收后端；CUDA/XPU 上的底层 kernel 计算不是本 PR 的新增目标。完整任务包需要在干净的历史 base 上确认 test patch 与 code patch 的应用顺序，并验证重编后的实际公开签名
