# 任务提案：PaddlePaddle__Paddle-78082

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78082`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78082
- PR 标题：`[API Compatibility] add method pop(), values() and keys() to paddle.nn.ParameterDict`
- `base_commit`：`ae907b878e91dbabf3582da99f8b05a46b588fc2`（squash 合入 commit `a2e4e5062dacbfef63cf4b08981b74b72ad21214` 的第一父提交）
- merged 时间：2026-03-09 13:48:16 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

`paddle.nn.ParameterDict` 原有接口支持按键访问、迭代和更新参数，但缺少 PyTorch `ParameterDict` 已提供的 `pop()`、`keys()` 和 `values()` 容器操作。该 PR 补齐这些公开方法，使用户可以按插入顺序查看键和值、移除并取得指定 Parameter，同时保持参数注册、训练和序列化行为不变。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自已合入的 Paddle API Compatibility 工作，解决用户在 Paddle 与 PyTorch 间迁移 Parameter 容器代码时缺少常用映射接口的真实兼容性问题。
- **代表性**：样本覆盖 Python API 语义、`Layer` 参数注册容器、插入顺序、Parameter 身份、动态修改、前向 / 反向计算以及 `state_dict` 序列化，是神经网络容器兼容改造的典型任务。
- **边界清楚**：`pop(key)` 应返回并移除对应 Parameter，缺失键继续抛出 `KeyError`；`keys()` 应按容器顺序返回现有键；`values()` 应按相同顺序返回实际注册的 Parameter。方法执行后 `len()`、迭代和后续更新应反映当前内容。现有初始化、索引、update、梯度、forward 和 state-dict roundtrip 行为不得改变；任务不增加 `dict.pop` 的 default 参数，也不改变 ParameterDict 的键和值类型约束。
- **非平凡性**：`ParameterDict` 不是普通 Python dict，而是参与 `Layer` 参数发现、属性注册、梯度传播和序列化的容器。实现需要在暴露熟悉的映射操作时维护底层参数注册表、顺序和 Parameter 对象身份，不能只维护一份与 Layer 状态脱节的辅助字典。
- **范围单一**：squash commit 仅修改 ParameterDict 实现及其对应测试文件，全部 129 行变更都服务于同一个容器 API 兼容目标，没有需要剔除的独立功能或清理 hunk。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, nn_layer, container, parameter_management, state_dict]`

## 5. 验证思路

- 目标测试文件 / 命令：

  ```bash
  python -m pytest \
    test/legacy_test/test_imperative_container_parameterdict.py::TestParameterDictPopKeysValues \
    -q
  ```

- 修复前预期：在 `base_commit + test_patch` 上，`ParameterDict` 不存在 `pop()`、`keys()` 和 `values()`；九个目标用例在调用这些方法时因 `AttributeError` 失败。原有索引、迭代、update、forward / backward 和 state-dict 测试仍应通过。
- 修复后预期：在 `base_commit + test_patch + code_patch` 上，目标类全部通过。`pop()` 返回正确 Parameter、缩短容器并在缺失键时抛出 `KeyError`；连续 pop 可清空容器；`keys()` 保持初始化及 update 后的顺序；`values()` 返回对应 Parameter、shape 和数量，并在 pop 后同步更新。
- P2P 候选：同文件新增的 `TestParameterDictStateDictRoundtrip` 在修复前后均应通过，保护 state-dict 键和值、加载后的前向输出一致性；原有 `TestParameterDictInit`、`TestParameterDictAccess`、`TestParameterDictUpdate`、`TestParameterDictRegistration` 和 `TestParameterDictForwardBackward` 可作为完整模块内的存量回归护栏。

## 6. 环境与资源

- 是否能提供 Docker：无；proposal 阶段暂无与该历史 commit 精确匹配的固定镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，配合与该 revision 兼容的本地 Paddle 包；patch 为纯 Python，可优先使用精确 base-compatible wheel 加源码 overlay，否则使用 base revision 的本地构建产物
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：暂未固定精确 wheel URL；完整任务阶段优先查找对应日期的 Linux x86_64 CPU nightly wheel，不使用无法确认 commit / ABI 兼容性的任意新版 wheel
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti
- patch 类型：纯 Python，不含 C++、CUDA、kernel 或 infermeta 编译改动
- 最小测试命令：`python -m pytest test/legacy_test/test_imperative_container_parameterdict.py::TestParameterDictPopKeysValues -q`
- 是否有 oracle 日志：有；合入 PR 的 CI 提供修复后测试记录，完整任务阶段可补充精确 base 环境的 fail-before / pass-after 本地日志

## 7. 风险自查

- **泄露风险**：公开方法名来自任务本身、不可避免；后续 `instruction.md` 只描述容器的可观察行为和不变式，不应暴露内部参数表、上下文管理方式、具体实现行或 gold diff。
- **环境风险**：patch 本身为纯 Python，但测试依赖与历史源码 revision 匹配的 Paddle compiled core。完整任务必须记录 `paddle.__file__`、版本和 commit，并固定 wheel 或本地构建来源，避免新版安装包掩盖 base 行为。
- **测试风险**：三个方法的实现代码较紧凑，若测试只检查属性存在会使任务过于机械。目标测试必须保留返回对象、缺失键错误、顺序、长度、update / pop 后同步和 Parameter 类型检查，并配合 state-dict、梯度与前向 P2P 测试验证容器集成语义。
- **范围风险**：任务只针对 `paddle.nn.ParameterDict`，不扩展到 `LayerDict`、`ParameterList` 或普通 dict 的完整接口，也不要求新增 `clear()`、`items()`、`setdefault()` 或带 default 的 `pop()` 重载。
- **版本风险**：不同 Python / PyTorch 版本对映射 view 与具体返回容器类型的细节可能不同；验收应以该 PR 的明确可观察契约为准，重点验证顺序、Parameter 身份和修改后的容器状态，不额外引入未来版本语义。
