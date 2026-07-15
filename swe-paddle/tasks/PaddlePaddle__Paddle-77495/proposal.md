# Task Proposal: PaddlePaddle__Paddle-77495

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-77495`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/77495
- PR 标题：`【Hackathon 10th Spring No.1】Add dilation option for MaxPool1D/2D/3D -part`
- `base_commit`：`0604f65af5397848b6803c2bf577b9b82b8d8e08`
- merged 时间：`Fri Feb 6 10:59:55 2026 +0800` -> `commit 3cc3127674c55ceb8c8d24b15b4c2e6504066d0a`
- 你的身份：原 PR 作者
- 后续联系人： Github @jinyouzhi

## 2. 问题一句话

为 `MaxPool1D/2D/3D` 及 `F.max_pool1d/2d/3d` 算子添加 `dilation` 参数，使其支持空洞池化。补齐相关 API 的语义，丰富其功能，进步增强算子库的兼容性。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 10th 框架开发 PR，不是合成任务。
- **代表性**：它覆盖了 CPU/GPU kernel 开发（包括正向反向算子）、API 语义拓展、infermeta 和 kernel 注册，以及算子分发逻辑；同时关联 PaConvert 的 PyTorch API 映射和参数顺序兼容。
- **边界清楚**：开发目标明确，是对系列算子的同一特征的增强，限于该算子内部，边界清晰，作为兼容性增强性任务通过相关转换样本库判定目标达成。
- **非平凡性**：需要同时对完成三个细节各有不同 API 的功能增强，并且要开发相关的 CPU/GPU kernel；作为一个参数功能比较复杂的 API 算子注册处理机制有一定复杂性，需要熟悉框架；算子内部的分发逻辑上需要处理好参数间的耦合性且兼容已有的逻辑。
- **区分度潜力**：除了上述困难点，潜在的分歧有：竞品 `F.max_pool*d` 和 `MaxPool*d` 参数顺序有差异需要注意；如果把 `dilation` 插入 `return_mask` 之前，会破坏已有位置参数调用，因此需要将 Paddle 的 `dilation` 放在参数列表末尾，并通过兼容装饰器识别 PyTorch 风格的位置参数。模型还需要区分 `return_mask` 与 PyTorch 的 `return_indices` 语义，以及处理 `ceil_mode` 和返回索引参数的顺序差异。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu` / `cuda`（XPU 暂不实现，底层 SDK 不支持时应保持明确报错）
- 设备范围：`cpu_only` / `single_gpu`
- 模块标签：`[python_api, api_compatibility, infermeta, operator_kernel, cuda_kernel, legacy_test]`

## 5. 验证思路

- 目标测试文件 / 命令：
    - UT：`test/legacy_test/test_imperative_layers.py`
    - UT：`test/legacy_test/test_max_pool_dilation.py`
    - UT：`test/legacy_test/test_pool_max_op.py`
    - PaConvert 的 `MaxPool` 转换样本和 API 映射测试
- 修复前预期：`base_commit` + `test_patch` 下，`MaxPool1D/2D/3D` 及 `F.max_pool1d/2d/3d` 传入 `dilation` 时应出现不支持该参数的 `TypeError` 或参数不匹配；即使直接加入参数，也会因放置在 `return_mask` 之前而破坏旧的位置参数调用。PaConvert 中使用 PyTorch 风格 `dilation` 或参数顺序的样本应转换失败或产生 mismatch。
- 修复后预期：`base_commit` + `test_patch` + `code_patch` 后，1D/2D/3D 的 CPU/GPU 正反向结果和形状计算均正确；Paddle 风格和 PyTorch 风格的 PaConvert 样本均通过。XPU 不作为本样本的实现验收范围。
- P2P 候选：`test_max_pool_dilation.py` 中 1D/2D/3D 的默认参数、不同 kernel/stride/padding/dilation 组合、`return_mask` 和 `ceil_mode` 用例，以及 `test_imperative_layers.py` 和 `test_pool_max_op.py` 中已有 `MaxPool` 用例。

## 6. 环境与资源

- 资源需求：CPU；GPU 单卡（CUDA）用于覆盖 CUDA kernel，XPU 不作为本样本的实现验收范围
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`；PaConvert 使用对应的已合入版本
- patch 类型：含 Python API、C++/infermeta、CPU kernel、CUDA kernel 及测试修改，不能只依赖已有 wheel
- 最小测试命令：`bash tests/test.sh`（完整任务包阶段补充构建命令和目标测试筛选）
- 是否有 oracle 日志：无独立归档，PR CI 结果可作为参考


## 7. 风险自查

- 泄露风险：较低；正式 `instruction.md` 只描述目标行为、兼容约束和验收标准，不直接暴露 gold patch 的文件落点。
- 环境风险：需要选择确定的基线版本和竞品代码版本，以免 API 接口变换引入变化；C++/CUDA kernel 改动不能只依赖已有 wheel，且 XPU 测试最好明确排除。
- flaky 风险：应无。
- 拆分风险：较低，应当整体开发；PaConvert #812 是配套兼容性验证，不宜脱离 Paddle API 变更单独作为本样本的核心修复。需要特别区分 Paddle#77495 的原始功能 patch 与其后续 API 兼容性修复（关联 Paddle#77789）：若完整任务要求保留旧位置参数调用，需将后续修复纳入 gold patch，或在任务边界中明确说明其不属于本样本。
