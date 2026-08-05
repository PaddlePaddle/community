# Task Proposal: PaddlePaddle__Paddle-59374

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59374`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59374
- PR 标题：`【Hackathon No.7】为 Paddle 新增 apply API -part`
- `base_commit`：`4af8ecca447eba12cf57597d95935b0b5f4311b1`（#59374 合入的父提交）
- gold commit：`9fab1fe754744eaaee8c829b89bbfc9ce230ab19`
- merged 时间：`2023-12-26`
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

为 Paddle Tensor / Variable 新增 `apply` / `apply_` API，使用户能对 Tensor 元素逐元素应用自定义可调用对象，并在动态图与相关静态路径下可用、可测。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 Hackathon No.7 新 API PR（关联 RFC），不是合成任务。
- **代表性**：覆盖 Tensor 方法暴露（Python patch / pybind）、inplace 变体与单测契约，属于典型的框架 API 新增样本。
- **边界清楚**：目标是提供正确的逐元素 `apply` / `apply_` 行为，并保持既有 Tensor API 不受影响；不扩展到无关高阶变换 API。
- **非平凡性**：需要同时打通 eager / 静态相关绑定与 Python 侧方法挂载；只改测试或只加空壳接口不够。
- **区分度潜力**：漏掉 inplace、绑定路径或错误处理，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, tensor_api, apply, inplace, pybind, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_apply.py`，以及 `test/legacy_test/test_inplace.py` 中与 `apply` 相关的用例
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`apply` / `apply_` 相关 F2P 用例 fail 或 error（API 不存在或行为不正确）。
- 修复后预期：继续应用 `solution/code.patch` 并重新构建后，F2P 与相关 P2P 均通过。
- P2P 护栏：同目录中既有 inplace / Tensor 无关用例继续通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否
- patch 类型：含 Python + C++ pybind 改动，**需 source build**
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：涉及 pybind / eager 绑定，历史 commit 复现需要 source build，不能只依赖已有 wheel overlay。
- flaky 风险：主路径应为确定性小规模 Tensor 比较；verifier 应抽取稳定 nodeid。
- 拆分风险：该 PR 目标集中在 `apply` / `apply_` API 新增，适合作为一个独立样本。
