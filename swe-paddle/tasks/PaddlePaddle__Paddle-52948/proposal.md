# Task Proposal: PaddlePaddle__Paddle-52948

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-52948`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/52948 （主 PR）；https://github.com/PaddlePaddle/Paddle/pull/53572 （follow-up，一并纳入 gold）
- PR 标题：`【Hackathon No.91】` / `【Hackathon No.91】Following updates`
- `base_commit`：`cf6cbc347970a1fd2c9d76e427880139789497af`
- gold endpoint：`f3f3d57a159caf3b77f93a4d86cb233e6a1c159a`（#53572 合入后）
- merged 时间：`2023-04-27`（#52948）、`2023-05-08`（#53572）
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

在静态图与动转静（`to_static`）场景下支持 `Tensor.register_hook`，使反向 hook 能正确触发，且梯度结果与动态图一致。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 PaddlePaddle Hackathon 第四期 No.91 框架开发 PR，不是合成任务；合入前相关单测显式断言该路径不可用。
- **代表性**：覆盖静态图 Tensor hook 语义补齐，以及动转静路径下 hook 行为与动态图对齐，属于框架级 dy2static / autograd 能力。
- **边界清楚**：目标行为是静态模式与 `to_static` 下 `register_hook` 可运行，且梯度与动态图一致；不要求覆盖嵌套内部函数中的 hook 注册，也不要求实现 `hook.remove`。
- **非平凡性**：仅放开静态接口或只改测试期望不够；需要同时保证静态图运行时可挂接 hook，以及动转静后 hook 仍作用在正确对象与时机上。
- **区分度潜力**：只修静态侧、只修动转静侧，或只放宽测试，都会被完整验收用例拦住。
- **双 PR 合一**：#53572 按 review 意见完成规范化接入；本样本 gold 取两者净效果，避免停留在中间实现形态。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[dy2static, register_hook, autograd, static_graph, python_api, unittest]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/dygraph_to_static/test_tensor_hook.py`，以及 `python/paddle/fluid/tests/unittests/test_tensor_register_hook.py` 中与 static / dy2static 相关的用例
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，静态模式 / `to_static` 下的 `register_hook` 用例 fail 或 error；既有动态图 hook 行为不受影响的 P2P 用例可通过。
- 修复后预期：继续应用 `solution/code.patch` 后，F2P 与 P2P 用例均通过。
- P2P 护栏：同模块中原有动态图 `register_hook` 存量用例继续通过，避免只改目标测试却破坏既有 dygraph hook 语义。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否
- patch 类型：纯 Python（#52948 + #53572 净变更）
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：历史 commit（2023-04）对应 wheel 可能较难精确获取；优先匹配时期 wheel，否则 source build。
- flaky 风险：主要为确定性小张量比较；含随机输入的用例需在抽取稳定 nodeid 时注意 seed。
- 拆分风险：两 PR 同属 Hackathon No.91 同一目标，合并为一个样本更合理；gold 使用相对 `base_commit` 的净效果，不保留被 follow-up 替换的中间实现。
