# Task Proposal: PaddlePaddle__Paddle-57741

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-57741`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/57741
- PR 标题：`【PIR】add memcpy in PIR`
- `base_commit`：`f984ed1a56960aeee0059c67b965406984565356`（#57741 合入的父提交）
- gold commit：`4288e25e07895e2fd9985b7a2ec94baedac39159`
- merged 时间：`2023-10-23`
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

在 PIR 路径下补齐 `memcpy` 算子支持，使动转静场景下 Tensor 在设备间拷贝相关用例可在 PIR 下正确运行。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 PIR 算子迁移 PR，不是合成任务。
- **代表性**：覆盖 PIR op YAML、compat、kernel pass 与动转静 memcpy 测试，属于基础算子迁 PIR 样本。
- **边界清楚**：目标是 `memcpy` 在 PIR 下可用；不扩展到其他设备管理 / 拷贝 API。
- **非平凡性**：需要同时处理算子声明与 PIR 到 kernel 的 lowering；只改测试白名单不够。
- **区分度潜力**：漏掉 pass / YAML / 测试开启任一侧，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pir, memcpy, dy2static, yaml_config, kernel_pass]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/dygraph_to_static/test_tensor_memcpy_on_cpu.py`（及同主题 GPU 用例若环境允许）
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，PIR 下 memcpy 相关 F2P 用例 fail 或 error。
- 修复后预期：继续应用 `solution/code.patch` 并重新构建后，F2P 与相关 P2P 均通过。
- P2P 护栏：同目录中既有无关动转静用例继续通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否（CPU 用例足够作为主验证；PR 亦触及 GPU 测试文件）
- patch 类型：含 C++ / YAML / pass，**需 source build**
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：PIR + pass 改动需要 source build；GPU memcpy 用例可作为可选补充。
- flaky 风险：低；设备拷贝路径需固定 place。
- 拆分风险：该 PR 目标集中在 `memcpy` 的 PIR 支持，适合作为一个独立样本。
