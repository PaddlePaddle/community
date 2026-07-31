# Task Proposal: PaddlePaddle__Paddle-79275

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-79275`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79275
* PR 标题：`[API Compatibility] Align torch.nn.attention.flex_attention.or_masks/and_masks`
* `base_commit`：`1d14ac949cd00747df9c828537f5fbff51b1f85f`
* merged 时间：`2026-06-12`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为 flex attention 新增 `or_masks` 和 `and_masks` 公共 API，使调用方能够组合一个或多个 mask 函数，并正确处理单个 mask、空输入和非法输入。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：任务来自已合入的 Paddle PR #79275，不是人工构造的需求。
* **代表性**：该任务涉及 attention 模块的公共 API、mask 函数组合和 PyTorch 接口兼容，可补充 SWE-Paddle 中以 bug fix 为主的任务类型。
* **边界清楚**：Gold production change 集中在 attention 包的模块导出和新增的 `flex_attention.py`，功能范围明确，相关行为可以通过独立测试直接验证。
* **非平凡性**：任务需要正确实现多个 mask 的 OR/AND 组合，保持四个调用参数不变，并处理单个 mask、空输入和非法输入等边界情况。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[attention, flex_attention, public_api, api_compatibility, mask, python_only]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr79275_flex_attention_masks.py`
* 修复前预期：`paddle.nn.attention` 现有公共接口的 P2P 测试应通过；`or_masks` 和 `and_masks` 的组合行为、空输入行为及非法输入处理测试应因目标 API 尚不存在而失败。
* 修复后预期：继续应用 production-only `solution/code.patch` 后，P2P 与全部 F2P 均应通过；新增 API 应能够正确组合 mask，并保持参数传递和边界输入行为符合预期。
* P2P 候选：`paddle.nn.attention` 中已有的 `SDPBackend`、`sdpa_kernel` 及相关公共导入行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述公共 API 的组合结果、参数约定和边界行为，不透露 Gold patch 的结果初始化方式、循环结构或具体实现步骤。
* 环境风险：测试不导入历史 source checkout 的完整 Paddle package，也不要求 native extension 与源码版本完全匹配。
* flaky 风险：测试使用固定的 mask 输入、Tensor 测试替身和确定性的布尔结果，不依赖随机数、GPU、并发或异步时序。
* 拆分风险：`or_masks` 和 `and_masks` 属于同一 flex attention mask 组合能力，且由同一新增模块提供，适合作为一个完整任务。
