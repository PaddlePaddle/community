# Task Proposal: PaddlePaddle__Paddle-74421

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-74421`
* Issue 链接：无独立 issue
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74421
* PR 标题：`[API compatibility] add msort api`
* `base_commit`：`0b7e62cbfe2368b51e554cc7c0deb3bd94cad8f7`
* merged 时间：`2025-08-08`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

Paddle 缺少与 `torch.msort` 对齐的公开接口，用户无法通过 `paddle.msort` 对 Tensor 沿第 0 维进行升序排序。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已经合入 Paddle `develop` 分支的 API compatibility PR #74421，不是合成任务。
* **代表性**：该任务涉及 Tensor 排序 API、公共命名空间导出，以及与其他深度学习框架的接口兼容。
* **边界清楚**：production change 集中在 `paddle`、`paddle.tensor` 的 API 导出，以及 `python/paddle/tensor/search.py` 中新增的公开接口，改动范围明确。
* **非平凡性**：任务不仅需要提供新的公开入口，还要保证接口固定沿第 0 维升序排序、支持 `input` 关键字调用，并且不影响现有 `sort` 的行为。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[api_compatibility, tensor, sorting, msort]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr74421_msort.py`
* 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，现有 `sort` 动态图路径的行为测试应继续通过；`msort` 相关测试应失败，因为 Base 中尚未提供该公开接口。
* 修复后预期：继续应用 `solution/code.patch` 后，现有 `sort` 回归测试应继续通过；`msort` 应同时在 `paddle` 和 `paddle.tensor` 公共命名空间中可用，支持通过 `input=` 调用，并能够对多维输入沿第 0 维进行升序排序。
* P2P 候选：现有 `sort` 在 dynamic/PIR 路径下仍能正确传递 `axis`、`descending` 和 `stable` 参数，并返回原有排序结果。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 环境建议：使用能够运行 `pytest` 的 Python 环境，通过 AST overlay 加载 source checkout 中的相关 Python 逻辑，并使用可控的测试替身补充运行依赖，无需编译 Paddle 源码。
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：测试仅验证公开 API 的可用性、排序语义和现有接口的回归行为，不匹配 Gold patch 中的源码文本、局部变量名或具体实现形式。
* 环境风险：低；测试无需导入历史版本的 Paddle native runtime，也无需编译源码。
* flaky 风险：低；测试使用固定输入和确定性的测试替身，不依赖随机数、并发执行或异步时序。
* 拆分风险：低；`msort` 的接口定义、公共命名空间导出和排序行为共同构成一项完整能力，适合作为一个独立样本。
