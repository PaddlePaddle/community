# Task Proposal: PaddlePaddle__Paddle-74421

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74421`
- Issue 链接：无独立 issue
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74421
- PR 标题：`[API compatibility] add msort api`
- `base_commit`：`0b7e62cbfe2368b51e554cc7c0deb3bd94cad8f7`
- merged 时间：`2025-08-08`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Paddle 缺少与 `torch.msort` 对齐的公开 API，用户无法通过 `paddle.msort` 将 Tensor 沿第 0 维按升序排序。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已经合入 Paddle `develop` 的 API compatibility PR #74421。
- **代表性**：覆盖公开 Tensor 排序 API、兼容性迁移以及已有排序能力的复用边界。
- **边界清楚**：production change 仅涉及 `paddle`/`paddle.tensor` 导出和 `python/paddle/tensor/search.py` 中的新 API。
- **非平凡性**：修复不仅要提供新入口，还必须保证固定沿 axis 0 升序排序，并保持现有 `sort` contract 不变。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[api_compatibility, tensor, sorting, msort]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr74421_msort.py`
- 修复前预期：已有 `sort` dynamic-path contract 继续通过，但 `msort` API 行为测试失败，因为 Base 尚未提供该入口。
- 修复后预期：已有 `sort` regression 继续通过，`msort` 在 `paddle` 和 `paddle.tensor` 公共命名空间可用，并对多维输入执行 axis 0 升序排序且支持 `input=` 调用。
- P2P 候选：已有 `sort` 在 dynamic/PIR 路径继续把 axis、descending、stable 参数传递给底层排序并返回排序结果。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 pytest、AST overlay 和 controlled doubles 直接验证 checkout 源码中的 Python control flow。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：测试只验证公开排序行为和已有 contract，不匹配 Gold 源码字符串或局部变量名。
- 环境风险：低；无需导入历史 Paddle native runtime，也无需编译源码。
- flaky 风险：低；测试使用固定数组和 deterministic doubles，不依赖随机数或并发。
- 拆分风险：低；新 API 行为与三个 production export/implementation 文件构成单一完整能力。
