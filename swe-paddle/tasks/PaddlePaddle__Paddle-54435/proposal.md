# Task Proposal: PaddlePaddle__Paddle-54435

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-54435`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/54435
- PR 标题：`[LAUNCH] enable sort ip in launch`
- `base_commit`：`56fd25b87196b84523b3cf25cc1637d1ca1b0d75`
- merged 时间：`2023-06-08T09:21:24Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

分布式 launch 缺少按节点 IPv4 地址稳定确定节点顺序和 rank 的可选能力。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：多节点任务需要稳定、可预期的节点 rank，启动和注册顺序不应影响启用 IP 排序后的结果。
- **代表性**：任务覆盖 launch 参数、环境变量以及两种 master 同步实现之间的一致性。
- **边界清楚**：只调整未显式指定 rank 时的可选节点排序逻辑，不涉及训练计算、算子或设备执行。
- **非平凡性**：需要同时处理配置入口、IPv4 数值排序、节点列表返回值和当前节点 rank，并保持原有分支不变。
- **环境友好性**：测试使用内存 fake 模拟 HTTP 与 ETCD 客户端，不启动服务、不访问网络，也不需要 GPU。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed_launch, rendezvous, rank_assignment, configuration]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr54435_sort_ip.py`
- 修复前预期：已有 rank 参数和未启用 IP 排序时的节点顺序保持通过；新增配置入口、HTTP master IP 排序和 ETCD master IP 排序三个目标场景失败。
- 修复后预期：全部 P2P/F2P 通过，HTTP 与 ETCD 路径返回相同的数值 IP 顺序和本节点 rank。
- P2P 候选：已有 `--rank` 参数解析；关闭 IP 排序时保留按 rendezvous key 排序的行为。
- F2P 候选：`--sort_ip` 与环境变量映射；HTTP master 数值 IP 排序；ETCD master 数值 IP 排序。
- 测试来源：来源 PR 未修改单测，因此新增与 PR 行为直接对应的独立测试。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：无需编译 Paddle；测试通过 AST 执行 checkout 中的真实参数解析和 master 同步控制流，并使用内存 fake 隔离网络与 etcd 依赖。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述用户可见的排序配置、节点顺序和 rank 结果，不说明具体 helper 或代码修改位置。
- 环境风险：历史 launch 模块与当前 Paddle wheel 可能存在导入差异；测试只提取并执行目标源码节点，不覆盖已安装 Paddle 文件。
- flaky 风险：所有 peer 数据和客户端响应均为确定性的内存数据，不依赖真实网络、etcd、进程时序或随机重试。
- 拆分风险：低。两个 production 文件共同完成同一个 IP 排序配置和节点 rank 行为。
