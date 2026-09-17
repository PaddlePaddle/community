# 任务提案：PaddlePaddle__Paddle-79612

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79612`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79612
- PR 标题：`[Flex_CheckPoint]: Support high-dim-reshard for flex_ckpt`
- `base_commit`：`58354a509a8d60b2cb3cdf6ead63a6c845eefd23`
- merged 时间：`2026-08-12`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：`wwaawwaaee`

## 2. 问题一句话

`flex_ckpt` 在 checkpoint reshard 时需把扁平化的半开区间重新映射回原张量的最小 N 维 slice，但 rank >= 3 且跨越高维轴边界时现有逻辑会错误收缩低阶轴，导致下游 offset、数据内容和 shape 一起带偏。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的分布式 checkpoint 修复 PR，对应 `flex_ckpt` 多卡保存与重分片加载中的真实错误。
- **代表性**：覆盖把线性索引语义与多维 shape 语义重新对齐，并喂给 broadcast、send_recv、grouped_send_recv 等数据搬运路径。
- **边界清楚**：目标是把 half open range 映射回最小 N 维 bounding slice；rank >= 3 时任一高阶轴分歧后，所有低阶轴必须扩成完整范围，即使紧邻坐标碰巧相同也不能继续收紧。1D、2D、同一前缀内及精确落在边界上的区间保持既有语义。
- **非平凡性**：难点在多维索引不变量，错误 slice 会让本 rank 读到错误数据或目标块 shape 不一致，修复者必须理解高维展开与回映射关系。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cuda`
- 设备范围：`multi_gpu`
- 模块标签：`[distributed, flex_checkpoint, reshard, indexing, shape_semantics]`

## 5. 验证思路

- 目标测试文件：`minimal_nd_slice` 所在模块的单测文件，以及两卡 save/load reshard 端到端用例。
- 修复前预期：1D、2D 或同一前缀内的简单区间可能通过，但 rank >= 3 的跨轴区间会暴露错误，表现为低阶轴未扩满、offset 被污染或 load 阶段 shape 不一致。
- 修复后预期：`minimal_nd_slice` 稳定返回最小包围语义的 N 维 slice，两卡 reshard 下 broadcast、send_recv、grouped_send_recv 均得到正确数据和 shape。
- F2P 候选：shape `(2, 10, 12)` 下 cross axis、exact boundary、last element 的确定性 helper 单测；两卡 save/load 端到端测试选一种代表性通信方式作为核心 F2P，其余通信方式作为补充 F2P / integration coverage 保留，不表述为 P2P。
- P2P 候选：现有 1D、2D range 测试；区间完全落在同一前缀内的切片测试；现有 checkpoint merge 与 save/load 测试作为回归护栏。
- 注册要求：端到端用例必须登记到 `test/auto_parallel/hybrid_strategy/testslist.csv` 及对应 `CMakeLists.txt`，确保 CI 真实收集执行。

## 6. 环境与资源

- 资源需求：Linux + 至少 2 张 GPU，用于 save/load reshard 多卡验收。
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`。
- 是否能提供 Docker：暂无公开 Docker 信息。
- Dockerfile 或镜像地址：无。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：proposal 阶段未固定，后续完整任务包需根据可用 Paddle 运行时补齐。
- OS / Python / CUDA / cuDNN / 其他关键依赖：端到端验收需要 Linux、多进程分布式运行能力和可用 CUDA 多卡环境；纯 Python helper 测试可在普通 Python 环境先隔离索引逻辑。
- patch 类型：纯 Python。
- 最小测试命令：一组确定性本地 helper 单测命令，加一组 Linux + 两卡分布式验收命令。
- 是否有 oracle 日志：无。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述高维 reshard 的可观察错误现象与期望语义，不给出逐维放宽 slice 的实现步骤。
- 环境风险：patch 虽为纯 Python，但端到端验收依赖 Linux + 两卡 CUDA 环境，本地 helper 测试不能替代分布式 acceptance evidence。
- flaky 风险：测试使用固定 shape、固定 shard 布局和可重复数据内容，避免随机数据与异步时序放大分布式噪声。
- 拆分风险：目标聚焦高维 half open range 到最小 N 维 slice 的正确映射及其 reshard 后果，无需拆分。