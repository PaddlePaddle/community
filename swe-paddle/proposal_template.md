# SWE-Paddle Task Proposal 模板

> 第一阶段只需要提交本文件。proposal 通过后，再在同一目录补齐 `instruction.md`、patch 和环境说明。

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-xxxxx`
- PR 链接：
- PR 标题：
- `base_commit`：
- merged 时间：
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：GitHub ID / 其他联系方式

## 2. 问题一句话

用 1-2 句话说明这个 PR 解决了什么真实问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：这个问题来自真实用户、真实研发需求、CI 失败、API 对齐，还是其他来源？
- **代表性**：它能代表 Paddle 哪类研发能力？例如 API 语义、infermeta、kernel、autograd、dtype、静态图 / PIR、稀疏 / 复数等。
- **边界清楚**：目标行为和非目标行为是否清楚？有没有容易误修的边界？
- **非平凡性**：为什么它不是纯格式化、纯重命名或一眼能改完的机械任务？

## 4. 任务类型和标签

- 任务类型：`bug_fix` / `feature_enhancement` / `feature_implementation` / `flaky_fix` / `refactor` / `performance`
- 执行后端：`cpu` / `cuda` / `xpu` / `npu` / `rocm_dcu` / `tensorrt`
- 设备范围：`cpu_only` / `single_gpu` / `multi_gpu` / `multi_node` / `special_hardware`
- 模块标签：例如 `[python_api, infermeta, operator_kernel, sparse, complex, autograd]`

## 5. 验证思路

- 目标测试文件 / 命令：
- 修复前预期：base + test_patch 后应该 fail / error 的现象
- 修复后预期：base + test_patch + code_patch 后应该 pass 的现象
- P2P 候选：同文件或同模块中哪些存量测试可作为回归护栏

## 6. 环境与资源

- 是否能提供 Docker：有 / 无
- Dockerfile 或镜像地址：
- Paddle 来源：nightly wheel URL / 发布版 wheel / source build / 其他
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：
- OS / Python / CUDA / cuDNN / 其他关键依赖：
- 硬件：CPU / GPU 卡型、卡数、显存：
- patch 类型：纯 Python / 含 C++ / 含 CUDA / 含 kernel 或 infermeta 编译：
- 最小测试命令：
- 是否有 oracle 日志：有 / 无

说明：

- 纯 Python patch 优先提供可安装 wheel；可以优先查 Paddle nightly wheel：`https://www.paddlepaddle.org.cn/packages/nightly/`。
- 涉及 C++ / CUDA / kernel / infermeta 编译时，优先提供 Dockerfile、基础镜像和编译命令。
- 如果找不到对应 wheel 或镜像，请尽量提供完整环境配方，维护组后续尝试复现。

## 7. 风险自查

- 泄露风险：题面是否会暴露具体修改行、diff、函数名以外的答案路径？
- 环境风险：是否依赖老版本 wheel、特殊镜像、外部服务或不可固定下载？
- flaky 风险：是否需要固定 seed、多卡同步或特殊数据？
- 拆分风险：这个 PR 是否混入多个目标，是否需要拆成多个样本？
- 其他不确定点：
