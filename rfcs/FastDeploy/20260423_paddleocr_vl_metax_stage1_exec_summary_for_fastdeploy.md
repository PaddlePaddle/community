# FastDeploy PaddleOCR-VL Metax 适配阶段一执行摘要

| 任务名称 | FastDeploy PaddleOCR-VL Metax 适配阶段一执行摘要 |
|------|------|
| 提交作者 | Dryoung |
| 提交时间 | 2026-04-23 |
| 版本号 | V1.0 |
| 文件名 | 20260423_paddleocr_vl_metax_stage1_exec_summary_for_fastdeploy.md |
| 说明 | 本文为阶段一主报告的 1 页摘要版本，用于评审快速阅读 |

## 1. 结论

阶段一已经达到提交基础。`patch15` 已完成 build、install、smoke 闭环验证；基于已安装包的 PaddleOCR-VL 在线服务已经跑通；单请求 OCR、benchmark 小矩阵、正式小规模 benchmark 和轻量 profiling 证据均已具备。

需要明确的是：深度 profiling 仍未完成，当前主要受 `ptrace/py-spy` 权限限制。

## 2. 环境与对象

- 环境：`Ubuntu 24.04.1 LTS`，内核 `5.15.0-58-generic`
- venv：`/data/venvs/paddleocr_vl_metax_fd24`
- FastDeploy：`2.4.0`
- Paddle：`3.4.0.dev20251223`
- 模型：`/mnt/moark-models/PaddleOCR-VL`
- 当前在线服务：`http://127.0.0.1:8288`

## 3. patch15 收口结论

`patch15` 共 6 个改动文件：

- 5 个必需文件：
  - `build.sh`
  - `custom_ops/setup_ops.py`
  - `fastdeploy/entrypoints/llm.py`
  - `fastdeploy/model_executor/layers/backends/metax/attention/flash_attn_backend.py`
  - `fastdeploy/worker/metax_model_runner.py`
- 1 个环境兜底文件：
  - `fastdeploy/worker/metax_worker.py`

这 5 个必需文件构成了 PaddleOCR-VL 在 Metax 上 build/install/smoke 成功的最小主链路；`metax_worker.py` 用于 `pymxsml` 缺失时的显存探测兜底。

## 4. 服务与功能验证

- patch15 smoke 已成功返回文本：`相思那得夢魂来。`
- 当前在线服务 `/health` 返回 `200`
- 单请求 OCR 成功：
  - 图片：`/data/ocr_demo.jpg`
  - 返回文本：`相思那得夢魂来。`
  - 首次请求耗时：`0.6402s`

结论：当前安装包已经具备稳定最小在线服务能力。

## 5. benchmark 结果

### 5.1 小矩阵

扫描范围：

- `max_num_seqs`：`1`、`2`、`4`
- `max_model_len`：`1024`、`2048`

结果：

- 6 组全部成功
- 最优组：`seqs_4_len_2048`
  - 平均时延：`0.3561s`
  - P50：`0.3403s`
  - P95：`0.4319s`
  - 吞吐：`2.8078 req/s`
- 异常组：`seqs_1_len_1024`
  - 首轮矩阵出现一次 `8s` 级 P95

### 5.2 正式 benchmark

固定最佳参数 `seqs_4_len_2048` 后得到：

- 顺序 50 次：
  - 成功率：`100%`
  - 平均时延：`0.3572s`
  - P50：`0.3475s`
  - P95：`0.3869s`
  - P99：`0.5300s`
  - 吞吐：`2.7987 req/s`
- 小并发补充组（并发 4，总请求 20）：
  - 成功率：`100%`
  - 平均时延：`0.7347s`
  - P50：`0.7380s`
  - P95：`0.7893s`
  - P99：`0.7905s`
  - 吞吐：`5.3036 req/s`

## 6. 长尾复核

针对 `seqs_1_len_1024` 进行了 20 次顺序复测：

- 平均时延：`0.3495s`
- P95：`0.3751s`
- P99：`0.5221s`
- 最慢样本：第 1 次请求 `0.5588s`
- `tail_reproduced=false`

结论：8 秒级异常没有稳定复现，更像冷态或偶发扰动，不支持认定为稳定长尾瓶颈。

## 7. profiling 现状

当前已经拿到轻量 profiling 证据：

- `mx-smi` 连续采样期间 10/10 请求成功
- profiling 期间平均请求时延：`0.3607s`
- 最大 GPU 利用率：`10%`
- 显存使用：约 `23416004 KB`

当前限制：

- `py-spy` 已安装，但 attach 在线服务失败
- 根因：`ptrace_scope=1`，且 `/proc/sys` 只读
- 因此深度 profiling 仍受 `ptrace/py-spy` 权限限制

## 8. 下一步建议

1. 保持 `seqs_4_len_2048` 作为后续默认稳态参数。
2. 优先协调 root/平台管理员解锁 `ptrace_scope`，打通 `py-spy` attach。
3. 若不能调整系统权限，则改走 `py-spy record --subprocesses` 或 `cProfile` 包裹专用 profiling 副本。
4. 在现有正式 benchmark 基础上继续扩展更长时长和更高并发档位。
