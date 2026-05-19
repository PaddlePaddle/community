# FastDeploy PaddleOCR-VL Metax 适配阶段一证据清单

| 任务名称 | FastDeploy PaddleOCR-VL Metax 适配阶段一证据清单 |
|------|------|
| 提交作者 | Dryoung |
| 提交时间 | 2026-04-23 |
| 版本号 | V1.0 |
| 文件名 | 20260423_paddleocr_vl_metax_stage1_evidence_manifest_for_fastdeploy.md |
| 说明 | 本文用于汇总阶段一主报告对应的本地证据产物路径，便于 PR 描述或评审追溯 |

## 1. 提交包内容

本文档已落位于 `community/rfcs/FastDeploy/`，作为阶段一正式报告的配套证据清单，不修改原始事实材料，仅引用已生成产物。

当前落位文件包含：

- `20260423_paddleocr_vl_metax_stage1_report_for_fastdeploy.md`
- `20260423_paddleocr_vl_metax_stage1_exec_summary_for_fastdeploy.md`
- `20260423_paddleocr_vl_metax_stage1_evidence_manifest_for_fastdeploy.md`

## 2. 当前社区仓库落位文件

- 正式报告：
  - `/data/community/rfcs/FastDeploy/20260423_paddleocr_vl_metax_stage1_report_for_fastdeploy.md`
- 一页摘要：
  - `/data/community/rfcs/FastDeploy/20260423_paddleocr_vl_metax_stage1_exec_summary_for_fastdeploy.md`
- 证据清单：
  - `/data/community/rfcs/FastDeploy/20260423_paddleocr_vl_metax_stage1_evidence_manifest_for_fastdeploy.md`

## 3. patch15 说明与 patch

- `patch15` 正式变更说明：
  - `/data/FastDeploy_release_2_4_metax/patch15_artifacts/patch15_change_note_zh.md`
- 全量统一 patch：
  - `/data/FastDeploy_release_2_4_metax/patch15_artifacts/patch15_unified.patch`
- 5 个必需文件 patch：
  - `/data/FastDeploy_release_2_4_metax/patch15_artifacts/patch15_required.patch`
- 1 个环境兜底文件 patch：
  - `/data/FastDeploy_release_2_4_metax/patch15_artifacts/patch15_env_fallback.patch`
- patch15 产物清单：
  - `/data/FastDeploy_release_2_4_metax/patch15_artifacts/patch15_manifest.txt`
- patch15 收口摘要：
  - `/data/FastDeploy_release_2_4_metax/patch15_artifacts/patch15_summary_zh.md`

## 4. 服务验证结果

- 最小服务日志：
  - `/data/FastDeploy_release_2_4_metax/run_stage_next/service.log`
- 最小服务启动命令：
  - `/data/FastDeploy_release_2_4_metax/run_stage_next/service_command.txt`
- 最小服务状态：
  - `/data/FastDeploy_release_2_4_metax/run_stage_next/service_status.json`
- 单请求 OCR 结果：
  - `/data/FastDeploy_release_2_4_metax/run_stage_next/request_result.json`
- 最小 benchmark 基线：
  - `/data/FastDeploy_release_2_4_metax/run_stage_next/benchmark.json`
- 阶段推进总结：
  - `/data/FastDeploy_release_2_4_metax/run_stage_next/stage_next_summary.md`

## 5. benchmark matrix

- 成功配置说明：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/success_config.md`
- benchmark matrix 原始结果：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/benchmark_matrix.json`
- benchmark matrix 摘要：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/benchmark_matrix_summary.md`

## 6. 正式 benchmark

- 当前正式服务状态：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/formal_service_status.json`
- 当前正式服务命令：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/formal_service_command.txt`
- 当前正式服务日志：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/formal_service.log`
- 正式 benchmark 原始结果：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/benchmark_formal.json`
- 正式 benchmark 日志：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/benchmark_formal.log`
- 正式 benchmark 摘要：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/benchmark_formal_summary.md`

## 7. 长尾复核

- 长尾复核原始结果：
  - `/data/FastDeploy_release_2_4_metax/run_stage_tailcheck/tailcheck_seq1_len1024.json`
- 长尾复核摘要：
  - `/data/FastDeploy_release_2_4_metax/run_stage_tailcheck/tailcheck_summary.md`
- 长尾复核服务状态：
  - `/data/FastDeploy_release_2_4_metax/run_stage_tailcheck/tailcheck_service_status.json`
- 长尾复核服务日志：
  - `/data/FastDeploy_release_2_4_metax/run_stage_tailcheck/tailcheck_service.log`

## 8. profiling 证据

- profiler 工具检查：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/profiler_check.md`
- 轻量 profiling 汇总：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/profiling/profiling_summary.json`
- `mx-smi` 采样结果：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/profiling/mxsmi_trace.json`
- `mx-smi` 采样日志：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/profiling/mxsmi_trace.log`
- profiling 期间请求结果：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/profiling/profiling_requests.json`
- 阶段一分析初稿：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/stage1_draft_zh.md`

## 9. py-spy / ptrace 限制说明

- `py-spy` 解锁方案：
  - `/data/FastDeploy_release_2_4_metax/run_stage_benchfinal/pyspy_unblock_plan.md`
- `py-spy` attach 报错证据：
  - `/data/FastDeploy_release_2_4_metax/run_stage_formal/profiling/pyspy_attach_error.txt`

## 10. 当前落位状态

当前阶段一材料已完成 community 仓库落位，已包含：

- 正式报告
- 一页摘要
- `patch15` 说明与补丁
- 服务跑通证据
- benchmark 小矩阵结果
- 正式 benchmark 结果
- 长尾复核结论
- profiling 轻量证据
- `py-spy/ptrace` 限制说明

说明：

- 本清单中的证据路径大多仍指向本地工作目录，用于 PR 说明、评审复核或后续附件上传。
- 当前已完成“文档落位”，仓库级提交流程状态以 `community` 仓库当前 `git status` 为准。

## 11. PR 前仍需完成的动作

1. 检查并确认 PR 目标分支与标题
   - 当前文件已经按 `rfcs/FastDeploy` 目录现有命名风格落位。
   - 开 PR 前仍需确认目标分支、PR 标题和评审人要求。

2. 审核是否需要补充 PR 描述中的证据引用
   - 当前主文档、摘要和证据清单已齐备。
   - 若评审习惯要求把关键日志、patch 或 benchmark 结果节选写入 PR 描述，还需整理到 PR 文本中。

3. 执行 git 提交流程
   - 在 `community` 仓库内完成分支创建、diff 检查、commit、push 与 PR 创建。
