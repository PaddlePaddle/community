# 分布式训练自动并行与通信相关的老 IR 逻辑清理设计文档

|任务名称 | 分布式训练自动并行与通信相关的老 IR 逻辑清理                     | 
|---|-------------------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | aquagull                                                   | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2025-02-17                                            | 
|版本号 | 1.0                                                   | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                | 
|文件名 | 2025217_design_for_remove_oldIRtest.md | 

# 一、概述
## 1、相关背景

Paddle 自动并行、通信库相关代码中，同时存在新老IR代码，老IR将逐渐不维护，因此需要清理掉老IR代码逻辑，同时清理相关单测。

[详细背景链接](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_8th/%E3%80%90Hackathon_8th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no3-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E8%87%AA%E5%8A%A8%E5%B9%B6%E8%A1%8C%E4%B8%8E%E9%80%9A%E4%BF%A1%E7%9B%B8%E5%85%B3%E7%9A%84%E8%80%81-ir-%E9%80%BB%E8%BE%91%E6%B8%85%E7%90%86)

## 2、功能目标

删除 Paddle 自动并行、通信库中与老 IR 逻辑，以及相关的单测，并清理与之关联的框架代码。

# 二、设计思路与实现方案

## 整体全貌 

### 1.1 Communtication (python/paddle/distributed/communication)

| API 名称         | 新 IR 分支 | 新 IR 单测                          |
|------------------|-----------|------------------------------------|
| all_gather       | 否        |                                    |
| all_to_all       | 否        |                                    |
| gather           | 否        |                                    |
| recv             | 否        |                                    |
| reduce_scatter   | 否        |                                    |
| reduce           | 否        |                                    |
| scatter          | 否        |                                    |
| send             | 否        |                                    |
| all_reduce       | 是        | test/collective/process_group_nccl_pir.py |
| broadcast        | 是        | test/collective/process_group_nccl_pir.py |

### 1.2 Auto_Parallel (python/paddle/distributed/auto_parallel)

| 模块/方法                      | 文件:行号              | 新 IR 单测                                      | 备注                     |
|-------------------------------|-----------------------|------------------------------------------------|--------------------------|
| reshard                       | api.py:858            | test/auto_parallel/pir/test_reshard.py         |                          |
| shard_tensor                  | api.py:360            | test/auto_parallel/test_shard_tensor_api.py    |                          |
| unshard_dtensor               | api.py:3218           | test/auto_parallel/semi_auto_parallel_unshard_dtensor_api.py |              |
| to_static                     | api.py:3119           | -                                              | 兼容老 IR 分支            |
| _build_distributed_state_dict | api.py:2756           | -                                              | 调用 get_dist_attr        |
| DistModel::state_dict         | api.py:2674           | -                                              |                          |
| DistModel::set_state_dict     | api.py:2896           | -                                              |                          |
| get_dist_attr                 | static/utils.py:876   | -                                              |                          |

#### 1.2.1 Engine (static/engine.py)

| 模块/方法                   | 行号   | 备注 |
|:--------------------------|-------:|:----|
| \_\_init\_\_               | 305    |      |
| _prepare_fetch             | 572    |      |
| _prepare_program           | 1017   |      |
| _build                     | 1153   |      |
| _init_comm                 | 1323   |      |
| _initialize                | 1369   |      |
| run                        | 2095   |      |
| get_dist_main_program...   | 2599   |      |

### 1.3 Pass (python/paddle/distributed/passes)

#### 1.3.1 pass_utils.py

| 函数/方法                        | 行号  | 新 IR 分支/说明               | 备注                                      |
|---------------------------------|-------:|------------------------------|-------------------------------------------|
| set_skip_gc_vars                | 277    | 是                           |                                           |
| _set_skip_gc_vars_in_old_ir     | 283    | _set_skip_gc_vars_in_pir     |                                           |
| shadow_var_between_sub_programs | 370    | 仅被老 IR 调用                |                                           |
| _overlap_send_recv              | 645    | _pir_overlap_send_recv       |                                           |
| _get_backward_op_type           | 1450   | _pir_get_backward_op_type     |                                           |
| _program_for_vpp                | 1207   | _pir_program_for_vpp          |                                           |
| split_matmul_grad_to_matmul     | 1782   | _pir_split_matmul_grad_to_matmul |                                        |
| _program_for_fthenb_and_1f1b    | 679    | 仅被老 IR 调用                | 在 pipeline_eager_1f1b 中被调用，保留 |

#### 1.3.2 Scheduler (pipeline_scheduler_pass)

##### (1) pipeline_1f1b.py

| 方法路径                    | 行号 | 新 IR 分支/说明            | 备注                         |
|----------------------------|------|----------------------------|-----------------------------|
| _create_job_list           | 183  | _create_job_list_in_pir    |                             |
| _partial_programs          | 375  | _partial_pir_programs      | 调用 _program_for_fthenb_and_1f1b    |
| _backward_forward_overlap  | 72   | 仅被老 IR 调用              | PIR 不支持 1F1B 重叠通信     |

对应单测: test/auto_parallel/pir/test_pipeline_scheduler_1f1b_pir.py

##### (2) pipeline_fthenb.py

| 方法路径           | 行号 | 新 IR 分支/说明          | 备注                         |
|--------------------|------|-------------------------|-----------------------------|
| _partial_programs | 59   | _partial_pir_programs   | 调用 _program_for_fthenb_and_1f1b      |

##### (3) pipeline_pass_base.py

| 方法路径             | 行号 | 新 IR 分支/说明          | 备注                          |
|--------------------|-----|-------------------------|------------------------------|
| _apply_impl        | 69  | 是                      |                               |
| _apply_single_impl | 847 | _apply_pir_single_impl  |  `auto_parallel_replace_with_parallel_cross_entropy.py`仅实现这个接口，未适配PIR，保留 |
| _partial_programs  | 49  | _partial_pir_programs   |  `pipeline_eager_1f1b.py`中仅实现了这个接口，未适配PIR，保留   |

将不清除`pipeline_pass_base.py`中的老IR逻辑，原因是：
- 目前仍有个别的pipeline未适配PIR，为了兼容老IR，因此选择保留。

##### (4) pipeline_vpp.py

| 方法路径                          | 行号 | 新 IR 分支/说明                | 备注                  |
|----------------------------------|------|-------------------------------|----------------------|
| _split_matmul_grad_ops_to_matmul | 174  | _pir_split_matmul_grad_ops_to_matmul |                   |
| _partial_programs                | 305  | _partial_pir_programs         | PIR 不支持 VPP 重叠通信 |

对应单测: test/auto_parallel/pir/test_pipeline_scheduler_vpp_pir.py

#### 1.3.3 auto_parallel_gradient_merge.py

| 方法路径                             | 行号 | 新 IR 分支/说明                | 备注  |
|-------------------------------------|------|-------------------------------|-------|
| _append_gradient_merge_backward_op | 161  | _pir_append_gradient_merge_backward_op |     |
| _remove_cast_for_master_grad        | 472  | _pir_remove_cast_for_master_grad |       |
| parse_program                       | 648  | _pir_parse_program            |       |
| _apply_single_impl                  | 847  | 是                            |       |

#### 1.3.4 auto_parallel_recompute.py

虽然有新IR的recomputePass： `auto_parallel_recompute_pir.py`，但老的仍然被使用，如`optimization_tuner.py`、`parallelizer.py`、`parallelizer_v2.py`，保留。

## 实施计划

因通信库下只有`all_reduce`、`broadcast`API适配了新IR，经讨论，现暂时保存老IR及单测。

为确保出现错误的时候方便定位问题，拟拆分成一下几个PR进行删除：
1.删除api.py下的老IR逻辑。
2.删除engine相关的的老IR逻辑。
3.删除pipeline相关的老IR逻辑
4.删除剩余的老IR逻辑

# 三、测试和验收的考量

- Paddle 自动并行、通信库老 IR 及相关单测删除。
- 与上述单测相关的框架代码逻辑清理。

# 四、影响面

仅删除已适配新IR逻辑的老IR代码，不会产生影响。

# 五、排期规划

- 2025-02-17 ~ 2025-02-23：进一步确认是否有遗漏单测、框架代码，完善RFC
- 于 2025-03-01 前完成任务

