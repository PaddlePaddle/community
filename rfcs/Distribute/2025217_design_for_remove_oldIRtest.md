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

在`python/paddle/distributed/communication`下，各API新老IR情况:

| API名称 | 是否有新IR分支 |  是否有新IR单测 | 老IR单测 |
|----------|----------|----------|----------|
|`all_gather`| 否 |  |
|`all_to_all`| 否 |  |
|`gather`| 否 |  |
|`recv`| 否 |  |
|`reduce_scatter`| 否 |  |
|`reduce`| 否 |  |
|`scatter`| 否 |  |
|`send`| 否 |  |
|`all_reduce`| 是 | `test/collective/process_group_nccl_pir.py` | `test_collective_allreduce_api.py` |
|`broadcast`| 是 | `test/collective/process_group_nccl_pir.py` | `test_collective_broadcast_api.py` |

在`python/paddle/distributed/auto_parallel`下，:

| 函数名称：文件名（位置） | 是否有新IR分支 |  是否有新IR单测 | 额外说明 |
|----------|----------|----------|----------|
|`reshard: api.py(858 - 905)`| 是 | `test/auto_parallel/pir/test_reshard.py` |
|`shard_tensor: api.py(360 - 363)`| 是 | `test/auto_parallel/test_shard_tensor_api.py` |
|`unshard_dtensor: api.py(3218 - 3236)`| 是 | `test/auto_parallel/semi_auto_parallel_unshard_dtensor_api.py` |
|`to_static: api.py(3119 - 3151)` | 是 | | 这一部分是兼容老IR分支 |
|`get_dist_attr: static/utils.py(876 - 896)` | 是 |  |  |
|`_build_distributed_state_dict: api.py(2756 - 2760)` | 是 |  | 调用了`get_dist_attr` |
|`DistModel::state_dict: api.py(2674 - 2677)` | 是 |  |  |
|`DistModel::set_state_dict: api.py(2896 - 2899)` | 是 |  |  |

## 实施计划

# 三、测试和验收的考量

- Paddle 自动并行、通信库老 IR 及相关单测删除。
- 与上述单测相关的框架代码逻辑清理。

# 四、影响面



# 五、排期规划

- 2025-02-17 ~ 2025-02-20：进一步确认是否有遗漏单测、框架代码，完善RFC
- 于 2025-02-27 前完成任务

