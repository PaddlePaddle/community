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

飞桨分布式训练在老通信算子退场后，其相关的单测、框架代码尚还留存，并且这些单测会因为老通信算子退场而报错。

[详细背景链接](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_8th/%E3%80%90Hackathon_8th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no3-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E8%87%AA%E5%8A%A8%E5%B9%B6%E8%A1%8C%E4%B8%8E%E9%80%9A%E4%BF%A1%E7%9B%B8%E5%85%B3%E7%9A%84%E8%80%81-ir-%E9%80%BB%E8%BE%91%E6%B8%85%E7%90%86)
## 2、功能目标

删除 Paddle 自动并行通信库中与老 IR 相关的单测，并清理与之关联的框架代码逻辑。

# 二、设计思路与实现方案

## 整体全貌 && 实施计划

`test/deprecated/auto_parallel`下报错单测:

   -    | 单测名称 | 报错原因 | 处理方法 |
        |----------|----------|----------|
        | `test_auto_parallel_amp_pass_deprecated` | ModuleNotFoundError: No module named 'test_auto_parallel_amp_pass_deprecated' | 直接删除 |
        | `test_auto_parallel_recompute_pass_deprecated` | ModuleNotFoundError: No module named 'test_auto_parallel_recompute_pass_deprecated' | 直接删除 |
        | `test_auto_parallel_sharding_pass_deprecated` | ModuleNotFoundError: No module named 'test_auto_parallel_sharding_pass_deprecated' | 直接删除 |
        | `test_auto_parallel_fp16_pass_deprecated` | ModuleNotFoundError: No module named 'test_auto_parallel_fp16_pass_deprecated' | 直接删除 |
        | `test_auto_parallel_gradient_merge_pass_deprecated` | ModuleNotFoundError: No module named 'test_auto_parallel_gradient_merge_pass_deprecated' | 直接删除 |
        | `test_auto_parallel_data_parallel_optimization_pass_deprecated` | ModuleNotFoundError: No module named 'test_auto_parallel_data_parallel_optimization_pass_deprecated' | 直接删除 |
        | `test_amp_o2_pass_deprecated` | Operator "c_allreduce_max" has not been registered. | 直接删除 |

前六个单测中，对`auto_parallel_pass_test_base_deprecated.py`的`AutoParallelPassTestBase`进行引用，并且只有这六个测试引用，因此删除。
类似，单测`test_amp_o2_pass_deprecated` 引用`amp_o2_pass.py`，因此删除。

`test/deprecated/collective/fleet`下报错单测:

   -    | 单测名称 | 报错原因 | 处理方法 |
        |----------|----------|----------|
        | `test_auto_parallel_parallelizer_deprecated` | ImportError: /home/aistudio/Paddle/build/python/paddle/base/libpaddle.so: undefined symbol: PyCMethod_New | 直接删除 |
        | `test_communicator_sync_deprecated` | Operator "send_barrier" has not been registered | 直接删除 |

`test/deprecated/legacy_test`下报错单测:

   -    | 单测名称 | 报错原因 | 处理方法 |
        |----------|----------|----------|
        | `test_auto_parallel_parallelizer_deprecated` | ImportError: /home/aistudio/Paddle/build/python/paddle/base/libpaddle.so: undefined symbol: PyCMethod_New | 直接删除 |
        | `test_auto_parallel_data_unshard_deprecated` | ImportError: /home/aistudio/Paddle/build/python/paddle/base/libpaddle.so: undefined symbol: PyCMethod_New | 直接删除 |
        | `test_auto_parallel_save_load_deprecated` | ImportError: /home/aistudio/Paddle/build/python/paddle/base/libpaddle.so: undefined symbol: PyCMethod_New | 直接删除 |


# 三、测试和验收的考量

- Paddle 自动并行通信库老 IR 相关单测删除。
- 与上述单测相关的框架代码逻辑清理。

# 四、影响面

deprecated单测清理不会影响框架总体功能跟用户使用。

# 五、排期规划

- 2025-02-17 ~ 2025-02-20：进一步确认是否有遗漏单测、框架代码，完善RFC
- 于 2025-02-27 前完成任务

