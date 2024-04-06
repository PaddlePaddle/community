# Paddle 框架旧执行器功能退场

|任务名称|Paddle 框架旧执行器功能退场|
|------|------|
|提交作者|@ccsuzzh (张正海)|
|提交时间|2024-03-23|
|版本号|v0.1|
|依赖飞桨版本|develop|
|文件名| Paddle 框架旧执行器功能退场.md|

## 一、概述

### 1、相关背景

飞桨 Paddle 自 2.0 版本以来，进行了多个重大机制改造。包括：高可复用算子库 PHI、全新的动态图体系、全新的静态图执行引擎等。随着新机制的发布使用，旧机制和功能代码需要进行退场和移除，保持架构清晰和代码库的条理性，为内外部开发者提供更好的二次开发环境。这就包括了 Operators 算子库的清理、旧动态图机制代码的清理、旧静态图执行引擎的清理。

2.0 版本后静态图虽然是非默认形态，但以 Interpreter 为中心的内核执行器正式取代了旧的 ParallelExecutor 执行器，提供了更优调度策略的新执行器。因此飞桨考虑虑将此系列旧执行器进行退场处理。

### 2、意义

保持架构清晰和代码库的条理性，为内外部开发者提供更好的二次开发环境。


## 二、飞桨现状

根据飞桨代码库 `develop` 分支，目前存在如下旧执行器：ParallelExecutor、SSAGraphExecutor（其派生相关执行器：AsyncSSAGraphExecutor、BindThreadedSSAGraphExecutor、FastThreadedSSAGraphExecutor、ParallelSSAGraphExecutor、ScopeBufferedSSAGraphExecutor、ThreadedSSAGraphExecutor）、AsyncExecutor。

## 三、旧执行器遗留代码分析

### ParallelExecutor 执行器

#### 1、Python端
经过Paddle内部的python端退场之后，已经移除了部分API、依赖和单测。目前还有`static.BuildStrategy()`、`static.ExecutionStrategy()`和 `static.CompiledProgram` 与`ParallelExecutor`C++底层实现绑定，在进行C++端代码退场前必须先解除这些依赖。

##### 1.1 解除`ExecutionStrategy()`依赖
`ExecutionStrategy`在新执行器中已经被废弃可以直接移除相应代码，官方文档也需同步删除。

##### 1.2 解除`BuildStrategy()`依赖
通过对`BuildStrategy`进一步调研发现，存在许多地方直接依赖`BuildStrategy`，并不能直接将其移除。考虑删除`BuildStrategy`中无用的属性，将其与`ParallelExecutor`解耦后再迁移到别的地方进行兼容，最后官方文档中`BuildStrategy`也需同步更新。

目前`BuildStrategy`存在依赖的地方如下：
- 在`jit.to_static()`中作为参数传入，其`build_cinn_pass`属性用来控制是否使用CINN编译。
- 在分布式`DistributedStrategy`API中被用于实现一些特定功能，如`fp16_allreduce`、`fuse_all_reduce_ops`等。
- 在`static.CompiledProgram`中作为参数传入，在编译`program`或`graph`支持选择一些特殊的编译策略。

##### 1.3 解除`CompiledProgram()`依赖
在新执行器动转静的API`add_build_strategy_for`中仍然复用`CompiledProgram`接口实现，而其内部的真正实现函数`_compile_data_parallel`与ParallelExecutor直接绑定，因此需要将这个部分功能从ParallelExecutor中提取出来兼容，完成兼容后python端就不再依赖ParallelExecutor。

#### 2、C++端

##### 2.1 解除 CINN 依赖
目前 CINN 已经迁移到 Paddle 框架中并且支持了ParallelExecutor作为它的一种执行方式，所以首先需要将其从 CINN 中退场。

##### 2.2 迁移 ParallelExecutor 编译模块
由于ParallelExecutor旧执行器中compile功能并没有在框架中彻底废除，因此需要将这部分代码从提取出来进行兼容。完成完成兼容后，意味着ParallelExecutor与python端彻底解绑，不再依赖ParallelExecutor中的任何代码。

##### 2.3 实现类和组件全面退场
ParallelExecutor底层的实现类在paddle/fluid/framework/parallel_executor.cc，而在该执行器中实际调用的也是SSAGraphExecutor，针对不同的构建策略和硬件设备使用了不同的SSAGraphExecutor（其中涉及到的派生相关执行器有：AsyncSSAGraphExecutor、ParallelSSAGraphExecutor、ThreadedSSAGraphExecutor、BindThreadedSSAGraphExecutor、FastThreadedSSAGraphExecutor），而另一个派生类执行器ScopeBufferedSSAGraphExecutor，也通过DropLocalExeScopes和NeedCreateLocalExeScope函数来控制是否使用，同时相关组件也需要同步移除。


### SSAGraphExecutor 执行器

SSAGraphExecutor 执行器在paddle/fluid/framework/ssagraph_executor.cc中，其派生类有：AsyncSSAGraphExecutor、BindThreadedSSAGraphExecutor、FastThreadedSSAGraphExecutor、ParallelSSAGraphExecutor、ScopeBufferedSSAGraphExecutor、ThreadedSSAGraphExecutor。它们才是ParallelExecutor 执行器的核心部分，需要与ParallelExecutor 执行器一并移除。

### AsyncExecutor 执行器

AsyncExecutor 执行器在Python端的API和单元测试已经删除，所以只需要清理C++端的残留代码。另外，其中涉及到的DataFeedDesc相关类暂时保留。


## 四、可行性分析与排期计划

Paddle 框架旧执行器功能退场可分为如下几步进行：

### 1、移除旧执行器单元测试
首先移除旧执行器相关单元测试，方便后续进行代码退场。根据单测的实际情况来决定单测的处理方式。移除的单测文件还需要清理对应编译依赖。

参考PR：
- [remove flags_enable_parallel_graph](https://github.com/PaddlePaddle/Paddle/pull/51375)
- [remove parallel_executor related unit tests](https://github.com/PaddlePaddle/Paddle/pull/51632)

处理方式|单测文件统计
:------:|:------
部分删除|cinn_launch_context_test.cc
直接删除|share_varinfo_into_cinn_pass_test.cc
直接删除|test_reference_count_pass_last_lived_ops.cc
直接删除|seresnext_test_base.py
直接删除|test_fuse_all_reduce_pass.py
部分删除|test_fuse_elewise_add_act_pass.py
直接删除|test_fuse_optimizer_pass.py
直接删除|test_fuse_relu_depthwise_conv_pass.py
直接删除|test_ir_inplace_pass.py
直接删除|test_ir_memory_optimize_pass.py
直接删除|test_ir_memory_optimize_transformer.py
直接删除|test_mix_precision_all_reduce_fuse.py
直接删除|test_parallel_executor_run_cinn.py
直接删除|test_parallel_executor_seresnext_base_cpu.py
直接删除|test_parallel_executor_seresnext_base_gpu.py
直接删除|test_parallel_executor_seresnext_with_fuse_all_reduce_cpu.py
直接删除|test_parallel_executor_seresnext_with_fuse_all_reduce_gpu.py
直接删除|test_parallel_executor_seresnext_with_reduce_cpu.py
直接删除|test_parallel_executor_seresnext_with_reduce_gpu.py
直接删除|test_parallel_executor_transformer_auto_growth.py
直接删除|test_parallel_executor_transformer.py
部分删除|test_py_func_op.py
部分删除|test_standalone_executor.py

### 2、完成旧执行器中compile功能的迁移
为了兼容python端的`BuildStrategy`和`CompiledProgram`API，需要将其旧执行器的pybind的代码迁移到新的地方重新绑定，同时还需要提取出ParallelExecutor的comilpe功能到新的地方兼容。完成C++端代码迁移后，只需要将python端的绑定切换至新绑定的模块即可。

### 3、移除`ExecutionStrategy`
主要清理python端的`ExecutionStrategy`代码，解除与C++端依赖，官方文档也需同步删除，而C++端的实现代码后续跟随`ParallelExecutor`执行器代码一并删除。

### 4、CINN 中旧执行器代码退场
目前 CINN 在 Paddle 框架中已经完成迁移，并且兼容了旧执行器，其中涉及旧执行器相关代码全部集中在C++端可以直接退场，主要分为以下几个部分：
- 移除旧执行器分支代码
- 移除share_varinfo_into_cinn_pass
- 移除旧执行器相关单测代码(如有)
- 移除旧执行器flag`FLAGS_enable_pe_launch_cinn`
- CMakeLists.txt 删除对应编译依赖

### 5、移除旧执行器相关代码
- 移除`ParallelExecutor`执行器的`OpHandle`组件
- 移除旧执行器的底层实现类以及其派生类
- CMakeLists.txt 删除对应编译依赖

## 五、测试和验收的考量

- Paddle 框架无旧执行器关类、函数和单测
- 上下游关联模块同步删除，CMakeLists.txt 删除对应编译依赖

## 六、影响面

- 对用户的影响

  框架默认使用新的执行器，移除旧执行器不会有任何影响。

- 对 Paddle 框架开发者的影响

  架构更加清晰、代码库更加有条理性，为内外部开发者提供更好的二次开发环境。

## 参考资料

1. [飞桨静态图执行流程](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/static_graph_execution)
2. [飞桨全新执行器升级](https://www.paddlepaddle.org.cn/documentation/docs/zh/release_note_cn.html#jingtaituxinzhixingqiquanmianshangxian)
3. [Paddle 训练框架应用 CINN 进行编译优化加速](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/cinn/paddle2cinn_intro_cn.md)
4. [remove execution_strategy doc](https://github.com/PaddlePaddle/Paddle/pull/53668)
5. [Dy2Static support new_executor](https://github.com/PaddlePaddle/Paddle/pull/44450)