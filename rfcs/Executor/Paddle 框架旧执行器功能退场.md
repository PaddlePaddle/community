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
目前已经移除了相关API，不再对外暴露。但在python/paddle/base/compiler.py中`ExecutionStrategy`、`BuildStrategy`仍然使用的是ParallelExecutor的执行策略和构建策略，在paddle/fluid/pybind/parallel_executor.cc中绑定，涉及到的API 有static.BuildStrategy()、static.ExecutionStrategy()。该执行器相关的python端代码暂时保留（不确定这个两个API是否考虑直接删除，根据[PR53668](https://github.com/PaddlePaddle/Paddle/pull/53668)，未来可能会重构它们去使用StandaloneExecutor)。

#### 2、C++端
ParallelExecutor底层的实现类在paddle/fluid/framework/parallel_executor.cc，而在该执行器中实际调用的也是SSAGraphExecutor，针对不同的构建策略和硬件设备使用了不同的SSAGraphExecutor（其中涉及到的派生相关执行器有：AsyncSSAGraphExecutor、ParallelSSAGraphExecutor、ThreadedSSAGraphExecutor、BindThreadedSSAGraphExecutor、FastThreadedSSAGraphExecutor），而另一个派生类执行器ScopeBufferedSSAGraphExecutor，也通过DropLocalExeScopes和NeedCreateLocalExeScope函数来控制是否使用。


### SSAGraphExecutor 执行器

SSAGraphExecutor 执行器在paddle/fluid/framework/ssagraph_executor.cc中，其派生类有：AsyncSSAGraphExecutor、BindThreadedSSAGraphExecutor、FastThreadedSSAGraphExecutor、ParallelSSAGraphExecutor、ScopeBufferedSSAGraphExecutor、ThreadedSSAGraphExecutor。它们才是ParallelExecutor 执行器的核心部分，需要与ParallelExecutor 执行器一并移除。

### AsyncExecutor 执行器

AsyncExecutor 执行器在Python端的API和单元测试已经删除，所以只需要清理C++端的残留代码。另外，其中涉及到的DataFeedDesc相关类暂时保留。


## 四、可行性分析与排期计划

Paddle 框架旧执行器功能退场可分为如下几步进行：

### 1. 移除 CINN 中旧执行器相关代码

目前 CINN 执行器在 Paddle 框架中已经完成迁移，其中涉及旧执行器相关代码全部集中在C++端可以直接退场，主要分为以下几个部分：

- 移除旧执行器分支代码
- 移除share_varinfo_into_cinn_pass
- 移除旧执行器相关单测代码
- 移除旧执行器flag：FLAGS_enable_pe_launch_cinn
- CMakeLists.txt 删除对应编译依赖

### 2. 移除旧执行器相关 Python 端代码（待细化）

- 移除旧执行器在 pybind 中的绑定，解除python端对旧执行器的依赖(此处有两个API`BuildStrategy`和`CompiledProgram`的去留问题需要研发大哥确认。如果是去除，则移除相关API的代码；如果是保留，则从旧执行器中提出相关的模块进行兼容)
- 移除旧执行器相关的python单测

|相关单元测试统计
:------
cinn_launch_context_test.cc
share_varinfo_into_cinn_pass_test.cc
test_reference_count_pass_last_lived_ops.cc
seresnext_test_base.py
test_fuse_all_reduce_pass.py
test_fuse_elewise_add_act_pass.py
test_fuse_optimizer_pass.py
test_fuse_relu_depthwise_conv_pass.py
test_ir_inplace_pass.py
test_ir_memory_optimize_pass.py
test_ir_memory_optimize_transformer.py
test_mix_precision_all_reduce_fuse.py
test_parallel_executor_run_cinn.py
test_parallel_executor_seresnext_base_cpu.py
test_parallel_executor_seresnext_base_gpu.py
test_parallel_executor_seresnext_with_fuse_all_reduce_cpu.py
test_parallel_executor_seresnext_with_fuse_all_reduce_gpu.py
test_parallel_executor_seresnext_with_reduce_cpu.py
test_parallel_executor_seresnext_with_reduce_gpu.py
test_parallel_executor_transformer_auto_growth.py
test_parallel_executor_transformer.py
test_py_func_op.py
test_standalone_executor.py

### 3. 移除旧执行器相关 C++ 端代码（待细化）

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