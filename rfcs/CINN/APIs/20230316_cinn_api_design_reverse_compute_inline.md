# CINN ReverseComputeInline 设计文档
|API名称 | ReverseComputeInline | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | zrr1999 |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-16 |
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230226_cinn_api_design_reverse_compute_inline.md<br> |


# 一、概述

## 1、相关背景
CINN是一种在不改变模型代码的条件下加速飞桨模型运行速度的深度学习编译器。
在对接上层框架时，编译器会将上层的框架算子进一步拆分为若干基础算子，这样做的目的一方面是为了减少算子开发的工作量，
仅实现有限的基础算子便可以组合出大量的上层框架算子；
另一方面便于算子融合技术在编译器中可以实现跨算子自动融合，减少最终执行时的kernel数目和访存开销，达到更好的性能。

Schedule 原语是 CINN 编译器优化算子计算实现的接口，目前已经实现了Split、Fuse、Reorder等常用原语，
其中 ComputeInline 原语操作是将一个 tensor 的计算过程内联到其消费者中，
而 ReverseComputeInline 原语操作与其相反，是将一个 tensor 的计算过程内联到其生产者中。

## 2、名词解释
NCHW ：一种图的数据格式。N 指 Batch，C 指 Channel，H 指 Height，W 指 width。

## 3、功能目标
参考已有的 ComputeInline 操作和 CINN 调度原语开发说明文档，添加 ReverseComputeInline 原语，实现将一个 tensor 的计算内联到其生产者中。

## 4、意义
添加 ReverseComputeInline 原语，实现将一个 tensor 的计算内联到其生产者中。

# 二、CINN现状
CINN框架暂不支持 `ReverseComputeInline` 原语，需要实现。

# 三、业内方案调研
**TVM 的 `ReverseComputeInline` 原语**

在 TVM 中，核心代码见[ReverseComputeInline的核心代码](https://github.com/apache/tvm/blob/422ca2855a74bf0d0d88f1aa66343015f4326ac1/src/tir/schedule/primitive/compute_inline.cc)，
其主要步骤如下：
1. 获取需要内联的块所在的作用域块。
2. 检查需要内联的块是否完整。
3. 检查需要内联的块是否只有一个完整的生产者，并且生产者不是输出块。
4. 分析块的 body，在分析过程中会进行模式匹配，判断是否允许内联。
5. 创建一个计划，将需要内联的块从 AST 中移除。
6. 创建一个新的 AST，将内联后的块插入到相应的位置，并更新其他块的引用。
7. 在 AST 中进行实际的变更。
8. 更新一些缓存的标志信息。


# 四、对比分析
TVM 的 `ReverseComputeInline` 原语实现比较清晰，可作为参考。
本次任务计划参考已有的 ComputeInline 操作、TVM 的 `ReverseComputeInline` 原语实现以及 CINN 调度原语开发说明文档，实现 ReverseComputeInline。

# 五、设计思路与实现方案

## 原语API设计
在 `cinn/ir/ir_schedule.h` 中新增 `ReverseComputeInline` 原语。
```c++
  /**
   * \brief Mark an schedule block as inlined.
   * @param schedule_block the schedule block to be inlined.
   */
  void ReverseComputeInline(const Expr& schedule_block);
```

## API实现方案
ReverseComputeInline 原语：分别添加接口及实现至 cinn/ir/ir_schedule.h、cinn/ir/ir_schedule.cc
支持新增原语 Trace 重放：在 cinn/ir/schedule_desc.cc 中使用CINN_BUILD_STEP_KIND 注册 ReverseComputeInline 原语的重放函数
使用类python的伪代码实现：
```python
def reverse_compute_inline(schedule_block):
    #1. 获取 scope block
    scope_root = get_scope_root(schedule_block, True)

    #2. 检查完整性
    if not is_complete(schedule_block, scope_root):
        raise ValueError("Block is not complete")

    #3. 检查消费者是否只有一个完整的生产者且生产者不是输出块
    producer_block = get_single_producer(schedule_block, scope_root)
    if producer_block is None:
        raise ValueError("Consumer has no single complete producer")
    if is_output_block(producer_block, scope_root):
        raise ValueError("Producer is an output block")

    #4. 分析块体
    inliner = ReverseComputeInliner(schedule_block, scope_root)
    if not inliner.body_pattern_allow_inline():
        raise ValueError("Block body pattern does not allow inline")

    #5. 创建删除叶块的计划
    inliner.create_leaf_block_removal_plan()

    #6. 创建新的
    tgt = inliner.get_tgt_stmt()
```

# 六、测试和验收的考量。
ReverseComputeInline 原语单测添加至 cinn/backends/ir_schedule_test.cc
新增原语 Trace 重放单测添加至 cinn/ir/schedule_desc_test.cc

# 七、可行性分析和排期规划
- 可行性分析

CINN中已经实现了许多其他原语，在现有的框架基础上能够很好地添加其他原语功能。

- 排期规划

3月28日 ~ 4月10日完成基本开发。

4月10日 ~ 4月15日完成调试和测试代码的编写。

# 八、影响面
本次任务影响模块如下，

`cinn\backends`，`cinn\ir`，`cinn\hlir`。

均是在原模块内增加代码，不影响原模块的已有功能。

# 附件及参考资料
1. [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
2. [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
3. [CINN调度原语开发](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/CINN/CINN_ir_schedule.md) 
