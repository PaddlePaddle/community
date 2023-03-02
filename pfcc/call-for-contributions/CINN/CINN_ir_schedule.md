# CINN调度原语开发

> This project will be mentored by [@zhhsplendid](https://github.com/zhhsplendid) and [@CtfGo](https://github.com/CtfGo)


### 背景

[CINN](https://github.com/PaddlePaddle/CINN)是一种在不改变模型代码的条件下加速飞桨模型运行速度的深度学习编译器。CINN致力于创造训推一体自动调优、分布式编译加速等特色功能，对深度学习模型提供全自动、极致的性能优化，并在科研界和工业界建立影响力。

深度学习框架的算子由具备领域特定专业知识的高性能专家实现，而深度学习编译器将深度学习计算图转换为内部的IR，该IR可以对深度学习计算图中的算子计算进行表达和概括，接着自动生成特定硬件可运行的指令或代码。为了生成高性能的代码，需要对IR进行调度变换，如：计算分块、读写数据缓存、线程绑定等，这些操作需要通过调度原语实现。此外，自动调优技术依赖调度原语的健全和鲁棒性，可以说调度原语是优化神经网络计算的关键。

### 主要工作

经过调研，初步筛选出5个调度原语，由易至难分为：Unannotate、GetChildBlocks、SampleCategorical、SamplePerfectTile、ReverseComputeInline。

若想了解CINN IR，可参考[CINN IR](https://github.com/PaddlePaddle/CINN/pull/775/files);

调度原语开发示例情况可参考[CINN ir_schedule](https://github.com/PaddlePaddle/CINN/blob/develop/cinn/ir/ir_schedule.cc);

若想详细了解CINN设计思路及算子开发介绍，可观看视频课程：[深度学习编译器算子应用与开发介绍](https://aistudio.baidu.com/aistudio/education/lessonvideo/3186683)。

### 原语描述

#### **1.Unannotate**

**描述:**  按照键ann_key取消一个Block或Loop内的Annotation

```
// before Unannotate:
ScheduleBlock(root)
{
  attrs(auto_unroll_max_step:32)
  {
    serial for (i, 0, 32)
    {
      ScheduleBlock(B)
      {
        i0 = axis.bind(i)
        B[i0] = (1 + A[i0])
      }
    }
  }
}

// after Unannotate: ir_schedule.Unannotate(node, ir::attr::auto_unroll_max_step)
ScheduleBlock(root)
{
  {
    serial for (i, 0, 32)
    {
      ScheduleBlock(B)
      {
        i0 = axis.bind(i)
        B[i0] = (1 + A[i0])
      }
    }
  }
}
```

**难度：** 简单

**接口：** 

| 类别   | 类型   | 实际节点类型             | 名称        | 描述                        |
| ------ | ------ | ------------------------ | ----------- | --------------------------- |
| 参数   | Expr   | ScheduleBlockRealize/For | source_node | 要取消Annotate的Block或Loop |
| 参数   | string | string                   | ann_key     | 要取消的Annotate key        |
| 返回值 | void   |                          |             |                             |

 

#### **2.GetChildBlocks**

**描述：** 获取某ScheduleBlock或Loop下的所有子Block

```
// before GetChildBlocks:
ScheduleBlock(root)
{
  {
    serial for (i, 0, 32)
    {
      ScheduleBlock(B)
      {
        i0 = axis.bind(i)
        B[i0] = (1 + A[i0])
      }
    }
  }
}

// auto child_blocks = ir_schedule.GetChildBlocks(root_block)
// child_blocks[0]:
ScheduleBlock(B)
{
  i0 = axis.bind(i)
  B[i0] = (1 + A[i0])
}
```

**难度：** 简单

**接口：**

| 类别   | 类型         | 实际节点类型             | 名称        | 描述                           |
| ------ | ------------ | ------------------------ | ----------- | ------------------------------ |
| 参数   | Expr         | ScheduleBlockRealize/For | source_node | 外层Block或Loop                |
| 返回值 | vector<Expr> | ScheduleBlockRealize     |             | 所有子ScheduleBlockRealize节点 |

#### **3.SampleCategorical**

**描述:  按给定分布随机采样一个整数**

```
std::vector<int> candidates = {1, 2, 3};
std::vecror<float> probs = {1.0, 2.0, 3.0};
Expr result;
for (int i = 0; i < 6; ++i) {
  result = ir_schedule.SampleCategorical(candidates, probs);
} 
// possible result: 1 3 2 3 3 2
```

**难度：** 简单

**接口：**

| 类别   | 类型         | 实际节点类型 | 名称       | 描述                 |
| ------ | ------------ | ------------ | ---------- | -------------------- |
| 参数   | vector<int>  | int          | candidates | 候选整数集           |
| 参数   | vector<float> | float       | probs      | 候选整数集的概率分布   |
| 返回值 | int          |              |            | 采样到的随机变量      |



#### **4.SamplePerfectTile**

**描述:**  为将要进行分裂的循环随机采样循环次数

```
// The loop to be split
serial for (i, 0, 1024)
{
    ...
}

// auto result = ir_schedule.SamplePerfectTile(for_node, 2, 64)
// result: [32, 32]  or  [64, 16] ...

// auto result = ir_schedule.SamplePerfectTile(for_node, 3, 64)
// result: [8, 8, 16]  or  [4, 16, 16] ...
```

**难度：** 中等

**接口：**

| 类别   | 类型         | 实际节点类型 | 名称                 | 描述                           |
| ------ | ------------ | ------------ | -------------------- | ------------------------------ |
| 参数   | Expr         | For          | loop                 | 要进行分裂的Loop               |
| 参数   | int          | int          | n                    | 要分裂的循环层数               |
| 参数   | int          | int          | max_innermost_factor | 最内层循环的次数限制           |
| 返回值 | vector<int>  | int           |                      | 采样到的tile size           |

#### **5.ReverseComputeInline**

**描述：** 将一个block中tensor的计算内联到其生产者中。

```
// before ReverseComputeInline:
{
  serial for (i, 0, 32)
  {
    serial for (j, 0, 64)
    {
      ScheduleBlock(B)
      {
        i0, i1 = axis.bind(i, j)
        B[i0, i1] = (1 + A[i0, i1])
      }
    }
  }
  serial for (i, 0, 64)
  {
    serial for (j, 0, 32)
    {
      ScheduleBlock(C)
      {
        i0, i1 = axis.bind(i, j)
        C[i0, i1] = (2 * B[i1, i0])
      }
    }
  }
}

// after ReverseComputeInline:  ir_schedule.ReverseComputeInline(ir_schedule.GetBlock("C"))
{
  serial for (i, 0, 32)
  {
    serial for (j, 0, 64)
    {
      ScheduleBlock(B)
      {
        i0, i1 = axis.bind(i, j)
        C[i1, i0] = (2 * (1 + A[i0, i1]))
      }
    }
  }
}
```

**难度：** 中等

**接口：**

| 类别   | 类型 | 实际节点类型         | 名称           | 描述                |
| ------ | ---- | -------------------- | -------------- | ------------------- |
| 参数   | Expr | ScheduleBlockRealize | schedule_block | 需要被Inline的block |
| 返回值 | void |                      |                |                     |
