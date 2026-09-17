# Paddle自动调优器设计实现

| 任务名称 | 自动调优器                               |
| -------- | ---------------------------------------- |
| 提交作者 | chenxianjie                              |
| 提交时间 | 2024-06-03                               |
| 版本号   | V0.1                                     |
| 飞桨版本 | develop                                  |
| 文件名   | 20250602_cinn_autotuner_design_implement |

# 一、概述

随着深度学习模型的多样化与硬件平台的不断创新，现有深度学习框架在支持跨平台、高效执行方面面临挑战。模型部署面临硬件适配困难、算子优化效率低下、缺乏自动化调优机制的问题。为了避免面向硬件编程的复杂性，自动调优算法在深度学习编译器的基础上，通过搜索调优参数加速原有算子。

# 二、飞桨现状
Paddle 目前采用启发式方法根据经验从输入数据形状计算得到具体的各硬件参数


# 三、业内方案调研
AutoTVM通过引入Schedule 模板概念，结合了内存访问、线程模式和硬件原语，建立搜索空间，保证可能的人工工程优化全部包含在这个搜索空间里面。TVM通过随机顺序、网格搜索、遗传算法或	机器学习算法快速搜索这个搜索空间，生成可部署代码。其中机器学习算法旨在通过来学习程序空间的代价估价函数。利用可迁移的模型学习已经看到过的算子优化记录来预测新的目标的代价。然而，对于包含复杂计算的DNN算子，需要开发者针对算子手写TE（Tensor Expression，张量表达式），这需要对硬件有较为深刻的理解，同时，这个过程重复性较强，需付出大量的时间成本。

# 四、设计思路与实现方案

## 设计思路

分为三部分：
- 搜索空间：搜索空间定义了可用于优化的所有配置项的集合。
- 调优算法：调优算法的任务是在庞大的搜索空间中寻找最优或近似最优的配置组合。
- 调优记录：调优过程中的每次试验都会生成一条记录，称为调优记录。

![](/run/media/roy1994/083E-662F/bishe/pic/自动调优模块.png)

## 使用流程

### 调优方式

1. 用户首先需要在Python层书写算子计算声明、调优参数空间定义
2. 初始化自动调优框架，一方面接收用户编写的算子定义及其相关的元数据信息，另一方面需要配置调优策略，包括调优算法的选择、期望输入形状的范围等。
3. 对于动态形状算子，对输入形状进行拆分，生成测试项，用于后续依据拆分后形状子集和权重信息对性能测试结果进行采样。
4. 初始化性能测试模块。此模块负责构建测试所需的数据，并调度底层C++实现的性能测试器执行实际的算子运行任务，系统会依据形状子集和权重信息进行采样。
5. 准备调优参数空间与相关的约束项。明确调优参数以及这些参数的上下限、依赖关系等约束条件。
6. 启动调优流程

```python
candidate_range = {
    "warp_num": (1, 32),
    "tree_reduce_num": (32, 1024),
    # "grid_reduce_num": (32, 512),
    "spatial_inner_num": (1, 4),
}
constraints = {
    lambda x: all((n & (n - 1)) == 0 for n in x),
}


shape = [(2, 4096), (2, 4096)]
name = "layernorm"
layout= "SR"
shape_name = ['S', 'R']


searcher = ModelSearcher(
    name=name,
    shape=shape,
    shape_name=shape_name,
    layout=layout,
    # program_builder=resnet.build_resnet50_program(dtype="float32"),
    # program_builder=resnet.build_conv2d_program(dtype="float32"),
    program_builder=ops.build_layernorm_program(dtype="float16"),
    candidate_generator=BFGenerator(
        candidate_range=candidate_range,
        constraints=constraints,
    )
)

search_option = SearchOption(
    num_measure_trials=100,  # 调优步数
    repeat=5,  # 重复测量次数
    timeout=10, # 任务超时时间(s)
    target=common.DefaultTarget()
)

searcher.search(search_option)

```

### 应用调优参数方式
```python
op =  searcher.apply("record.json")
```


## 实现

1. **python/paddle/cinn/auototuner/bench_func.py**
   - 性能测试、应用参数部分
2. **python/paddle/cinn/auototuner/candidate_generator.py**
   - 调优空间生成
3. **python/paddle/cinn/auototuner/candidate_searcher.py**
   - 搜索算法实现
4. **python/paddle/cinn/auototuner/model_searcher.py**
   - 主体搜索部分
5. **paddle/cinn/pybind/autotuner**
   - Python C++绑定部分
6. **paddle/cinn/ir/group_schedule/search**
   - 搜索和Apply部分，提供Operator用于Apply

# 五、测试和验收的考量
- 添加 API 的文档
- 添加对Apply API 的测试

# 六、可行性分析和排期规划

根据现有参考，可行。目前功能部分大致已完成，主要在测试和文档部分。

# 七、影响面

和现有启发式调优的兼容方案