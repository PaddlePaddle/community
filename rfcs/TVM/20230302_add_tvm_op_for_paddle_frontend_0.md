
# 在TVM中为paddle框架新增7个不支持的算子

|任务名称 | TVM项目5-为Paddle框架新增TVM算子 |
|---|---|
|提交作者 | 郑学贵 |
|提交时间 | 2023-3-2 |
|版本号 | V0.0 |
|依赖飞桨版本 | v2.4.2 |
|文件名 | add_tvm_op_for_paddle_frontend_0.md |

# 一、方案名称

tvm前端支持paddle算子

# 二、方案描述

tvm前端目前暂不支持paddle框架的`tile`、`stack`、`mish`、`unstack`、`silu`、`softshrink`、`where`算子，需要在tvm前端中适配这些算子，以支撑更多的paddle模型通过tvm进行部署。

# 三、方案流程

## 流程设计

1. 调研paddle中`tile`、`stack`、`mish`、`unstack`、`silu`、`softshrink`、`where`接口的实现，了解具体的计算逻辑和公式
1. 调用并参考paddle2onnx的流程。
2. 在tvm中新增相应的convert函数，对于不支持的算子通过Relay IR组合实现。
3. 根据paddle框架中算子参数的可能情况，构建测试函数，覆盖所有使用场景。

## 算子实现

### 1.tile

tvm relay中也有相应的`tile`函数，因此只需要针对输入进行处理，再调用`_op.tile`即可。输入的`repeat_times`有三种类型：

   - Tensor: 存储在`op.input("RepeatTimes")`，需要`infer`常量值。
   - list|tuple且元素为Tensor：存储在`op.input("repeat_times_tensor")`，需要逐个`infer`常量值，再拼接。
   - list|tuple且元素为整数：存储在`op.attr("repeat_times")`

### 2. mish

激活函数，根据API文档中计算公式，通过`Relay`中已有的`exp`、`mul`、`log`等函数组合实现

### 3. stack

通过Relay中已有的`stack`实现

### 4. unstack

Relay中没有实现`unstack`，可以采用`split`和`squeeze`组合实现

### 5.silu

激活函数，根据API文档中计算公式，通过`Relay`中已有的`sigmoid`、`mul`函数组合实现

### 6. softshrink

激活函数，根据API文档中计算公式，该函数是个三段的分段函数，可以通过组合`where`和`add`等函数实现

### 7. where

通过Relay中已有的`where`实现

# 四、方案运行效果

## 测试用例

根据API的参数所有可能的类型进行组合，输入通过随机以及手工构造边界样例生成不同`shape`的Tensor，覆盖所有使用场景。

## 运行结果

paddle框架中`tile`、`stack`、`mish`、`unstack`、`silu`、`softshrink`、`where`算子能够导入tvm并执行，计算结果和paddle框架保持一致。

# 五、项目提交时间计划

3-1日已完成代码，通过单测并提交到tvm [pr地址](https://github.com/apache/tvm/pull/14160)
