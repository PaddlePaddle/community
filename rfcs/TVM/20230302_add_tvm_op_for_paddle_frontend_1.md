
# 在TVM中为paddle框架新增6个不支持的算子

|任务名称 | TVM项目4-为Paddle框架新增TVM算子 |
|---|---|
|提交作者 | 郑学贵 |
|提交时间 | 2023-3-2 |
|版本号 | V0.0 |
|依赖飞桨版本 | v2.4.2 |
|文件名 | add_tvm_op_for_paddle_frontend_1.md |

# 一、方案名称

tvm前端支持paddle算子

# 二、方案描述

tvm前端目前暂不支持paddle框架的`thresholded_relu`、`index_select`、`eye`、`linspace`、`take_alone_axis`、`dist`算子，需要在tvm前端中适配这些算子，以支撑更多的paddle模型通过tvm进行部署。

# 三、方案流程

## 流程设计

1. 调研paddle中`thresholded_relu`、`index_select`、`eye`、`linspace`、`take_alone_axis`、`dist`接口的实现，了解具体的计算逻辑和公式
2. 调用并参考paddle2onnx的流程。
3. 在tvm中新增相应的convert函数，对于不支持的算子通过Relay IR组合实现。
4. 根据paddle框架中算子参数的可能情况，构建测试函数，覆盖所有使用场景。

## 算子实现

### 1.thresholded_relu

激活函数，可以通过Relay IR中`where`、`greater`等函数组合实现

### 2.convert_index_select

Relay中由功能更强的`take`函数，使用`warp`模式实现

### 3.eye

可以先构建一个全0的Tensor，再通过`scatter_nd`在对角线上再赋值1

### 4.linspace

需要实现在区间上均匀采用`num`个值，需要考虑两种情况：

- `num`为1，则输出为`start`
- `num`大于1，可以计算出间隔步长，再通过组合`arrange`等函数进行实现

### 5.take_along_axis

可以通过`gather`函数实现

### 6. dist

计算`p-范数`，需要根据`p`的值进行讨论：

- `p == inf`：计算最大值
- `p == -inf`：计算最小值
- `p == 0`：计算非零元素数量
- 其余情况：根据计算公式，组合Relay中相应的函数进行实现

# 四、方案运行效果

## 测试用例

根据API的参数所有可能的类型进行组合，输入通过随机以及手工构造边界样例生成不同`shape`的Tensor，覆盖所有使用场景。

## 运行结果

paddle框架中`thresholded_relu`、`index_select`、`eye`、`linspace`、`take_alone_axis`、`dist`算子能够导入tvm并执行，计算结果和paddle框架保持一致。

# 五、项目提交时间计划

3-2日已完成代码，通过单测并提交到tvm [pr地址](https://github.com/apache/tvm/pull/14172)
