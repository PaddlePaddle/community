|API名称 | paddle.masked_fill, paddle.masked_fill_, Tensor.masked_fill, Tensor.masked_fill_ | 
|---|---|
|提交作者 | robinbg | 
|提交时间 | 2023-09-16| 
|版本号 | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20230916_api_design_for_masked_fill.md | 

## 一、概述
### 1、相关背景
在深度学习和数据处理中，经常需要按照某些条件修改tensor的值。例如，在数据增强、数据处理和某些特定的算法实现中，我们可能需要根据一个mask tensor来决定如何修改原始tensor。为此，我们提议添加一个`masked_fill`函数，使得用户可以轻松地实现这一功能。

### 2、功能目标
- 提供`paddle.masked_fill`作为独立的函数调用，非inplace
- 提供`paddle.masked_fill_`，作为独立的函数，inplace地修改输入
- 提供`Tensor.masked_fill`，作为Tensor的方法使用，非inplace
- 提供`Tensor.masked_fill_`，作为Tensor的方法使用，inplace修改输入

### 3、意义
提供一个简洁且高效的方式，使得用户可以根据条件来修改tensor的值，从而增强Paddle的功能性和灵活性。

## 二、飞桨现状
目前，飞桨尚未提供与`masked_fill`功能相似的API。用户需要使用低级API手动实现此功能，这既不方便，也可能导致效率低下。

## 三、业内方案调研
PyTorch提供了一个名为`masked_fill`的API，允许用户根据一个mask tensor来修改原始tensor的值。这个功能在PyTorch社区中被广泛使用，反映出其实际应用的价值。

## 四、对比分析
与PyTorch的实现相比，我们的设计目标是提供一个功能相同但使用更为简洁的API。我们还提供了inplace和非inplace两种方式，以满足不同的使用需求。

## 五、设计思路与实现方案
### 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)

#### 参数
- `input`(Tensor) - 输入的Tensor
- `mask`(Tensor) - 与输入Tensor形状相同的mask tensor，数据类型为bool。
- `value`(scalar) - 用于填充的值。
- `inplace`(bool, 可选) - 是否进行inplace操作，默认为False。

### 底层OP设计
我们可以考虑使用现有的element-wise操作来实现这个功能。基本的思路是利用mask tensor和原始tensor进行element-wise乘法，然后再加上一个与mask tensor相反的tensor和value的乘积。

### API实现方案
在`python/paddle/tensor/manipulation.py`文件中实现上述API，并提供相关的文档说明。

## 六、测试和验收的考量
- 在`Paddle repo`的`test/`目录下提供单测代码，确保在各种条件下都能正确工作。
- 在`paddle/test/legacy_test/test_inplace.py`中新增对应的inplace api单测，确保inplace功能正确无误。
- 验收标准应包括功能测试、性能测试和边界条件测试。
