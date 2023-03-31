### p_norm_grad OP性能优化设计文档


| 基本信息                                                     | 内容                                        |
| ------------------------------------------------------------ | ------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | zerorains                                   |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-31                                  |
| 版本号                                                       | V1.0                                        |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                               |
| 文件名                                                       | 20230331_p_norm_grad_op_optimization.md<br> |


# 1 背景与意义

目前 Paddle 内`p_norm_grad`算子 GPU 计算采用了 CUDA Kernel 与 Eigen 混合的模式，虽然使用Eigen库描述`p_norm_grad`算子十分方便，但是性能上仍有一定的提升空间。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

目前的实现有一定的性能优化空间，可以加入一些`ElementWise`的向量化读写和计算融合的技巧对性能进行优化。当前`p_norm_grad`算子纯backword的性能如下表：

| Case No. | device | input_shape |input_type|porder|axis|  keepdim | Paddle Perf(ms) |
|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 | [300L, 250L, 250L] |float32| 2.0 | -1 |False| 0.89023| 
| 2 | Tesla V100 | [300L, 250L, 250L] |float32| 3.0 | -1 |False| 0.88928|
| 3 | Tesla V100 | [300L, 250L, 250L] |float32| fro | -1 |False| 0.88506|
| 4 | Tesla V100 | [300L, 250L, 250L] |float16| 2.0 | -1 |False| 0.88407| 
| 5 | Tesla V100 | [300L, 250L, 250L] |float16| 3.0 | -1 |False| 0.89285|
| 6 | Tesla V100 | [300L, 250L, 250L] |float16| fro | -1 |False| 0.88684|



API文档 [norm-API文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/norm_cn.html#norm)

## 1.2 业内方案调研

Pytorch中无单独的文件专门用于`p_norm_grad`算子计算。在查找相关源码时,发现Pytoch中的[torch.nn.functional.pdist](https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html?highlight=pdist#torch.nn.functional.pdist)计算p距离时，算的就是`p_norm`，由此，可以推断两者共用一套计算体系。[torch.nn.functional.pdist和torch.norm的源码](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/DistanceKernel.cu#L70)中定义了`p_norm`在GPU上元素级别的forward和backward的计算方式，并采用二维线程的GPU计算方式其进行计算。`p_norm_grad`算子backward的整体性能如下(基于pytorch　v1.12.1)：

| Case No. | device | input_shape |input_type|porder|axis|  keepdim | Pytorch Perf(ms) |
|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 | [300L, 250L, 250L] |float32| 2.0 | -1 |False| 0.20349| 
| 2 | Tesla V100 | [300L, 250L, 250L] |float32| 3.0 | -1 |False| 0.66578|
| 3 | Tesla V100 | [300L, 250L, 250L] |float32| fro | -1 |False| 0.20724|
| 4 | Tesla V100 | [300L, 250L, 250L] |float16| 2.0 | -1 |False| 0.14398| 
| 5 | Tesla V100 | [300L, 250L, 250L] |float16| 3.0 | -1 |False| 0.38483|
| 6 | Tesla V100 | [300L, 250L, 250L] |float16| fro | -1 |False| 0.14636|

## 1.3 对比分析

目前Paddle中主要使用Eigen实现`p_norm_grad`算子在GPU上的计算。Pytorch在实现`p_norm`算子的前向和反向上，对参数`porder`的情况分别设计了6种特定的实现方案，但是Paddle中实现方式则是采用通用的处理方式。在测试样本中，Case No.2和Case No.5对应pytorch中的通用方案。两者直接的差距在于，Paddle采用Eigen实现的通用计算方式处理所有情况，而Pytorch则使用了二维线程的GPU计算方法，并设计了6种不同的计算方案。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

`p_norm_grad`算子的性能瓶颈在于Eigen实现了整个计算过程，在查阅了相关源码之后，确定可以使用`ElementWiseKernel`和`BroadcastKernel`对Eigen的实现进行替换，并结合一些计算融合的方法，减少`Kernel`的调用，提高`p_norm_grad`算子在GPU的计算性能。

## 2.2 Host端计算流程

将输入按照传入的参数调整成对应的shape，为Device端的计算提供输出维度。

## 2.4 Device端计算流程

`p_norm_grad`算子的计算公式为:

$$dx = \frac {abs(x)^p * sign(x)*broadcast(dy)}{broadcast((y+eps)^p)}$$

其中$abs(·)$是绝对值函数，$sign(·)$是取符号函数，$broadcast(·)$是广播函数，因为使用除法可能会导致除0的情况，因此`p_norm_grad`算子的计算公式可以更新为

$$dx = {abs(x)^p * sign(x)*broadcast(dy)}*{broadcast((y+eps)^{-p})}$$

基于Kernel调用次数和计算量的权衡，在实际实现过程中，${broadcast(dy)}*{broadcast((y+eps)^{-p})}$的部分可以转变成${broadcast(dy*(y+eps)^{-p})}$进行计算，即先算乘法再进行广播，这样可以有效减小计算量，同时也减少了Kernel的调用次数。这个部分预计需要一个`ElementWiseKernel`的调用和一个`BroadcastKernel`的调用。这部分的结果称为$y_{part}$。

然后计算公式变成：

$$dx = {abs(x)^p * sign(x)}*y_{part}$$

这里可以看到实际计算涉及到了$x$和$y_{part}$两个变量，因此这部分的计算操作可以借助`ElementWiseKernel`实现，并自定义计算过程进行操作融合，以减少Kernel的调用次数，最终调用的Kernel数为3。

还有一种方案是将所有元素操作融入到一个`ElementWiseKernel`中，但这个方案是不现实的，因为$dy$和$y$要经过`broadcast`操作才能与$x$进行计算，如果想要融合到一个`ElementWiseKernel`中则需要先对$dy$和$y$进行广播，这就已经引入了两个Kernel了，在加上一个`ElementWiseKernel`就是3个kernel，但是经过广播之后再执行ElementWise的计算无疑会增加$y_{part}$部分的计算量。

因此可以认为使用两个`ElementWiseKernel`和一个`BroadcastKernel`实现`p_norm_grad`算子的GPU计算过程是所有采用`Kernel`实现方案中的最优解了。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 03.27~03.31 |
| 2 | 完成开发文档设计  | 03.31~04.01 |
| 3 | 提交PR进行后续迭代 | 04.01~活动结束 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)

