# paddle.nn.BatchNorm1D 优化设计文档

|API名称 | BatchNorm1D |
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yaozihang |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-05-22 |
|版本号 | V1.0 |
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 |
|文件名 | 20220522_api_optim_batchnorm1d.md |


# 一、概述
## 1、相关背景
测试发现paddle.nn.BatchNorm1D比torch.nn.BatchNorm1d的能力差，体现在支持的数据规模比torch的小，性能比较差，因此需要进一步优化paddle.nn.BatchNorm1D
## 2、功能目标

BatchNorm1D算子在CUDA GPU上至少在性能上与竞品打平

## 3、意义

BatchNorm1D是一个被广泛使用的算子，提升BatchNorm1D算子性能将使得许多模型的性能得到提升

# 二、飞桨现状
目前paddle直接使用了cudnn库来处理BatchNorm1D的逻辑


# 三、业内方案调研
Pytorch方案：在满足一定条件时（training batch size <= 880801 or eval batch size <= 65535....）调用cudnn库, 在满足一定条件时（CUDA使用了MIOpen）使用MIOpen进行计算，其余使用自己开发的batchnorm算子进行计算。

# 四、对比分析
这个PR的任务主要是优化BatchNorm1D，所以在测试的时候采用[num_batch, num_feature]的shape来测试，NCHW这种shape留在之后的优化任务中。
我简单的调用了pytorch和paddle中的batchnorm1d进行性能评测：

```
### torch 代码 ###

import torch
import torch.nn
import numpy as np

shape=[126000, 16]
device = torch.device("cuda")
torch_x = torch.tensor(x.numpy(), device=device)

torch_bn = torch.nn.BatchNorm1d(16, device=device)

#warm up
torch_out = torch_bn(torch_x)
torch.cuda.synchronize(device)

t0 = time.time()
for i in range(100):
    torch_out = torch_bn(torch_x)
# torch.cuda.synchronize(device)
t1 = time.time()
print("torch time : ", t1-t0)
```

```
### paddle 代码 ###
import paddle

shape=[126000, 16]
x = paddle.randn(shape).cuda()

bn = paddle.nn.BatchNorm1D(16)

#warm up
out = bn(x)
paddle.device.cuda.synchronize()

t0 = time.time()
for i in range(100):
    out = bn(x)
paddle.device.cuda.synchronize()
t1 = time.time()
print("paddle time : ", t1-t0)
```

```
### oneflow 代码 ###
import oneflow as flow

device = flow.device("cuda")
flow_x = flow.tensor(x.numpy(), device=device)

flow_bn = flow.nn.BatchNorm1d(16).cuda()

#warm up
flow_out = flow_bn(flow_x)
flow.device.cuda.synchronize()

t0 = time.time()
for i in range(100):
    flow_out = flow_bn(flow_x)
flow.cuda.synchronize(device)
t1 = time.time()
print("oneflow time : ", t1-t0)
```
调研了这三种框架的源码之后，发现三种框架采取了不同的kernel策略，其中torch应为最优的解决方案（kernel策略均使用nvprof证实了运行了不同的kernel）。

### torch
+ 在[N, C]的输入下，直接使用自己编写的CUDA kernel（后面统称native kernel）
+ 在[N, C, L]及[N, C, H, W]（所有dim>=3）的shape下，根据N的值进行判断，小于阈值时使用cudnn的库kernel，大于阈值时使用native kernel。特别的，train mode阈值为880801，eval mode为65535.

### oneflow
+ 在所有输入下，全部使用CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode进行cudnn计算，这样的坏处是特定模型输入下会产生精度问题，该问题在cudnn文档中有说明：https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormMode_t。

### paddle
+ 在FLAGS_cudnn_batchnorm_spatial_persistent开启时且cudnn版本满足时，全部使用CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode
+ 在[N, C]的输入下，使用CUDNN_BATCHNORM_PER_ACTIVATION mode，测试可以发现在这个shape下的输入下，使用CUDNN_BATCHNORM_PER_ACTIVATION比CUDNN_BATCHNORM_SPATIAL的性能要更好，参考issue：https://github.com/PaddlePaddle/Paddle/pull/33887
+ 在其余shape输入下，使用CUDNN_BATCHNORM_SPATIAL进行计算


特别的，paddle在[136000, 16]的配置下报了错误(对应PER_ACTIVATION mode)，在[2100000, 256, 4]的配置下报了错误(对应SPATIAL mode和SPATIAL_PERSISTENT mode)，查阅资料可以发现，这个问题是由于过大的batch size导致的cudnn报错，torch也有相关的issue汇报了这一情况：https://github.com/pytorch/pytorch/issues/29744

```
Traceback (most recent call last):
  File "prof_paddle_bn.py", line 10, in <module>
    batch_norm_out = batch_norm(x)
  File "/usr/local/python3.7.0/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 930, in __call__
    return self._dygraph_call_func(*inputs, **kwargs)
  File "/usr/local/python3.7.0/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 915, in _dygraph_call_func
    outputs = self.forward(*inputs, **kwargs)
  File "/usr/local/python3.7.0/lib/python3.7/site-packages/paddle/nn/layer/norm.py", line 666, in forward
    use_global_stats=self._use_global_stats)
  File "/usr/local/python3.7.0/lib/python3.7/site-packages/paddle/nn/functional/norm.py", line 207, in batch_norm
    variance_out, *attrs)
OSError: (External) CUDNN error(9), CUDNN_STATUS_NOT_SUPPORTED.
  [Hint: 'CUDNN_STATUS_NOT_SUPPORTED'.  The functionality requested is not presently supported by cuDNN.  ] (at /paddle/paddle/phi/kernels/gpu/batch_norm_kernel.cu:532)
  [operator < batch_norm > error]
```

# 五、设计思路与实现方案

## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计

### 性能问题解决方案
编写native CUDA kernel完成batchnorm1d的计算。

### 报错问题解决方案
+ 方案1：全部使用native kernel（因为暂时没发现使用cudnn的好处）
+ 方案2：进行判断，当batch size大于一定阈值后使用native kernel

## API实现方案

本任务中无需更改API

## 方案迭代

实现前的baseline测试，使用对比分析里的测试代码，shape分别为NC:[126000, 16]、NCL:[1000000, 16, 16] ，测试结果：
```
# [126000, 16]
paddle time :  0.7727415561676025
torch time :  0.011563777923583984
oneflow time :  0.08174824714660645

# [1000000, 16, 16]
paddle time :  7.080853462219238
torch time :  3.952852487564087
oneflow time :  7.155768871307373
```

针对BatchNorm1D的native kernel，有以下优化点需要尝试：

- [ ] 使用welford算法在线计算方差和均值
- [ ] 使用4-way循环展开提升内存吞吐，隐藏时延
- [ ] 针对channel-last形状张量与其他形状张量（e.g. NCL）分别实现kernel


### Native kernel第一版

+ block.dim：C
+ block中的所有thread处理每个channel的N\*H\*W元素,使用cub blockreduce计算平均值和方差。
+ 使用计算得到的方差、均值、weight、bias进行elementwise计算

```
# [126000, 16]
paddle time :  0.06167960166931152
torch time :  0.012059926986694336
oneflow time :  0.08394908905029297

# [1000000, 16, 16]
paddle time :  6.160153388977051
torch time :  3.9497106075286865
oneflow time :  7.15362024307251
```

### Native kernel第二版

使用wellford算法完成均值和方差的计算，公式：
$$
\overline{x_{n+1}}=\overline{x_{n}}+\frac{x_{n+1}-\overline{x_{n}}}{n+1}
$$
$$
\sigma_{n+1}^{2}=\sigma_{n}^{2}+\frac{\left(x_{n+1}-\overline{x_{n}}\right)\left(x_{n+1}-\overline{x_{n+1}}\right)-\sigma_{n}^{2}}{n+1}
$$

```
# [126000, 16]
paddle time :  0.0656125545501709
torch time :  0.011774539947509766
oneflow time :  0.08371567726135254

# [1000000, 16, 16]
paddle time :  7.180898189544678
torch time :  3.950751543045044
oneflow time :  7.150729656219482
```

结论：将求和算法（batch update）替换为wellford算法（iterative update）并未获得性能提升，需要进一步分析pytorch性能提升的来源。
# 六、测试和验收的考量

参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
1、5月22日 提交rfc 设计文档至community repo
2、6月3日  提及代码至paddlepaddle repo（包括API、OP、中英文文档、单测）
3、6月10日 完成验收，合入代码

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料