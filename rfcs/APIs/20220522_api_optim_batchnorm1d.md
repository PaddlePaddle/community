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
我简单的调用了pytorch和paddle中的batchnorm1d进行性能评测：

```
### torch 代码 ###

import torch
import torch.nn
import numpy as np

np.random.seed(123)
device = torch.device("cuda")
x_data = np.random.random(size=(200000, 1, 3)).astype('float32')
x = torch.from_numpy(x_data).to(device)
batch_norm = torch.nn.BatchNorm1d(1, device=device)
batch_norm_out = batch_norm(x)
```

```
### paddle 代码 ###

import paddle
import numpy as np

np.random.seed(123)
x_data = np.random.random(size=(2000000, 1, 3)).astype('float32')
x = paddle.to_tensor(x_data).cuda()
batch_norm = paddle.nn.BatchNorm1D(1)
batch_norm_out = batch_norm(x)
```

简单的进行端到端实验测试，可以发现在数据规模较小时，torch和paddle都使用cudnn进行计算，而当数据规模大于一定阈值后，torch使用自己开发的算子进行计算使得计算时延小于paddle，进一步使用nsight compute查看profile结果：

```
paddle  
	bn_fw_tr_1C11_kernel_NCHW kernel   133.06 ms
```

```
torch
	batch_norm_collect_statistics_kernel 71.51 ms
	unrolled_elementwise_kernel_for_multi_outputs 4.4 us
	batch_norm_transform_input_kernel 298 us
```



# 五、设计思路与实现方案

## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
需要进一步研究讨论

## API实现方案

需要进一步研究讨论

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