# paddle.nn.SoftMarginLoss设计文档

|API名称 | SoftMarginLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 杜渺 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-13 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220313_api_design_for_SotftMarginLoss.md<br> | 


# 一、概述
## 1、相关背景
SotftMarginLoss是一个用于二分类的损失函数。
loss的公式为
$$loss(x,y)= \sum _i 
\frac{log(1+exp(−y[i]∗x[i]))}{x.nelement()}
)$$

原始论文不可考，但是PyTorch的官网有详细的 [介绍.](https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html)
## 2、功能目标
在paddle内实现类似于Pytorch SoftMarginLoss的功能。


## 3、意义
为paddle增加SoftMarginLoss，以后对于二分类问题会有更多的选择。

# 二、飞桨现状
paddle目前没有SoftMarginLoss这个损失函数，可以根据上层api搭建，但是性能估计会有影响；


# 三、业内方案调研
### PyTorch
中SoftMarginLoss中cpu的实现位于pytorch/aten/src/ATen/native/Loss.cpp这个文件中
##### 前向传播
```cpp 
Tensor& soft_margin_loss_out(const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output) {
  // compute inplace variant of: output = at::log(1. + at::exp(-input * target));
  at::neg_out(output, input).mul_(target).exp_().add_(1.).log_();
  if (reduction != Reduction::None) {
    auto tmp = apply_loss_reduction(output, reduction);
    output.resize_({});
    output.copy_(tmp);
  }
  return output;
}
```

##### 反向传播

```cpp
Tensor& soft_margin_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto z = at::exp(-target * input);
  // inplace version of: grad_input = -norm * target * z / (1. + z) * grad_output;
  at::mul_out(grad_input, target, z).mul_(-norm);
  z.add_(1);
  grad_input.div_(z).mul_(grad_output);
  return grad_input;
}
```

### tensorflow

tensorflow没有官方大算子实现，需要外层搭建小算子实现。
# 四、对比分析
两种方案对比：
- 内部实现SoftMarginLoss大算子的前向传播和反向传播，优点是直接操作矩阵计算流，性能最优；缺点是实现起来相对麻烦。
- 调用小算子拼接，优点是实现速度比较快，不用单独实现反向传播；缺点是有潜在的可能存在性能损失。

# 五、设计思路与实现方案
为paddle phi计算库内部添加SoftMarginLoss的前向传播和反向传播大算子（CPU和GPU各自单独实现）。然后为paddle 动态图和静态图分别添加SoftMarginLoss的API。
## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
添加到paddle最新维护的phi计算库中。
## API实现方案
新增两个api，调用路径为：
paddle.nn.SoftMarginLoss
和
paddle.nn.functional.soft
_margin_loss
# 六、测试和验收的考量
- Loss准确度的测试。
- 1D，2D tensor的表现行为和pytorch表现一致
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
- 3月20号合入rfcs
- 3月30号合入实现和测试，过ci


# 八、影响面
无

# 名词解释
无
# 附件及参考资料
https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html