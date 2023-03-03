# paddle.nn.SoftMarginLoss设计文档

|API名称 | SoftMarginLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yangguohao | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-04-16 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220416_api_design_for_SotftMarginLoss.md<br> | 


# 一、概述
## 1、相关背景
SotftMarginLoss是一个用于二分类的损失函数。
loss的公式为
$$loss(x,y)= \sum _i 
\frac{log(1+exp(−y[i]∗x[i]))}{x.nelement()}
)$$

## 2、功能目标
在paddle内实现类似于paddle.nn.SoftMarginLoss以及paddle.nn.functional.soft_margin_loss的功能。


## 3、意义
为paddle增加新的API计算loss

# 二、飞桨现状
paddle目前没有SoftMarginLoss损失函数；


# 三、业内方案调研
### PyTorch
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

tensorflow没有官方实现。

# 四、对比分析
两种方案对比：
- 采用pytorch方法，但是通过飞桨已有算子对其进行设计。

# 五、设计思路与实现方案
## 命名与参数设计
- paddle.nn.SoftMarginLoss(reduction(str,可选)，name(str，可选)) -> Tensor:

- paddle.nn.functional.soft_margin_loss(input, label, reduction: str = "mean", name:str=None, ) -> Tensor:
    - input:Tensor, 维度为[N,*],其中N是batch_size， `*` 是任意其他维度。数据类型是float32、float64。
    - label:Tensor, 维度与输入 input 相同，数据类型为int32, int64, float32, float64，数值为-1或1。
    - reduction:str，可选，指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始Loss。

## 底层OP设计
核心部分现改为用已有的 API 算子进行组合
主要为以下几个原因：
- 1. 开发成本比较高，修改代码多
- 2. 面临多硬件适配的问题，cpu, gpu, xpu, npu都需要写一份算子
- 3. 和其他生态环境适配的问题，无法和tensorRT和ONNX适配
- 4. 高阶自动微分的问题，需要在高阶自动微分添加lowering逻辑
- 5. 编译器对接的问题，需要在编译器添加lowering逻辑


## API实现方案

在python中调用算子来实现paddle.nn.SoftMarginLoss和paddle.nn.functional.soft_margin_loss

- 检查参数
  - 检查 reduction 有效性（同其余 functional loss 中的实现）
  - 检查输入的 dtype（含 input、label）（同其余 functional loss 中的实现）
  - 对 label 的 dtype 进行转换，尽量与 input 一致。
  - 检查输入的input、label维度是否相同
- 计算
  - 调用 paddle.log 以及 paddle.exp 计算loss
- 根据 reduction，输出 loss（同其余 functional loss 中的实现）

# 六、测试和验收的考量
- 验证在 CPU 以及 GPU 环境下，计算结果需要与 numpy 一致。
- 验证 reduction 检查的有效性。
- 验证输入维度检查的有效性。
- 验证对 dtype 检查的有效性。



# 七、可行性分析和排期规划
- 争取在29号之前完成代码开发工作


# 八、影响面
无

# 名词解释
无
