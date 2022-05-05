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
paddle目前没有SoftMarginLoss损失函数并且要搭建CPU以及GPU算子；


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
- 采用pytorch方法，对算子进行CPU以及GPU设计

# 五、设计思路与实现方案
## 命名与参数设计
- paddle.nn.SoftMarginLoss(reduction(str,可选)，name(str，可选)) -> Tensor:

- paddle.nn.functional.soft_margin_loss(input, target, reduction: str = "mean", name:str=None, ) -> Tensor:
    - input:Tensor, 维度为[N,*],其中N是batch_size， `*` 是任意其他维度。数据类型是float32、float64。
    - label:Tensor, 维度为[batchsize,num_classes]维度、数据类型与输入 input 相同。
    - reduction:str，可选，指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始Loss。

## 底层OP设计
核心部分需要分别完成 softmarginloss.cc softmarginloss.cu 前向计算以及反向传播的算子kernel。

### 前向计算kernel
```namespace phi {

template <typename T, typename Context>
void SoftMarginLossKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& label,
                   DenseTensor* out) {
  auto x_data = input.data<T>();
  auto label_data = label.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto x_numel = input.numel();

  // out = ln(1+exp(-label * x)/(x_numel)
  for (int64_t i = 0; i < x_numel; ++i) {
    PADDLE_ENFORCE_GE(
        x_data[i],
        static_cast<T>(0),
        phi::errors::InvalidArgument(
            "Illegal input, input must be greater than  or equal to 0"));
    PADDLE_ENFORCE_LE(
        x_data[i],
        static_cast<T>(1),
        phi::errors::InvalidArgument(
            "Illegal input, input must be less than or equal to 1"));
    out_data[i] =paddle::operators::real_log(static_cast<T>(1) + std::exp(-label_data[i]* x_data[i]))/x_numel;
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(
    soft_margin_loss, CPU, ALL_LAYOUT, phi::SoftMarginLossKernel, float, double) {}```
 
### 反向计算kernel

```namespace phi {

template <typename T, typename Context>
void SoftMarginLossGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& label,
                       const DenseTensor& out_grad,
                       DenseTensor* input_grad) {
  auto dx_data = dev_ctx.template Alloc<T>(input_grad);
  auto dout_data = out_grad.data<T>();
  auto x_data = input.data<T>();
  auto label_data = label.data<T>();

  int x_numel = input.numel();

  // dx = dout * (-label * exp(-label * x))/(1 + exp(-label * x ))
  for (int i = 0; i < x_numel; ++i) {
    dx_data[i] =
        dout_data[i] * ((- label_data[i]*std::exp(-label_data[i]*x_data[i] )) /
                        std::max((static_cast<T>(1) + std::exp(-label_data[i]*x_data[i])),
                                 static_cast<T>(1e-12)));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    soft_margin_loss_grad, CPU, ALL_LAYOUT, phi::SoftMarginLossGradKernel, float, double) {}
```
以及GPU版本的算子kernel。
同时还需要实现算子的描述及定义，以及InferMeta函数。

## API实现方案

在python中调用算子来实现paddle.nn.SoftMarginLoss和paddle.nn.functional.soft_margin_loss

- 检查参数
  - 检查 reduction 有效性（同其余 functional loss 中的实现）
  - 检查输入的 dtype（含 input、target）（同其余 functional loss 中的实现）
  - 检查输入的input、target维度是否相同
- 计算
  - 调用OP计算loss
- 根据 reduction，输出 loss（同其余 functional loss 中的实现）

# 六、测试和验收的考量
- CPU算子与numpy结果一致
- GPU算子与numpy结果一致
- CPU算子与GPU算子结果一致
- 验证反向传播正确



# 七、可行性分析和排期规划
- 争取在29号之前完成代码开发工作


# 八、影响面
无

# 名词解释
无
