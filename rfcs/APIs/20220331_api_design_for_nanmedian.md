# paddle.Tensor.nanmedian 设计文档

|API名称 | paddle.nanmedian                         | 
|---|------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | thunder95                                | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-31                               | 
|版本号 | V1.0                                     | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                  | 
|文件名 | 20220311_api_design_for_nanmedian.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.nanmedian`以扩展paddle.median API 的功能。
## 2、功能目标
增加API`paddle.nanmedian`，实现沿指定轴计算算术中位数并且忽略nan的功能。
## 3、意义
飞桨支持沿指定轴计算中位数，而忽略NaN。

# 二、飞桨现状
目前paddle缺少相关功能实现。

API方面，已有相关功能的API，[paddle.nansum](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py#L910), 由于容易实现，所以在Paddle中是一个由其他API组合成的API，没有实现自己的OP，其主要实现逻辑为：
1. 通过`paddle.isnan()`和`paddle.where()`可以获取到轴上nan的数量.
2. `paddle.sort()`经测试，也对NAN无法很好支持，可先通过`paddle.isnan()`和`paddle.where()`将NAN替换为最小值.
3. 由于每行NAN数量不同，非NAN的元素无法再轴上对齐，可以进行遍历对指定轴获取中位数。
4. 用组合API的方式效率较低， 操作较为繁琐。

参考paddle.quantile, 支持axis是None, int, list或tuple类型, 计算步骤可基于下方代码模拟实现:


```Python
import paddle
import numpy as np

def nan_median(x, axis=None):
    dims = len(x.shape)
    ori_axis = [d for d in range(dims)]

    is_axis_single = True

    if isinstance(axis, (list, tuple)):
        axis_src, axis_dst = [], []
        for axis_single in axis:
            if axis_single < 0:
                axis_single = axis_single + dims
            axis_src.append(axis_single)
        axis_dst = list(range(-len(axis), 0))
        x = paddle.moveaxis(x, axis_src, axis_dst)
        x = paddle.flatten(x, axis_dst[0], axis_dst[-1])
        axis = axis_dst[0]
        is_axis_single = False
    else:
        if axis is None:
            x = paddle.flatten(x)
            axis = 0
        elif isinstance(axis, int):
            if axis < 0:
                axis += dims
        else:
            return False

    mask = x.isnan().logical_not()
    out = paddle.masked_select(x, mask)
    x0 = paddle.expand_as(paddle.min(out), x)
    x2 = paddle.where(paddle.isnan(x), x0, x)

    if is_axis_single:
        ori_axis.append(ori_axis[axis])
        del ori_axis[axis]
        x2 = paddle.transpose(x2, perm=ori_axis)

    sorted_t = paddle.sort(x2, descending=True)
    out_shape = sorted_t.shape[:-1]

    if is_axis_single and dims > 1:
        sorted_t = sorted_t.flatten(stop_axis=dims - 2)
    if len(sorted_t.shape) == 1:
        sorted_t = paddle.unsqueeze(sorted_t, axis=0)

    sum_t = mask.astype("float64").sum(axis=axis, keepdim=True).flatten().numpy()
    for i in range(sum_t.shape[0]):
        sz = int(sum_t[i])
        if sz == 0:
            sum_t[i] = float('nan')
        else:
            kth = sz >> 1
            if sz & 1 == 0:
                sum_t[i] = (sorted_t[i, kth - 1] + sorted_t[i, kth]) / 2.0
            else:
                sum_t[i] = sorted_t[i, kth]

    if len(out_shape):
        return paddle.to_tensor(sum_t).reshape(out_shape).numpy()
    return paddle.to_tensor(sum_t).numpy()


y = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
y[0, 1, 1] = -10
y[0, 1, 0] = np.nan
y[0, 1, 2] = np.nan
y[1, 1, :2] = np.nan
x = paddle.to_tensor(y)
axis_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]
for a in axis_list:
    print("running axis: ", a)
    c1 = np.nanmedian(y, axis=a).astype("float64")
    c2 = nan_median(x, axis=a).astype("float64")
    print(np.allclose(c1, c2, equal_nan=True))
```

# 三、业内方案调研
## Numpy 
### 实现方法
Numpy已内置nanmedian功能的API，[文档位置](https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html).
[源码位置](https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/nanfunctions.py#L1127-L1223)
其中核心代码为：

```Python
def _nanmedian1d(arr1d, overwrite_input=False):
    """
    Private function for rank 1 arrays. Compute the median ignoring NaNs.
    See nanmedian for parameter usage
    """
    arr1d_parsed, overwrite_input = _remove_nan_1d(
        arr1d, overwrite_input=overwrite_input,
    )

    if arr1d_parsed.size == 0:
        # Ensure that a nan-esque scalar of the appropriate type (and unit)
        # is returned for `timedelta64` and `complexfloating`
        return arr1d[-1]

    return np.median(arr1d_parsed, overwrite_input=overwrite_input)
```
整体逻辑为：

- 通过`np.apply_along_axis`, 对axis上分别执行_nanmedian1d。
- 然后利用`_remove_nan_1d`移出掉nan元素。
- 最后通过常规的中位数获取方式`np.median`得到结果

## Pytorch
Pytorch中有API`torch.nanmedian(input, dim=- 1, keepdim=False, *, out=None)`。

在pytorch中，[文档地址](https://pytorch.org/docs/stable/generated/torch.nanmedian.html?highlight=nanmedian#torch.nanmedian), 介绍为：
```
Returns the median of the values in input, ignoring NaN values.
This function is identical to torch.median() when there are no NaN values in input. When input has one or more NaN values, torch.median() will always return NaN, while this function will return the median of the non-NaN elements in input. If all the elements in input are NaN it will also return NaN.
```
在底层实现上torch.nanmedian和torch.median是基于相同核函数实现，通过ignore_nan参数控制。

### 实现方法
在实现方法上，Pytorch的整体逻辑与Numpy一致。其中核心代码为：
```c++
std::tuple<Tensor&, Tensor&> median_out_cuda(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

Tensor median_cuda(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> nanmedian_out_cuda(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/true);
}
}
```

## TensorFlow
tensorflow没有支持median接口，但可以通过percentile接口， 也没有支持nanmedian功能。
在tensorflow里也可以通过tf.experimental.numpy直接调用numpy函数

```Python
tf.contrib.distributions.percentile(
    x,
    q,
    axis=None,
    interpolation=None,
    keep_dims=False,
    validate_args=False,
    name=None
)
```

# 四、对比分析
- 使用场景与功能：在维度支持上，Pytorch只支持一个轴，但是Numpy支持指向多个轴。
- Pytorch支持返回索引， Numpy只支持返回中位数的值。

# 五、方案设计
## 命名与参数设计
API设计为`paddle.nanmedian(x, axis=None, keepdim=False, name=None)`
命名与参数顺序为：形参名`input`->`x`和`dim`->`axis`,  与paddle其他API保持一致性，不影响实际功能使用。
参数类型中，`axis`支持`int|tuple|list`输入， keepdim支持返回保持原来的形状。

## 底层OP设计
现有API对NAN元素支持比较有限, 需要单独设计一个OP, 支持cpu和cuda, 对最后一维度进行nanmedian算子操作.
参考Pytorch不做反向传播梯度计算。

## API实现方案
主要按下列步骤进行组合实现,实现位置为`paddle/tensor/math.py`与`sum`,`nansum`等方法放在一起：
1. 如果axis是多个轴, 参考`paddle.quantile`算子对输入进行轴的转换操作 。
2. 如果axis是一个轴, 当不是最后一个轴, 先使用`paddle.transpose`获取axis上的目标元素。
3. 设计cpu和gpu核函数, 对最后的轴进行nanmedian计算。
4. 将最终结果reshape到目标维度
5. 对`keepdim`参数的处理，对标Numpy融合到各个API当中。

反向计算主要逻辑参考MaxOrMinGradFunctor
1. 对median计算有贡献的元素梯度为1， 其他元素梯度为0, 
2. 考虑到奇偶数量需要缓存中间输出变量, 若是奇数存储两个一样的中间数值, 若是偶数存储左右两边的数值
3. 分别对两个数值进行equals判断

## 代码实现文件路径

CPU中正向和反向计算： paddle/phi/kernels/cpu/nanmedian_kernel.cc paddle/phi/kernels/cpu/nanmedian_grad_kernel.cc
GPU中正向和反向计算： paddle/phi/kernels/gpu/nanmedian_kernel.cu paddle/phi/kernels/gpu/nanmedian_grad_kernel.cu


```c++
template <typename T, typename Context>
void NanMedianKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out);
                          
```


算子注册路径：
paddle/fluid/operators/nanmedian_op.cc

函数API实现路径: python/paddle/tensor/stat.py
单元测试路径： python/paddle/fluid/tests/unittests/test_nanmedian.py

# 六、测试和验收的考量
测试考虑的case如下：

- 和numpy结果的数值的一致性, `paddle.nanmedian`,和`np.nanmdian`结果是否一致；
- 参数`axis`校验参数类型int，tuple以及list，判断axis合法，并进行边界检查；
- axis测试需覆盖None, int, tuple, list, 数量需要覆盖一维，多维，以及全部维度；
- `keepdim`参数的正确性，输出结果的正确性；
- 输入含`NaN`结果的正确性；
- 输入所有轴上都不含`NaN`结果的正确性；
- 输入轴上不同数量的`NaN`结果的正确性；
- 输入少量或大量的`NaN`结果的正确性；
- 输入axis上全为`NaN`结果的正确性；
- 测试在进行反向梯度计算时结果的正确性(包含nan值和非nan值位置的梯度)；
- 错误检查：输入`x`不是Tensor时,能否正确抛出错误；

# 七、可行性分析及规划排期

方案实施难度可控，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无

