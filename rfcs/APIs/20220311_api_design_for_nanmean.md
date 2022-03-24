# paddle.Tensor.nanmean 设计文档

|API名称 | paddle.nanmean | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 李芳钰 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-11 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220311_api_design_for_nanmean.md.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.nanmean`以扩展paddle.mean API 的功能。
## 2、功能目标
增加API`paddle.nanmean`，实现沿指定轴计算算术平均值并且忽略nan的功能。
## 3、意义
飞桨支持计算算术平均值并且忽略NaN。

# 二、飞桨现状
目前paddle缺少相关功能实现。

API方面，已有相关功能的API，[paddle.nansum](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py#L910), 由于容易实现，所以在Paddle中是一个由其他API组合成的API，没有实现自己的OP，其主要实现逻辑为：
1. 获取一个和输入x维度一致的全零Tensor(zero_tensor).
2. 通过`paddle.isnan()`获取输入x的nan值所在位置，可以视为nan_mask。
3. 通过`paddle.where()`将输入x中的nan值替换成0，得到temp_tensor。
4. 最后将替换nan值的temp_tensor以及相应的参数，作为`paddle.sum`的输入。

在实际实现时，可以获取输入tensor在指定轴上的非nan值的统计个数，在结合API`paddle.nansum`即可实现`paddle.nanmean`的功能。

# 三、业内方案调研
## Numpy 
### 实现方法
以现有numpy python API组合实现，[代码位置](https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/nanfunctions.py#L953-L1056).
其中核心代码为：
```Python
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                       where=where)
    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims,
                 where=where)
    tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)
    avg = _divide_by_count(tot, cnt, out=out)
    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=3)
        # NaN is the only possible bad value, so no further
        # action is needed to handle bad results.
    return avg
```
整体逻辑为：

- 通过`_replace_nan`获取nan的mask，以及将nan替换成0后的arr。
- 然后利用`np.sum`和`~mask`获取指定轴的非nan值的计数值cnt。
- 再通过`np.sum`和去除nan的`arr`获取指定轴上元素的总和tot。
- 最后利用`_divide_by_count`将tot/cnt,得到最终结果avg。
- 需要注意的是当`(cnt == 0).any() == True`时说明在指定轴上，存在元素全为nan的情况，<br>这时候numpy的做法是抛出警告，且该元素上的均值任然为nan。


## Pytorch
Pytorch中有API`torch.nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None) → Tensor`。在pytorch中，介绍为：
```
Computes the mean of all non-NaN elements along the specified dimensions.
This function is identical to torch.mean() when there are no NaN values in the input tensor. In the presence of NaN, torch.mean() will propagate the NaN to the output whereas torch.nanmean() will ignore the NaN values (torch.nanmean(a) is equivalent to torch.mean(a[~a.isnan()])).
If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
```

### 实现方法
在实现方法上，Pytorch的整体逻辑与Numpy一致，[代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp#L1232-L1244)。其中核心代码为：
```c++
    Tensor nanmean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  TORCH_CHECK(
      self.is_floating_point(),
      "nanmean(): expected input to have floating point dtype but got ",
      self.scalar_type());
  const auto factor =
      at::native::isnan(self.detach()).logical_not_().sum(dim, keepdim);
  return at::nansum(self, dim, keepdim, opt_dtype).div_(factor);
}
```
整体逻辑为：
- 通过`isnan`获取张量nan值的mask。
- 然后利用`logical_not_`,`sum`结合`mask`获取指定轴的非nan值的计数值factor。
- 再通过`nansum`获取指定轴上张量非nan值的总和。
- 最后利用`div_`除以factor(对标Numpy的cnt)得到张量在指定轴上的算数平均值。



# 四、对比分析
- 使用场景与功能：在维度支持上，Pytorch和Numpy都支持指向多个轴，但Numpy在指定多轴时指支持tuple输入，这里对标Pytorch支持tuple输入以及python:ints。
- 需要注意的是Numpy当`(cnt == 0).any() == True`时说明在指定轴上，存在元素全为nan的情况，这时候Numpy会额外抛出一个警告，且该元素上的均值任然为nan。

# 五、方案设计
## 命名与参数设计
API设计为`paddle.nanmean(x, axis=None, keepdim=False, name=None)`
命名与参数顺序为：形参名`input`->`x`和`dim`->`axis`,  与paddle其他API保持一致性，不影响实际功能使用。
参数类型中，`axis`支持`int|list|tuple`输入,以同时支持一维和多维的场景。

## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案
主要按下列步骤进行组合实现,实现位置为`paddle/tensor/math.py`与`sum`,`nansum`等方法放在一起：
1. 使用`paddle.nansum`得到忽略nan值的元素总和.
2. 使用`paddle.isnan`以及`paddle.sum`得到输入Tensor的nan mask，以及指定轴的非nan值的计数值cnt.
3. 使用`paddle.divide`得到忽略nan的输入张量的算术平均值。

- 对`keepdim`参数的处理，对标Numpy融合到各个API当中。

# 六、测试和验收的考量
测试考虑的case如下：

- 和numpy结果的数值的一致性, `paddle.nanmean`,和`np.nanmean`结果是否一致；
- 参数`axis`为int,tuple和list时输出的正确性；
- `keepdim`参数的正确性；
- 未输入维度时的输出正确性；
- 输入含`NaN`结果的正确性；
- 输入在指定轴上存在元素都为NaN时,结果的正确性；
- 测试在进行反向梯度计算时结果的正确性(包含nan值和非nan值位置的梯度)；
- 错误检查：输入`x`不是Tensor时,能否正确抛出错误；
- 错误检查：`axis`所指维度在当前Tensor中不合法时能正确抛出错误。

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，且依赖的`paddle.nansum`已经在 Paddle repo 的 python/paddle/tensor/math.py [目录中](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py#L910)。工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无

