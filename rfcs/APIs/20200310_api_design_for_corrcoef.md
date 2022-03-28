# paddle.linalg.corrcoef 设计文档

|API名称 | paddle.linalg.corrcoef | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 李啟铜 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-11 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20200310_design_for_corrcoef.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.linalg.corrcoef`。
## 2、功能目标
增加API`paddle.linalg.corrcoef`,实现求皮尔逊积矩相关系数功能。

## 3、意义
飞桨支持计算皮尔逊积矩相关系数。

# 二、飞桨现状
目前paddle缺少相关功能实现。

API方面,要实现的API是在`paddle.linalg.cov`实现的基础上实现的，没有实现的自己的OP，其主要功能为：

1.计算协方差

在实际实现时,可以通过调用`paddle.linalg.cov`，再结合公式进行实现`paddle.linalg.corrcoef`。

# 三、业内方案调研

## Numpy 
### 实现方法
以现有numpy python API`numpy.cov`，进行实现`numpy.corrcoef`。
其中核心代码为：
```Python
  def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *,
             dtype=None):
    """
    Return Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `R`, and the
    covariance matrix, `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    R : ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    cov : Covariance matrix

    Notes
    -----
    Due to floating point rounding the resulting array may not be Hermitian,
    the diagonal elements may not be 1, and the elements may not satisfy the
    inequality abs(a) <= 1. The real and imaginary parts are clipped to the
    interval [-1,  1] in an attempt to improve on that situation but is not
    much help in the complex case.

    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    In this example we generate two random arrays, ``xarr`` and ``yarr``, and
    compute the row-wise and column-wise Pearson correlation coefficients,
    ``R``. Since ``rowvar`` is  true by  default, we first find the row-wise
    Pearson correlation coefficients between the variables of ``xarr``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> xarr = rng.random((3, 3))
    >>> xarr
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])
    >>> R1 = np.corrcoef(xarr)
    >>> R1
    array([[ 1.        ,  0.99256089, -0.68080986],
           [ 0.99256089,  1.        , -0.76492172],
           [-0.68080986, -0.76492172,  1.        ]])

    If we add another set of variables and observations ``yarr``, we can
    compute the row-wise Pearson correlation coefficients between the
    variables in ``xarr`` and ``yarr``.

    >>> yarr = rng.random((3, 3))
    >>> yarr
    array([[0.45038594, 0.37079802, 0.92676499],
           [0.64386512, 0.82276161, 0.4434142 ],
           [0.22723872, 0.55458479, 0.06381726]])
    >>> R2 = np.corrcoef(xarr, yarr)
    >>> R2
    array([[ 1.        ,  0.99256089, -0.68080986,  0.75008178, -0.934284  ,
            -0.99004057],
           [ 0.99256089,  1.        , -0.76492172,  0.82502011, -0.97074098,
            -0.99981569],
           [-0.68080986, -0.76492172,  1.        , -0.99507202,  0.89721355,
             0.77714685],
           [ 0.75008178,  0.82502011, -0.99507202,  1.        , -0.93657855,
            -0.83571711],
           [-0.934284  , -0.97074098,  0.89721355, -0.93657855,  1.        ,
             0.97517215],
           [-0.99004057, -0.99981569,  0.77714685, -0.83571711,  0.97517215,
             1.        ]])

    Finally if we use the option ``rowvar=False``, the columns are now
    being treated as the variables and we will find the column-wise Pearson
    correlation coefficients between variables in ``xarr`` and ``yarr``.

    >>> R3 = np.corrcoef(xarr, yarr, rowvar=False)
    >>> R3
    array([[ 1.        ,  0.77598074, -0.47458546, -0.75078643, -0.9665554 ,
             0.22423734],
           [ 0.77598074,  1.        , -0.92346708, -0.99923895, -0.58826587,
            -0.44069024],
           [-0.47458546, -0.92346708,  1.        ,  0.93773029,  0.23297648,
             0.75137473],
           [-0.75078643, -0.99923895,  0.93773029,  1.        ,  0.55627469,
             0.47536961],
           [-0.9665554 , -0.58826587,  0.23297648,  0.55627469,  1.        ,
            -0.46666491],
           [ 0.22423734, -0.44069024,  0.75137473,  0.47536961, -0.46666491,
             1.        ]])

    """
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=3)
    c = cov(x, y, rowvar, dtype=dtype)
    try:
        d = diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c
```
整体逻辑为：

- 因为计算皮尔逊积矩相关系数和ddof无关，所以若输入中的ddof不为False，发出警告。
- 使用cov计算协方差。
- 得到对角线的值(分母)，如果值错误返回1。
- 计算分母部分，使用实部进行计算，然后用协方差分别除以分母的两个部分。
- 判断是否是复数，并进行值的范围控制。

## Pyrotch
### 实现方法
以torch现有的API实现，并且是用c++实现.
其中核心代码为：
```C++
  Tensor corrcoef(const Tensor& self) {
  TORCH_CHECK(
      self.ndimension() <= 2,
      "corrcoef(): expected input to have two or fewer dimensions but got an input with ",
      self.ndimension(),
      " dimensions");

  auto c = at::cov(self);

  if (c.ndimension() == 0) {
    // scalar covariance, return nan if c in {nan, inf, 0}, 1 otherwise
    return c / c;
  }

  // normalize covariance
  const auto d = c.diag();
  const auto stddev = at::sqrt(d.is_complex() ? at::real(d) : d);
  c = c / stddev.view({-1, 1});
  c = c / stddev.view({1, -1});
  
  // due to floating point rounding the values may be not within [-1, 1], so
  // to improve the result we clip the values just as NumPy does.
  return c.is_complex()
      ? at::complex(at::real(c).clip(-1, 1), at::imag(c).clip(-1, 1))
      : c.clip(-1, 1);
}
```
可以看到整体的逻辑和numpy是一样的。

整体逻辑为：

- 首先计算cov。
- 接着判断输入数据是否合法，不合法返回1。
- 接着获取其对角线元素。
- 然后判断是否是complex类型，如果是就对实部进行平方，不是的话就正常平方。
- 接着计算分母和numpy的方式一样，判断是否是complex类型。
- 最后使用clip将结果控制在[-1,1]。


# 四、对比分析
- 使用场景与功能：通过调用函数计算皮尔逊积矩相关系数。
- 实现对比：paddle的cov函数已经实现的相对完全，只需要和numpy一样，在cov的基础上进行修改，就可实现该功能,并且numpy是基础的库，pytorch中的实现也是1.10以上版本才有，其实现
  也是参考numpy，所以我们在实现的时候也需要参考numpy，在改动的方式上参考一些pytorch，由于pytorch是用c++写的，而且没有使用cuda加速，所以速度不会快很多，综合实现的复杂度和
  性能，使用python实现，参照numpy，模仿pytorch实现numpy的方式进行实现。



# 五、方案设计
## 命名与参数设计
API设计为`corrcoef(x,rowvar=True,ddof=False,name=None)`
命名与参数顺序为：形参名`x`->`x`,`rowvar`->`rowvar`,`ddof`->`ddof`,  与paddle的covAPI保持一致性，不影响实际功能使用。
- **x** (Tensor) - 一个N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数rowvar设置。
- **rowvar** (bool, 可选) - 若是True，则每行作为一个观测变量；若是False，则每列作为一个观测变量。默认True。
- **ddof** (bool, 可选) - 在计算中不起作用，不需要。默认False。
- **name** (str, 可选) - 与paddle其他API保持一致性，不影响实际功能使用。

## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案
主要按下列步骤进行组合实现,实现位置为`python/paddle/tensor/linalg.py`与`cov`等方法放在一起：
1. 由于在计算时，和ddof参数无关，所以在其设为TRUE时，进行警告，"ddof  have no effect"。
2. 使用`paddle.linalg.cov`得到协方差。
3. 使用`paddle.diag`获取对角线元素,如果发生值错误 返回1。
4. 对上述结果判断是否是complex类型，并使用`paddle.sqrt`求平方根。
5. 然后使用协方差分别除C{ii}，C{jj}。
6. 判断是否是complex类型，如果是，对实部虚部分别处理，不是就单独处理，最后用`paddle.clip`进行裁剪范围[-1，1]。
 
# 六、测试和验收的考量
测试考虑的case如下：

- 和numpy结果的数值的一致性, `paddle.linalg.corrcoef`和`numpy.corrcoef`的结果是否一致；
- x的输入值如果输入错误,在cov的时候会报错,所以无需实现值判断；
- 输入数据类型如果是complex类型也要输出正确的结果；
- ddof在实现时并不需要,如果用户输入,则发出警告；

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，且依赖的`paddle.cov`已经实现。工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无
