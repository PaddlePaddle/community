# paddle_ormqr 设计文档

| API 名称     | paddle.distribution.chi2 / paddle.distribution.LKJCholesky                     |
| ------------ | -------------------------------- |
| 提交作者     | Cmcamdy                     |
| 提交时间     | 2024-04-15                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20240415_api_design_for_Chi2_LKJCholesky.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，需要为飞桨扩充 API `paddle.distribution.chi2` 和 `paddle.distribution.LKJCholesky  `

本 API 属于飞桨开源个人贡献赛 API 开发任务[NO.5 为 Paddle 新增 Chi2 / LKJCholesky API](https://github.com/PaddlePaddle/Paddle/issues/62905)的任务。

## 2、功能目标

- 实现卡方分布。调用路径为:paddle.distribution.chi2，作为独立的函数调用。
- 实现相关矩阵的下三角 Choleskey 因子的 LJK 分布。调用路径为:paddle.distribution.LKJCholesky，作为独立的函数调用。

## 3、意义

为飞桨增加更多概率分布，提升飞桨 API 丰富度。

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

- PyTorch 中的 torch.distributions.chi2 [API文档](https://pytorch.org/docs/stable/distributions.html#chi2)
- PyTorch 中的 torch.distributions.LKJCholesky [API文档](https://pytorch.org/docs/stable/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky)
- Pytorch 中的 [对于分布的测试代码](https://github.com/pytorch/pytorch/blob/e3ac61587aa368c613ef01df1f328a396b64cd5d/test/distributions/test_distributions.py)

### 实现
```
class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """
    arg_constraints = {"df": constraints.positive}

    def __init__(self, df, validate_args=None):
        super().__init__(0.5 * df, 0.5, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Chi2, _instance)
        return super().expand(batch_shape, new)

    @property
    def df(self):
        return self.concentration * 2

```

# 四、对比分析
- paddle.distribution.chi2
在 Pytorch 的chi2是通过继承Gamma实现的，chi2分布实际上等价于Gamma(alpha=0.5*df, beta=0.5)，在Paddle中已经有了[Gamma](https://github.com/PaddlePaddle/Paddle/blob/fba5029777f79c289003a24dbf736fdb6465d92a/python/paddle/distribution/gamma.py#L24)可以用相似的思路构造一下即可。

----------------------------------------
- paddle.distribution.LKJCholesky


# 五、设计思路与实现方案

paddle 目前的算子已经支持矩阵的转置,行列计算等操作，因此，可以使用 paddle 已有算子实现 `ormqr` 。

## 命名与参数设计

添加 Python API:

```python
paddle.orqmr(input, tau, other, left=True, transpose=False)
```

参数表：

- input: (Tensor) shape（\*，mn，k），当 left 为 True 时， mn 的值等于 m，否则 mn 的值等于 n。 \*表示 Tensor 在轴 0 上的长度为 0 或者大于 0。
- tau: (Tensor) shape（\*，min（mn，k）），其中 \_ 表示 Tensor 在轴 0 上的长度为 0 或者大于 0，其类型与 input 相同。
- other: (Tensor) shape（\*，m，n），其中 \* 表示 Tensor 在轴 0 上的长度为 0 或者大于 0，其类型与 input 相同。
- left: (bool, 可选) 决定了矩阵乘积运算的顺序。如果 left 为 True ，计算顺序为 op(Q) ∗ other ，否则，计算顺序为 other \* op(Q)。默认值：True。
- transpose: (bool, 可选) 如果为 True ，对矩阵 Q 进行共轭转置变换，否则，不对矩阵 Q 进行共轭转置变换。默认值： False。

## 底层 OP 设计

不涉及底层 OP。

# 六、测试和验收的考量

paddle.distribution.chi2, paddle.distribution.LKJCholesky：
 - 正确性验证：可以与 Pytorch 的结果对齐；
   - 不同 shape；
   - 前向计算；
   - 计算dtype类型：验证 `float16`，`float32`，`float64`；
 - 不同计算设备：覆盖 CPU 和 GPU 等实现；
 - 错误检查：输入数据类型不支持。

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关 PythonAPI 均有实现，可以在开源贡献个人挑战赛期间完成。
2024/04/15 - 2024/04/20 完成 API 主体实现；
2024/04/20 - 2024/04/25 完成单测；


# 八、影响面

对其他模块暂无影响

