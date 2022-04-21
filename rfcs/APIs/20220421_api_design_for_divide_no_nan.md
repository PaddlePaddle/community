# paddle.divide_no_nan 设计文档

|API名称 | paddle.divide_no_nan | 
|---|---|
|提交作者 | Karshilov | 
|提交时间 | 2022-04-21 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20220421_api_design_for_divide_no_nan.md | 


# 一、概述
## 1、相关背景
[Issue41978](https://github.com/PaddlePaddle/Paddle/issues/41978)中提出了Feature Request，这个操作用于在发生除以0的事件时不返回NaN而是对齐tf中的对应api返回0
## 2、功能目标

在飞桨中实现对齐tf的divide_no_nan API

## 3、意义

飞桨中将直接支持divide_no_nan而不需要通过组合API实现本功能

# 二、飞桨现状
飞桨目前不支持此功能，但是可以通过组合API的方式实现此功能（在原issue下由[@wawltor](https://github.com/wawltor)提出）
```py
def divide_no_nan(x, y):
    z = x / y
    z_nan = paddle.isnan(z)
    z_inf = paddle.isinf(z)
    z_true = paddle.logical_or(z_nan, z_inf)
    z = paddle.where(z_true,  paddle.zeros_like(z), z)
    return z
```


# 三、业内方案调研
TensorFlow中已有`tensorflow.math.divide_no_nan`（文档详见[此处](https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan)）

python层的代码是:
```py
def div_no_nan(x, y, name=None):
  with ops.name_scope(name, "div_no_nan", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    return gen_math_ops.div_no_nan(x, y, name=name)
```

其中`gen_math_ops.div_no_nan`是来自于cpp的，代码是:
```cpp
template <typename T>
struct div_no_nan_op<T, /*IsComplex=*/true> {
  EIGEN_EMPTY_STRUCT_CTOR(div_no_nan_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b == T(0)) {
      return T(0);
    } else {
      // If the numerator is zero, then the result must be zero even if |b|^2
      // underflows to zero.
      const T numerator =
          scalar_product_op<T>()(a, scalar_conjugate_op<T>()(b));
      if (numerator == T(0)) {
        return T(0);
      }
    }
    return scalar_quotient_op<T>()(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    const Packet numerator = pmul(a, pconj(b));
    const Packet mask = por(pcmp_eq(b, pzero(a)), pcmp_eq(numerator, pzero(a)));
    const Packet quotient = pdiv(a, b);
    return pandnot(quotient, mask);
  }
};
```
# 四、对比分析

功能目的上，Feature Request仅需要一个效果和tf一致的接口，并且就运算量而言，本接口应该并没有一定要修改kernel的必要，社区的方案也支持直接将API组合的方式设计成官方API

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.divide_no_nan`，

- **x** (Tensor) - 被除数
- **y** (Tensor) - 除数

## 底层OP设计

方案主要依赖paddle现有op组合而成

## API实现方案

参考原issue下由[@wawltor](https://github.com/wawltor)提出的做法

# 六、测试和验收的考量

- 输入的`y`为0，返回0

# 七、可行性分析和排期规划

基本可仿照[@wawltor](https://github.com/wawltor)在原issue提出的做法，待文档通过验收后可短时间内提交

# 八、影响面
`paddle.divide_no_nan`受参与组合的API影响，不主动影响其他模块

# 名词解释

无

# 附件及参考资料

无