# paddle.ldexp 设计文档

|API名称 | paddle.ldexp                     | 
|---|----------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | longranger2                      | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-02-22                       | 
|版本号 | V1.0                             | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develp                           | 
|文件名 | 20230222_api_design_for_ldexp.md | 

# 一、概述
## 1、相关背景
为了提升飞桨API丰富度，Paddle需要扩充API `paddle.ldexp` 和 `paddle.Tensor.ldexp`。
任务 Issue: [No.6：为 Paddle 新增 ldexp API](https://github.com/PaddlePaddle/Paddle/issues/50630#task6)

## 2、功能目标
增加API `paddle.ldexp` 和 `paddle.Tensor.ldexp`，实现 `ldexp(x, y)`函数，函数返回 x 乘以 2 的 y 次方，因为 y 通常为整数，所以 2 的 y 次方为整数幂。

## 3、意义
飞桨将支持 `paddle.ldexp` API

# 二、飞桨现状

API方面，已有类似功能的API，`paddle.pow`, 在Paddle中是一个由多个其他API组合成的API，没有实现自己的OP，其主要实现逻辑为：

1. 使用`paddle.pow`对输入 y，进行以 2 为底数的指数运算，得到整数幂
2. 将步骤1得到的值乘以 x 即可


# 三、业内方案调研
## Pytorch
pytorch中有API `torch.ldexp(input, other, *, out=None) → Tensor`，文档链接
### 源代码
在实现方法上，PyTorch 是通过 C++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/71ad1005f66c9a53a2fe28d24b95c4e828aa944e/aten/src/ATen/native/BinaryOps.cpp#L1556-L1566)
```C++
Tensor& ldexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::mul_out(result, self, at::pow(2.0, other));
}

Tensor ldexp(const Tensor& self, const Tensor& other) {
  return at::mul(self, at::pow(2.0, other));
}

Tensor& ldexp_(Tensor& self, const Tensor& other) {
  return at::ldexp_out(self, self, other);
}
```

## Numpy
Numpy 中包含 API `ldexp(x1, x2, *args, **kwargs)`,
文档链接: [ldexp](https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html#numpy.ldexp)
### 源代码
在实现方法上，Numpy 是通过 C++ API 组合实现的，[代码位置](https://github.com/python/cpython/blob/1c49e61b9b18d550b9c5cff69a1dd3bb218e544a/Modules/mathmodule.c#L2042)
```C++

/*[clinic input]
math.ldexp

    x: double
    i: object
    /

Return x * (2**i).

This is essentially the inverse of frexp().
[clinic start generated code]*/

static PyObject *
math_ldexp_impl(PyObject *module, double x, PyObject *i)
/*[clinic end generated code: output=b6892f3c2df9cc6a input=17d5970c1a40a8c1]*/
{
    double r;
    long exp;
    int overflow;

    if (PyLong_Check(i)) {
        /* on overflow, replace exponent with either LONG_MAX
           or LONG_MIN, depending on the sign. */
        exp = PyLong_AsLongAndOverflow(i, &overflow);
        if (exp == -1 && PyErr_Occurred())
            return NULL;
        if (overflow)
            exp = overflow < 0 ? LONG_MIN : LONG_MAX;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "Expected an int as second argument to ldexp.");
        return NULL;
    }

    if (x == 0. || !Py_IS_FINITE(x)) {
        /* NaNs, zeros and infinities are returned unchanged */
        r = x;
        errno = 0;
    } else if (exp > INT_MAX) {
        /* overflow */
        r = copysign(Py_HUGE_VAL, x);
        errno = ERANGE;
    } else if (exp < INT_MIN) {
        /* underflow to +-0 */
        r = copysign(0., x);
        errno = 0;
    } else {
        errno = 0;
        r = ldexp(x, (int)exp);
        if (Py_IS_INFINITY(r))
            errno = ERANGE;
    }

    if (errno && is_error(r))
        return NULL;
    return PyFloat_FromDouble(r);
}
```

# 四、对比分析
## 不同点
- `pytorch` 调用现有的 `API` 实现的，而 `numpy` 是通过 `C++` 实现的

# 五、设计思路与实现方案

## 命名与参数设计
添加 API
```python
paddle.ldexp(
    x: N-D Tensor，数据类型为 float32、float64、int32 或 int64 
    y: N-D Tensor，通常为整数
)
```
注意：输出类型为float32或者float64，跟pytorch进行对齐
```python
>>> a = torch.ones([2], dtype=torch.float32)
>>> torch.ldexp(a, torch.tensor([-2])).dtype
torch.float32

>>> a = torch.ones([2], dtype=torch.float64)
>>> torch.ldexp(a, torch.tensor([-2])).dtype
torch.float64
```

## 底层OP设计
使用已有 API 组合实现，不再单独设计OP

## API实现方案
该 API 实现于 python/paddle/tensor/math.py，计算公式为：

$$
 x * 2^{y}
$$

通过调研发现，Paddle 本身已实现 paddle.pow 可以计算2的整数次幂函数，可利用paddle.pow API 与 输入 x 相乘实现 paddle.ldexp。
而 Paddle 中已有 paddle.pow API 的具体实现逻辑，位于 python/paddle/tensor/math.py 下的 pow 函数中。

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)
测试需要考虑的 case 如下：
- 输出数值结果的一致性和数据类型是否正确，使用 numpy 作为参考标准
- 参数 input 的数据类型准确性判断
- 参数 other 的数据类型准确性判断

## 七、可行性分析及规划排期
具体规划：
- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.ldexp` 和 `paddle.Tensor.ldexp` 的单元测试
- 阶段三：该API书写中英文档

# 八、影响面
对其他模块没有影响

# 名词解释

# 附件及参考资料
1. [numpy.ldexp](https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html#numpy.ldexp)
2. [torch.ldexp](https://pytorch.org/docs/stable/generated/torch.ldexp.html#torch.ldexp)