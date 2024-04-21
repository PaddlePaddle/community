# paddle.isposinf，paddle.isneginf，paddle.isreal设计文档

|API名称 | paddle.isposinf /paddle.isneginf /paddle.isreal | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-04-18 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240418_api_design_for_isposinf_isneginf_isreal.md<br> | 


# 一、概述
## 1、相关背景
[NO.10 为 Paddle 新增 isposinf / isneginf / isreal / isin API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/【Hackathon%206th】开源贡献个人挑战赛框架开发任务合集.md#no10-为-paddle-新增-isposinf--isneginf--isreal--isin-api)

## 2、功能目标
- 实现 paddle.isposinf 作为独立的函数调用，Tensor.isposinf(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为正无穷大。
- 实现 paddle.isneginf 作为独立的函数调用，Tensor.isneginf(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为负无穷大。
- 实现 paddle.isreal 作为独立的函数调用，Tensor.isreal(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为实值。


## 3、意义
新增 paddle.isposinf，paddle.isneginf，paddle.isreal 方法，丰富 paddle API

# 二、飞桨现状
对于 paddle.isposinf，paddle.isneginf，paddle 目前有相似的API paddle.isinf；
对于 paddle.isreal 目前有相似的API paddle.is_complex；

# 三、业内方案调研

### PyTorch
- PyTorch 中的 torch.isposinf [API文档](https://pytorch.org/docs/stable/generated/torch.isposinf.html#torch-isposinf)
- PyTorch 中的 torch.isneginf [API文档](https://pytorch.org/docs/stable/generated/torch.isneginf.html#torch-isneginf)
- PyTorch 中的 torch.isreal [API文档](https://pytorch.org/docs/stable/generated/torch.isreal.html#torch-isreal)

### Numpy
- Numpy 中的 numpy.isposinf [API文档](https://numpy.org/doc/stable/reference/generated/numpy.isposinf.html)
- Numpy 中的 numpy.isneginf [API文档](https://numpy.org/doc/stable/reference/generated/numpy.isneginf.html)
- Numpy 中的 numpy.isreal [API文档](https://numpy.org/doc/stable/reference/generated/numpy.isreal.html)

### 实现方法
- isposinf
    - pytorch
    ```cpp
    static void isposinf_kernel_impl(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); });
    });
    }
    ```
    - numpy
    ```python
    def isposinf(x, out=None):
        is_inf = nx.isinf(x)
        try:
            signbit = ~nx.signbit(x)
        except TypeError as e:
            dtype = nx.asanyarray(x).dtype
            raise TypeError(f'This operation is not supported for {dtype} values '
                            'because it would be ambiguous.') from e
        else:
            return nx.logical_and(is_inf, signbit, out)
    ```

- isneginf
    - pytorch
    ```cpp
    static void isneginf_kernel_impl(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); });
    });
    }
    ```
    - numpy
    ```python
    def isneginf(x, out=None):
        is_inf = nx.isinf(x)
        try:
            signbit = nx.signbit(x)
        except TypeError as e:
            dtype = nx.asanyarray(x).dtype
            raise TypeError(f'This operation is not supported for {dtype} values '
                            'because it would be ambiguous.') from e
        else:
            return nx.logical_and(is_inf, signbit, out)
    ```


- isreal
    - pytorch
    ```cpp
    Tensor isreal(const Tensor& self) {
    // Note: Integral and Floating tensor values are always real
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) ||
        c10::isFloatingType(self.scalar_type())) {
        return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
    }

    return at::imag(self) == 0;
    }
    ```
    - numpy
    ```python
    def isreal(x):
        return imag(x) == 0
    ```


# 四、对比分析

Numpy 中 isposinf，isneginf，isreal 的实现比较直接。可通过 Paddle 现有的 API 组合实现。


# 五、设计思路与实现方案

## 命名与参数设计
API `paddle.isposinf(x, name)`
paddle.isposinf
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回输入 `x` 的每一个元素是否为 `+INF` 。

API `paddle.isneginf(x, name)`
paddle.isneginf
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回输入 `x` 的每一个元素是否为 `-INF` 。

API `paddle.isreal(x, name)`
paddle.isreal
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回输入 `x` 的每一个元素是否为实数。


## 底层OP设计
用现有API组合实现

## API实现方案
1. paddle.isposinf
利用 paddle.isinf 与 paddle.signbit 组合实现。

2. paddle.isneginf
利用 paddle.isinf 与 paddle.signbit 组合实现。

3. paddle.isreal
利用Tensor数据类型判断和 paddle.imag 实现


# 六、测试和验收的考量

测试case：

paddle.isposinf，paddle.isneginf
- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算；
  - 计算dtype类型：验证 `float16`，`float32`，`float64`，`int8`，`int16`，`int32`，`int64`，`uint8`；
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入类型异常。

paddle.isreal：
- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算；
  - 计算dtype类型：验证 `float16`，`float32`，`float64`，`bool`，`int16`，`int32`，`int64`，`uint16`，`complex64`，`complex128`(paddle.ones_like 支持 `float16`，`float32`，`float64`，`bool`，`int16`，`int32`，`int64`，`uint16`)；
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入类型异常。

# 七、可行性分析和排期规划

2024/04/01 - 2024/04/07 完成 API 主体实现；
2024/04/08 - 2024/04/15 完成单测；

# 八、影响面
丰富 paddle API，对其他模块没有影响