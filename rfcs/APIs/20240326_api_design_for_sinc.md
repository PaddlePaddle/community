# paddle.sinc，paddle.sinc_ 设计文档

|API名称 | paddle.sinc /paddle.sinc_ | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-03-26 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240326_api_design_for_sinc.md<br> | 


# 一、概述
## 1、相关背景
[NO.7 为 Paddle 新增 sinc / sinc_ API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/【Hackathon%206th】开源贡献个人挑战赛框架开发任务合集.md#no7-为-paddle-新增-sinc--sinc_-api)

## 2、功能目标
1. 计算输入的归一化 sinc 函数。需要实现paddle.sinc、Tensor.sinc，及对应的 inplace 函数（paddle.sinc_、Tensor.sinc_）。

## 3、意义
新增 paddle.sinc，paddle.sinc_ 方法，丰富 paddle API

# 二、飞桨现状
飞桨框架目前不支持sinc，需新增kernel


# 三、业内方案调研

### PyTorch
PyTorch 中的 torch.sinc API文档 (https://pytorch.org/docs/stable/special.html#torch.special.sinc)
PyTorch 中的 Tensor.sinc_ API文档 (https://pytorch.org/docs/stable/generated/torch.Tensor.sinc_.html)

### Numpy
Numpy 中的 numpy.sinc API文档 (https://numpy.org/doc/stable/reference/generated/numpy.sinc.html)

底层实现：
- sinc
    pytorch
    ```cpp
    template <typename T>
    T sinc(T a) {
      if (a == T(0)) {
      return T(1);
      } else {
      constexpr T pi = T(3.14159265358979323846L);
      T product = pi * a;
      return std::sin(product) / product;
      }
    }
    ```
    numpy
    ```python
    def sinc(x):
      x = np.asanyarray(x)
      y = pi * where(x == 0, 1.0e-20, x)
      return sin(y)/y
    ```

# 四、对比分析

对比 PyTorch 与 Numpy:

- 实现方式不同

  PyTorch 通过 c++ 实现；Numpy 通过 python 实现。

- 实现逻辑类似

  PyTorch 是逐元素计算； Numpy 是用 API 在 ndarray 层面整体计算。


# 五、设计思路与实现方案

## 命名与参数设计
1. 
    ```python
    paddle.sinc(x, name=None)
    ```

    参数表：

    - x: (Tensor) 支持任意维度的 Tensor。数据类型支持 float16，float32，float64。
    - name: (Optional[str]) op 名称

2. 
    ```python
    paddle.sinc_(x, name=None)
    ```

    参数表：

    - x: (Tensor) 支持任意维度的 Tensor。数据类型支持 float16，float32，float64。
    - name: (Optional[str]) op 名称

## 底层OP设计

不涉及底层OP

## API实现方案

计算逻辑使用 paddle.sin 和 paddle.where 组合实现。

```python
def sinc(x, name=None):
  y = math.pi * paddle.where(x == 0, 1.0e-20, x)
  return paddle.sin(y)/y
```

```python
def sinc(x, name=None):
  paddle.where_(x != 0, x, paddle.full_like(x, 1.0e-20))
  paddle.multiply_(x, paddle.to_tensor(math.pi, dtype=x.dtype))
  paddle.sin_(x)
  tmp = paddle.asin(x)
  return paddle.divide_(x, tmp)
```


# 六、测试和验收的考量
测试case：

paddle.sinc, paddle.sinc_：
- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算；
  - 计算dtype类型：验证 `float16`，`float32`，`float64`；
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入数据类型不支持。

# 七、可行性分析和排期规划

2024/03/31 - 2024/04/07 完成 API 主体实现；
2024/04/08 - 2024/04/15 完成单测；

# 八、影响面

丰富 paddle API，对其他模块没有影响
