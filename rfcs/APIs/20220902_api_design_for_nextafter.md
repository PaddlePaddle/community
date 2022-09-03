# paddle.Tensor.nextafter 设计文档

| API名称      | paddle.nextafter                     |
| ------------ | ------------------------------------ |
| 提交作者     | 小张1998                             |
| 提交时间     | 2022-09-02                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | v2.2.0                               |
| 文件名       | 20200819_api_design_for_nextafter.md |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.nextafter`

## 2、功能目标

增加API`paddle.nextafter`，将输入后的下一个浮点值返回给其他元素，输入和其他 shape 必须是可广播的。

## 3、意义

Paddle 将可以使用 `paddle.nextafter` 找到当前输入的下一个浮点值，丰富了API。

# 二、飞桨现状

- 目前paddle缺少相关功能实现，因此可以使用c，或者几个飞桨api算子结合的办法来进行实现。

# 三、业内方案调研

**Java.Math：**

主要的逻辑在于

`int transducer = Float.floatToRawIntBits(start + 0.0f);` 和 `transducer = transducer + (transducer >= 0 ? 1:-1);`

其实就是在浮点数的尾数根据需要上+1或-1

```java
public static float nextAfter(float start, double direction) {
    /*
         * The cases:
         *
         * nextAfter(+infinity, 0)  == MAX_VALUE
         * nextAfter(+infinity, +infinity)  == +infinity
         * nextAfter(-infinity, 0)  == -MAX_VALUE
         * nextAfter(-infinity, -infinity)  == -infinity
         *
         * are naturally handled without any additional testing
         */

    // First check for NaN values
    if (Float.isNaN(start) || Double.isNaN(direction)) {
        // return a NaN derived from the input NaN(s)
        return start + (float)direction;
    } else if (start == direction) {
        return (float)direction;
    } else {        // start > direction or start < direction
        // Add +0.0 to get rid of a -0.0 (+0.0 + -0.0 => +0.0)
        // then bitwise convert start to integer.
        int transducer = Float.floatToRawIntBits(start + 0.0f);

        /*
             * IEEE 754 floating-point numbers are lexicographically
             * ordered if treated as signed- magnitude integers .
             * Since Java's integers are two's complement,
             * incrementing" the two's complement representation of a
             * logically negative floating-point value *decrements*
             * the signed-magnitude representation. Therefore, when
             * the integer representation of a floating-point values
             * is less than zero, the adjustment to the representation
             * is in the opposite direction than would be expected at
             * first.
             */
        if (direction > start) {// Calculate next greater value
            transducer = transducer + (transducer >= 0 ? 1:-1);
        } else  { // Calculate next lesser value
            assert direction < start;
            if (transducer > 0)
                --transducer;
            else
                if (transducer < 0 )
                    ++transducer;
            /*
                     * transducer==0, the result is -MIN_VALUE
                     *
                     * The transition from zero (implicitly
                     * positive) to the smallest negative
                     * signed magnitude value must be done
                     * explicitly.
                     */
            else
                transducer = FloatConsts.SIGN_BIT_MASK | 1;
        }

        return Float.intBitsToFloat(transducer);
    }
}
```

**numpy:**

底层是调用C++算子来实现

`numpy.``nextafter`(*x1*, *x2*, */*, *out=None*, ***, *where=True*, *casting='same_kind'*, *order='K'*, *dtype=None*, *subok=True*[, *signature*, *extobj*]) *= <ufunc 'nextafter'>*

Return the next floating-point value after x1 towards x2, element-wise.

| Parameters: | **x1** : array_likeValues to find the next representable value of.**x2** : array_likeThe direction where to look for the next representable value of *x1*.**out** : ndarray, None, or tuple of ndarray and None, optionalA location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or *None*, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.**where** : array_like, optionalValues of True indicate to calculate the ufunc at that position, values of False indicate to leave the value in the output alone.***\*kwargs**For other keyword-only arguments. |
| :---------- | ------------------------------------------------------------ |
| Returns:    | **out** : ndarray or scalarThe next representable values of *x1* in the direction of *x2*. This is a scalar if both *x1* and *x2* are scalars. |

# 四、对比分析

其实最直观的做法是像Java.Math一样，把浮点数转换成二进制形式，对尾数进行操作，根据第二个参数和第一个参数的大小关系，判断尾数是+1或-1或不变化

编写C++算子的话，其实只需要把指针 reinterpret_cast<int32*> 或 reinterpret_cast<int64*>，然后根据方向进行+1或-1，修改内存之后返回指向的值（float或double）即可

# 五、方案设计

## 命名与参数设计

- 函数名称: paddle.nextafter(start, direction)
- 功能描述: paddle.nextafter(start, direction), 将输入后的下一个浮点值返回给其他元素
- 输入参数
  - start: 任意数值, 类型是paddle.float32或paddle.float64
  - direction:任意大于等于a的数值, 类型是paddle.float32或paddle.float64
- 返回值:
  - 和start类型的返回值

## 底层OP设计

不涉及底层op

## API实现方案

1. 在 python/paddle/tensor/math.h 里添加 def nextafter(start, direction)

   由于需要进行广播操作，可以借助 `paddle.broadcast_tensors()` 

```python
def vnextafter(start, direction):
    """
    Returns the next representable value after start in the direction of direction.

    Args:
    	start(Tensor): Base value
    	direction(Tensor):Value toward which the return value is approximated
    
    Returns:
        Tensor: The next representable value after start in the direction of direction.
    If both parameters compare equal, the function returns start.
    
    Example:
        .. code-block:: python
            
            import paddle
            
            # start is a Tensor of shape [2, 1, 3]
            start = paddle.rand([2, 1, 3])
            
            direction = paddle.rand([2, 1, 3])
            out = paddle.nextafter(start, direction)
            print(out.shape) # [2, 1, 3]
            
            direction = paddle.rand([4, 1])
            out = paddle.nextafter(start, direction)
            print(out.shape) # [2, 4, 3]
            
            direction = paddle.rand([1, 3])
            out = paddle.nextafter(start, direction)
            print(out.shape) # [2, 1, 3]
    """
    
    out_start, out_direction = paddle.broadcast_tensors(input=[start, direction])
    
    return nextafter(out_start, out_direction)
```

2. 在 python/paddle/tensor/\__init__.py 中导入 nextafter

3. 添加 python/paddle/fluid/tests/test_nextafter_api.py 进行测试

# 六、测试和验收的考量

- 模型使用 paddle 已经实现的api进行组合，因此一下场景无需考虑
  - 硬件场景
  - 编程范式场景
- ensor 精度场景
  - 支持 FP32、FP64

- 计算精度：
  - 前向计算：通过 numpy.nextafter 实现的函数的对比结果。
  - 反向计算：无需考虑

- 异常测试：需对于参数异常值输入，应该有友好的报错信息及异常反馈。
- 参数组合场景：常规覆盖 API 的全部入参，需要对全部入参进行参数有效性和边界值测试，同时可选参数也需有相应的测试覆盖。

# 七、可行性分析及规划排期

实现阶段：实现方案较为简单，可以快速实现 书写中英文文档：英文文档需要花费更多时间，但仍在合理范围之内

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无