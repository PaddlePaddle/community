## Inplace 介绍 & 使用介绍
-------------
#### 什么是Inplace

在深度学习框架中，以y = op(x) 为例，该op可分为三种：
* view 形式的op:  op 操作后y 与 x 共享相同的数据， 但有不同的元数据(shape/stride) ，y与x是两个变量。
* inplace 形式的op:  op 操作后 y 与 x 共享相同的数据，且有相同的元数据(shape/stride)，y与x是同一个变量。
* 普通的op：op操作后 y 与 x 是不同的数据，有不同的元数据(shape/stride)，y与x是两个变量。



Inplace操作可以带来两个好处：
* 减少网络训练过程中的显存开销。
  * 例如，y = op(x)，如果x与y不进行Inplace，则显存开销为2倍于x的存储空间（y与x大小相同）；如果采用Inplace操作，则只需要1倍于x的存储空间。
* 减少显存分配和数据拷贝等的时间开销。
  * 在Inplace的情况下，输出不需要重新分配空间，减少了这部分代码逻辑的开销。
  * 原来，部分输入和输出Tensor Buffer相同的op（如reshape）是通过Tensor Copy完成的；支持Inplace后，这部分Tensor Copy可以省略，可以减少D2D数据拷贝的开销。


#### Paddle Inplace的使用
-------------
Paddle 的inplace api 都会以 '\_' 收尾，例如 `paddle.add_`, `paddle.abs_` 等，inplace 版本的 api 除了会在输入上进行原地操作以外，与非inplace版本的 api 本身没有区别,
除了支持以paddle 开头的api 形式以外，也同样支持`Tensor.add_` 即  `x.add_`这种形式的调用，而后者是更为常见的形式。

```python
import paddle
a = paddle.randn([3, 4])
a.stop_gradient=False
x = a.scale(2.0)
print(x)
# Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[ 0.21718574,  2.82289553, -4.42365265, -4.73598146],
#         [-5.24532938, -4.69758368, -1.76940155,  4.37858629],
#         [ 0.04942622,  2.00082755, -3.87005639,  2.72555494]])
print(id(x))
#140110962616016
y = x.scale_(2.0)
print(y)
# Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[ 0.43437147 ,  5.64579105 , -8.84730530 , -9.47196293 ],
#         [-10.49065876, -9.39516735 , -3.53880310 ,  8.75717258 ],
#         [ 0.09885245 ,  4.00165510 , -7.74011278 ,  5.45110989 ]])
print(id(y))
# 140110962616016
print(y is x)
# True
print(x)
# Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[ 0.43437147 ,  5.64579105 , -8.84730530 , -9.47196293 ],
#         [-10.49065876, -9.39516735 , -3.53880310 ,  8.75717258 ],
#         [ 0.09885245 ,  4.00165510 , -7.74011278 ,  5.45110989 ]])
y.backward()
print(a.grad)
# Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[4., 4., 4., 4.],
#         [4., 4., 4., 4.],
#         [4., 4., 4., 4.]])
```
通过上面的例子可以看出，`y = x.scale_(2.0)` 这个操作实际是x原地的进行了一个`scale` 操作，并没有为y分配额外的存储空间，且此时也是可以正常的进行反向传播的

这里可能会有一个疑问，为什么不直接在a上进行inpalce操作`a.scale_(2.0)`呢，如果按照这个来
```python
import paddle
a = paddle.randn([3, 4])
a.stop_gradient=False
x = a.scale_(2.0)
# ValueError: (InvalidArgument) Leaf Var () that doesn't stop gradient can't use inplace strategy.
```
至于为什么叶子结点不能直接进行inpalce操作呢，就由大家自己去发掘了
