功能名称： 增加paddle作为DeepXDE的 backend

开始日期：2022/03/10

GitHub Issue：[deepxde](https://github.com/lululxvi/deepxde/issues/559)

# 总结

增加paddle作为DeepXDE的 backend

# 使用指南

## requirements

- paddlepaddle-develop版本

# 开发说明

paddlepaddle暂时不支持`L-BFGS` 、`L-BFGS-B`

相关issue:
https://github.com/PaddlePaddle/Paddle/issues/38444
https://github.com/PaddlePaddle/Paddle/issues/36002

在需要L-BFGS降低Loss时，应予以注释

例如：

https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Burgers.py 案例中

```python
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
```

使用`L-BFGS`降低Loss。可修改为如下继续预测：
```python
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=15000)
```

# 未解决的问题

- 在训练时，如果损失函数为`L-BFGS`，`L-BFGS-B`，则会出现错误
- [deepxdeL31](https://github.com/lululxvi/deepxde/blob/master/deepxde/icbc/initial_conditions.py#L31)以及[L71](https://github.com/lululxvi/deepxde/blob/master/deepxde/icbc/boundary_conditions.py#L71)中判断tensor的维度，当tensor的维度为1时，会出现错误，问题来源于paddlepaddle中没有零维向量。暂行的结局方案是当tensor的维度为1时，则返回的ndim为0。如下是该问题的代码例子：
```python
import paddle
const =paddle.to_tensor(1) 
lists = paddle.to_tensor([1])
print(const.ndim) # 1
print(lists.ndim) # 1
import torch 
const = torch.tensor(1)
lists = torch.tensor([1])
print(const.ndim) # 0
print(lists.ndim) # 1
```

- net.apply_output_transform()出现错误
> 该问题定位于是paddle不支持部分算子的高阶导数，解决方式可以像torch一样返回1