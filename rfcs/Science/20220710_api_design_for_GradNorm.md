# PaddleScience.network.GradNorm设计文档

| API名称                                    | paddlescience.network.GradNorm          |
| ---------------------------------------- | --------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden"> | Asthestarsfalll                         |
| 提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-10                              |
| 版本号                                      | V1.0                                    |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                 |
| 文件名                                      | 20220710_api_design_for_GradNorm.md<br> |

# 一、概述

## 1、相关背景

PINNs方法中损失函数由 PDE Loss、初值 Loss、边值 Loss及 data Loss 组成，其中每项损失函数有不同的权重。

[GradNorm](https://arxiv.org/abs/1711.02257)方法可以根据损失对各个任务的梯度来动态分配损失的权重，从而达到均衡各个损失梯度、稳定不同任务收敛速度的效果。

顾名思义，GradNorm即为利用梯度来对梯度进行归一化，论文中选用了所有损失的最后一层共享层，记为$W$。

其整体计算流程如下：

首先初始化可学习的损失权重，记为$w$，选取超参$\alpha$ 用于控制学习速度， $\alpha$ 越大，对训练速度的平衡限制越强。

在训练过程的第一步时，储存所有loss，记为$L(0)$，用于后续步骤。

1. 求和所有loss$\sum_iL_i(t)$，反向传播，获得所有参数的梯度，这一步需要保留计算图；
2. 取出最后一个共享层的参数$w$，分别求解每个loss相对于$W$的梯度$ \nabla WL_i(t)$ ，与对应的权重相乘取l2范数：$||\nabla Ww_i(t)L_i(t)||_2$， 记为$G^{(i)}_w(t) $ ；
3. 计算当前loss相对于初始loss的比例：$\widetilde{L}_i=L_i(t)/L_i(0)$ ，计算相对反向训练速度（inverse training rate）$r_i(t)=\widetilde{L}_i/E(\widetilde{L}_i)$ ，这里需要将其当做一个常数，因此在实现过程中需要转换为numpy计算;
4. 计算$\overline{G}_W(t) =E(G_W^i(t))$，此处同样作为常数。
5. 计算Norm Loss：$\sum_i|G_W^i(t)-\overline{G}_W(t)\times (r_i(t))^{\alpha}|_1$；
6. 通过Norm Loss求解相对于相对于$w$的梯度，手动更新$w$的梯度；
7. 更新参数即可。

## 2、功能目标

为 PaddleScience 新增 GradNorm Loss 权重自适应功能，集成入 PaddleScience 作为 API 调用。

## 3、意义

使得PaddleScience支持损失函数权重自适应的功能。

# 二、飞桨现状

Paddle拥有实现GradNorm所需的基础API。

# 三、业内方案调研

## pytorch-grad-norm

核心代码如下：

[初始化权重及获取最后一层共享层](https://github.com/brianlan/pytorch-grad-norm/blob/master/model.py#L24-L43)：

```python
class RegressionTrain(torch.nn.Module):
    def __init__(self, model):
        # initialize the module using super() constructor
        super(RegressionTrain, self).__init__()
        # assign the architectures
        self.model = model
        # assign the weights for each task
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        # loss function
        self.mse_loss = MSELoss()

    def forward(self, x, ts):
        B, n_tasks = ts.shape[:2]
        ys = self.model(x)
        
        # check if the number of tasks is equal to this size
        assert(ys.size()[1] == n_tasks)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append( self.mse_loss(ys[:,i,:], ts[:,i,:]) )
        task_loss = torch.stack(task_loss)

        return task_loss

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()
```

[Norm Loss计算过程](https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py#L67-L138)：

```python
            if t == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                initial_task_loss = initial_task_loss.numpy()

            # get the total loss
            loss = torch.sum(weighted_task_loss)
            # clear the gradients
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            # This is equivalent to compute each \nabla_W L_i(t)
            loss.backward(retain_graph=True)

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            #print('Before turning to 0: {}'.format(model.weights.grad))
            model.weights.grad.data = model.weights.grad.data * 0.0
            #print('Turning to 0: {}'.format(model.weights.grad))


            # switch for each weighting algorithm:
            # --> grad norm
            if args.mode == 'grad_norm':
                
                # get layer of shared weights
                W = model.get_last_shared_layer()

                # get the gradient norms for each of the tasks
                # G^{(i)}_w(t) 
                norms = []
                for i in range(len(task_loss)):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    # compute the norm
                    norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
                norms = torch.stack(norms)
                #print('G_w(t): {}'.format(norms))


                # compute the inverse training rate r_i(t) 
                # \curl{L}_i 
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                #print('r_i(t): {}'.format(inverse_train_rate))


                # compute the mean norm \tilde{G}_w(t) 
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                #print('tilde G_w(t): {}'.format(mean_norm))


                # compute the GradNorm loss 
                # this term has to remain constant
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                #print('Constant term: {}'.format(constant_term))
                # this is the GradNorm loss itself
                grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
                #print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
```

# 四、对比分析

实现清晰，与论文一致，按照上述代码实现即可。

# 五、设计思路与实现方案

## 命名与参数设计

API设计为`paddlescience.network.GradNorm(net, n_loss, alpha, weight_attr=None)`

## 底层OP设计

无需设计底层OP。

## API实现方案

paddlescience.network.GradNorm实现于paddlescience\network\grad_norm.py文件中，初步实现如下：

```python
import numpy as np
import paddle
import paddle.nn
from paddle.nn.initializer import Assign
from .network_base import NetworkBase

class GradNorm(NetworkBase):
    r"""
    Gradient normalization for adaptive loss balancing.
    Parameters:
        net(NetworkBase): The network which must have "get_shared_layer" method.
        n_loss(int): The number of loss, must be greater than 1.
        alpha(float): The hyperparameter which controls learning rate.
        weight_attr(list, tuple): The inital weights for "loss_weights". If not specified, "loss_weights" will be initialized with 1.
    """
    def __init__(self, net, n_loss, alpha, weight_attr=None):
        super().__init__()
        if n_loss <= 1:
            raise ValueError("'n_loss' must be greater than 1, but got {}".format(n_loss))
        if alpha < 0:
            raise ValueError("'alpha' is a positive number, but got {}".format(alpha))
        if weight_attr is not None:
            if len(weight_attr) != n_loss:
                raise ValueError("weight_attr must have same length with loss weights.")

        self.net = net
        self.loss_weights = self.create_parameter(
            shape=[n_loss], attr=Assign(weight_attr if weight_attr else [1] * n_loss),
            dtype=self._dtype, is_bias=False)
        self.set_grad()
        self.alpha = float(alpha)
        self.initial_losses = None

    def nn_func(self, ins):
        return self.net.nn_func(ins)

    def __getattr__(self, __name):
        try:
            return super().__getattr__(__name)
        except:
            return getattr(self.net, __name)

    def get_grad_norm_loss(self, losses):
        if isinstance(losses, list):
            losses = paddle.concat(losses)
        if self.initial_losses is None:
            self.initial_losses = losses.numpy()

        W = self.net.get_last_shared_layer()
        if self.loss_weights.grad is not None
            self.loss_weights.grad.set_value(paddle.zeros_like(self.loss_weights))

        norms = []
        for i in range(losses.shape[0]):
            grad = paddle.autograd.grad(losses[i], W, retain_graph=True)
            norms.append(paddle.norm(self.loss_weights[i]*grad[0], p=2))
        norms = paddle.concat(norms)

        loss_ratio = losses.numpy() / self.initial_losses
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        mean_norm = np.mean(norms.numpy())

        constant_term = paddle.to_tensor(mean_norm * np.power(inverse_train_rate, self.alpha), dtype=self._dtype)

        grad_norm_loss = paddle.norm(norms - constant_term, p=1)
        self.loss_weights.grad.set_value(paddle.autograd.grad(grad_norm_loss, self.loss_weights)[0])

        return grad_norm_loss

    def reset_initial_losses(self):
        self.initial_losses = None

    def set_grad(self):
        x = paddle.ones_like(self.loss_weights)
        x *= self.loss_weights
        x.backward()
        self.loss_weights.grad.set_value(paddle.zeros_like(self.loss_weights))

    def get_weights(self):
        return self.loss_weights.numpy()
```

这里对实现的要点进行说明：

1. GradNorm接受一个网络实例，需要为`NetworkBase`的子类，且必须要有`get_shared_layer`方法，该方法可以返回任意共享层，论文中为最后一层。为了拓展功能，后续可以考虑如何开放接口让用户手动选择。
2. 需要输入loss的数量，当`weight_attr`未指定时，权重会全部初始化未1，否则按照给定权值初始化。
3. 重写了`__getattr__`方法，可以通过`.`直接调用传入网络的方法，使得用户在使用GradNorm前后都可以使用相同的方式调用网络的方法，而无需修改。
4. `set_grad()` 用于初始化权重的梯度，由于在paddle中无法直接对梯度使用赋值语句，需要使用`Tensor.grad.set_value()`，而刚初始化时grad为None，无法调用set_value方法，因此需要使用`backward()` 来获得初始权重。

# 六、测试和验收的考量

1. 保证前向反向与论文中所描述的一致；
2. 保证训练、测试等流程不出现错误。

# 七、可行性分析和排期规划

已完成部分工作，规定时间内可全部完成。

# 八、影响面

需要为网络添加一个`get_shared_layer()`方法。

此外当`GradNorm` 拥有与传入网络实例相同名称的属性或方法时，以GradNorm为准，可能会对已有代码造成一些影响，修改后即可使用。

# 名词解释

无

# 附件及参考资料

[GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/pdf/1711.02257.pdf)
