# paddle.optimizer.ASGD 设计文档


| API名称                                                      | paddle.optimizer.ASGD               |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | 我的名字连起来就是王豆豆            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-27                          |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                             |
| 文件名                                                       | 20220327_api_design_for_ASGD.md<br> |


# 一、概述
## 1、相关背景
对应 Issue：https://github.com/PaddlePaddle/Paddle/issues/40314

Averaged SGD 是一种对 SGD 优化算法的改进。可以证明[1]这个优化算法在理论上和使用 Hessian 矩阵的二阶随机梯度下降法有相同的收敛速度，但在实现上要比二阶随机梯度下降法简单的多。Averaged SGD 所做的事情和 SGD 完全一样，只是从某一次 iteration t0 之后，它会开始维护模型参数从 t0 时刻到现在的所有版本的平均值，并以这个平均值作为它优化得到的模型参数。

## 2、功能目标

在飞桨中增加 `paddle.optimizer.ASGD` 优化器

## 3、意义
飞桨用户将可以使用 `paddle.optimizer.ASGD` 优化器。

# 二、飞桨现状
飞桨目前不支持直接使用 ASGD 优化器，但用户仍可以自己用 SGD 优化器实现相同的功能，只需要在每次迭代时维护参数的平均值即可。但如果想用 ASGD 优化器取得理想的效果，一个合理的学习率策略非常重要[2]，所以如果能以开箱即用的方式提供 ASGD 优化器，对用户来说会是很好的体验。


# 三、业内方案调研
PyTorch 有 ASGD 优化器的实现，文档在 https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html ，代码在 https://github.com/pytorch/pytorch/blob/master/torch/optim/asgd.py 。

PyTorch 的实现其实是**有问题**的，它最初版的实现参考了 [bottou-sgd](https://github.com/npinto/bottou-sgd) （在 PyTorch ASGD 的[最初一个版本](https://github.com/pytorch/pytorch/commit/554a1d83365cf80d8676686e8fcc190c0c95d1a9)中有说明），当时的 PyTorch 开发者可能没有特别重视这个优化器，因此没有消化它的原理而是囫囵吞枣的照搬公式和术语，导致它的文档和代码出现了问题，如：文档中 “eta update” 中的 “eta” 其实就是学习率 lr；参数`lambd` 其实就是 l2 regularization 的系数，和 `weight_decay` 参数的功能是高度重叠的。

这里对照着 PyTorch ASGD 源码的逻辑把 [bottou-sgd](https://github.com/npinto/bottou-sgd) README 里的内容转述如下：

我们要优化一个带 l2 正则项的函数 Obj(w)，

![img](https://latex.codecogs.com/gif.latex?%5Clarge%20Obj%28w%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Clambda%20w%5E2%20&plus;%20loss%28w%29)

按照梯度下降法的规则，更新量是 Obj(w) 的梯度乘以学习率（学习率称为 eta_t）（latex 在 codecogs 上渲染的，没有很好的排版功能，见谅！）

![img](https://latex.codecogs.com/gif.latex?%5Clarge%20Obj%27%28w%29%20%3D%20%28lambda%20*%20w%20&plus;%20loss%27%28w%29%29%20*%20%5Ceta_t%20%3D%20lambda%20*%20w%20*%20%5Ceta_t%20&plus;%20w.grad%20*%20%5Ceta_t)

等号最右边的两项里，第一项 lambda * w * eta_t 对应于 PyTorch 实现的 188-189 行

```python
        # decay term
        param.mul_(1 - lambd * eta.item())
```

第二项 w.grad * eta_t 对应于 PyTorch 实现的 191-192 行

```python
        # update parameter
        param.add_(grad, alpha=-eta.item())
```

PyTorch 源码中接下来的 194-198 行和 202-203 行，是 on-the-fly 地计算平均值的经典算法，ax 在 t0 时刻之前是当前权重，在 t0 时刻之后是从 t0 时刻开始到当前为止的权重平均值。

```python
        # averaging
        if mu.item() != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)

        new_mu = torch.tensor(1 / max(1, step - t0))
        mu.copy_(new_mu)
```



200-201 行，是更新学习率（在 PyTorch 的实现里，eta 就是学习率，而 lr 是一个常量，专指用户设置的初始学习率）的策略，这个更新策略是照搬自 bottou-sgd，bottou-sgd 参考自 [2]。和其它的优化器不一样，ASGD 优化器的学习率并不能由 lr scheduler 控制，这可能也是它被不经消化地加入 PyTorch 的一个表现。

```python
        new_eta = torch.tensor(lr / math.pow((1 + lambd * lr * step), alpha))
        eta.copy_(new_eta)
```

再看看 PyTorch 的实现，除了上述的几段代码之外，还有一个名叫 weight_decay 的参数，在上面的推导里我们已经了解到，其实 lambda 就是 l2 正则项的系数，也就是 “weight decay”，因此不应该再有另一个 weight_decay 参数了。再看看相关的代码：

```python
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # decay term
        param.mul_(1 - lambd * eta.item())
        
        # update parameter
        param.add_(grad, alpha=-eta.item())
```

经过一些简单的数学变换，不难发现在这段代码里 `weight_decay` 和 `lambd` 的用法虽然形式上差异很大，但作用是完全一模一样的。

`lambd` 和 `weight_decay` 唯一不同的地方，是在 200-201 行学习率更新的策略里用到了 `lambd` 而没有用到 `weight_decay`。但 ASGD 作为一个优化方法并不应该和某种具体的学习率更新策略耦合。如果改由外部某个 lr scheduler 来控制 ASGD 的学习率，那么 ASGD 内的 `lambd` 和 `weight_decay` 就完全可以只留一个了。

PyTorch 的 ASGD 还同时存在着 single_tensor 和 multi_tensor 两种实现，其它 PyTorch 优化器也是一样。和 ASGD 本身无关。multi_tensor 使用了 PyTorch 的 foreach API，效率更高，但没有默认启用。

到现在，PyTorch 的代码已经分析完成，我们也明白了 ASGD 的实现：它和普通的 SGD 可以说完全一样，只是在 `ax` 里保存了一份权重的平均值而已。由于相关作者的囫囵吞枣，它和其它优化器的实现风格格格不入，这才阻碍了对它的理解。而 ASGD 并没有一定要使用某一种特定的学习率更新策略，举例来说，PyTorch ASGD 和 bottou-sgd 所用的学习率更新策略 [2] 是比 ASGD 本身 [1] 更晚提出的。这一点也可以从 [维基百科](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) 和 [这个课件](https://courses.cs.washington.edu/courses/cse547/18sp/slides/sgd_averaging.pdf) 对 ASGD 的描述里证实 —— ASGD 只是记录参数的平均值而已。

注意：PyTorch 和 TensorFlow 也实现了 Stochastic Weight Averaging，它和 Averaged SGD 并不是相同的概念。具体可以参考 https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch。

# 四、对比分析
经过上面的分析可以发现 PyTorch 的实现是很有问题的。在飞桨里它可以以更加优雅和一致的方式实现。

# 五、设计思路与实现方案

## 命名与参数设计
```python
class paddle.fluid.optimizer.ASGDOptimizer(learning_rate, parameter_list=None, regularization=None, name=None)
```

和飞桨中其它优化器的风格保持一致。weight_decay 通过 `regularization` 参数设置，支持 L1/L2 正则。而学习率用 LR Scheduler 来控制，不内置在优化器内。并新增一个 LRScheduler 实现 [2] 中提出的学习率更新策略（具体名字可以后续决定），用户也可以通过使用其它 LRScheduler 或者 LambdaLR，自由选择其它的学习率更新策略。



## 底层OP设计

基本可以仿照飞桨 SGD 优化器的实现，实现 paddle/fluid/operators/optimizers/asgd_op.cc 和相应的 asgd_kernel.h/.cc/.cu，注册 ASGDOP 和 CPU 与 CUDA 版的 ASGDOpKernel。

飞桨中 SGD 和 Adam 等常见的优化器的 CPU 版的实现是通过代码生成机制在运行时生成并加载的，而 ASGD 是较冷门的优化器，可以类似于飞桨中的 RMSProp 等优化器，通过 Eigen 库实现 CPU Kernel，通过 for_range + Functor 实现 CUDA Kernel 即可。

具体来讲，如上文所述，ASGD 和普通的 SGD 可以说完全一样，只是额外保存了一份权重沿时间的平均值而已。因此 CPU 版 Kernel 的伪代码如下：

```c++
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    const auto *param = ctx.Input<framework::Tensor>("Param");
    // 相比 SGD 优化器增加一个 AveragedParam 输入，表示该权重到目前为止的平均值
    const auto *averaged_param = ctx.Input<framework::Tensor>("AveragedParam");
    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    float regularization_coeff = ctx.Attr<float>("regularization_coeff");
    int64_t t0 = ctx.Attr<int64_t>("t0");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    // 相比 SGD 优化器增加一个 AveragedParamOut 输出，表示经过本次更新之后的该权重的新平均值
    auto *averaged_param_out = ctx.Output<framework::Tensor>("AveragedParamOut");
    const auto *grad = ctx.Input<framework::Tensor>("Grad");
    
    // 省略构造 EigenVector 的代码
    // ...
    
    auto &place =
        *execution_context.template device_context<DeviceContext>().eigen_device();
    if (regularization_method == "l2_decay") {
      param_out.device(place) = param * (1 - learning_rate * weight_decay); // 处理 weight decay
    } else if (regularization_method == "l1_decay") {
      ...
    }
    param_out -= learning_rate * grad;  // 梯度下降

    // 与普通 SGD 的关键区别，维护参数平均值：
    if (current_step() < t0) {
      averaged_param_out.device(place) = param_out;
    } else {
      // 与 PyTorch 中的方法相同，更新平均值
      averaged_param_out.device(place) = update_average(averaged_param, param_out, current_step(), t0);
    }
```

CUDA Kernel 的实现也将是类似的。

ASGD 优化器计划暂不支持 SelectedRows 等稀疏张量和 AMP，毕竟这个优化器实在是冷门，即使是用户量多如 PyTorch，它的 ASGD 优化器可能也没有用户真的使用过。

新增的 LR Scheduler 将是纯 Python 代码（和其它 LR Scheduler 相同），不涉及新增底层 OP。

## API实现方案

基本可以仿照 python/paddle/optimizer/sgd.py，只是把调用的 op 从 sgd 变成 asgd，并在 outputs 中增加一个输出 “AveragedParamOut”，并提供一个 `GetAveragedParameters` 方法。

# 六、测试和验收的考量
增加完善的测试和文档，本地测试和 PyTorch 的结果一致。构造基于 Paddle SGD、在 Python 中计算参数平均值的参考实现，作为 CI 对比中的 baseline。

# 七、可行性分析和排期规划
前两周：实现相关代码、测试用例和文档。

第三周：Code Review 和迭代 PR。

# 八、影响面
ASGD 对其它模块没有影响。PyTorch ASGD 的问题已经向 PyTorch 提交 issue：https://github.com/pytorch/pytorch/issues/74884

# 名词解释

# 附件及参考资料

[1] http://dl.acm.org/citation.cfm?id=131098 提出 ASGD 算法的 Paper

[2] https://arxiv.org/abs/1107.2490 提出 PyTorch 和 bottou-sgd 所用的学习率更新策略的 Paper

[3] https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/

[4] https://en.wikipedia.org/wiki/Stochastic_gradient_descent SGD 的维基百科，里面有介绍 Averaged SGD

[5] https://courses.cs.washington.edu/courses/cse547/18sp/slides/sgd_averaging.pdf 讲述 ASGD 原理的课件

[6] https://github.com/npinto/bottou-sgd/blob/master/README.txt bottou-sgd README
