# paddle.tril_indices 设计文档

| API 名称     | paddle.tril_indices                |
| ------------ | ---------------------------------- |
| 提交作者     | 小猫                               |
| 提交时间     | 2022-03-28                         |
| 版本号       | V-0.1                              |
| 依赖飞桨版本 | develop                            |
| 文件名       | 20220328_design_for_fsl_metaopt.md |

# 一、概述

## 1、相关背景

Meta Learning 旨在通过学习不同的任务，从而拥有快速学习的能力，此过程也是在模拟人类学习的过程。

## 2、功能目标

重新设计 PaddleFSL 中 MAML 和 ANIL 算法，使其可转化为任务模型可调用的算法，在框架层面实现开箱即用，不添加任何依赖性的库，最终可直接用于 Paddle、PaddleNLP、PaddleSpeech 等库中的模型。

## 3、意义

MetaLearning 作为一种模型训练范式，如果 Paddle 开源生态中的其它项目可直接使用，将会给其它项目提供更多的研究和工程落地的工具，同时也能够形成一种良性促进，与其它项目之间共同发展和完善。

# 二、飞桨现状

MAML 是一个 Gradient Descent Based 的优化方法，可是在 FSL Meta 库当中将整个数据集、模型训练过程全部绑定到一起，无法实现插件化以及开箱即用的功能，换言之无法与其他模型配合使用，方法的封装欠缺扩展性，无法发挥 MAML 的作用，以下是目前 FSL Meta 库当中当前的方法。

```python
# refer:
def meta_training(train_dataset,
                  valid_dataset,
                  model,
                  meta_lr=0.002,
                  inner_lr=0.4,
                  iterations=60000,
                  meta_batch_size=32,
                  ways=5,
                  shots=5,
                  inner_adapt_steps=1,
                  approximate=True,
                  report_iter=10,
                  save_model_iter=5000,
                  save_model_root='~/paddlefsl_models'):
    # Set training configuration information and
    module_info = utils.get_info_str('maml', train_dataset, model, str(ways) + 'ways', str(shots) + 'shots')
    train_info = utils.get_info_str('metaLR' + str(meta_lr), 'innerLR' + str(inner_lr),
                                    'batchSize' + str(meta_batch_size), 'adaptSteps' + str(inner_adapt_steps),
                                    'approximate' if approximate else '')
    # Make directory to save report and parameters
    module_dir = utils.process_root(save_model_root, module_info)
    train_dir = utils.process_root(module_dir, train_info)
    report_file = train_dir + '/training_report.txt'
    utils.clear_file(report_file)
    # Set dataset, meta optimizer and loss function
    model.train()
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=meta_lr, T_max=iterations)
    meta_opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler)
    loss_fn = paddle.nn.CrossEntropyLoss()
    # Meta training iterations
    for iteration in range(iterations):
        # Clear gradient, loss and accuracy
        meta_opt.clear_grad()
        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0
        for task_i in range(meta_batch_size):
            # Clone a model in order to keep connected with the original computation graph
            model_cloned = utils.clone_model(model)
            # Sample a task from dataset
            task = train_dataset.sample_task_set(ways=ways, shots=shots)
            # Do inner adaptation
            data = (task.support_data, task.support_labels), (task.query_data, task.query_labels)
            inner_valid_loss, inner_valid_acc = inner_adapt(model=model_cloned,
                                                            data=data,
                                                            loss_fn=loss_fn,
                                                            inner_lr=inner_lr,
                                                            steps=inner_adapt_steps,
                                                            approximate=approximate)
            # Renew original model parameters using inner validation loss
            inner_valid_loss.backward(retain_graph=True)
            # Accumulate inner validation loss and inner validation accuracy
            train_loss += inner_valid_loss.numpy()[0]
            train_acc += inner_valid_acc
            # Do the same adaptation using validation dataset
            if (iteration + 1) % report_iter == 0 or iteration + 1 == iterations:
                model_cloned = utils.clone_model(model)
                task = valid_dataset.sample_task_set(ways, shots)
                data = (task.support_data, task.support_labels), (task.query_data, task.query_labels)
                loss_acc = inner_adapt(model_cloned, data, loss_fn, inner_lr, inner_adapt_steps * 2, approximate)
                valid_loss += loss_acc[0].numpy()[0]
                valid_acc += loss_acc[1]
        meta_opt.step()
        scheduler.step()
        # Average the accumulation through batches
        train_loss, train_acc = train_loss / meta_batch_size, train_acc / meta_batch_size
        valid_loss, valid_acc = valid_loss / meta_batch_size, valid_acc / meta_batch_size
        # Print report and save report
        if (iteration + 1) % report_iter == 0 or iteration + 1 == iterations:
            utils.print_training_info(iteration + 1, train_loss, train_acc, valid_loss, valid_acc,
                                      report_file=report_file, info=[module_info, train_info])
        # Save model parameters
        if (iteration + 1) % save_model_iter == 0 or iteration + 1 == iterations:
            paddle.save(model.state_dict(), train_dir + '/iteration' + str(iteration + 1) + '.params')
    return train_dir
```

```python
# refer to:  https://github.com/tata1661/FSL-Mate/blob/master/PaddleFSL/examples/image_classification/maml_image_classification.py
    train_dir = maml.meta_training(train_dataset=TRAIN_DATASET,
                                   valid_dataset=VALID_DATASET,
                                   ways=WAYS,
                                   shots=SHOTS,
                                   model=MODEL,
                                   meta_lr=META_LR,
                                   inner_lr=INNER_LR,
                                   iterations=ITERATIONS,
                                   meta_batch_size=META_BATCH_SIZE,
                                   inner_adapt_steps=TRAIN_INNER_ADAPT_STEPS,
                                   approximate=APPROXIMATE,
                                   report_iter=REPORT_ITER,
                                   save_model_iter=SAVE_MODEL_ITER,
                                   save_model_root=SAVE_MODEL_ROOT)
```

通过以上代码可看出，使用 MAML 的方法过于简单粗暴，无法实现良好的扩展。

# 三、业内方案调研

Pytorch 开源项目当中有两个非常好的库： [learn2learn](https://github.com/learnables/learn2learn), [higher](https://github.com/facebookresearch/
higher).

## Learn2Learn

此库对于优化方法的封装比较好，只可惜已经很长时间没有更新了。此库当中对于优化算法以`Learner`的概念呈现，不同的优化算法可封装为`Learner`的派生类。其核心代码如下所示：

```python
class BaseLearner(nn.Module):

    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.module = module

    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class MAML(BaseLearner):
    def __init__(self,
                 model,
                 lr,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0
            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')
        # Update the module
        self.module = maml_update(self.module, self.lr, gradients)
    
    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)
```

和其它模型的使用方法如下所示：

```python
# 1. 定义Meta Model
meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
for _ in tqdm_bar:
    iteration_error = 0.0
    iteration_acc = 0.0
    for _ in range(tps):
        # Clone 学习器
        learner = meta_model.clone()
        train_task, valid_task = train_gen.sample(), validation_gen.sample()

        # 进行多轮学习，并调整根据损失来调整模型梯度
        for _ in range(fas):
            train_error, _ = compute_loss(train_task, roberta, device, learner, loss_func, batch=shots * ways)
            learner.adapt(train_error)

        # 计算验证集上面的损失
        valid_error, valid_acc = compute_loss(valid_task, roberta, device, learner, loss_func,
                                              batch=shots * ways)
        iteration_error += valid_error
        iteration_acc += valid_acc

    iteration_error /= tps
    iteration_acc /= tps
    tqdm_bar.set_description("Loss : {:.3f} Acc : {:.3f}".format(iteration_error.item(), iteration_acc))
    accs.append(iteration_acc)
    
    # 执行MetaLearning部分
    opt.zero_grad()
    iteration_error.backward()
    opt.step()
```

## Higher

在此库当中，Gradient Descent的方法都是直接封装于Optimzier当中，与平常的优化器方法使用一样，只是通过参数来控制不同的处理逻辑。

```python
refer to: https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py#L137
# 清空Grad
meta_opt.zero_grad()
for i in range(task_num):

    # Clone Model & 构造 Learner
    with higher.innerloop_ctx(
        net, inner_opt, copy_initial_weights=False
    ) as (fnet, diffopt):
        
        # 在支撑集上计算损失并执行学习器步
        for _ in range(n_inner_iter):
            spt_logits = fnet(x_spt[i])
            spt_loss = F.cross_entropy(spt_logits, y_spt[i])
            diffopt.step(spt_loss)

        # 在查询集上面的损失同时也会应用于模型的参数更新上
        qry_logits = fnet(x_qry[i])
        qry_loss = F.cross_entropy(qry_logits, y_qry[i])
        qry_losses.append(qry_loss.detach())
        qry_acc = (qry_logits.argmax(
            dim=1) == y_qry[i]).sum().item() / querysz
        qry_accs.append(qry_acc)
        qry_loss.backward()
meta_opt.step()
```

# 四、对比分析

此两者的设计各有优缺点：

* Learn2Learn在设计上扩展性更高，侵入性也更小，能够在更小程度上与其他库进行融合使用
* Higher 使用Context 来创建优化方法，大大减小了代码量。

# 五、设计思路与实现方案

通过分析对比`learn2learn`和`higher`两个库的优缺点，准备采用以下实现方案。

> 当然以下是初步设计思路，更细节的实现方案可在后续的沟通过程中确定。

## 命名与参数设计

针对于MAML和ANIL两个优化方法，API接口设计为：`paddlefsl.metaopt.maml`和`paddlefsl.metaopt.anil`。

```python

class BaseLearner(ABC):
    """Abstract Base Learner Class"""
    def __init__(self, module: Layer) -> None:
        """The constructor of BaseLearner

        Args:
            module (Layer): the model to be trained
        """
        super().__init__()
        self._source_module = module

    @property
    def model(self,) -> Layer:
        """get the cloned model and keep the computation gragh

        Returns:
            Layer: the cloned model
        """
        return clone_model(self._source_module)

    @abstractmethod
    def adapt(self, train_loss: Tensor) -> None:
        """Adapt the model to the current training loss

        Args:
            train_loss (Tensor): the current training loss
        """
        raise NotImplementedError


    @abstractmethod 
    def step(self, validation_loss: Tensor) -> None:
        """Perform a step of training

        Args:
            loss (float): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
```

MAML的相关方法只需要继承并实现对应抽象方法即可。

```python
class MAML(BaseLearner):
    def __init__(self, module: Layer) -> None:
        super().__init__(module)

    def adapt(self, train_loss: Tensor) -> None:
        pass

    def step(self, validation_loss: Tensor) -> None:
        pass
```

> 具体实现有待后续进行深入分析。

## 文档编写

我发现本项目目前不存在一定的pythonic 的文档，后续可使用mkdocs编写文档并发布到readthedoc上面去，进而不断完善项目。

# 六、测试和验收的考量

在Omniglot和miniImageNet数据集的5-way 1-shot任务和5-way 5-shot任务进行测试。使用算法api版的MAML, ANIL获得与原汇报结果一致或更高的效果。

# 七、可行性分析和排期规划

* 1-th week: 完成初步沟通和整体设计方案
* 2-th week: 完成MAML Learner的优化
* 3-th week: 完成example test并改善代码
* 4-th week: 完成在Omniglot和miniImageNet数据集上的效果测试
* 5-th week: 深入讨论项目后续开发计划

# 名词解释

无

# 附件及参考资料

无
