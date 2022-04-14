# paddle.nn.AdaptiveLogSoftmaxWithLoss 设计文档

|API名称 | paddle.nn.AdaptiveLogSoftmaxWithLoss             | 
|---|------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | PeachML                            | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-22                         | 
|版本号 | V1.0                               | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                             | 
|文件名 | 20200322_api_design_for_AdaptiveLogSoftmaxWithLoss.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，AdaptiveLogSoftmaxWithLoss 来源于 Efficient softmax approximation for GPUs，
其中的 Adaptive Softmax 方法是对一种高效实现 softmax 函数近似计算的方法。
Paddle需要扩充API,新增 AdaptiveLogSoftmaxWithLoss API，
调用路径为：`paddle.nn.AdaptiveLogSoftmaxWithLoss` 和 `paddle.nn.functional.adaptive_log_softmax_with_loss`。
实现Softmax快速近似计算的功能。

## 3、意义
在自然语言处理中，当字典维度过大时，embedding 将占据模型大部分参数量。
例如机器翻译任务中，词表维度大约是2^17，embedding维度取1024，那么就会产生将近1亿参数量，
如果不共享embedding矩阵和softmax映射的矩阵，将会再多出1亿参数量。

这样会引起常见的两个问题：

- 参数量巨大会直接影响线上部署显存占用，单点部署的进程数就会收到限制，云上GPU是很贵的
- 自然语言中单词的分布服从齐夫定律(Zipf law)，少部分单词频数和占据总频数的大部分。
这使得出现频数少的单词没法得到充分的训练。

Facebook在Efficient softmax approximation for GPUs中提出了Adaptive Softmax，
可以很好的解决以上两个问题。大致思想就是按照每个单词在语料中出现的频数从高到低排序并分组，
针对频数高的组设置大的embedding维度，频数低的组设置小的embedding维度。


# 二、飞桨现状
目前paddle缺少相关功能实现。

# 三、业内方案调研
## Pytorch
Pytorch中有API`torch.nn.AdaptiveLogSoftmaxWithLoss
(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, device=None, dtype=None)`，
在pytorch中，介绍为：
```
Efficient softmax approximation as described in
    `Efficient softmax approximation for GPUs`_ by Edouard Grave, Armand Joulin,
    Moustapha Cissé, David Grangier, and Hervé Jégou.

    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{in\_features}{div\_value^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted accoridng to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, in\_features)`
        - target: :math:`(N)` where each value satisfies :math:`0 <= target[i] <= n\_classes`
        - output1: :math:`(N)`
        - output2: ``Scalar``


```

### 实现方法
在实现方法上, Pytorch是通过纯Python API组合实现的, [代码位置](https://github.com/pytorch/pytorch/blob/bceb1db885cafa87fe8d037d8f22ae9649a1bba0/torch/nn/modules/adaptive.py#L18)。

整体逻辑为：

1. 初始化参数
    - in_features (int): 输入特征数
    - n_classes (int): 数据集中类别数
    - cutoffs 表示低频词clusters的分界值list
    - div_value (float, optional): 计算cluster的大小的指数
    - head_bias 表示第一层的softmax是否需要bias

然后定义了第一层softmax的线性转换`self.head` 和第二层softmax的线性转换`self.tail`，
且第二层的线性转换`self.tail`先对输入进行了降维来加速计算。
由于这里是低频词的预测，所以降维造成的效果损失应该可以容忍。

2. 训练
```python
def forward(self, input, target):
    # input的shape为[batch_size * bptt, hidden_size]
    # target的shape为[batch_size * bptt, 1]
    if input.size(0) != target.size(0):
        raise RuntimeError('Input and target should have the same size '
                            'in the batch dimension.')
    # 用来统计多个cluster计算的batch，然后求和，保证最终等于batch_size
    used_rows = 0
    batch_size = target.size(0)
    # 用来记录在target位置的 logprob 
    output = input.new_zeros(batch_size)
    # 用来记录batch样本在第一层对应的类别
    gather_inds = target.new_empty(batch_size)

    cutoff_values = [0] + self.cutoffs
    for i in range(len(cutoff_values) - 1):

        low_idx = cutoff_values[i]
        high_idx = cutoff_values[i + 1]
        # 找到当前cluster的样本对应的index
        target_mask = (target >= low_idx) & (target < high_idx)
        row_indices = target_mask.nonzero().squeeze()
        # 如果当前cluster没有样本，则没有loss
        if row_indices.numel() == 0:
            continue
        # target对应高频词，这里只用来记录batch对应的target，高频词的预测在后面 self.head
        if i == 0:
            gather_inds.index_copy_(0, row_indices, target[target_mask])
        # target对应低频词
        else:
            # 获取低频cluster对应的target的相对位置
            relative_target = target[target_mask] - low_idx
            # 获取对应cluster的input
            input_subset = input.index_select(0, row_indices)
            # 经过线性变换 得到 [batch_size_i, target_i]
            cluster_output = self.tail[i - 1](input_subset)
            # 当前cluster对应第一层权重元素的类别
            cluster_index = self.shortlist_size + i - 1
            # 记录对应第一层的类别
            gather_inds.index_fill_(0, row_indices, cluster_index)
            # 计算当前cluster的log_prob
            cluster_logprob = log_softmax(cluster_output, dim=1)
            # 获取对应target位置的log_prob
            local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
            # 将结果记录到对应的batch中
            output.index_copy_(0, row_indices, local_logprob.squeeze(1))

        used_rows += row_indices.numel()

    if used_rows != batch_size:
        raise RuntimeError("Target values should be in [0, {}], "
                            "but values in range [{}, {}] "
                            "were found. ".format(self.n_classes - 1,
                                                    target.min().item(),
                                                    target.max().item()))
    # 第一层的线性变换，因为无论高频和低频词都需要计算第一层，所以放到了这里统一计算
    head_output = self.head(input)
    # 取log_prob
    head_logprob = log_softmax(head_output, dim=1)
    # 这里是第一层的log_prob和第二层的log_prob加起来作为最后的输出
    # tips: 对于属于第一层的样本，只需要计算第一层的log_prob就好
    #       对于属于第二层的样本，需要将第一层计算得到的cluster对应类别的log_prob和
            第二层cluster内计算得到的log_prob加起来，所以是output += 
    output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
    loss = (-output).mean()
    # 返回一个nametuple
    return _ASMoutput(output, loss)
```


3. 预测
```python
def predict(self, input):
    """
    Args:
        input (Tensor): a minibatch of examples
    Returns:
        output (Tensor): a class with the highest probability for each example
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N)`
    """
    # 第一层的线性转化
    head_output = self.head(input)
    # 记录预测target的位置
    output = torch.argmax(head_output, dim=1)
    # 判断预测的位置是否都是低频词
    not_in_shortlist = (output >= self.shortlist_size)
    # 获取预测高频词的样本index
    all_in_shortlist = not (not_in_shortlist.any())
    # 如果预测的结果都为高频词，则直接返回结果
    if all_in_shortlist:
        return output
    # 如果预测的结果都为低频词
    elif not_in_shortlist.all():
        # 计算低频词对应cluster中target对应的log_prob
        log_prob = self._get_full_log_prob(input, head_output)
        return torch.argmax(log_prob, dim=1)
    # 如果预测的结果既有高频词，也有低频词
    else:
        # 只对低频词进行对应cluser的预测
        log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                            head_output[not_in_shortlist])
        output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
        return output

# 计算低频词对应cluster中target对应的log_prob
def _get_full_log_prob(self, input, head_output):
    """ Given input tensor, and output of `self.head`,
    compute the log of the full distribution """

    out = input.new_empty((head_output.size(0), self.n_classes))
    head_logprob = log_softmax(head_output, dim=1)

    out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

    for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
        cluster_output = self.tail[i](input)
        cluster_logprob = log_softmax(cluster_output, dim=1)
        output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

        out[:, start_idx:stop_idx] = output_logprob

    return out
```


# 四、对比分析
无其它框架实现

# 五、方案设计
## 命名与参数设计
API设计为`paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)`及
`paddle.nn.functional.adaptive_log_softmax_with_loss(input, label, 
in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)`, 返回为`NamedTuple` 包含 `output` 和 `loss`字段



## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案
主要参考pytorch实现，替换掉部分paddle没有的api
 
# 六、测试和验收的考量
测试考虑的case如下：

- 数值正确性
- 错误检查：`cutoff`的唯一性，数据类型，数值大于零小于`n_classes - 1`
- 错误检查：`input`尺寸与`in_features`一致


# 七、可行性分析及规划排期
方案主要依赖paddle现有api组合而成

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无
