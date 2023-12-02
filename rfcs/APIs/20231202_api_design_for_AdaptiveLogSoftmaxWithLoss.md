# paddle.nn.AdaptiveLogSoftmaxWithLoss 设计文档

|API名称 | paddle.nn.AdaptiveLogSoftmaxWithLoss             |
|---|------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | netpunk                            |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-12-02                         |
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

## 2、功能目标

为飞桨补充 AdaptiveLogSoftmaxWithLoss API，该API实现 softmax 函数近似计算

adaptive_log_softmax_with_loss的计算分步骤如下

$\text{head_output} = \text{linear}(\text{input}, \text{head_weight}, \text{head_bias})$

$\text{head_logprob} = \text{log_softmax}(\text{head_output}, \text{axis}=1)$ 

$\text{output} += \text{take_along_axis}(\text{head_logprob}, \text{gather_inds.unsqueeze(1)}, \text{axis}=1).\text{squeeze()}$ 

$\text{loss} = -\text{output.mean()}$

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
def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput:
    targ_dim = target_.dim()

    if targ_dim == 1:
        if input_.size(0) != target_.size(0):
            raise RuntimeError('Input and target should have the same size '
                                'in the batch dimension.')
        if input_.dim() != 2:
            raise RuntimeError('1D target tensor expects 2D input tensors, '
                                'but found inputs with size', input_.size())
    elif targ_dim == 0:
        if input_.dim() != 1:
            raise RuntimeError('0D target tensor expects 1D input tensors, '
                                'but found inputs with size', input_.size())
    else:
        raise RuntimeError('0D or 1D target tensor expected, '
                            'multi-target not supported')

    is_batched = targ_dim > 0
    input = input_ if is_batched else input_.unsqueeze(0)
    target = target_ if is_batched else target_.unsqueeze(0)

    used_rows = 0
    batch_size = target.size(0)

    output = input.new_zeros(batch_size)
    gather_inds = target.new_empty(batch_size)

    cutoff_values = [0] + self.cutoffs
    for i in range(len(cutoff_values) - 1):

        low_idx = cutoff_values[i]
        high_idx = cutoff_values[i + 1]

        target_mask = (target >= low_idx) & (target < high_idx)
        row_indices = target_mask.nonzero().squeeze()

        if row_indices.numel() == 0:
            continue

        if i == 0:
            gather_inds.index_copy_(0, row_indices, target[target_mask])

        else:
            relative_target = target[target_mask] - low_idx
            input_subset = input.index_select(0, row_indices)

            cluster_output = self.tail[i - 1](input_subset)
            cluster_index = self.shortlist_size + i - 1

            gather_inds.index_fill_(0, row_indices, cluster_index)
            cluster_logprob = log_softmax(cluster_output, dim=1)
            local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
            output.index_copy_(0, row_indices, local_logprob.squeeze(1))

        used_rows += row_indices.numel()

    if used_rows != batch_size:
        raise RuntimeError(f"Target values should be in [0, {self.n_classes - 1}], "
                            f"but values in range [{target.min().item()}, {target.max().item()}] "
                            "were found. ")

    head_output = self.head(input)
    head_logprob = log_softmax(head_output, dim=1)
    output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
    loss = (-output).mean()

    if not is_batched:
        output = output.squeeze(0)

    return _ASMoutput(output, loss)
```


3. 预测
```python
def predict(self, input: Tensor) -> Tensor:
    r""" This is equivalent to `self.log_prob(input).argmax(dim=1)`,
    but is more efficient in some cases.

    Args:
        input (Tensor): a minibatch of examples

    Returns:
        output (Tensor): a class with the highest probability for each example

    Shape:
        - Input: :math:`(N, \texttt{in\_features})`
        - Output: :math:`(N)`
    """

    head_output = self.head(input)
    output = torch.argmax(head_output, dim=1)
    not_in_shortlist = (output >= self.shortlist_size)
    all_in_shortlist = not (not_in_shortlist.any())

    if all_in_shortlist:
        return output

    elif not_in_shortlist.all():
        log_prob = self._get_full_log_prob(input, head_output)
        return torch.argmax(log_prob, dim=1)

    else:
        log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                            head_output[not_in_shortlist])
        output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
        return output

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

def log_prob(self, input: Tensor) -> Tensor:
    r""" Computes log probabilities for all :math:`\texttt{n\_classes}`

    Args:
        input (Tensor): a minibatch of examples

    Returns:
        log-probabilities of for each class :math:`c`
        in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
        parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

    Shape:
        - Input: :math:`(N, \texttt{in\_features})`
        - Output: :math:`(N, \texttt{n\_classes})`

    """

    head_output = self.head(input)
    return self._get_full_log_prob(input, head_output)
```


# 四、对比分析
无其它框架实现

# 五、设计思路与实现方案
## 命名与参数设计

layer层类API：`paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)`，包含两个主要方法：
    - forward(self, input, label)，用于训练，返回为`output` 和 `loss`
    - predict(self, input),用于预测

- in_features (int): 输入tensor的特征数量。
- n_classes (int): 数据集中类型的个数。
- cutoffs (Sequence): 用于将label分配到不同存储桶的截断值。
- div_value (float, 可选): 用于计算簇大小的指数值. 默认值：4.0。
- head_bias (bool, 可选): 如果为 ``True``，向自适应 softmax 的头部添加偏置项. 默认值：``False``.
- name (str, 可选): 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

function API：`paddle.nn.functional.adaptive_log_softmax_with_loss(input, label, head_weight, tail_weights, cutoffs, head_bias=None)` 用于训练计算

- input (Tensor): 输入张量，数据类型为 float32 或 float64。
- label (Tensor): 标签张量，数据类型为 float32 或 float64。
- head_weight (Tensor): 用于线性计算的权重矩阵，数据类型为 float32 或 float64。
- tail_weights (Tensor): 用于线性计算的权重矩阵，数据类型为 float32 或 float64。
- cutoffs (Sequence): 用于将label分配到不同存储桶的截断值。
- head_bias (Tensor, 可选): 用于线性计算的偏置矩阵，数据类型为 float32 或 float64。
- name (str, 可选): 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案

计算逻辑参考pytorch实现，并基于paddle API进行重组与封装：
- function API：`paddle.nn.functional.adaptive_log_softmax_with_loss(input, label, head_weight, tail_weights, cutoffs, head_bias=None)`，使用已有api进行组合实现，

- layer层类API：`paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)`，包含两个主要方法：
    - forward(self, input, target)，用于训练，返回为`output` 和 `loss`
    - predict(self, input), 用于预测，其计算与forward共享权重但是计算逻辑存在差异，故使用已有API组合实现的方式单独实现

# 六、测试和验收的考量
测试考虑的case如下：

- 数值正确性（CPU、GPU、动态图、静态图）
- 错误检查：`cutoff`的唯一性，数据类型，数值大于零小于`n_classes - 1`
- 错误检查：`input`尺寸与`in_features`一致


# 七、可行性分析及规划排期

paddle.gather与torch.gather存在差异，使用paddle.take_along_axis替换实现。实现无明显难点，可以按期完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无