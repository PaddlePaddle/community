

# 飞桨适配 nueralop

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |         Kai-Qi     |
| 队伍名称      |         qikai123     |
| 提交时间      |       2024-10-25   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本   | 3.0-beta                |
| 文件名        | 20241025_paddle_for_neuraloperator_qikai123.md |

## 1. 概述

### 1.1 相关背景

> [飞桨科学计算工具组件开发大赛](https://github.com/PaddlePaddle/PaddleScience/issues/1000)


`neuraloperator`是一个用于在 PyTorch 中学习神经运算符的综合库。它是 Fourier Neural Operators 和 Tensorized Neural Operators 的官方实现。并且`neuraloperator`支持在函数空间之间学习映射。

为`neuraloperator`工具组件支持飞桨后端，可以提高飞桨用户开发科学计算模型的开发效率。

### 1.2 功能目标


1. 整理 `neuraloperator` 的所有公开 API；
2. 使用 paddle 的 python API 等价组合实现上述公开 API 的功能；
3. 参考 PyTorch 后端已有代码，撰写飞桨后端的单测文件，并自测通过。

### 1.3 意义


为`neuraloperator`支持飞桨后端，从而提高飞桨用户开发科学计算模型的开发效率。

## 2. PaddleScience 现状


当前的PaddleScience有`neuraloperator`相关的[神经算子](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/neuraloperator)实现。

但是为了确保代码的合入和比赛结束后成果的可维护性，本方案将不采用 PaddleScience 中现有的 `neuraloperator` 实现，而是基于 `neuraloperator` 的源码，独立编写支持飞桨后端的文件。

## 3. 目标调研


参考的源代码为：[https://github.com/neuraloperator/neuraloperator/tree/0.3.0](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)

`neuraloperator`源代码实现的8个模型分别为, `FNO1d`, `FNO2d`, `FNO3d`, `SFNO`, `UNO`, `TFNO1d`, `TFNO2d`, `TFNO3d`。

为`neuraloperator`支持飞桨后端，主要的难点在于：

（1）傅里叶卷积层中复数权重的转换。

（2）`TFNO`中使用tltorch进行张量分解，需要利用tensorly的API组合实现张量分解的功能。
## 4. 设计思路与实现方案


比赛要求：所有文件组织结构必须与原有代码保持一致（新增文件除外），原有的注释、换行、空格、开源协议等内容不能随意变动（新增内容除外），否则会严重影响代码合入和比赛结束后成果代码的维护。


所以本项目按照[`neuraloperator`](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)的源码来撰写支持飞桨后端的文件，并且严格保持文件结构、名称、注释、空格等都相同。

### 4.1[`neuraloperator`](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)核心API和对应单测文件列表


| API列表                                  | 单测                              | API列表                                  | 单测                              | API列表                                  | 单测                              |
| ---------------------------------------- | ------------------------------------- | ---------------------------------------- | ------------------------------------- | ---------------------------------------- | ------------------------------------- |
| FNOBlocks                                | test_fno_block.py                    | BaseSpectralConv                         | 无                                   | einsum_complexhalf_two_input             | 无                                   |
| legacy_spectral_convolution_SpectralConv | test_legacy_spectral_convolution.py  | einsum_complexhalf                       | 无                                   | PositionalEmbedding                      | 无                                   |
| legacy_spectral_convolution_SpectralConv1d | test_legacy_spectral_convolution.py  | SubModule                                | 无                                   | FCLegendre                               | 无                                   |
| legacy_spectral_convolution_SpectralConv2d | test_legacy_spectral_convolution.py  | IntegralTransform                        | 无                                   | _contract_dense                          | 无                                   |
| legacy_spectral_convolution_SpectralConv3d | test_legacy_spectral_convolution.py  | _contract_dense_separable                | 无                                   | _contract_cp                             | 无                                   |
| DomainPadding                            | test_padding.py                      | _contract_tucker                         | 无                                   | _contract_tt                             | 无                                   |
| resample                                 | test_resample.py                     | get_contract_fun                         | 无                                   | SubConv                                  | 无                                   |
| spectral_convolution_SpectralConv        | test_spectral_convolution.py         | MLP                                      | 无                                   | MLPLinear                                | 无                                   |
| spectral_convolution_SpectralConv1d      | test_spectral_convolution.py         | AdaIN                                    | 无                                   | iterative_resample                       | 无                                   |
| spectral_convolution_SpectralConv2d      | test_spectral_convolution.py         | segment_csr                              | 无                                   | simple_neighbor_search                   | 无                                   |
| spectral_convolution_SpectralConv3d      | test_spectral_convolution.py         | skip_connection                          | 无                                   | SoftGating                               | 无                                   |
| SHT                                      | test_spherical_convolution.py        | central_diff_1d                          | 无                                   | central_diff_2d                          | 无                                   |
| SphericalConv                            | test_spherical_convolution.py        | central_diff_3d                          | 无                                   | LpLoss                                   | 无                                   |
| FNO                                      | test_fno.py                          | H1Loss                                   | 无                                   | IregularLpqLoss                          | 无                                   |
| FNO1d                                    | test_fno.py                          | WeightedL2DragLoss                       | 无                                   | BurgersEqnLoss                           | 无                                   |
| FNO2d                                    | test_fno.py                          | ICLoss                                   | 无                                   | FieldwiseAggregatorLoss                 | 无                                   |
| FNO3d                                    | test_fno.py                          | WeightedSumLoss                          | 无                                   | load_burgers_1d                          | 无                                   |
| TFNO                                     | test_fno.py                          | load_burgers_1dtime                      | 无                                   | load_darcy_flow_small                    | 无                                   |
| TFNO1d                                   | test_fno.py                          | load_darcy_pt                            | 无                                   | MGPatchingDataProcessor                  | 无                                   |
| TFNO2d                                   | test_fno.py                          | H5pyDataset                              | 无                                   | load_navier_stokes_pt                    | 无                                   |
| TFNO3d                                   | test_fno.py                          | _load_navier_stokes_test_HR              | 无                                   | OutputEncoder                            | 无                                   |
| UNO                                      | test_uno.py                          | MultipleFieldOutputEncoder               | 无                                   | DictTransform                            | 无                                   |
| DefaultDataProcessor                     | test_data_processor.py               | load_pt_traintestsplit                   | 无                                   | load_spherical_swe                       | 无                                   |
| UnitGaussianNormalizer                   | test_output_encoder.py               | SphericalSWEDataset                      | 无                                   | TensorDataset                            | 无                                   |
|                                           |                                       | GeneralTensorDataset                     | 无                                   | Transform                                | 无                                   |
|                                           |                                       | Normalizer                               | 无                                   | Composite                                | 无                                   |
|                                           |                                       | MGPatchingTransform                      | 无                                   | RandomMGPatch                            | 无                                   |
|                                           |                                       | MGPTensorDataset                         | 无                                   | regular_grid                             | 无                                   |
|                                           |                                       | PositionalEmbedding2D                    | 无                                   | ZarrDataset                              | 无                                   |


## 5. 测试和验收的考量


所有测试结果均在 NVIDIA RTX 3090 GPU 上进行，操作系统为 Ubuntu 20.04，。测试分为三个独立的环境：环境1 使用 PyTorch 2.4.0（CUDA 12.1），环境2 使用 Paddle 3.0-beta（CUDA 11.8），环境1和环境2主要用于代码测试；环境3 包括 PyTorch 2.5.0（CUDA 11.8） 和 Paddle 3.0-beta（CUDA 11.8），用于模型初始化权重参数的转换。

### 5.1 模型前向对齐

#### 5.1.1 权重转化

为了保证模型前向对齐不受到模型参数不一致的影响，本项目对模型采用相同的权重参数进行初始化。生成相同权重参数的流程主要包括以下三个步骤：

&nbsp;&nbsp;（1）随机初始化neuraloperator-pytorch的官方模型参数并保存成 pytorch_model.pth；

&nbsp;&nbsp;（2）使用Paddle/models/tutorials/mobilenetv3_prod/Step1-5中的 torch2paddle.py 将 pytorch_model.pth 转化为 paddle_model.pdparams；

&nbsp;&nbsp;（3）将生成的 paddle_model.pdparams 加载到 neuraloperator-paddle模型中。


在模型转换过程中，PyTorch 和 Paddle 之间的一些参数需要特别处理，
尤其是傅里叶卷积层中的复数权重。由于这类权重在两个框架中的表示方式不同，
转换时需要将复数权重的实部和虚部分别提取、保存，并在加载时按相同方式进行处理，以确保模型参数的一致性。

#### 5.1.2 模型前向对齐验证（以FNO2d为例）

模型前向对齐验证主要分为以下四个步骤：

&nbsp;&nbsp;（1）将数据集如darcy_train_16.pt、darcy_test_16.pt、darcy_test_32.pt等转化为ndarray格式；（[https://github.com/neuraloperator/neuraloperator/tree/0.3.0/neuralop/datasets/data](https://github.com/neuraloperator/neuraloperator/tree/0.3.0/neuralop/datasets/datad)）

&nbsp;&nbsp;（2）在 Paddle 和 PyTorch 中分别对模型的 DataLoader 和 Datasets 进行对齐处理，确保两者一致；

&nbsp;&nbsp;（3）PyTorch 前向传播：定义 PyTorch 模型，加载权重，固定随机种子，基于 numpy 生成随机数，并转换为 `torch.Tensor`，送入网络，得到输出`y_pytorch`；

&nbsp;&nbsp;（4）飞桨前向传播：定义飞桨模型，加载步骤 5.1.1 中转换后的权重，将步骤（3）中的生成的随机数，转换为 `paddle.Tensor`，送入网络，获取输出`y_paddle`。

最终定义`y_diff=|y_pytorch-y_paddle|`。最终不同模型的结果如下，其中参数`layers`表示Fourier Layers的数目。

| 模型   | Max(y_diff)  | Min(y_diff)   | MSE(y_pytorch, y_paddle)   |备注|
|:-----:|:-----:|:-----:|:-----:|:-----:|
| FNO1d | 2.78e-05 | 1.11e-08  | 9.45e-11 |单精度训练|
| FNO2d | 2.69e-05 | 0.00 | 3.44e-11 |单精度训练|
| FNO3d | 4.75e-05 |0.00   | 5.66e-11  |单精度训练|
| UFNO(layers=1) | 1.03e-09 | 0.00 | 6.91e-21 |双精度训练|
| UFNO(layers=5) | 1.33e-08 |8.05e-13  | 1.05e-17  |双精度训练|
| SFNO | 3.068e-09 | 1.81e-13  | 4.36e-19 |双精度训练|
| TFNO1d | &nbsp; |&nbsp;   | &nbsp;  |撰写中，具体难点见6.3节|
| TFNO2d | &nbsp; | &nbsp;  | &nbsp; |撰写中，具体难点见6.3节|
|TFNO3d | &nbsp; |&nbsp;   | &nbsp;  |撰写中，具体难点见6.3节|




### 5.2 模型训练对齐（以FNO2d为例）


FNO2d-torch的每个epoch训练loss（L2 norm）为`loss_pytorch`，学习率为`lr_pytorch`。FNO2d-paddle的训练loss（L2 norm）为`loss_paddle`. 学习率为`lr_paddle`。定义指标`loss_diff=|loss_pytorch-loss_paddle|`，`lr_diff=|lr_pytorch-lr_paddle|`。

FNO2d-torch与FNO2d-paddle使用数据集 darcy_train_16.npy并且设置`ntrains=32 `。

FNO2d-torch的优化器参数为：
```python
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=8e-3, 
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

```

FNO2d-paddle的优化器参数为：
```python
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.008, T_max=30)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), 
                                  learning_rate=scheduler, 
                                  weight_decay=0.0001)
```


FNO2d-torch与FNO2d-paddle的训练loss和学习率对比结果如下：


<table >
  <tr>
    <th>epoch</th>
    <th colspan="3"style="text-align: center;">loss </th>
    <th colspan="3"style="text-align: center;">learning rate </th>
  </tr>
  <tr>
    <th></th>
    <th>loss_pytorch</th>
    <th>loss_paddle</th>
    <th>loss_diff</th>
    <th>lr_pytorch</th>
    <th>lr_paddle</th>
    <th>lr_diff</th>
  </tr>
  <tr>
    <td>0</td>
    <td>32.06811523</td>
    <td>32.06811523</td>
    <td>0.00</td>
    <td>0.00797809</td>
    <td>0.00797809</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>1</td>
    <td>31.38529968</td>
    <td>31.38491058</td>
    <td>3.89e-04</td>
    <td>0.00791259</td>
    <td>0.00791259</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>2</td>
    <td>27.89798164</td>
    <td>27.89617157</td>
    <td>1.81e-03</td>
    <td>0.00780423</td>
    <td>0.00780423</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>3</td>
    <td>30.61170769</td>
    <td>30.61891174</td>
    <td>7.20e-03</td>
    <td>0.00765418</td>
    <td>0.00765418</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>4</td>
    <td>21.70384979</td>
    <td>21.70200921</td>
    <td>1.84e-03</td>
    <td>0.0074641</td>
    <td>0.0074641</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>5</td>
    <td>21.37845802</td>
    <td>21.37961197</td>
    <td>1.15e-03</td>
    <td>0.00723607</td>
    <td>0.00723607</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>6</td>
    <td>20.06527138</td>
    <td>20.06911851</td>
    <td>3.85e-03</td>
    <td>0.00697258</td>
    <td>0.00697258</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>7</td>
    <td>16.89506531</td>
    <td>16.89888763</td>
    <td>3.82e-03</td>
    <td>0.00667652</td>
    <td>0.00667652</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>8</td>
    <td>15.51529121</td>
    <td>15.51611137</td>
    <td>8.20e-04</td>
    <td>0.00635114</td>
    <td>0.00635114</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>9</td>
    <td>14.76096916</td>
    <td>14.75966835</td>
    <td>1.30e-03</td>
    <td>0.00600000</td>
    <td>0.00600000</td>
    <td>0.00</td>
  </tr>
</table>


### 5.3 模型训练表现评估

#### 5.3.1 不同模型在训练时使用的数据
| 模型   | 数据集名称 |方程类型 |数据维度 |
|:-----:|:-----:|:-----:|:-----:|
| FNO1d |  burgers_lowres.mat |Burgers Equation |[32,1,16]|
| FNO2d | darcy_train_16.npy |Darcy Flow| [32,3,16,16]|
| FNO3d | NavierStokes_V1e-5_N1200_T20.mat |Navier-Stokes Equation |[32,1,64,64,10]|
| UFNO | darcy_train_16.npy |Darcy Flow| [32,3,16,16]|
| SFNO |  darcy_train_16.npy |Darcy Flow| [32,3,16,16]|
| TFNO1d | burgers_lowres.mat |Burgers Equation   |[32,1,16]  |
| TFNO2d |darcy_train_16.npy | Darcy Flow  | [32,3,16,16]|
| TFNO3d | NavierStokes_V1e-5_N1200_T20.mat |Navier-Stokes Equation   | [32,1,64,64,10] |

#### 5.3.2 不同模型训练loss的差异
| 模型   | Max(loss_diff)  | Min(loss_diff)   | MSE(loss_pytorch, loss_paddle)   | lr_diff   | 备注|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| FNO1d | 1.72e-02 | 3.05e-05  | 6.12e-05 |0.00 |单精度训练|
| FNO2d | 7.20e-03 | 0.00 | 9.18e-06 |0.00 |单精度训练|
| FNO3d | 7.50e-03 |1.14e-05   | 6.96e-06  |0.00 |单精度训练|
| UFNO(layers=1) | 1.10e-06 | 1.53e-08  | 4.62e-13 |0.00 |双精度训练|
| UFNO(layers=5) | 1.23e-04 |6.77e-09  | 2.71e-09  |0.00|双精度训练|
| SFNO | 2.17e-06 | 1.65e-08  |1.05e-12 |0.00 |双精度训练|
| TFNO1d | &nbsp; |&nbsp;   | &nbsp;  ||撰写中，具体见6.3节|
| TFNO2d | &nbsp; | &nbsp;  | &nbsp; ||撰写中，具体见6.3节|
|TFNO3d | &nbsp; |&nbsp;   | &nbsp;  ||撰写中，具体见6.3节|


#### 5.3.3 不同模型训练耗时 
使用 `timeit` 模块中的 `default_timer` 进行计时，各个模型在训练 10个epochs 时的耗时为 `10_epochs_time`，数据读取耗时为 `data_reader_time`，其中包括所有的数据预处理步骤（例如对 y 进行UnitGaussianNormalizer）,各个模型的耗时（s）如下：

(fno2d,ufno,sfno)
<table>
  <tr>
    <th>模型</th>
    <th colspan="2" style="text-align: center;">10_epochs_time</th>
    <th colspan="2" style="text-align: center;">data_reader_time</th>
  </tr>
  <tr>
    <th></th>
    <th>torch</th>
    <th>paddle</th>
    <th>torch</th>
    <th>paddle</th>
  </tr>
  <tr>
    <td>FNO1d</td>
    <td>7.19e-01</td>
    <td>3.18e-01</td>
    <td>2.32e-02</td>
    <td>8.79e-02</td>
  </tr>
  <tr>
    <td>FNO2d</td>
    <td>7.81e-01</td>
    <td>3.13e-01</td>
    <td>2.68e-02</td>
    <td>1.14e-01</td>
  </tr>
  <tr>
    <td>FNO3d</td>
    <td>1.88</td>
    <td>1.57</td>
    <td>3.26e-02</td>
    <td>1.62e-01</td>
  </tr>
  <tr>
    <td>UFNO(layers=1)</td>
    <td>6.30e-01</td>
    <td>2.86e-01</td>
    <td>2.68e-02</td>
    <td>8.03e-02</td>
  </tr>
  <tr>
    <td>UFNO(layers=5)</td>
    <td>8.94e-01</td>
    <td>5.26e-01</td>
    <td>3.12e-02</td>
    <td>9.05e-02</td>
  </tr>
  <tr>
    <td>SFNO</td>
    <td>8.11e-01</td>
    <td>4.44e-01</td>
    <td>3.19e-02</td>
    <td>1.04e-01</td>
  </tr>
  <tr>
    <td>TFNO1d</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>撰写中</td>
  </tr>
  <tr>
    <td>TFNO2d</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>撰写中</td>
  </tr>
  <tr>
    <td>TFNO3d</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>撰写中</td>
  </tr>
</table>




## 6. 已被解决的关键问题

## 6.1 UFNO(layers=5)（已解决）
在训练模型`UFNO(layers=5)`时，Pytorch版本和Paddle版本的训练损失在epoch=2时出现了显著差异，具体表现如下

<table>
    <tr>
    <th>epoch</th>
    <th colspan="3"style="text-align: center;">loss </th>
    <th colspan="3"style="text-align: center;">learning rate </th>
    </tr>
        <tr>
            <th></th>
            <th>loss_pytorch</th>
            <th>loss_paddle</th>
            <th>loss_diff</th>
            <th>lr_pytorch</th>
            <th>lr_paddle</th>
            <th>lr_diff</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>31.99346241</td>
            <td>32.02302024</td>
            <td>2.96e-02</td>
            <td>0.00797809</td>
            <td>0.00797809</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>1</td>
            <td>28.79406888</td>
            <td>28.28792398</td>
            <td>5.06e-01</td>
            <td>0.00791259</td>
            <td>0.00791259</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>2</td>
            <td>47.18949491</td>
            <td>62.09492780</td>
            <td>1.49e+01</td>
            <td>0.00780423</td>
            <td>0.00780423</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>3</td>
            <td>32.03622775</td>
            <td>25.50253190</td>
            <td>6.53e+00</td>
            <td>0.00765418</td>
            <td>0.00765418</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>4</td>
            <td>27.85857869</td>
            <td>28.94802254</td>
            <td>1.09e+00</td>
            <td>0.0074641</td>
            <td>0.0074641</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>5</td>
            <td>27.42604250</td>
            <td>29.43171277</td>
            <td>2.01e+00</td>
            <td>0.00723607</td>
            <td>0.00723607</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>6</td>
            <td>30.13455886</td>
            <td>28.78272349</td>
            <td>1.35e+00</td>
            <td>0.00697258</td>
            <td>0.00697258</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>7</td>
            <td>25.39807468</td>
            <td>26.31120726</td>
            <td>9.13e-01</td>
            <td>0.00667652</td>
            <td>0.00667652</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>8</td>
            <td>24.83115153</td>
            <td>22.55190781</td>
            <td>2.28e+00</td>
            <td>0.00635114</td>
            <td>0.00635114</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>9</td>
            <td>22.83329250</td>
            <td>20.45876442</td>
            <td>2.37e+00</td>
            <td>0.00600000</td>
            <td>0.00600000</td>
            <td>0.00</td>
        </tr>
    </tbody>
</table>




解决过程： 从1开始逐渐增大UFNO的layers数量。最终发现当layers>1，且UFNO模型的放缩参数uno_scalings经过0.5倍返回1倍时，UFNO中间网络层所调用函数validate_scaling_factor()返回值出错，经过修改后，UFNO的适配过程正常。

## 6.2 SFNO（已解决）

在训练模型`SFNO`时，Pytorch版本和Paddle版本的训练损失在epoch=3时出现了显著差异，具体表现如下

<table>
    <thead>
        <tr>
            <th>epoch</th>
            <th colspan="3" style="text-align: center;">loss</th>
            <th colspan="3" style="text-align: center;">learning rate</th>
        </tr>
        <tr>
            <th></th>
            <th>loss_pytorch</th>
            <th>loss_paddle</th>
            <th>loss_diff</th>
            <th>lr_pytorch</th>
            <th>lr_paddle</th>
            <th>lr_diff</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>32.06262686</td>
            <td>32.14197592</td>
            <td>7.93e-02</td>
            <td>0.00797809</td>
            <td>0.00797809</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>1</td>
            <td>31.48523259</td>
            <td>30.92742936</td>
            <td>5.58e-01</td>
            <td>0.00791259</td>
            <td>0.00791259</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>2</td>
            <td>27.51574327</td>
            <td>24.31375748</td>
            <td>3.20e+00</td>
            <td>0.00780423</td>
            <td>0.00780423</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>3</td>
            <td>28.54495172</td>
            <td>35.39358056</td>
            <td>6.85e+00</td>
            <td>0.00765418</td>
            <td>0.00765418</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>4</td>
            <td>22.02489485</td>
            <td>17.31783389</td>
            <td>4.71e+00</td>
            <td>0.0074641</td>
            <td>0.0074641</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>5</td>
            <td>22.83375406</td>
            <td>20.14916569</td>
            <td>2.68e+00</td>
            <td>0.00723607</td>
            <td>0.00723607</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>6</td>
            <td>22.05632114</td>
            <td>19.80994133</td>
            <td>2.25e+00</td>
            <td>0.00697258</td>
            <td>0.00697258</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>7</td>
            <td>21.20442744</td>
            <td>17.54419624</td>
            <td>3.66e+00</td>
            <td>0.00667652</td>
            <td>0.00667652</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>8</td>
            <td>19.56874356</td>
            <td>14.98493694</td>
            <td>4.58e+00</td>
            <td>0.00635114</td>
            <td>0.00635114</td>
            <td>0.00</td>
        </tr>
        <tr>
            <td>9</td>
            <td>18.70933591</td>
            <td>13.35811956</td>
            <td>5.35e+00</td>
            <td>0.00600000</td>
            <td>0.00600000</td>
            <td>0.00</td>
        </tr>
    </tbody>
</table>

解决过程：排查代码，最终发现由于粗心，SFNO中的谱卷积层类型出错，对应修正后，SFNO的适配过程正常。


## 7. 未解决的问题：TFNO中的张量分解过程

`TFNO`中使用了tltorch进行张量分解，tltorch是tensorly的PyTorch版。tensorly源代码([https://github.com/tensorly/tensorly](https://github.com/tensorly/tensorly))中说明了
```
    You can change the backend to perform computation with a different framework. 
    By default, the backend is NumPy, but you can also perform the computation using PyTorch, TensorFlow, JAX, CuPy or Paddle (requires to have installed them first). 
```
tensorly支持Paddle后端，但是在编写代码时发生了如下错误：

```python
import tensorly as tl
tl.set_backend('paddle')

ValueError:Unknown backend name 'paddle', known backends are ['numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy', 'jax']
```


## 8. 可行性分析和排期规划


| 里程碑        |  时间点     |
| :-------------:| :------------: | 
| 提交RFC      |     2024.10.25        |  
| 完成全部代码撰写并通过相关单测验证  |    2024.10.25-2024.11.15        |
| 提交PR，修改代码完成合入 |  2024.11.15-2024.11.25       | 
## 9. 影响面


为 PaddleScience 添加`neuraloperator`库。
