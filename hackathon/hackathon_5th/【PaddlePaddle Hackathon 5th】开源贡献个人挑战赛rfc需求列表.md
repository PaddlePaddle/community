## 热身赛

| **序号** | **难度** | **任务 ISSUE**                                               | **是否需要提交rfc** |
| -------- | -------- | ------------------------------------------------------------ | ---------------------------- |
| 1        | 🌟        | 为 Paddle 新增 copysign API | 是                           |
| 2        | 🌟        | 为 Paddle 新增 index_fill API | 是                           |
| 3        | 🌟        | 为 Paddle 新增 masked_fill API | 是                           |
| 4        | 🌟        | 为 Paddle 新增 masked_scatter API | 是                           |
| 5        | 🌟        | 为 Paddle 增强scatter API | 是                           |
| 6        | 🌟        | 为 Paddle 增强put_along_axis API | 是                           |
| 7        | 🌟        | 为 Paddle 新增 apply API | 是                           |
| 8        | 🌟        | 为 Paddle 新增 hypot API | 是                           |
| 9        | 🌟        | 为 Paddle 新增 multigammaln API | 是                           |
| 10       | 🌟🌟       | 为 Paddle 新增 bernoulli_ / log_normal_ / log_normal API | 是                           |
| 11       | 🌟🌟🌟      | 为 Paddle 新增 igamma 和 igammac API | 是                           |

## 框架 API 开发任务

| **序号** | **难度** | **任务 ISSUE**                                               | **是否需要提交rfc** |
| -------- | -------- | ------------------------------------------------------------ | ---------------------------- |
| 12       | 🌟        | 为 Paddle 新增 AdaptiveLogSoftmaxWithLoss API | 是                           |
| 13       | 🌟        | 为 Paddle 新增 signbit API | 是                           |
| 14       | 🌟        | 为 Paddle 新增 combinations API | 是                           |
| 15       | 🌟        | 为 Paddle 新增 Tensor.to / Layer.astype API | 是                           |
| 16       | 🌟        | 为 Paddle 新增 EmbeddingBag API | 是                           |
| 17       | 🌟        | 为 Paddle 新增 pdist API | 是                           |
| 18       | 🌟        | 为 Paddle 新增 Binomial 和 Poisson API | 是                           |
| 19       | 🌟        | 为 Paddle 新增 ContinuousBernoulli 和 MultivariateNormal API | 是                           |
| 20       | 🌟        | 为 Paddle 新增 Exponential 和 Gamma API | 是                           |
| 21       | 🌟        | 为 Paddle 新增 LinearLR API | 是                           |
| 22       | 🌟        | 为 Paddle 新增 CosineAnnealingWarmRestarts API | 是                           |
| 23       | 🌟        | 为 Paddle 新增 ConcatDataset API | 是                           |
| 24       | 🌟        | 为 Paddle 新增 SubsetRandomSampler API | 是                           |
| 25       | 🌟        | 为 Paddle 新增 gammaln API | 是                           |
| 26       | 🌟        | 为 Paddle 新增 diagonal_scatter API | 是                           |
| 27       | 🌟        | 为 Paddle 新增 select_scatter API                            | 是                           |
| 28       | 🌟        | 为 Paddle 新增 slice_scatter API                             | 是                           |
| 29       | 🌟        | 为 Paddle 新增 cholesky_inverse API                          | 是                           |
| 30       | 🌟        | 为 Paddle 新增 vdot API                                      | 是                           |
| 31       | 🌟🌟       | 为 Paddle 新增 column_stack / row_stack / dstack / hstack / vstack API | 是                           |
| 32       | 🌟🌟       | 为 Paddle 新增 tensor_split / hsplit / dsplit API            | 是                           |
| 33       | 🌟🌟       | 为 Paddle 新增 atleast_1d / atleast_2d / atleast_3d API      | 是                           |
| 34       | 🌟🌟       | 为 Paddle 新增 bitwise_right_shift / bitwise_right_shift_ / bitwise_left_shift / bitwise_left_shift_ API | 是                           |
| 35       | 🌟🌟       | 为 Paddle 新增 histogramdd API                               | 是                           |
| 36       | 🌟🌟       | 为 Paddle 新增 matrix_exp API                                | 是                           |
| 37       | 🌟🌟       | 为 Paddle 新增 householder_product API                       | 是                           |
| 38       | 🌟🌟🌟     | 为 Paddle 新增 FractionalMaxPool2d / FractionalMaxPool3d API | 是                           |
| 39       | 🌟🌟🌟     | 为 Paddle 新增 LPPool1D / LPPool2D API                       | 是                           |
| 40       | 🌟🌟🌟     | 为 Paddle 新增 ASGD API                                      | 是                           |
| 41       | 🌟🌟🌟     | 为 Paddle 新增 Rprop API                                     | 是                           |
| 110      | 🌟         | 为 Paddle 增强 sparse.matmul API                              | 是                           |


## 框架其他开发任务

| **序号** | **难度** | **任务 ISSUE**                                               | **是否需要提交rfc** |
| -------- | -------- | ------------------------------------------------------------ | ---------------------------- |
| 42       | 🌟        | 为Paddle代码转换工具新增API转换规则                      | 否                           |
| 43       | 🌟        | 为Paddle代码转换工具新增API转换规则                      | 否                           |
| 44       | 🌟        | 为Paddle代码转换工具新增API转换规则                      | 否                           |
| 45       | 🌟        | 为Paddle代码转换工具新增API转换规则                      | 否                           |
| 46       | 🌟        | 为Paddle代码转换工具新增API转换规则                      | 否                           |
| 47       | 🌟        | 为Paddle代码转换工具新增API转换规则                      | 否                           |
| 48       | 🌟        | ContiguousKernel、StridedCopyKernel算子CPU、GPU性能优化  | 否                           |
| 49       | 🌟🌟       | python端补齐OpResult的patch方法                          | 否                           |
| 50       | 🌟        | 为 Paddle 新增 slice 的 spmd 切分推导规则                | 否                           |
| 51       | 🌟        | 为 Paddle 新增 flatten 的 spmd 切分推导规则              | 否                           |
| 52       | 🌟🌟       | 为 Paddle 新增 squeeze 和 unsqueeze 的 spmd 切分推导规则 | 否                           |
| 101  | 🌟    | 将paddle内部的fused_multi_transformer/fused_multi_transformer_int8算子及其kernel实现从fluid下迁移到phi下 |  否    |
| 102  | 🌟    | 将paddle内部的fused_embedding_eltwise_layernorm、fusion_transpose_flatten_concat和fused_fc_elementwise_layernorm算子及其kernel实现从fluid下迁移到phi下 | 否     |
| 103  | 🌟    | 将paddle内部的skip_layernorm、fc和fused_bias_dropout_residual_layer_norm算子及其kernel实现从fluid下迁移到phi下 |  否    |
| 104  | 🌟    | 将paddle内部的self_dp_attention和fusion_repeated_fc_relu/fusion_squared_mat_sub算子及其kernel实现从fluid下迁移到phi下 |  否    |
| 105  | 🌟    | 将paddle内部的fusion_gru、fusion_seqconv_eltadd_relu和fusion_seqexpand_concat_fc算子及其kernel实现从fluid下迁移到phi下 |   否   |
| 112  | 🌟    | 将paddle内部的read_file、fused_gemm_epilogue算子及其kernel实现从fluid下迁移到phi下；添加identity_loss的yaml配置 |   否   |
| 113  | 🌟    | 为paddle.nn.functional.embedding增加参数max_norm/norm_type/scale_grad_by_freq |   否   |
| 114  | 🌟    | 为paddle.linalg.norm进行功能对齐与功能增强 |   否   |
| 115  | 🌟    | 为paddle.nn.LSTM/RNNBase/paddle.quantile/nanquantile功能增强 |   否   |
| 116  | 🌟    | 为paddle.histogram/paddle.nn.functional.threshold进行功能对齐与功能增强 |   否   |
| 117  | 🌟    | 为paddle.nn.functional.upsample/paddle.nn.initializer.XavierNormal/XavierUniform/KaimingNormal/KaimingUniform进行功能增强 |   否   |
| 118  | 🌟    | 为paddle.io.RandomSampler/random_split/Layer.clear_gradients进行功能增强 |   否   |
| 119  | 🌟    | 为paddle.round/paddle.nn.functional.max_pool1d/max_pool2d/max_pool3d进行功能增强 |   否   |
| 120  | 🌟    | 为paddle.nn.functional.max_unpool1d/max_unpool2d/max_unpool3d/paddle.nn.functional.kl_div进行功能增强或Bug修复 |   否   |
| 121  | 🌟    | 为paddle.nn.functional.max_pool1d/max_pool2d/max_pool3d/paddle.signal.stft进行Bug修复 |   否   |

## 科学计算模型复现

| **序号** | **难度** | **任务 ISSUE**                                               | **是否需要提交rfc** |
| -------- | -------- | ------------------------------------------------------------ | ---------------------------- |
| 53       | 🌟        | NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations | 是                           |
| 54       | 🌟        | NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations | 是                           |
| 55       | 🌟        | Fourier-MIONet: Fourier-enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration | 是                           |
| 56       | 🌟        | A Physics-Informed Neural Network to solve 2D steady-state heat equation | 是                           |
| 57       | 🌟        | Neural networks for topology optimization                    | 是                           |
| 58       | 🌟🌟       | A physics-informed deep neural network for surrogate modeling in classical elasto-plasticity | 是                           |
| 59       | 🌟🌟       | Spline-PINN: Approaching PDEs without Data using Fast, Physics-Informed Hermite-Spline CNNs | 是                           |
| 60       | 🌟🌟       | PhyGeoNet: Physics-Informed Geometry-Adaptive Convolutional Neural Networks for Solving Parameterized Steady-State PDEs on Irregular Domain | 是                           |
| 61       | 🌟🌟       | Skillful nowcasting of extreme precipitation with NowcastNet | 是                           |
| 62       | 🌟🌟🌟      | GraphCast: Learning skillful medium-range global weather forecasting | 是                           |
| 63       | 🌟🌟       | PhyCRNet: Physics-informed Convolutional-Recurrent Network for Solving Spatiotemporal PDEs | 是                           |
## 套件开发任务

| **序号** | **难度** | **任务 ISSUE**                                               | **是否需要提交rfc** |
| -------- | -------- | ------------------------------------------------------------ | ---------------------------- |
| 64       | 🌟🌟🌟🌟🌟🌟        | 全套件模型接入动转静训练功能                      | 是                           |
| 65       | 🌟        | 版面恢复功能（恢复为docx或者excel）的c++版        | 是                           |
| 66       | 🌟🌟      | 生僻词模型训练                                    | 是                           |
| 67       | 🌟🌟       | 版面矫正网络DocTr++论文复现                       | 是                           |
| 68       | 🌟🌟       | 轻量语义分割网络PIDNet                            | 是                           |
| 69       | 🌟        | 分类大模型--人体视觉任务SOLIDER                   | 是                           |
| 70       | 🌟🌟        | DET重点模型支持实例分割                           | 是                           |
| 71       | 🌟        | 新增 bevfusion 部署链条                           | 是                           |
| 72       | 🌟🌟       | 新增模型TaskMatrix                                | 是                           |
| 73       | 🌟🌟🌟       | 新增模型Tree of Thoughts                          | 是                           |
| 74       | 🌟🌟🌟       | RetroMAE训练                                      | 是                           |
| 75       | 🌟🌟       | 新增模型InstructBlip                              | 是                           |
| 76       | 🌟        | 新增数据集训练和评估 (coco retrieval)             | 是                           |
| 77       | 🌟🌟🌟      | 新增模型kosmos2                                   | 是                           |
| 78       | 🌟        | minigpt-4 zeroshot评估                            | 是                           |
| 79       | 🌟🌟       | 新增模型openseed                                  | 是                           |
| 80       | 🌟        | 添加appflow以及对应模型单测                       | 否                           |
| 81       | 🌟        | applications应用gradio demo                       | 否                           |
| 82       | 🌟🌟       | 为Paddle2ONNX增加原生FP6 Paddle模型的转换能力     | 否                           |
| 83       | 🌟🌟       | PaddleMIX ppdiffusers models模块功能升级同步HF    | 否                           |
| 84       | 🌟🌟       | 新增模型视频生成模型MS-Image2Video+MS-Vid2Vid-XL  | 是                           |
| 85       | 🌟🌟       | 新增换装模型应用 DCI-VTON-Virtual-Try-On          | 是                           |
| 86       | 🌟🌟       | 新增图像组合模型应用TF-ICON                       | 是                           |
| 87       | 🌟🌟       | PaddleMIX ppdiffusers新增HF community应用pipeline | 否                           |

## 合作伙伴任务

| **序号** | **难度** | **任务 ISSUE**                                               | **是否需要提交rfc** |
| -------- | -------- | ------------------------------------------------------------ | ---------------------------- |
| 88       | 🌟        | Arm虚拟硬件上完成PaddleClas模型的部署验证                    | 是                           |
| 89       | 🌟        | Arm虚拟硬件上完成飞桨视觉模型的部署验证                      | 是                           |
| 90       | 🌟🌟       | Arm虚拟硬件上完成飞桨模型与Arm Ethos-U microNPU的适配与部署验证 | 是                           |
| 91       | 🌟🌟       | Arm虚拟硬件上完成飞桨模型的优化部署                          | 是                           |
| 92       | 🌟🌟       | 使用Arm smart vision configuration kit 在Arm虚拟硬件上部署飞桨模型 | 是                           |
| 93       | 🌟🌟🌟      | 为OpenVINO 实现 Paddle 算子max_pool3d_with_index与max_pool3d转换 | 否                           |
| 94       | 🌟🌟       | 为 OpenVINO 实现 Paddle 算子partial_sum与partial_concat转换  | 否                           |
| 95       | 🌟        | 为 OpenVINO 实现 Paddle 算子 unique转换                      | 否                           |
| 96       | 🌟        | 为 OpenVINO 实现 Paddle 算子unstack转换                      | 否                           |
| 97       | 🌟        | 为 OpenVINO 实现 Paddle 算子tanh_shrink转换                  | 否                           |
| 98       | 🌟🌟       | 完成PP-YOLOE在华为昇腾平台上的推理优化                       | 是                           |
| 99       | 🌟🌟       | 基于 Qualcomm SNPE SDK 开发 RMSNorm 算子                     | 是                           |
| 100      | 🌟        | 基于openKylin OS和X2paddle实现面向AI框架的统一推理接口，实现AI软件的适配与应用 | 是                           |
| 106  | 🌟    | Paddle模型适配InfiniTensor推理引擎                           | 否 |
| 107  | 🌟    | 基于InfiniTensor推理引擎的对话类示范应用                     | 否 |
| 108  | 🌟    | 为InfiniTensor推理引擎添加GeLU算子                           | 否 |
| 109  | 🌟    | InfiniTensor推理引擎的Windows系统适配                        | 否 |
| 111  | 🌟🌟🌟    | 基于PaddleSeg的纤维轮廓识别                        | 否 |
| 113  | 🌟   | OpenVINO 开放任务            | 否 |
