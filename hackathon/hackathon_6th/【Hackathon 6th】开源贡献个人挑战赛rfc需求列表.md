## 框架开发任务

| 序号 |  难度  | 任务标题                                                                                                   | 是否需要提交 rfc |
| :--: | :----: | ---------------------------------------------------------------------------------------------------------- | :--------------: |
|  1   |   ⭐   | 为 Paddle 新增 AdaptiveLogSoftmaxWithLoss API                                                              |        是        |
|  2   |   ⭐   | 为 Paddle 新增 cholesky_inverse API                                                                        |        是        |
|  3   |   ⭐   | 为 Paddle 新增 ZeroPad1D / ZeroPad3D / block_diag API                                                      |        是        |
|  4   |   ⭐   | 为 Paddle 新增 ormqr API                                                                                   |        是        |
|  5   |   ⭐   | 为 Paddle 新增 Chi2 / LKJCholesky API                                                                      |        是        |
|  6   |   ⭐   | 为 Paddle 新增 MultivariateNormal / StudentT API                                                           |        是        |
|  7   |  ⭐⭐  | 为 Paddle 新增 PolynomialLR / sinc / sinc\_ API                                                            |        是        |
|  8   |   ⭐   | 为 Paddle 新增 FeatureAlphaDropout API                                                                     |        是        |
|  9   |   ⭐   | 为 Paddle 新增 cartesian_prod API                                                                          |        是        |
|  10  |  ⭐⭐  | 为 Paddle 新增 isposinf / isneginf / isreal / isin API                                                     |        是        |
|  11  |  ⭐⭐  | 为 Paddle 新增 bernoulli\_ / log_normal\_ / log_normal API                                                 |        是        |
|  12  |  ⭐⭐  | 为 Paddle 新增 lu_solve API                                                                                |        是        |
|  13  | ⭐⭐⭐ | 为 Paddle 新增 RAdam / NAdam API                                                                           |        是        |
|  14  |  ⭐⭐  | 为 Paddle 新增 tensorinv / tensorsolve API                                                                 |        是        |
|  15  | ⭐⭐⭐ | 为 Paddle 新增 ldl_factor / ldl_solve API                                                                  |        是        |
|  16  | ⭐⭐⭐ | 为 Paddle 新增 LPPool1D / LPPool2D API                                                                     |        是        |
|  17  |   ⭐   | 为 Paddle 新增 sparse.mask_as API                                                                          |        是        |
|  18  |  ⭐⭐  | 为 Paddle 新增 sparse.concat API                                                                           |        是        |
|  19  |  ⭐⭐  | 为 Paddle 新增 sparse.stack API                                                                            |        是        |
|  20  |  ⭐⭐  | 为 Paddle 新增 sparse.nn.Conv2DTranspose / Conv3DTranspose API                                             |        是        |
|  21  |  ⭐⭐  | 为 Paddle 新增 sparse.nn.InverseConv2D / InverseConv3D API                                                 |        是        |
|  22  | ⭐⭐⭐ | 为 Paddle 增强 sparse.add / subtract / multiply / divide API                                               |        是        |
|  23  |   ⭐   | 为 paddle.nn.functional.embedding/paddle.nn.Embedding 增加参数 max_norm/norm_type/scale_grad_by_freq       |        否        |
|  24  |   ⭐   | 为 paddle.nn.LSTM/RNNBase /paddle.quantile/nanquantile 功能增强                                            |        否        |
|  25  |   ⭐   | 为 paddle.histogram/paddle.nn.functional.threshold 进行功能对齐与功能增强                                  |        否        |
|  26  |   ⭐   | 为 paddle.view/paddle.nn.initializer.XavierNormal/XavierUniform /KaimingNormal/KaimingUniform 进行功能增强 |        否        |
|  27  |   ⭐   | 为 paddle.io.RandomSampler/random_split /Layer.clear_gradients 进行功能增强                                |        否        |
|  28  |   ⭐   | 为 paddle.round/paddle.nn.functional.max_pool1d /max_pool2d/max_pool3d 进行功能增强                        |        否        |
|  29  |   ⭐   | 为 paddle.nn.functional.max_unpool1d/max_unpool2d /max_unpool3d/paddle.nn.functional.kl_div 进行功能增强   |        否        |
|  30  |   ⭐   | 为 paddle.nn.functional.max_pool1d/max_pool2d /max_pool3d/paddle.signal.stft 进行功能增强                  |        否        |
|  31  |  ⭐⭐  | paddle Normal 分布支持复数                                                                                 |        是        |
|  32  | ⭐⭐⭐ | paddle Adam 优化器支持复数                                                                                 |        是        |
|  33  |   ⭐   | 支持动态图流水并行设定多个损失函数，并返回多个 loss                                                        |        否        |
|  34  |   ⭐   | 支持动态图流水并行时返回 micro batch 的 loss                                                               |        否        |
|  35  |   ⭐   | 前向重计算函数在 use_reentrant == True 时支持以关键字参数的方式传入 Tensor                                 |        否        |
|  50  |   ⭐   | 将 PyLayer 机制迁移至 PIR 体系下                                 |        否        |

## 科学计算模型复现

| 序号 |    难度     | 任务标题                              | 是否需要提交 rfc |
| :--: | :---------: | ------------------------------------- | :--------------: |
|  36  |   ⭐️⭐️️   | CausalPINN 代码复现                   |        是        |
|  37  |     ⭐️     | GraphCastNet 代码迁移至 PaddleScience |        是        |
|  38  |   ⭐️⭐️️   | LDCast 代码复现                       |        是        |
|  39  |    ⭐️️     | XPINN 迁移至 PaddleScience            |        是        |
|  40  | ⭐️⭐️️⭐️️ | SDGD 优化器实现                       |        是        |
|  41  | ⭐️⭐️️⭐️️ | PIRATENETS 代码复现                   |        是        |
|  42  |  ⭐️⭐️⭐️  | AlphaGeometry 几何推理模型            |        是        |

## 合作伙伴任务（OpneVINO）

| 序号 |    难度    | 任务标题                                                       | 是否需要提交 rfc |
| :--: | :--------: | -------------------------------------------------------------- | :--------------: |
|  43  |  ⭐️⭐️️   | 为 OpenVINO 实现 Paddle 算子 tril/triu 转换                    |        否        |
|  44  |   ⭐️⭐️   | 为 OpenVINO 实现 Paddle 算子 rsqrt 转换                        |        否        |
|  45  |  ⭐️⭐️️   | 为 OpenVINO 实现 Paddle 算子 scaled_dot_product_attention 转换 |        否        |
|  46  | ⭐️⭐️️⭐️ | 为 Openvino 支持 Paddle 2.6.0                                  |        否        |
|  47  |  ⭐️⭐️️️  | 修复 OpenVINO 算子 set_value 问题                              |        否        |
|  48  |  ⭐️⭐️️️  | (预留)CPU 赛题，后续提供                                       |        否        |
|  49  | ⭐️⭐️⭐️  | (预留)CPU 赛题，后续提供                                       |        否        |
