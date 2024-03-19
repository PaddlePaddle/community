## 框架开发任务

| 序号 |  难度  | 任务标题                                                                                                   | 是否需要提交 rfc |
| :--: | :----: | ---------------------------------------------------------------------------------------------------------- | :--------------: |
|  1   |   ⭐   | 为 Paddle 新增 AdaptiveLogSoftmaxWithLoss API                                                              |                  |
|  2   |   ⭐   | 为 Paddle 新增 cholesky_inverse API                                                                        |                  |
|  3   |   ⭐   | 为 Paddle 新增 / ZeroPad3D API                                                                             |                  |
|  4   |   ⭐   | 为 Paddle 新增 block_diag / block_diag API                                                                 |                  |
|  5   |   ⭐   | 为 Paddle 新增 ormqr API                                                                                   |                  |
|  6   |   ⭐   | 为 Paddle 新增 Chi2 / LKJCholesky API                                                                      |                  |
|  7   |   ⭐   | 为 Paddle 新增 MultivariateNormal / StudentT API                                                           |                  |
|  8   |  ⭐⭐  | 为 Paddle 新增 PolynomialLR / sinc / sinc\_ API                                                            |                  |
|  9   |   ⭐   | 为 Paddle 新增 FeatureAlphaDropout API                                                                     |                  |
|  10  |   ⭐   | 为 Paddle 新增 cartesian_prod API                                                                          |                  |
|  11  |  ⭐⭐  | 为 Paddle 新增 isposinf / isneginf / isreal / isin API                                                     |                  |
|  12  |  ⭐⭐  | 为 Paddle 新增 bernoulli\_ / log_normal\_ / log_normal API                                                 |                  |
|  13  |  ⭐⭐  | 为 Paddle 新增 lu_solve API                                                                                |                  |
|  14  | ⭐⭐⭐ | 为 Paddle 新增 RAdam / NAdam API                                                                           |                  |
|  15  |  ⭐⭐  | 为 Paddle 新增 tensorinv / tensorsolve API                                                                 |                  |
|  16  | ⭐⭐⭐ | 为 Paddle 新增 ldl_factor / ldl_solve API                                                                  |                  |
|  17  | ⭐⭐⭐ | 为 Paddle 新增 LPPool1D / LPPool2D API                                                                     |                  |
|  18  |   ⭐   | 为 Paddle 新增 sparse.mask_as API                                                                          |                  |
|  19  |  ⭐⭐  | 为 Paddle 新增 sparse.concat API                                                                           |                  |
|  20  |  ⭐⭐  | 为 Paddle 新增 sparse.stack API                                                                            |                  |
|  21  |  ⭐⭐  | 为 Paddle 新增 sparse.nn.Conv2DTranspose / Conv3DTranspose API                                             |                  |
|  22  |  ⭐⭐  | 为 Paddle 新增 sparse.nn.InverseConv2D / InverseConv3D API                                                 |                  |
|  23  | ⭐⭐⭐ | 为 Paddle 增强 sparse.add / subtract / multiply / divide API                                               |                  |
|  24  |   ⭐   | 为 paddle.nn.functional.embedding/paddle.nn.Embedding 增加参数 max_norm/norm_type/scale_grad_by_freq       |                  |
|  25  |   ⭐   | 为 paddle.nn.LSTM/RNNBase /paddle.quantile/nanquantile 功能增强                                            |                  |
|  26  |   ⭐   | 为 paddle.histogram/paddle.nn.functional.threshold 进行功能对齐与功能增强                                  |                  |
|  27  |   ⭐   | 为 paddle.view/paddle.nn.initializer.XavierNormal/XavierUniform /KaimingNormal/KaimingUniform 进行功能增强 |                  |
|  28  |   ⭐   | 为 paddle.io.RandomSampler/random_split /Layer.clear_gradients 进行功能增强                                |                  |
|  29  |   ⭐   | 为 paddle.round/paddle.nn.functional.max_pool1d /max_pool2d/max_pool3d 进行功能增强                        |                  |
|  30  |   ⭐   | 为 paddle.nn.functional.max_unpool1d/max_unpool2d /max_unpool3d/paddle.nn.functional.kl_div 进行功能增强   |                  |
|  31  |   ⭐   | 为 paddle.nn.functional.max_pool1d/max_pool2d /max_pool3d/paddle.signal.stft 进行功能增强                  |                  |
|  32  |  ⭐⭐  | paddle Normal 分布支持复数                                                                                 |                  |
|  33  | ⭐⭐⭐ | paddle Adam 优化器支持复数                                                                                 |                  |
|  34  |   ⭐   | 支持动态图流水并行设定多个损失函数，并返回多个 loss                                                        |                  |
|  35  |   ⭐   | 支持动态图流水并行时返回 micro batch 的 loss                                                               |                  |
|  36  |   ⭐   | 前向重计算函数在 use_reentrant == True 时支持以关键字参数的方式传入 Tensor                                 |                  |

## 科学计算模型复现

| 序号 |    难度     | 任务标题                              | 是否需要提交 rfc |
| :--: | :---------: | ------------------------------------- | :--------------: |
|  1   |   ⭐️⭐️️   | CausalPINN 代码复现                   |                  |
|  2   |     ⭐️     | GraphCastNet 代码迁移至 PaddleScience |                  |
|  3   |   ⭐️⭐️️   | LDCast 代码复现                       |                  |
|  4   |    ⭐️️     | XPINN 迁移至 PaddleScience            |                  |
|  5   | ⭐️⭐️️⭐️️ | SDGD 优化器实现                       |                  |
|  6   | ⭐️⭐️️⭐️️ | PIRATENETS 代码复现                   |                  |
|  7   |  ⭐️⭐️⭐️  | AlphaGeometry 几何推理模型            |                  |
