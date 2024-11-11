## 科学计算模型复现

| 序号 |   难度    | 任务标题                                                                                        | 是否需要提交 RFC |
| :--: | :-------: | ----------------------------------------------------------------------------------------------- | :--------------: |
|  1   |    ⭐     | 为开源符号回归库进行 paddle 适配                                                                |        是        |
|  2   |  ⭐️⭐️   | Transolver 论文复现                                                                             |        是        |
|  3   |  ⭐️⭐️   | DrivAerNet ++ 论文复现                                                                          |        是        |
|  4   |  ⭐️⭐️   | DrivAerNet 论文复现                                                                             |        是        |
|  5   |  ⭐️⭐️   | Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations 论文复现 |        是        |
|  6   |  ⭐️⭐️   | Synthetic Lagrangian turbulence by generative diffusion models 论文复现                         |        是        |
|  7   |  ⭐️⭐️   | AI-aided geometric design of anti-infection catheters 论文复现                                  |        是        |
|  8   |  ⭐️⭐️   | A physics-informed diffusion model for high-fidelity flow field reconstruction 论文复现         |        是        |
|  9   |    ⭐     | DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting 论文复现      |        是        |
|  10  |  ⭐️⭐️   | Neural General Circulation Models for Weather and Climate 论文复现                              |        是        |
|  11  |    ⭐     | FuXi: A cascade machine learning forecasting system for 15-day global weather forecast 论文复现 |        是        |
|  12  |  ⭐️⭐️   | Adam、AdamW 优化器支持 amsgrad 任务                                                             |        是        |
|  13  |  ⭐️⭐️   | put_along_axis 反向算子实现静态图一阶拆解任务                                                   |        是        |
|  14  |  ⭐️⭐️   | Crystal Diffusion Variational AutoEncoder 论文复现                                              |        是        |
|  15  |  ⭐️⭐️   | SchNet 论文复现                                                                                 |        是        |
|  16  | ⭐️⭐️⭐️ | MACE 论文复现                                                                                   |        是        |
|  17  |  ⭐️⭐️   | PIKAN 论文复现                                                                                  |        是        |

## 框架开发任务

| 序号 |  难度  | 任务标题                                                                                                       | 是否需要提交 RFC |
| :--: | :----: | -------------------------------------------------------------------------------------------------------------- | :--------------: |
|  18  | ⭐️⭐️ | 为稀疏计算添加复数支持                                                                                         |        否        |
|  19  |   ⭐   | 为 Paddle 新增 load_state_dict_from_url API                                                                    |        是        |
|  20  |   ⭐   | 为 Paddle 新增 Tensor.set* / Tensor.resize* API                                                                |        是        |
|  21  |  ⭐⭐  | 为 Paddle 新增 reset_peak_memory_stats/reset_max_memory_allocated/memory_stats API                             |        是        |
|  22  |   ⭐   | 在 paddle.audio.functional.get_window 中支持 bartlett 、 kaiser 和 nuttall 窗函数                              |        是        |
|  23  |   ⭐   | 为 Paddle 新增 ParameterDict API                                                                               |        是        |
|  24  |  ⭐⭐  | 为 Paddle 新增 EmbeddingBag API                                                                                |        是        |
|  25  |   ⭐   | 为 Paddle 新增 is_coalesced/sparse_dim/dense_dim API                                                           |        是        |
|  26  |  ⭐⭐  | 为 Paddle 新增 lu_solve API                                                                                    |        是        |
|  27  | ⭐⭐⭐ | 为 Paddle 新增 register_parametrization/remove_parametrizations/cached/ParametrizationList/is_parametrized API |        是        |
|  28  |   ⭐   | 为 `paddle.clip` 进行功能增强                                                                                  |        否        |
|  29  |   ⭐   | 为 `paddle.grad` 进行功能增强                                                                                  |        否        |
|  30  |   ⭐   | 为 `paddle.divide` 进行功能增强                                                                                |        否        |
|  31  |   ⭐   | 为 `paddle.sparse.sparse_csr_tensor`进行功能增强                                                               |        否        |
|  32  |   ⭐   | 为 `paddle.nn.functional.scaled_dot_product_attention` 进行功能增强                                            |        否        |
|  33  |   ⭐   | 为 `paddle.nn.MaxPool1D/MaxPool2D/MaxPool3D` 及其对应 functional API 增加 dilation 参数                        |        否        |
|  34  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 1 组）                                                             |        否        |
|  35  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 2 组）                                                             |        否        |
|  36  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 3 组）                                                             |        否        |
|  37  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 4 组）                                                             |        否        |
|  38  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 5 组）                                                             |        否        |
|  39  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 6 组）                                                             |        否        |
|  40  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 7 组）                                                             |        否        |
|  41  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 8 组）                                                             |        否        |
|  42  |   ⭐   | 为 Paddle 代码转换工具新增 API 转换规则（第 9 组）                                                             |        否        |

## 套件开发任务

| 序号 |   难度    | 任务标题                                          | 是否需要提交 RFC |
| :--: | :-------: | ------------------------------------------------- | :--------------: |
|  43  | ⭐⭐️⭐️  | 完善 TokenizerFast 功能支持                       |        是        |
|  44  | ⭐️⭐️⭐️ | 大模型 4D 并行框架全自动构建                      |        是        |
|  45  | ⭐️⭐️⭐️ | 添加 FunctionCall 功能                            |        是        |
|  46  |    ⭐     | Paddle2ONNX 添加对返回常量的 IfElse 算子的支持    |        否        |
|  47  |  ⭐️⭐️   | Paddle2ONNX 添加对 While 算子的支持               |        否        |
|  48  |   ⭐⭐️   | Paddle2ONNX 添加对 Windows 平台自动发包机制的支持 |        否        |
|  49  |  ⭐️⭐️   | PaddleX 重要模型的量化能力验证和优化              |        是        |
|  50  |  ⭐️⭐️   | PaddleX 重要模型 Android Demo 支持                |        是        |
|  51  | ⭐️⭐️⭐️ | 在 PaddleOCR 中复现 MixTeX 模型                   |        是        |
|  52  | ⭐️⭐️⭐️ | 论文复现：OmniParser                              |        是        |
|  53  |  ⭐️⭐️   | 在 PaddleOCR 中复现 TrOCR-Formula-Rec 模型        |        是        |
|  54  | ⭐️⭐️⭐️ | 在 PaddleSpeech 中实现 Whisper 的 Finetune        |        否        |
|  55  | ⭐️⭐️    | 在 PaddleSpeech 中实现 DAC 的训练中使用的第三方库 audiotools |        是        |
|  56  | ⭐️      | 在 PaddleSpeech 中复现 DAC 的训练需要用到的 loss (依赖 55)   |        是        |
|  57  | ⭐️⭐️    | 在 PaddleSpeech 中复现 DAC 模型（依赖 55、56）      |        是        |
|  58  | ⭐️⭐️    | VisualDL PIR 可视化产品形态改进      |        否        |
