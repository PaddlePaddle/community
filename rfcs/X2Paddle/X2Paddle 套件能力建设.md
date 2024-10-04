# X2Paddle 套件能力建设

| 领域         | X2Paddle 套件能力建设    |
| ------------ | ---------------------------------- |
| 提交作者     | megemini (柳顺)                    |
| 提交时间     | 2024-10-04                         |
| 版本号       | V1.0                               |
| 依赖飞桨版本 | develop 分支                       |
| 文件名       | X2Paddle 套件能力建设.md |

# 一、概述

## 1、相关背景

X2Paddle 是飞桨生态下的模型转换工具，致力于帮助其它深度学习框架（Caffe/TensorFlow/ONNX/PyTorch）用户快速迁移至飞桨框架。

由于 Paddle 即将发布的 3.0 版本较之 X2Paddle (目前的版本 v1.5.0) 所支持的 Paddle 最高版本 2.4.2 存在较大的特性变更，如 `paddle.fluid` 全面退场、0-D tensor 的引入等，因此，需要对 X2Paddle 进行一次全面的版本适配升级，并保证在新版本的依赖环境下，模型转换工具能够正常运转，转换后模型的功能与精度不受损失。

## 2、功能目标

本次任务的目标为：

- 适配最新的 Paddle 与其他深度学习框架的版本 (截止 2024年9月)

  - python == 3.8
  - paddlepaddle == 3.0.0 (beta)
  - tensorflow == 2.16 (最低 2.0 版本)
  - onnx == 1.17 (最低 1.6 版本)
  - torch == 2.4 (最低 2.0 版本)
  - caffe == 1.0

相应的验收标准为：

- test_benchmark 目录下模型适配测试通过。

## 3、意义

通过适配 Paddle 与其他深度学习框架的最新版本，可以保证 X2Paddle 的持续可用性，以适应用户对于模型迁移的需求，以及未来新模型的转换等工作。

# 二、现状

目前 X2Paddle 的依赖适配环境为：

- python >= 3.5
- paddlepaddle >= 2.2.2 (官方验证到2.4.2)
- tensorflow == 1.14 (如需转换TensorFlow模型)
- onnx >= 1.6.0 (如需转换ONNX模型)
- torch >= 1.5.0 (如需转换PyTorch模型)

# 三、设计思路与实现方案

## 实施方案

本次任务以 `目标驱动` 的方式进行适配升级。

进行版本的适配升级任务，至少可以有两种实施方案：

- 版本驱动

  即，根据新版本发布的特性，逐个批量改动软件中相应的接口。

  如，ONNX 发布新的 opset，X2Paddle 则增加对应的接口映射。

- 目标驱动

  即，根据目标任务的需要，仅改动特定范围内的接口。

  如，ONNX 发布新的 opset，X2Paddle 中已验证支持的模型无需增加接口映射。

本次任务的 `目标驱动` 是指：

- 以完成 test_benchmark 目录下模型的适配，并通过测试为目标进行修改。

其原因为

- `版本驱动` 改动量过大，`目标驱动` 在保证兼容性的前提下更具有可行性。

以 ONNX 为例，X2Paddle 目前支持的版本为 `onnx >= 1.6.0` ，ONNX 最新的版本为 `1.17` ，其中，`1.6.0` 版本的 opset ai.onnx version 为 `11` ，`1.17` 版本的 opset 为 `22` ，期间修改与新增了较多的算子，如果采用 `版本驱动` 方式进行升级，则需要逐个对各版本的算子添加映射关系以及测试验证，工作量较大。而采用 `目标驱动` 的方式，如无必要（新模型、新特性等），不修改或增加接口的支持，基本可以保证原有算子映射的兼容性。

## 实施原则

具体适配过程中，遵从以下原则：

- 破坏性升级

  破坏性升级是指，遇到兼容性问题，直接以新版本接口代替；

  非破坏性升级是指，遇到兼容性问题，根据版本不同调用相应接口。

  如，`paddle.fluid` 在 Paddle 2.6.0 版本已经全面退场，破坏性升级，如以下代码：

  ``` python
  import paddle.fluid as fluid
  [prog, inputs, outputs] = fluid.io.load_inference_model(
      dirname="pd_model_trace/inference_model/",
      executor=exe,
      model_filename="model.pdmodel",
      params_filename="model.pdiparams")
  ```

  修改为：

  ``` python
  import paddle
  [prog, inputs, outputs] = paddle.static.load_inference_model(
      path_prefix='pd_model_trace/inference_model/model', executor=exe)

  ```

  非破坏性升级，则修改为：

  ``` python
  import paddle

  def match_version(paddle_version: str, required: tuple(int, int, int)) -> bool:
    """ check paddle version match the required. """
    ...

  compatible_flag = False
  if not match_version(paddle.__version__, (2, 6, 0)):
    import paddle.fluid as fluid
    compatible_flag = True

  if compatible_flag:
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_trace/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
  else:
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix='pd_model_trace/inference_model/model', executor=exe)

  ```

  以上代码中，首先检查当前 Paddle 的版本是否满足 `>= 2.6.0` ，若不满足，则导入 `paddle.fluid` 模块，并置 `compatible_flag = True` 。后续遇到 `fluid` 接口时，根据标志位走不同的逻辑路径。

  > **说明:** 本次方案建议采用 `破坏性升级` ，但仍需根据 CI 的设置等问题进行决策。

- 以 Paddle 版本为切入点

  由于适配升级涉及 Paddle，PyTorch，ONNX 等多个框架的版本改动，为了保证适配动作的一致性，需要以 Paddle 的版本为切入点，即，

  - 首先保证 Paddle 版本 (3.0.0 beta) 不存在特性不兼容的情况
  - 其他框架以最新版本为基准 (参考上文 `功能目标` 中的版本)
  - 其他框架在适配的过程中，如果遇到不兼容的情况，可视情况采取以下方式：

    - 修改 X2Paddle 进行适配
    - 降子版本，如 `2.3` 降为 `2.2` (不得小于任务目标最低版本要求)
    - 降主版本，如 `2.3` 降为 `1.x` (不得小于任务目标最低版本要求，非特殊情况不采用此方式)


## 实施步骤


TODO:
- 统计 fluid 的数量，test 中，x2paddle 源码中
- 开总 issue
- 开子 issue 跟踪 fluid
- 第一轮，先统一修改 fluid，如果遇到其他问题，先标记
- 第二轮，根据第一轮的问题统一、逐批次修复


# 四、测试和验收的考量


# 五、排期规划


# 六、影响面
