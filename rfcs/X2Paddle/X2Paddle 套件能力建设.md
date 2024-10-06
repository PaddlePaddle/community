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

X2Paddle 是飞桨生态下的模型转换工具，致力于帮助其它深度学习框
架（Caffe/TensorFlow/ONNX/PyTorch）用户快速迁移至飞桨框架。

由于 Paddle 即将发布的 `3.0.0` 版本较之 X2Paddle (目前的版本 v1.5.0) 所支持的 Paddle 最高版本 `2.4.2` 存在较大的特性变更，如 `paddle.fluid` 全面退场、0-D tensor 的引入等，因此，需要对 X2Paddle 进行一次全面的版本适配升级，并保证在新版本的依赖环境下，模型转换工具能够正常运转，转换后模型的功能与精度不受损失。

## 2、功能目标

本次任务的目标为：

- 适配最新的 Paddle 版本 `3.0.0 (beta)`
-

验收标准为：

- `test_benchmark` 目录下模型适配测试通过

`test_benchmark` 中所涉及的模型：

- `Caffe`： `20` 个模型
- `ONNX`： `57` 个模型
- `PyTorch`： `33` 个模型
- `TensorFlow`： `28` 个模型

共涉及 `4` 个框架，`138` 个模型。

验收的测试环境为 (截止 2024年9月 统计)

- python == 3.8
- paddlepaddle == 3.0.0 (beta)
- tensorflow == 2.16 (不低于 1.14 版本)
- onnx == 1.17 (不低于 1.6 版本)
- torch == 2.4 (不低于 1.5 版本)
- caffe == 1.0

不改变 X2Paddle 对外声明的深度学习框架支持版本

- tensorflow == 1.14
- onnx >= 1.6.0
- torch >= 1.5.0

> **注意：** 代码中 ONNX 支持的版本应该为 `1.10` ，PyTorch 为 `1.7` 。此处仍按照 `README.md` 中的文档为准。

需要说明的是，上述的测试验收环境，不同于 X2Paddle 可以声明的全面支持的版本，区别在于：

- 验证的测试环境采用最新版本

  保证 X2Paddle 可以在此环境下工作，至少能够完成 `test_benchmark` 内模型的转换

- 不改变原有声明的支持版本

  是因为，此次任务不涉及全面支持其他深度学习框架最新版本所涉及的特性。如，不涉及添加支持 ONNX 新的 opset 接口映射。

## 3、意义

通过适配 Paddle 与其他深度学习框架的最新版本，可以保证 X2Paddle 的持续可用性，以适应用户对于模型迁移的需求，以及未来新模型的转换等工作。

# 二、现状

目前 X2Paddle 的依赖适配环境为：

- python >= 3.5
- paddlepaddle >= 2.2.2 (官方验证到2.4.2)
- tensorflow == 1.14 (如需转换TensorFlow模型)
- onnx >= 1.6.0 (如需转换ONNX模型)
- torch >= 1.5.0 (如需转换PyTorch模型)

软件中仍存在 Paddle 旧版本的软件接口，如 `paddle.fluid` ，从而无法在 Paddle 3.0.0 (beta) 的环境中使用。

另外，软件所声明的依赖环境也较老旧，需要验证在较新的环境中，X2Paddle 仍然可以正常使用，也为未来全面升级提供一定的参考条件。

# 三、设计思路与实现方案

## 实施范围

- `X2Padde/test_benchmark`

  基线模型，需要修改不兼容的接口，通过测试

- `X2Padde/x2paddle`

  逻辑代码，需要修改不兼容的接口，通过测试

- `X2Padde/docs`

  文档，修改 `paddlepaddle` 的版本描述为 `3.0.0 (beta)`

- `X2Paddle/docker`

  docker 配置文件，修改 `paddlepaddle` 的版本为 `3.0.0 (beta)`

- 其他

  如 `README.md` 等，修改 `paddlepaddle` 的版本描述为 `3.0.0 (beta)`

本次任务主要目标为 `test_benchmark` 目录下模型适配测试通过，下文的实现方案以修改 `test_benchmark` 的 Paddle 兼容性为主。

## 实施方案

本次任务以 `目标驱动` 的方式进行适配升级。

进行版本的适配升级任务，至少可以有两种实施方案：

- 版本驱动

  即，根据新版本发布的特性，逐个批量改动或新增软件中相应的接口。

  类似 X2Paddle 之类的 "中间件" ，理应跟随各个框架一同升级，如，每次 ONNX 发布新的版本或 opset ，则 X2Paddle 一同提供接口映射。

  但是，由于历史遗留原因，X2Paddle 停留在了较早的框架版本上，一次性跨越多个版本做全面的适配升级不具备短时间内的可操作性。

- 目标驱动

  即，根据目标任务的需要，仅改动特定范围内的接口。

  如，虽然 ONNX 发布了新的 opset，X2Paddle 中已验证支持的模型不涉及，则无需增加接口映射。

本次任务为 `目标驱动` ：

- 以完成 `test_benchmark` 目录下模型的适配，并通过测试为目标进行修改。

上面提到，`版本驱动` 的方案改动量过大，`目标驱动` 在保证兼容性的前提下更具有可行性。

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

  def match_version(paddle_version: str, required: tuple[int, int, int]) -> bool:
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
  - 其他框架以最新版本为基准 (参考上文 `功能目标` 中的测试验证版本)
  - 其他框架在适配的过程中，如果遇到不兼容的情况，可视情况采取以下方式：

    - 修改 X2Paddle 进行适配
    - 降子版本，如 `2.3` 降为 `2.2` (不得小于任务目标最低版本要求)
    - 降主版本，如 `2.3` 降为 `1.x` (不得小于任务目标最低版本要求，非特殊情况不采用此方式)

## 实施步骤

上文提到，`以 Paddle 版本为切入点` ，X2Paddle 目前支持的 Paddle 版本为 `2.4.2` ，相较于 `3.0.0 (beta)` 版本，至少存在以下不兼容特性升级：

- `paddle.fluid` API 全面退场
- 飞桨 API 支持 0 维 tensor
- 飞桨 API 支持隐式类型提升
- 将 `paddle.nn.functional.diag_embed` 精简为 `paddle.diag_embed`

> **说明：** 参考官网 `Release Note` 中的 `不兼容升级` ，仅列举与本任务相关的特性。

其中，能够比较直观进行修改的为第一个特性：`paddle.fluid API 全面退场` ，即，修改所有 `paddle.fluid` 接口为 3.0.0 (beta) 中对应的接口。

后几个不兼容特性，可以通过验证 `test_benchmark` 中模型转换后的功能与精度来进行修改。

因此，本次任务采用 `批量多轮、修改验证` 的步骤进行：

- 第一轮

  - 修改： `paddle.fluid` 接口
  - 验证： `test_benchmark` 不通过则统一放到下一轮

- 第二轮

  - 修改： 第一轮中 `test_benchmark` 不通过的模型，分析最多共性的问题
  - 验证： `test_benchmark` 不通过则统一放到下一轮

- 第三轮

  - 重复 `第二轮` 的步骤，直至问题收敛。

因此，需要创建 tracking issue 跟踪任务进展：

- 一个 `总 tracking issue`，跟踪整体任务进展，如，多个轮次的修改进展
- 多个 `子 tracking issue`，如，修改 `paddle.fluid` 接口

由于后面多轮的子任务目前不能确定，这里仅阐述第一轮 `paddle.fluid` 接口的修改。

### `paddle.fluid` 接口的修改

统计目前 X2Paddle 中遗留的 `paddle.fluid` 接口：

- `X2Padde/test_benchmark` 目录共涉及 `220` 个文件
- `X2Padde/x2paddle` 目录共涉及 `4` 个文件
- `X2Padde/docs` 目录共涉及 `4` 个文件

其中，`X2Padde/test_benchmark` 中涉及的 `paddle.fluid` 接口改动量最大，主要为加载推理模型，即 `fluid.io.load_inference_model`。

参考上文中 `破坏性升级原则` ，可将

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

`X2Padde/x2paddle` 与 `X2Padde/docs` 中涉及的接口较少，可以在上述 `X2Padde/test_benchmark` 修改完之后一并修改。

# 四、测试和验收的考量

本次任务以 `test_benchmark` 中模型的功能与精度不受损失为验收标准。

# 五、排期规划

- 完成 `paddle.fluid` 接口的修改，大约 2 ～ 3 周
- 完成 `X2Padde/x2paddle` 与 `X2Padde/docs` 的修改，大约 1 ～ 2 周
- 完成其他未知兼容性错误的修改，大约 2 ～ 3 周 (这部分工作量目前较难评估)
- 完成 `docs`、`docker` 等修改与配置，大约 1 周

# 六、影响面

若采用 `破坏性升级` ，则存在以下影响：

**对用户的影响：** 使用旧环境的用户 (如 Paddle 的 2.4.2 版本)，无法使用新的接口

**对开发者的影响：** 开发者后续统一使用 Paddle 的 3.0.0 (beta) 的接口开发框架

**其他风险：** 修改过程中，CI 如何配置测试环境需要重点考虑
