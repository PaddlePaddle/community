### 升级paddlenlp.transformers内的模型结构并且增加基础单测 设计文档

| API名称                                                      | 新增API名称                                      |
| ------------------------------------------------------------ | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">   | ZwhElliott                                       |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-21                                       |
| 版本号                                                       | V1.0                                             |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本                                      |
| 文件名                                                       | 20230221_paddlenlp_transformers_configuration.md |


# 一、概述
## 1、相关背景
[【PaddlePaddle Hackathon 4】模型套件开源贡献任务合集 · Issue #50631 · PaddlePaddle/Paddle (github.com)](https://github.com/PaddlePaddle/Paddle/issues/50631#task111)

## 2、功能目标

- 升级[MegatronBERT](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/megatronbert),[artist](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/artist),[dallebart](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/dallebart)模型的PaddleNLP模型结构，每个模型的主要工作为：
  - 为模型结构增加configuration.py, 对齐huggingface/transformers的config，并且适配在模型代码中适配config
  - 为模型增加单测, 并且做到单测通过

## 3、意义

使得Paddle能够直接使用该模型应用到各项自然语言处理子任务。

# 二、飞桨现状
目前模型代码处虽然已有参数，但是未将参数统一集成至一个文件，在修改难度和可读性上仍有缺陷。


# 三、业内方案调研
haggleface中，将模型参数集成至configuration.py中的cofig对象中，模型只需要继承并微调相应参数即可。

# 四、对比分析
当前现有方法虽然可以使用，但是当需要修改参数时，需要在代码中找到相应位置，十分困难。同时对于初次阅读代码的人会造成困难，难以找到对应的模型参数。

# 五、设计思路与实现方案

## 命名与参数设计
- 将原model.py中定义的参数移植到configuration.py文件中，保留注释。如：

  num_hidden_layers等

- 同时参照现有文件中的命名方式定义新参数。如：

  - 模型下载文件map命名为：[模型名称]_PRETRAINED_RESOURCE_FILES_MAP

  - 模型各版本参数命名为: MegatronBert_PRETRAINED_INIT_CONFIGURATION

# 六、测试和验收的考量
根据model文件中的各项任务中需要的输入以及预期的输出，设计test_model文件。

# 七、可行性分析和排期规划
于2023年3月7日前，完成三个模型的参数移植，将现有参数移植至configuration.py文件中。

于2023年4月1日前，编写测试文件，检测模型,完成验收。

总体可以在活动时间内完成。

# 八、影响面
模型仍然为一定程度的封装形式，对其他模块无影响

