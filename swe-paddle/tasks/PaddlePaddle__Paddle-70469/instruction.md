# 修复 `fused_dropout_add` 的精度问题

## 详细描述

`paddle.incubate.nn.functional.fused_dropout_add` 当前存在精度问题，计算结果与先执行 `dropout`、再执行加法得到的结果不一致。

在该问题解决前，`fused_dropout_add` 应使用普通的 `dropout + add` 计算，避免继续调用存在问题的 fused 实现。

调用该接口时需要给出提示，说明当前会回退到普通实现。同一次程序运行中，该提示只需要出现一次。

## 验收说明

- `fused_dropout_add` 的结果应与使用相同参数执行 `dropout + add` 一致
- `p`、`training` 和 `mode` 参数应正常生效
- 训练和推理场景都应使用正确的计算结果
- 调用接口时应提示当前会回退到普通实现
- 同一次程序运行中，该提示不应重复出现
- `p=0` 等现有合法调用方式应保持正常

## 技术要求

- 熟悉 Python
- 了解 Paddle 的 `dropout` 和 `fused_dropout_add`
- 了解训练和推理模式下的 dropout 行为
- 了解 Python warning 的使用方式
