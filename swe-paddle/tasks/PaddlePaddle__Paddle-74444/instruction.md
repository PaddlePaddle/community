# 新增 `paddle.nn.functional.dropout1d`

## 详细描述

Paddle 目前没有 `paddle.nn.functional.dropout1d`。对于 `[C, L]` 或 `[N, C, L]` 形状的输入，用户无法直接按通道执行 dropout。

新增该接口后，训练时以概率 `p` 将整个通道置零，而不是随机对通道中的单个元素置零。输出 shape 应与输入保持一致；`training=False` 时不执行 dropout。

该接口只支持二维和三维输入，`p` 必须在 `[0, 1]` 范围内。`inplace` 参数暂时不生效，传入 `inplace=True` 时应给出警告，并按非 `inplace` 方式返回结果。

## 验收说明

- 可以通过 `paddle.nn.functional.dropout1d` 调用该接口
- 支持 `[C, L]` 和 `[N, C, L]` 形状的输入
- 训练时应按通道执行 dropout，并保持输出 shape 不变
- `training=False` 时，输出应与输入一致
- `p` 不在 `[0, 1]` 范围内时应报错
- 输入不是二维或三维 Tensor 时应报错
- `inplace=True` 时应给出警告，且不修改输入 Tensor
- 现有 `dropout`、`dropout2d` 和 `dropout3d` 的行为保持不变

## 技术要求

- 熟悉 Python
- 了解 Paddle `nn.functional` API
- 了解普通 dropout 和按通道 dropout 的区别
