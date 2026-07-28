# 为 nn.functional 增加 dropout1d API

## 详细描述

Paddle 的 `nn.functional` 缺少面向一维通道特征的 dropout 接口，导致需要兼容此类 API 的代码无法直接对 `[C, L]` 或 `[N, C, L]` 输入执行按通道丢弃。该接口需要在训练/推理参数下保持输入形状，并对不合法的概率和输入维度给出明确错误。

## 验收说明

- `paddle.nn.functional.dropout1d` 应支持 2D `[C, L]` 和 3D `[N, C, L]` 输入，并按通道维执行 dropout，同时保持输出维度与输入一致。
- 概率超出 `[0, 1]` 或输入维度不是 2D/3D 时应拒绝调用；当前不支持的 `inplace=True` 应以兼容方式处理而不能静默改变既有语义。
- 现有 dropout 相关 API 的有效行为应保持兼容。

## 技术要求

- 熟悉 Paddle `nn.functional` API 的公开导出方式。
- 理解 channel-wise dropout 与普通 element-wise dropout 的区别。
- 能维护输入维度、参数校验和训练模式相关行为的一致性。
