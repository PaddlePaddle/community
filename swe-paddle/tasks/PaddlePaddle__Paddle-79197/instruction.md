# 让 learning-rate scheduler 可以直接接收 optimizer

## 详细描述

很多训练代码会先创建 optimizer，再把这个 optimizer 传给 learning-rate scheduler。这样 scheduler 可以直接使用 optimizer 当前的学习率，并在后续训练中负责更新它。

目前 Paddle 的常用 scheduler 只接受一个数值形式的 `learning_rate`。开发者把 optimizer 作为位置参数或使用 `optimizer=` 传入时，会收到参数类型错误或不支持该参数的报错。即使手动取出学习率创建 scheduler，还需要额外处理 optimizer 和 scheduler 的关联，迁移代码比较麻烦。

需要让常用 scheduler 同时支持两种写法：继续接受原来的数值学习率，也可以直接接受已经创建好的 optimizer。使用 optimizer 创建 scheduler 后，scheduler 应从 optimizer 取得初始学习率，并由 optimizer 在训练过程中使用。

## 验收说明

- 常用 scheduler 可以通过位置参数或 `optimizer=` 接收已有 optimizer，并使用它当前的学习率。
- 创建完成后，optimizer 应使用这个 scheduler；调用 `step()` 时，学习率仍按各 scheduler 原有规则变化。
- 原有的 `learning_rate` 数值调用方式和计算结果保持不变。
- 同时传入 `learning_rate` 和 `optimizer` 时，应给出清楚的参数冲突错误。

## 技术要求

- 熟悉 PaddlePaddle optimizer 和 learning-rate scheduler 的使用方式。
- 熟悉 Python 函数的位置参数、关键字参数和参数校验。
- 能够验证 scheduler 与 optimizer 的关联以及学习率变化结果。
