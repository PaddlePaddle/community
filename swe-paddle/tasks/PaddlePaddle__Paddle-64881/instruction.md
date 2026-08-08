# 新增 FeatureAlphaDropout API

## 详细描述

为 Paddle 新增 `FeatureAlphaDropout` 层与 `feature_alpha_dropout` 函数式 API。Feature Alpha Dropout 以整个 channel(特征图)为单位随机置零,同时保持 Alpha Dropout 的自归一化(self-normalizing)性质;被置零的 channel 中激活值被设置为负饱和值,参考论文 *Self-Normalizing Neural Networks* (https://arxiv.org/abs/1706.02515)。

要求支持:

- `paddle.nn.FeatureAlphaDropout` 层,参数 `p`(置零概率,默认 0.5)与 `name`(可选)
- `paddle.nn.functional.feature_alpha_dropout(x, p, training, name)` 函数
- channel 级 dropout 行为:输入至少 2-D,按前两个维度(批次、channel)生成 mask,其余维度整体置零或保留;`x.ndim < 2` 时应报错
- 与现有 `alpha_dropout` 共享同一实现(`_feature_alpha_dropout_impl`),**`alpha_dropout` 的现有行为不能改变**
- `training=False` 时不做 dropout,直接返回输入

实现提示(来自 `_feature_alpha_dropout_impl` 的既有逻辑):

- 校验 `p` 必须为 float/int 且 ∈ [0, 1]
- 生成随机 mask 时,feature 模式下 `input_shape` 取 `list(x.shape[:2]) + [1] * len(x.shape[2:])`,使同一 channel 内元素共享同一个 mask 值
- 计算 `alpha_p = 1 - p`、`a = 1 / sqrt(alpha_p + alpha_p * (1 - alpha_p))`、`b = -a * alpha_p * p` 等缩放参数,对未被置零的激活做仿射缩放

## 验收说明

- `feature_alpha_dropout` / `FeatureAlphaDropout` 前向行为正确(channel 整体置零或保留)
- 输入不足 2-D 时抛出 `ValueError`
- 与 `alpha_dropout` 共用实现,`alpha_dropout` 行为保持不变
- 静态图与动态图下均可用

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
