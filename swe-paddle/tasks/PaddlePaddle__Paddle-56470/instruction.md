# 让 `UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 支持整数 `size`

## 详细描述

`UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 的 `size` 目前需要传入包含两个元素的 list 或 tuple。

这两个层还应支持直接传入一个整数。整数 `size` 表示输出的高度和宽度使用相同的值。例如，`size=12` 应按 `size=[12, 12]` 处理。

原有的 list、tuple 和 `scale_factor` 用法应保持不变。

## 验收说明

* `UpsamplingNearest2D` 支持整数 `size`
* `UpsamplingBilinear2D` 支持整数 `size`
* 整数 `size` 应转换为两个相同的尺寸值
* list 和 tuple 形式的 `size` 保持原有行为
* `scale_factor` 的现有用法保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle Layer API
* 了解二维上采样的 `size` 参数
