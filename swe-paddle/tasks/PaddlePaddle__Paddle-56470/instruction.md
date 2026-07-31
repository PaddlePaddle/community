# 支持 2D Upsampling 层使用单个整数指定输出尺寸

## 详细描述

`UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 当前要求通过包含两个元素的 list 或 tuple 指定输出尺寸。对于高度和宽度相同的输出，调用方仍需重复传入相同的数值。

请支持使用单个整数设置这两个层的 `size` 参数。当 `size` 为整数时，该值应同时作为输出的高度和宽度。

现有的 list、tuple 以及 `scale_factor` 参数用法不得受到影响。

## 验收说明

* `UpsamplingNearest2D` 应支持使用单个整数设置 `size`
* `UpsamplingBilinear2D` 应支持使用单个整数设置 `size`
* 当 `size` 为整数时，输出的高度和宽度均应使用该值
* 现有 list 和 tuple 形式的 `size` 参数应保持原有行为
* 现有 `scale_factor` 调用方式应保持原有行为
* 两个层现有的插值结果、参数校验和其他合法调用方式不得发生变化

## 技术要求

* 熟悉 Python 和 Paddle Layer API
* 了解二维图像插值及输出尺寸参数的语义
* 理解 `size` 和 `scale_factor` 的使用方式
