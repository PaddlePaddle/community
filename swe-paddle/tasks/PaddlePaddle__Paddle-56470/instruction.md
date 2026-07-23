# 为 2D Upsampling layers 支持单个整数 size

## 详细描述

`UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 的输出尺寸参数目前要求使用二元素 list 或 tuple。对于正方形输出，用户仍需重复填写相同的高度和宽度，降低了 API 易用性。两个 2D Upsampling layers 应支持直接传入单个整数，并将其解释为相同的输出高度和宽度。

## 验收说明

- `UpsamplingNearest2D` 接收单个整数 `size` 时，应在高度和宽度两个维度使用该值。
- `UpsamplingBilinear2D` 接收单个整数 `size` 时，应在高度和宽度两个维度使用该值。
- 现有 list、tuple 和 `scale_factor` 调用方式应保持兼容。

## 技术要求

- 熟悉 Python 和 Paddle layer API。
- 了解 2D image interpolation 的尺寸参数约定。
- 能够编写稳定的 API behavior tests。
