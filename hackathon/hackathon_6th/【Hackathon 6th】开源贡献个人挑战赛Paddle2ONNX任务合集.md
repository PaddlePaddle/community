## 任务背景

由社区用户 [chenwhql](https://github.com/chenwhql)、[luotao1](https://github.com/luotao1)、 [goocody](https://github.com/goocody)、[jeff41404](https://github.com/jeff41404)、 [jzhang553](https://github.com/jzhang533)、[ZhengBicheng](https://github.com/ZhengBicheng) 于 2024 年 03 月 28 日向 Paddle2ONNX PMC 捐赠共 10000 元人名币用于 Paddle2ONNX 的发展。
由 Paddle2ONNX PMC 决定，本次使用其中 4500 元人名币向社区发布三道黑客松的赛题。本次使用的奖励均为社区捐赠，因此最后的奖励由 Paddle2ONNX PMC 向社区热心开发者直接颁发。

## 【开源贡献个人挑战赛-Paddle2ONNX 方向】任务详情

### NO.56 赛题 1：为 Paddle2ONNX 添加 DeformConv 算子

**赛题简介**

本赛题需要社区开发者为 Paddle2ONNX 添加对 DeformConv 算子的支持，要求在 PaddlePaddle2.6 下将[PaddlePaddle deform_conv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/deform_conv2d_cn.html#deform-conv2d)转换为[ONNX DeformConv](https://onnx.ai/onnx/operators/onnx__DeformConv.html#l-onnx-doc-deformconv)，并添加单测。

**参考链接**

- Issues 链接：[Paddle2ONNX Issues 1183](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1183)
- ONNX 算子定义文档：[ONNX DeformConv](https://onnx.ai/onnx/operators/onnx__DeformConv.html#l-onnx-doc-deformconv)
- PaddlePaddle 算子定义文档：[PaddlePaddle deform_conv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/deform_conv2d_cn.html#deform-conv2d)
- Paddle2ONNX 对 Conv2d 算子的支持：[paddle2onnx/mapper/nn/conv2d.cc](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/mapper/nn/conv2d.cc)

### NO.57 赛题 2：为 PaddleOCRv4 Det 模型量化模型添加支持

**赛题简介**

本赛题需要社区开发者为 Paddle2ONNX 添加对两个插入的虚拟量化算子节点的支持，要求在 PaddlePaddle2.6 下将 PaddleOCRv4 Det 转换为 ONNX 模型并对齐精度。

**参考链接**

- Issues 链接：[Paddle2ONNX Issues 1141](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1141)
- Paddle2ONNX 对 Conv2d 算子的支持：[paddle2onnx/mapper/nn/conv2d.cc](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/mapper/nn/conv2d.cc)

### NO.58 赛题 3：为 Paddle2ONNX 添加半精度模型的支持

**赛题简介**

本赛题需要社区开发者为 Paddle2ONNX 添加对半精度 Paddle 模型的支持，赛题要求在 PaddlePaddle2.6 下实现将半精度 ResNet 模型成功转换为 ONNX 模型并实现结果的对齐，要求添加独立的单元测试。（可以不与现在的单元测试兼容）

> [!WARNING]
> 本赛题最后需要使用 ResNet 模型进行验证，考虑到 Paddle 和 ONNXRuntime 之间对相同参数的版精度模型的推理结果可能存在误差，因此本赛题允许接受最后可能存在的部分误差。

**参考步骤**

- 实现在 FP16 的情况下正确使用 PaddleDataTypeSize 读取 FP16 所占的内存字节大小
- 实现对 BatchNormalize 的 FP16 支持 -> 可以考虑通过升级 OP 版本实现
- 实现对 LayerNormalize 的 FP16 支持 -> 可以考虑通过删除部分参数的强制转换实现
- 实现对 Pooling 的 FP16 支持 -> 可以考虑通过删除部分参数的强制转换实现
- 实现对 MatMul 的 FP16 支持 -> 可以考虑通过添加强制转换来实现
- 实现对 FillConstant 的 FP16 支持 -> 可以考虑通过添加 FP32 参数转 FP16 参数来实现

**要求**

要求转换出来的 ONNX 模型尽量不要存在多余的无用节点且精度与 PaddleInference 推理结果相近
