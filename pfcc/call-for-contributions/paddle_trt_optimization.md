# Paddle-TensorRT 算子开发

> This project will be mentored by [@zhangjun](https://github.com/zhangjun)

## 1、背景与意义
TensorRT 是一个针对 NVIDIA GPU 及 Jetson 系列硬件的高性能机器学习推理 SDK，可以使得深度学习模型在这些硬件上的部署获得更好的性能。Paddle Inference 以子图方式集成了 TensorRT，将可用 TensorRT 加速的算子组成子图供给 TensorRT，以获取 TensorRT 加速的同时，保留 PaddlePaddle 即训即推的能力。

当模型被 Paddle Inference 加载后，神经网络被表示为由变量和运算节点组成的计算图。在图分析阶段，Paddle Inference 会对模型进行分析同时发现图中可以使用 TensorRT 优化的子图，并使用 TensorRT 节点替换它们(如下图)。在模型的推理期间，如果开启TensorRT开关，遇到 TensorRT 节点，Paddle Inference 会调用 TensorRT 对该节点进行执行，其它节点调用 GPU 原生推理，充分利用TensorRT性能优化。

![trt_engine](https://user-images.githubusercontent.com/1312389/203226862-a3cbc221-dc51-4a31-8108-5dd1c31ffca5.png)

NVIDIA TensorRT提供了42类Layer，包括91+个算子。Paddle-TensorRT 已经支持其中大部分Layer和算子，并且由于NVIDIA TensorRT新版本在持续更新迭代，仍然存在缺失的情形，如控制流算子支持、TensorRT v8.5版本新增IOneHotLayer、INMSLayer。目前Paddle主要通过三种机制对TensorRT进行支持：（1）Tensor Layer映射；（2）通用plugin机制（文档参见 [General plugin mechanism](https://github.com/PaddlePaddle/Paddle/pull/45355)）；（3）TensorRT OSS plugin映射。

![paddle_trt](https://user-images.githubusercontent.com/1312389/203226961-2d934d10-1c72-4c4c-96e9-8b1ac7814e85.png)

## 2、目标
完成 Paddle-TensorRT 算子开发及映射工作；通过通用plugin机制基于Phi算子库，完成对 Paddle-TensorRT 不支持的算子添加。
## 3、主要工作

|算子名称|类型|难度|API定义|OP定义|
| :----: | :---------: | :---: | :---: |:---: |
|conditional_block|TRT Layer映射|困难|[ConditionalBlock API](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/api/paddle/static/nn/cond_cn.html#cond)|[ConditionalBlock OP](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/controlflow/conditional_block_op.h#L100)|
|while|TRT Layer映射|困难|[While API](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/While_cn.html#while)|[While OP](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/controlflow/while_op.cc#L199)|
|inverse|通用plugin|中等|[inverse API](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/api/paddle/linalg/inv_cn.html#inv)|[inverse OP](https://github.com/PaddlePaddle/Paddle/blob/v2.4.0/paddle/fluid/operators/inverse_op.cc#L49)|
|one_hot_v2|TRT Layer映射|中等|[OneHot API](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/api/paddle/nn/functional/one_hot_cn.html#one-hot)|[OneHot OP](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/one_hot_v2_op.cc#L51)|
|range|TRT Layer映射|中等|[Range API](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/range_cn.html#range)|[Range OP](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/range_op.cc#L44)|
|set_value|TRT Layer映射|中等||[SetValue OP](https://github.com/PaddlePaddle/Paddle/blob/b546438c4e5cafb4a7a5d4967075004c4c8e6b5a/paddle/fluid/operators/set_value_op.cc#L69)|
|eye|TRT Layer映射|简单|[Eye API](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/api/paddle/eye_cn.html#eye)|[Eye OP](https://github.com/PaddlePaddle/Paddle/blob/b546438c4e5cafb4a7a5d4967075004c4c8e6b5a/paddle/fluid/operators/eye_op.cc#L45)|
|reduce_prod/reduce_min/reduce_max/reduce_any | TRT Layer映射|简单|[Reduce API](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/reduce_prod_cn.html)|[Reduce OP](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators/reduce_ops)|

算子开发示例：
* TRT Layer映射示例，参见 [where op映射示例](https://github.com/PaddlePaddle/Paddle/pull/47820)
* 通用plugin开发示例参见 [pad3d等算子开发示例](https://github.com/PaddlePaddle/Paddle/pull/47003)，设计文档参见 https://github.com/PaddlePaddle/Paddle/pull/45355

参考链接：
* [飞桨官网-贡献指南-代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)
