功能名称：在 Paddle-Lite-Demo 中添加 mobilenet_v3 模型在安卓上的 Demo


开始日期：2022-03-29


RFC PR：无


GitHub Issue：[Paddle-Lite#8601](https://github.com/PaddlePaddle/Paddle-Lite/issues/8601)


# 总结


以任务中给出的 mobilenet_v1 图像分类 android demo 代码为 BASE，把 PaddleClas 中已经训练好的 mobilenet_v3 模型，部署到 Android 应用中。包括开发相应的前处理，推理，后处理，以及 Runtime 选择（CPU/GPU）功能。


# 动机


为了丰富 Lite 的应用案例。
开发完成后，用户可以把这个 APP 安装到自己的 Android 手机，直观体验 mobilenet_v3模型的分类效果。 用户可以选择将模型运行在 CPU 或 GPU 上，比较两者在处理速度上的差别。


# 使用指南级别的说明


在配置好 AndroidStuido 环境的 PC 上，打开本工程，按照 ReadMe 文档的步骤就可以把 APP 安装到 Android 手机上。
在手机上运行APP，会显示一张照片和对照片的分类结果。
通过点击【是否使用GPU】按钮，对backend进行选择。也就是可以选将模型运行在 CPU 或 GPU 上。点击后，会重新用当前选择的 backend 进行推理，并重新显示推理结果，推理时间，使用的模型，线程数等信息。


# 参考文献级别的说明


* 模型取得与转换


可以从 https://github.com/PaddlePaddle/PaddleClas 下载，并通过OPT工具转换为后缀名为nb的Lite部署模型文件。转换的时候可以设置 target 为 CPU 或 GPU。


* 模型初始化


初始化时机：模型初始化时机有两个，一是在刚打开应用的时候，也就是 onResume() 的时候。二是 CPU/GPU 设定发生变化的时候，也就是 Switch 的监听函数被调用的时候。初始化动作：调用 predictor.init()，在 Native 侧的初始化中 new Pipeline，并设置 CPU/GPU 参数。最后，通过 CreatePaddlePredictor 构建预测器。


* 标签初始化


初始化时机：分类器构造的时候。初始化方法，通过读取标签数据文件载入。


* 输入数据


在 Java 侧通过 BitmapFactory 读取图片数据。读取时机为模型加载成功后。


* 前处理


在 Java 侧先把图片进行缩放。
之后在 Native 侧通过 OpenCV 进行 resize，RGBA2BGR，NHWC3ToNC3HW等操作，得到适合模型的输入数据。然后通过 GetInput 取得模型的输入 Tensor，并把处理好的数据放入 Tensor。


 - 推理


调用预测器的 run 方法，进行推理。


 - 后处理


通过 GetOutput 取得模型的输出 Tensor，解析出推理结果。把置信度最高的几个标签转换成字符串，作为结果输出。


 - 结果输出


在模型推理成功后，在 Java 侧，把以字符串形式返回的结果解析出来，通过 setText 把推理结果显示到界面上。


# 缺点


目前的实现还只能对默认的图片进行推理并显示结果。


# 理论依据和替代方案


无


# 现有技术


无


# 未解决的问题


无


# 未来的可能性


可以考虑通过摄像头输入与相册输入的方法，丰富可以用于推理的图片。


参考：[tvm/rfc-template](https://github.com/apache/tvm-rfcs/blob/main/0000-template.md)