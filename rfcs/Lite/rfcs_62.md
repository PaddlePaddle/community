功能名称：在 Paddle-Lite-Demo 中添加 PP-TinyPose 模型在安卓上的 Demo


开始日期：2022-03-24


RFC PR：无


GitHub Issue：[Paddle-Lite#8603](https://github.com/PaddlePaddle/Paddle-Lite/issues/8603)


# 总结


以任务中给出的 face_keypoints_detection_demo 代码为 BASE，把 PP-TinyPose 中已经训练好的模型，部署到 Android 应用中。包括开发相应的前处理，推理，后处理，以及 Runtime 选择（CPU/GPU）功能。


# 动机


为了丰富 Lite 的应用案例。
开发完成后，用户可以把这个 APP 安装到自己的 Android 手机，直观体验PP-TinyPose 模型的检测效果。 用户可以选择将模型运行在 CPU 或 GPU 上。


# 使用指南级别的说明


在配置好 AndroidStuido 环境的 PC 上，打开本工程，按照 ReadMe 文档的步骤就可以把 APP 安装到 Android 手机上。
在手机上运行APP，可以看到对摄像头捕捉的实时画面的检测结果。
通过点击【设置】按钮，可以打开设置界面，对backend进行选择。也就是可以选将模型运行在 CPU 或 GPU 上。


# 参考文献级别的说明


* 模型取得


可以从https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/keypoint/tiny_pose
下载后缀名为nb的Lite部署模型文件。


* 初始化


初始化时机：在刚打开应用的时候，和关闭设定页面的时候，MainActivity 的 onResume都会 被调用。初始化动作：调用 predictor.init()，在 Native 侧的初始化中 new Pipeline，并设置 CPU/GPU 参数。最后，通过 CreatePaddlePredictor 构建预测器。


* 视频流捕获


当 onDrawFrame 被调用时调用 onTextureChanged 方法，通过 nativeProcess 调用 Pipeline 的  Process，这时通过 CreateRGBAImageFromGLFBOTexture 方法得到 cv::Mat 格式的帧数据。


* 前处理


通过 OpenCV 进行 resize，RGBA2BGR，NHWC3ToNC3HW等操作，得到适合模型的输入数据。然后通过 GetInput 取得模型的输入 Tensor，并把处理好的数据放入 Tensor。


 - 推理


调用预测器的 run 方法，进行推理。


 - 后处理


通过 GetOutput 取得模型的输出 Tensor，解析出推理结果。


 - 结果描画


通过 OpenCV 的各种描画方法，把结果描画到 cv::Mat 格式的原始帧数据。这些数据将在 Native 的处理结束后，在 Java 层通过 GL 进行渲染。


 - 设置菜单


响应设置按钮的点击，打开设置画面，在这里提供 CPU/GPU 选择界面。推出后通过 checkAndUpdateSettings() 方法检查设置的变化，需要的话，根据新设置的值重新构建预测器。退出菜单后，返回主画面同时重新开始相机与预测器的动作。


# 缺点


暂无


# 理论依据和替代方案


无


# 现有技术


无


# 未解决的问题


无


# 未来的可能性


就 Android 应用角度来说，可以考虑把 camera 调用移动到 Native层，这样可以使得 Java 层与 Native 层的交互更简洁。


参考：[tvm/rfc-template](https://github.com/apache/tvm-rfcs/blob/main/0000-template.md)