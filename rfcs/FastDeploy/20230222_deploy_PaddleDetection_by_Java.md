# 使用java完成PaddleDetection部署设计文档
|属性 | 内容 |
|-|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | Tomoko-hjf |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-10 |
|版本号 | V0.2 |
|文件名 | 20230222_deploy_PaddleDetection_by_Java.md<br> |


# 一、概述
## 1、相关背景
随着人工智能应用的不断发展，不仅要求使用`C++`和`Python`进行部署，也需要使用`Java`进行部署。
## 2、功能目标
在`FastDeploy`中使用`java`完成`PaddleDetection`中`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等模型的部署。

## 3、意义
可以更加方便地使用`Java`部署`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等目标检测模型，丰富`FastDeploy`在`Java部署`方面的功能。

# 二、飞桨现状
在`使用Java部署目标检测模型`方面，`FastDeploy`目前缺乏其他的目标检测模型。

# 三、设计思路与实现方案

## 总体思路
在`Java`层面新增模型部署可以分为几个步骤：定义模型对应的调用本地`native`方法的`Java`类、使用C++实现`JNI`函数、修改`CMakeLists.txt`文件。

## 定义模型对应的调用本地`native`方法的`Java`类
* 在`FastDeploy/examples/application/java`文件夹下新建`com.baidu.paddle.fastdeploy.vision.detection`包，在该包下定义目标检测模型对应的Java类，如`PPYOLOE`类。
* 在类中定义`构造方法`，`predict方法`、`release方法`、`initialized方法`。
* 将公共类如`DetectionResult`放在`com.baidu.paddle.fastdeploy.vision`包下。

## 使用C++实现`JNI`函数
在`FastDeploy/examples/application/java`文件夹下新建`cpp/fastdeploy_jni/vision/detection/`文件夹，用于存放使用`C++`实现的对应模型的`JNI`函数(`Java`类中调用的`native`方法)。

## 编写部署文档
`FastDeploy/examples/application/java`文件夹下编写使用Java部署相关模型的示例文档。

# 四、测试和验收的考量
在`Java`端使用`Junit`编写新增检测模型测试用例，具体测试方法如下：
* 测试model的构造方法 `testModelInit()`
* 测试model的预测方法 `testPredict()`
* 测试model的release方法 `testRelease()`
* 测试model的initialized `testInitialized()`

# 五、可行性分析和排期规划
通过`JNI`调用`C++`函数，方案可行。
整体预计两周完成任务，一周编写model代码，一周测试接口功能的正确性和编写示例文档。

# 六、影响面
所有代码都放在`FastDeploy/examples/application/java`文件夹下，对其他模块没有影响。

# 附件及参考资料
参考仓库下已经实现的`PicoDet`。
