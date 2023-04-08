# 使用java完成PaddleDetection部署设计文档
|属性 | 内容 |
|-|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | Tomoko-hjf |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-04-08 |
|版本号 | V1 |
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
在`Java`层面新增模型部署可以分为几个步骤：定义模型对应的调用本地`native`方法的`Java`类、使用C++实现`JNI`函数。

下面以`yolov5`的部署过程示例：
* 在`FastDeploy/examples/application/java/yolov5/java`文件夹下新建`InferDemo`类，在该类中定义`infer`函数，该函数是一个`native`方法，会进一步调用`c++` API。
* 在`FastDeploy/examples/application/java/yolov5/cpp`文件夹下实现`infer`函数，创建动态链接库，供`Java`端调用。
* 编写中英文部署文档。

# 四、测试和验收的考量
通过命令行调用`java`程序进行预测。

# 五、可行性分析和排期规划
通过`JNI`调用`C++`函数，方案可行。
整体预计两周完成任务，一周编写model代码，一周测试接口功能的正确性和编写示例文档。

# 六、影响面
所有代码都放在`FastDeploy/examples/application/java`文件夹下，对其他模块没有影响。

# 附件及参考资料
参考仓库下已经实现的`PicoDet`。
