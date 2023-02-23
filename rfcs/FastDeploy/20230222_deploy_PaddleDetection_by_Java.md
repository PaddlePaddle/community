# 使用java完成PaddleDetection部署设计文档
|属性 | 内容 |
|-|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | Tomoko-hjf |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-02-22 |
|版本号 | V0.0 |
|文件名 | 20230222_deploy_PaddleDetection_by_Java.md<br> |


# 一、概述
## 1、相关背景
随着人工智能应用的不断发展，不仅要求在`服务端`进行部署，也需要在`移动安卓端`进行部署。
## 2、功能目标
在`FastDeploy`中使用`java`完成`PaddleDetection`中`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等模型的部署。

## 3、意义
可以使用Java在移动端部署`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等目标检测模型。

# 二、飞桨现状
在`使用Java部署目标检测模型`方面，`FastDeploy`目前只支持`PicoDet`，缺乏其他的目标检测模型。

# 三、设计思路与实现方案

## 总体思路
在`Java`层面新增模型部署可以分为几个步骤：定义模型对应的调用本地`native`方法的`Java`类、使用C++实现`JNI`函数、修改`CMakeLists.txt`文件。

## 定义模型对应的调用本地`native`方法的`Java`类
* 在`fastdeploy/src/main/java/com/baidu/paddle/fastdeploy/vision/detection`下定义目标检测模型对应的Java类，如`PPYOLOE`类。
* 在类中定义`构造方法`，`predict方法`、`release方法`。

## 使用C++实现`JNI`函数
在`fastdeploy/src/main/cpp/fastdeploy_jni/vision/detection/`下使用`C++`实现对应模型的`JNI`函数(`Java`类中调用的`native`方法)。

# 四、测试和验收的考量
在`Android`端部署增加的检测模型，并进行测试。

# 五、可行性分析和排期规划
通过`JNI`调用`C++`函数，方案可行。预计需要一周。

# 六、影响面
对其他模块没有影响。

# 附件及参考资料
参考仓库下已经实现的`PicoDet`。
