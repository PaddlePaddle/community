此文档展示 **PaddlePaddle Hackathon 第五期活动——开源贡献个人挑战赛合作伙伴任务** 详细介绍，更多详见  [PaddlePaddle Hackathon 说明](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_cn)。

## 【开源贡献个人挑战赛-合作伙伴】任务详情

### No.88：Arm虚拟硬件上完成PaddleClas模型的部署验证

**任务目标：**

将[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)模型库中的模型部署在[Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55)处理器上并使用Arm虚拟硬件[Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300)平台进行验证。

其中, 该任务涉及以下几个必要步骤：

- 选择合适的模型

可选模型库[Model Zoo](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5/docs/zh_CN/models),请避免重复选择已经完成适配的模型(eg. PP-LCNet,MobileNetV3_small_x0_35等) 。不局限于模型库的模型, 支持高性能自研模型。

- 使用TVM编译模型

训练模型(trained model)需导出为Paddleinference模型才可使用tvmc编译。同时, 请注意所选模型是否能成功地被TVM编译(部分算子目前不支持)。TVM帮助文档可查看[TVM官网](https://tvm.apache.org/docs/)或[GitHub仓库](https://github.com/apache/tvm)。

- 按照Open-CMSIS-Pack项目规范完成应用程序的开发

确保结果的可读性, 请正确地完成前后端的数据处理和结果展示。Open-CMSIS-Pack项目的规范可查看[帮助文档](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html)或[GitHub仓库](https://github.com/Open-CMSIS-Pack), 示例工程代码可参考[mlek-cmsis-pack-examples](https://github.com/Arm-Examples/mlek-cmsis-pack-examples/tree/main/object-detection)中目标检测的应用案例。

- 完成视频(video)虚拟数据流接口(VSI)驱动程序的开发

视频虚拟数据流接口的开发可参考VSI[帮助文档](https://arm-software.github.io/AVH/main/simulation/html/group__arm__vsi__video.html)及[示例代码](https://github.com/RobertRostohar/mlek-cmsis-pack-examples/tree/vsi_video/vsi/video)。

- 使用百度智能云Arm虚拟硬件镜像服务验证运行结果

订阅并远程登录[Arm虚拟硬件](https://market.baidu.com/product/detail/b5f9d5d0-3861-4fb8-a0e5-314bcc6617ce)BCC实例, 完成运行环境配置(部分软件可能需手动安装)并最终调用Corstone-300(VHT_MPS3_Corstone_SSE-300)平台验证应用程序的运行结果。

**提交内容:**

- 项目启动前, 请提交RFC文档(注意标明所选模型及来源)。
- PR代码至[GitHub仓库](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH)并创建新的分支(命名为: Open CMSIS PackProject)。代码合入规范请参考仓库中已有工程,

但是请注意涉及到第三方知识产权的图片等素材请注明相关的来源和使用许可证。

**技术要求:**

- 熟练使用 c/c++，Python 进行工程项目开发。
- 熟悉基础的 Linux 操作系统命令和在基于Arm的服务器上开发的经验。
- 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。

请与导师沟通获取更多技术参考资料和1v1指导, 更全面详细的产品和技术文档可访问 https://www.arm.com 或 https://developer.arm.com 了解。

### No.89：Arm虚拟硬件上完成飞桨视觉模型的部署验证

**详细描述:** 

任务目标为将2个飞桨视觉套件模型库中的模型部署在[Arm Cortex-M85](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m85)处理器上并使用Arm虚拟硬件[Corstone-310](https://www.arm.com/products/silicon-ip-subsystems/corstone-310)平台进行验证。

其中, 该任务涉及以下几个必要步骤:

- 选择合适的模型(2个)

可从飞桨提供视觉套件的模型库, 例如[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)等中进行选择(建议2个模型尽量不要针对同一应用场景)。不局限于模型库的模型, 支持高性能自研模型。

- 使用TVM编译模型

训练模型需导出为Paddle inference模型才可使用tvmc编译, 同时请注意所选模型是否能成功地被TVM编译(部分算子目前不支持)。TVM帮助文档可查看[TVM官网](https://tvm.apache.org/docs/)或[GitHub仓库](https://github.com/apache/tvm)。

- 应用程序编写

请根据相应的应用场景, 正确地完成应用程序的前后端数据地处理并确保最终结果的可读性。

- 使用百度智能云Arm虚拟硬件镜像服务验证运行结果

订阅并远程登录[Arm虚拟硬件](https://market.baidu.com/product/detail/b5f9d5d0-3861-4fb8-a0e5-314bcc6617ce)BCC实例, 完成运行环境配置(部分软件可能需手动安装)并最终调用Corstone-310(VHT_Corstone_SSE-310) 平台验证应用程序的运行结果。

**提交内容:**

- 项目启动前, 请提交RFC文档(注意标明所选模型及来源)。
- PR代码至GitHub仓库(tmp分支下)。代码合入规范请参考仓库中已有工程,但是请注意涉及到第三方知识产权的图片等素材请注明相关的来源和使用许可证。

**技术要求:**

- 熟练使用 c/c++，Python 进行工程项目开发。
- 熟悉基础的 Linux 操作系统命令和在基于Arm的服务器上开发的经验。
- 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。

请与导师沟通获取更多技术参考资料和1v1指导, 更全面详细的产品和技术文档可访问 https://www.arm.com 或 https://developer.arm.com 了解。

### No.90：Arm虚拟硬件上完成飞桨模型与Arm Ethos-U microNPU的适配与部署验证

**详细描述:** 

任务目标为将飞桨模型库中的模型部署在[Arm Ethos-U55 microNPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)处理器上并使用Arm虚拟硬件[Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300)平台(内含有Arm Ethos-U55

处理器)进行验证。

其中, 该任务涉及以下几个必要步骤：

- 选择合适的模型

可从飞桨套件提供的模型库, 例如[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)等模型库中进行选择, 但注意所选模型的大小。不局限于模型库的模型, 支持高性能自研模型。

- 使用TVM编译模型

训练模型需导出为Paddle inference模型才可使用tvmc编译, 同时请注意所选模型是否能成功地被TVM编译(部分算子目前不支持)。TVM帮助文档可查看[TVM官网](https://tvm.apache.org/docs/)或[GitHub仓库](https://github.com/apache/tvm)。

适配Ethos-U55的TVM编译步骤可参考示例代码(line 147-158)。同时，编译模型前请对模型进行适当的量化、压缩、剪枝等处理(Ethos-U55仅支持Int-8和Int-16数据类型) , 请确保算子尽可能地运行在Ethos-U55上, 部分不支持的算子可以运行在Cortex-M55处理器上。

- 应用程序编写

请根据选择模型的应用场景, 正确地完成应用程序的前后端数据的处理并确保结果的可读性。

- 使用百度智能云Arm虚拟硬件镜像服务验证运行结果

订阅并远程登录[Arm虚拟硬件](https://market.baidu.com/product/detail/b5f9d5d0-3861-4fb8-a0e5-314bcc6617ce)BCC实例, 完成运行环境配置(部分软件可能需手动安装)并最终调用Corstone-300(VHT_Corstone_SSE-300_Ethos-U55)平台验证应用程序的运行结果。

**提交内容:**

- 项目启动前, 请提交RFC文档(注意标明所选模型及来源)。
- PR代码至[GitHub仓库](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH)并创建新的分支(命名为: Ethos-U Project)。代码合入规范请参考仓库中已有工程,但是请注意涉及到第三方知识产权的图片等素材请注明相关的来源和使用许可证

**技术要求:**

- 熟练使用 c/c++，Python 进行工程项目开发。
- 熟悉基础的 Linux 操作系统命令和在基于Arm的服务器上开发的经验。
- 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。

请与导师沟通获取更多技术参考资料和1v1指导, 更全面详细的产品和技术文档可访问 https://www.arm.com 或 https://developer.arm.com 了解。

### No.91：Arm虚拟硬件上完成飞桨模型的优化部署

**详细描述:** 

任务目标为利用[Arm Helium](https://developer.arm.com/documentation/102102/0103/What-is-Helium-?lang=en)技术将飞桨模型库中的模型优化部署在[Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55)处理器上并使用Arm虚拟硬件[Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300)平台进行验证。

其中, 该任务涉及以下几个必要步骤：

- 选择合适的模型:

可从飞桨套件提供的模型库, 例如[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)等模型库中进行选择, 但注意所选模型的大小。不局限于模型库的模型, 支持高性能自研模型。

- 使用TVM编译模型

训练模型需导出为Paddle inference模型才可使用tvmc编译, 同时请注意所选模型是否能成功地被TVM编译(部分算子目前不支持)。TVM帮助文档可查看[TVM官网](https://tvm.apache.org/docs/)或[GitHub仓库](https://github.com/apache/tvm)。同时注意, 需要对模型进行适当地量化等操作, 从而确保尽可能多地算子可以调用Arm CMSIS-NN库支持算子(便于后续可以将部分算子运行在Helium上)。

- 应用程序编写

请根据选择模型的应用场景, 正确地完成应用程序的前后端数据处理并确保结果的可读性。

- 使用百度智能云Arm虚拟硬件镜像服务验证运行结果

订阅并远程登录[Arm虚拟硬件](https://market.baidu.com/product/detail/b5f9d5d0-3861-4fb8-a0e5-314bcc6617ce)BCC实例, 完成运行环境配置(部分软件需手动安装)并最终调用Corstone-300(VHT_Corstone_SSE-300_Ethos-U55)平台验证应用程序的运行结果。

同时, 需特别注意, 在编译应用和编译完成后在虚拟硬件上执行应用时, 请开启相应的Helium配置选项。具体Helium[技术介绍](https://developer.arm.com/documentation/102102/0103/What-is-Helium-?lang=en)及编程指南可参考相关[帮助文档](https://developer.arm.com/documentation/102095/0100/Enabling-Helium?lang=en)。

**提交内容:**

- 项目启动前, 请提交RFC文档(注意标明所选模型及来源)。
- PR代码至[GitHub仓库](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH)并创建新的分支(命名为: Helium Project)。代码合入规范请参考仓库中已有工程,但是请注意涉及到第三方知识产权的图片等素材请注明相关的来源和使用许可证

**技术要求:**

- 熟练使用 c/c++，Python 进行工程项目开发; 了解汇编语言。
- 熟悉基础的 Linux 操作系统命令和在基于Arm的服务器上开发的经验。
- 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。

请与导师沟通获取更多技术参考资料和1v1指导, 更全面详细的产品和技术文档可访问 https://www.arm.com 或 https://developer.arm.com 了解。

### No.92：使用Arm smart vision configuration kit 在Arm虚拟硬件上部署飞桨模型

**详细描述:** 

coming soon

### No.93：为OpenVINO 实现 Paddle 算子max_pool3d_with_index与max_pool3d转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

在这个任务中，由于max_pool3d_with_index与max_pool3d的转换实现较为相近，你需要为OpenVINO实现Paddle算子gaussian_random转换。该算子该函数是一个三维最大池化函数，根据输入参数 kernel_size, stride, padding 等参数对输入x做最大池化操作。该任务中的算子难度较高，Paddle2ONNX展示了如何将这些算子映射到ONNX的算子：https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/mapper/nn/pool3d.cc

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 5】字样
4. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且通过测试用例，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
5. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解OpenVINO和PaddlePaddle相关深度学习计算算子
- 了解OpenVINO推理引擎相关技术背景

**参考文档：**

OpenVINO算子库文档：https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset12.md

OpenVINO算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference

PaddlePaddle算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

PaddlePaddle算子参考实现：https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

Paddle2ONNX算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper ，https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性：https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

OpenVINO源码编译方法：

1. CMakeList中开启Paddle frontend测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9
2. 编译说明：https://github.com/openvinotoolkit/openvino/wiki

Ubuntu可参考：

```shell
$ git clone https://github.com/openvinotoolkit/openvino.git
$ cd openvino
$ git submodule update --init --recursive
$ chmod +x install_build_dependencies.sh
$./install_build_dependencies.sh
$ export OPENVINO_BASEDIR=`pwd`
$ mkdir build
$ cd build
$ cmake \
-DCMAKE_BUILD_TYPE= Release -DCMAKE_INSTALL_PREFIX="${OPENVINO_BASEDIR}/openvino_dist" \
-DPYTHON_EXECUTABLE=$(which python3) \
-DENABLE_MYRIAD=OFF \
-DENABLE_VPU=OFF \
-DENABLE_PYTHON=ON \
-DNGRAPH_PYTHON_BUILD_ENABLE=ON \
-DENABLE_DEBUG_CAPS=ON \
-DENABLE_TESTS=ON \
..
$ make -j$(nproc); make install
```

**单测测试方法：**

```shell
$ cd bin/intel64/Release
$ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*
```

### No.94：为 OpenVINO 实现 Paddle 算子partial_sum与partial_concat转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

在这个任务中，由于 partial_sum与partial_concat转换方式较为相近，你需要同时为OpenVINO实现这个两个算子转换，该算子将按指定起始位合并输入Tensor或为其求和。该任务中的算子难度中等，Paddle2ONNX展示了如何将这些算子映射到ONNX的算子：https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/mapper/tensor/partial_ops.cc。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 5】字样
4. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且通过测试用例，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
5. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解OpenVINO和PaddlePaddle相关深度学习计算算子
- 了解OpenVINO推理引擎相关技术背景

**参考文档：**

OpenVINO算子库文档：https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset12.md

OpenVINO算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference

PaddlePaddle算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

PaddlePaddle算子参考实现：https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

Paddle2ONNX算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper ，https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

OpenVINO源码编译方法：

1. CMakeList中开启Paddle frontend测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9
2. 编译说明：https://github.com/openvinotoolkit/openvino/wiki

Ubuntu可参考：

```shell
$ git clone https://github.com/openvinotoolkit/openvino.git
$ cd openvino
$ git submodule update --init --recursive
$ chmod +x install_build_dependencies.sh
$./install_build_dependencies.sh
$ export OPENVINO_BASEDIR=`pwd`
$ mkdir build
$ cd build
$ cmake \
-DCMAKE_BUILD_TYPE= Release -DCMAKE_INSTALL_PREFIX="${OPENVINO_BASEDIR}/openvino_dist" \
-DPYTHON_EXECUTABLE=$(which python3) \
-DENABLE_MYRIAD=OFF \
-DENABLE_VPU=OFF \
-DENABLE_PYTHON=ON \
-DNGRAPH_PYTHON_BUILD_ENABLE=ON \
-DENABLE_DEBUG_CAPS=ON \
-DENABLE_TESTS=ON \
..
$ make -j$(nproc); make install
```

**单测测试方法：**

```shell
$ cd bin/intel64/Release
$ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*
```

### No.95：为 OpenVINO 实现 Paddle 算子 unique转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

在这个任务中，你需要为OpenVINO实现Paddle算子unique转换，该算子返回 Tensor 按升序排序后的独有元素。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 5】字样
4. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且通过测试用例，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
5. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

### No.96：为 OpenVINO 实现 Paddle 算子unstack转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

在这个任务中，你需要为OpenVINO实现Paddle算子unstack转换，该算子将单个 dim 为 D 的 Tensor 沿 axis 轴 unpack 为 num 个 dim 为 (D-1) 的 Tensor。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 5】字样
4. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且通过测试用例，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
5. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解OpenVINO和PaddlePaddle相关深度学习计算算子
- 了解OpenVINO推理引擎相关技术背景

**参考文档：**

OpenVINO算子库文档：https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset12.md

OpenVINO算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference

PaddlePaddle算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

PaddlePaddle算子参考实现：https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

Paddle2ONNX算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper ，https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

OpenVINO源码编译方法：

1. CMakeList中开启Paddle frontend测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9
2. 编译说明：https://github.com/openvinotoolkit/openvino/wiki

Ubuntu可参考：

```shell
$ git clone https://github.com/openvinotoolkit/openvino.git
$ cd openvino
$ git submodule update --init --recursive
$ chmod +x install_build_dependencies.sh
$./install_build_dependencies.sh
$ export OPENVINO_BASEDIR=`pwd`
$ mkdir build
$ cd build
$ cmake \
-DCMAKE_BUILD_TYPE= Release -DCMAKE_INSTALL_PREFIX="${OPENVINO_BASEDIR}/openvino_dist" \
-DPYTHON_EXECUTABLE=$(which python3) \
-DENABLE_MYRIAD=OFF \
-DENABLE_VPU=OFF \
-DENABLE_PYTHON=ON \
-DNGRAPH_PYTHON_BUILD_ENABLE=ON \
-DENABLE_DEBUG_CAPS=ON \
-DENABLE_TESTS=ON \
..
$ make -j$(nproc); make install
```

**单测测试方法：**

```shell
$ cd bin/intel64/Release
$ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*
```

### No.97：为 OpenVINO 实现 Paddle 算子tanh_shrink转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

在这个任务中，你需要为OpenVINO实现Paddle算子tanh_shrink转换，该算子为激活层算子。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 5】字样
4. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且通过测试用例，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
5. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解OpenVINO和PaddlePaddle相关深度学习计算算子
- 了解OpenVINO推理引擎相关技术背景

**参考文档：**

OpenVINO算子库文档：https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset12.md

OpenVINO算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference

PaddlePaddle算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

PaddlePaddle算子参考实现：https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

Paddle2ONNX算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper ，https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

OpenVINO源码编译方法：

1. CMakeList中开启Paddle frontend测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9
2. 编译说明：https://github.com/openvinotoolkit/openvino/wiki

Ubuntu可参考：

```shell
$ git clone https://github.com/openvinotoolkit/openvino.git
$ cd openvino
$ git submodule update --init --recursive
$ chmod +x install_build_dependencies.sh
$./install_build_dependencies.sh
$ export OPENVINO_BASEDIR=`pwd`
$ mkdir build
$ cd build
$ cmake \
-DCMAKE_BUILD_TYPE= Release -DCMAKE_INSTALL_PREFIX="${OPENVINO_BASEDIR}/openvino_dist" \
-DPYTHON_EXECUTABLE=$(which python3) \
-DENABLE_MYRIAD=OFF \
-DENABLE_VPU=OFF \
-DENABLE_PYTHON=ON \
-DNGRAPH_PYTHON_BUILD_ENABLE=ON \
-DENABLE_DEBUG_CAPS=ON \
-DENABLE_TESTS=ON \
..
$ make -j$(nproc); make install
```

**单测测试方法：**

```shell
$ cd bin/intel64/Release
$ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*
```

### No.98：完成PP-YOLOE在华为昇腾平台上的推理优化

**技术标签：**

深度学习、算法部署、推理优化

**详细描述：**

- 优化PaddleYOLO套件中PP-YOLOE-L模型在华为昇腾平台上的推理速度：**使用**[**profiler工具**](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/profiling_model.html)**进行推理耗时打点，分析性能瓶颈，并提升模型推理性能20%**，给出调优前后的性能对比情况
- npu的profiler开启实例：

```python
import paddle.profiler as profiler
profiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CUSTOM_DEVICE], custom_device_types=['npu']);
paddle.set_device("npu")
profiler.start() ######  这里启动 profiler
# 需要打点的代码段
profiler.stop()  ######  这里停止 profiler
```

- 性能打点数据导出

```python
/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof --export=on --output=PROF_XXXXXX
# 导出后，timeline中为时间轴数据，可以通过chrome://traing工具查看。summary中为算子耗时统计数据表
```

- 调优参考思路：通过summary中op_statistic_0_1.csv数据表可以得知当前模型中TransData算子以及Cast算子耗时占比较多，可通过统一数据排布以及统一数据类型优化
- 开发流程和环境配置请参考 [CONTRIBUTING.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/CONTRIBUTING.md)
- 目标芯片为Ascend910

**提交内容：**

- pr：提交适配代码，及对应的中英文文档到：https://github.com/PaddlePaddle/PaddleCustomDevice
- 提交benchmark测试数据及精度对齐数据。

**技术要求：**

- 熟练掌握Python、C++开发
- 熟悉PaddleCustomDevice算子接入
- 熟悉昇腾技术栈

### No.99：基于 Qualcomm SNPE SDK 开发 RMSNorm  算子

**详细描述：**

基于高通AI软件栈 SNPE SDK，开发算子RMSNorm, 在高通HTP运行。

**开发流程：**

1. 了解RMSNorm的实现 [https://arxiv.org/pdf/1910.07467.pdf](https://mailshield.baidu.com/check?q=2nI8I5D6Z2WUW7FFuZ58dLMWX49bG5wQxUaZdOnUs1EWOWyV5GC69A%3d%3d)
2. 使用SNPE SDK开发自定义算子
   1. SNPE download: [https://zhuanlan.zhihu.com/p/641013796](https://mailshield.baidu.com/check?q=JA2lVQGIVatSjRzQJSkUkThfVVecVlcJ0ztrSvxtZrWx9eIy6J%2fuUpohpUQ%3d)
   2. SNPE getting start: [https://www.csdn.net/article/2022-04-08/124044583](https://mailshield.baidu.com/check?q=vlc512cCDZGnUNUqTet83t4ktm2MENnzxExrCfkILNngpKrNy4oHQnuzbckoCYoWxFCeBd6Ac3Q%3d)
3. 算子实现
   1. Python,CPU 代码实现
   2. DSP scalar FP32 实现
   3. HVX FP16 实现
   4. HVX UINT16 实现
   5. \#i和#ii为基本要求（对应奖金🌟），#iii和#iv为进阶要求（对应奖金🌟🌟）
4. 测试要求：在QNN HTP-simulator运行并验证精度

**提交内容：**

- API 的设计文档
- Dummy model, 用于测试
- 测试用例，精度比较脚本，生成精度比较结果
- 工程代码及运行步骤

**技术要求：**

- 熟练c/c++，Python
- 熟悉高通hexagon SDK
- 熟悉高通DSP/HVX 指令
- 熟练高通SNPE UDO 开发

### No.100：基于openKylin OS和X2paddle实现面向AI框架的统一推理接口，实现AI软件的适配与应用

**技术标签：** 

操作系统，AI

**提交内容：**

1.项目demo：任务伙伴提供Raspberry Pi 4B开发板，认领者基于开发板完成demo搭建

2.AIStudio项目：包括Python和C++实现代码 ，以及demo视频；

3.代码提交到https://gitee.com/openkylin

**验收标准：**

先提交支撑模块，然后审阅。

**技术要求：**

- 部署
- 熟练掌握C++、Python开发，了解AI算法

