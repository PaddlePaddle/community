# 【PaddlePaddle Hackathon 4】生态伙伴开源贡献任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

注：为飞桨框架新增一系列 API，提交流程请参考 [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项在任务列表后：

### No.205：为OpenVINO notebook新增demo示例 <a name='task205'></a>

- 技术标签：深度学习框架，Python，OpenVINO

- 任务难度：进阶

- 详细描述：作为深度学习工具套件，OpenVINO可以被广泛应用于不同的应用场景，实现AI模型的推理部署，为此我们也想收集更多基于PaddlePaddle模型所打造的优秀应用案例，丰富示例仓库。
  在这个任务中，你需要在OpenVINO notebook仓库新增一个notebook示例。本次任务评估将分为两个阶段，在第一阶段中，开发者需要提供一份RFC（附参考模板），用来描述本次任务的设计方案； 在第二阶段中，我们将从第一阶段提交的结果中，挑选出2份比较优秀的方案，并请相对应的开发者根据自己的方案提交PR。

- 提交内容**：**
  - 第一阶段：RFC方案提交 
    1. RFC提交方式：
    1）以issue的形式进行提交
    2）递交地址为https://github.com/openvinotoolkit/openvino_notebooks/issues
    3）需要标题处打上【PaddlePaddle Hackathon 4】字样
    4）RFC语言不做强制要求
    2. RFC方案基本要求：
    1）应用场景与现有notebook demo不重复
    2）该示例中需要使用最新版本的openvino完成所有模型的推理部署
    3. RFC筛选依据：
    1）该示例在真实场景下是否具有实际应用价值
    2）该示例的流程逻辑是否清晰
    3）运行结果是否符合预期
  
    第二阶段：PR代码提交
    1. PR提交地址：
    https://github.com/openvinotoolkit/openvino_notebooks
    2. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样，并在描述处链接之前的RFC地址
    3. 该PR需满足notebook贡献规范
    4.开发者需要及时根据review的结果进行PR修改
    5. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排
  
- 参考示例：
  
  - 考虑到通用性，选取的应用场景尽量以英文为主，推荐方案场景有：
    *PaddleDetection
    1）行为识别（打架，抽烟，接打电话  )
    2）车辆追踪
    3）高空抛物
    4）产线上包装盒内产品计数
    *PaddleSeg
    1）背景去除与替换
    *PaddleNLP
    1）为申请大学的学生写推荐信（推荐使用ERINE3.0 tiny）
    *PaddleGAN
    1）人脸表情迁移
    *PaddleOCR
    1）古籍电子化保存
    *PaddleSpeech
    1）口录医嘱（推荐使用U2++）
    2）会议，视频电话等背景噪音去除
  
- 技术要求**：**

  - 熟练掌握OpenVINO python API与其他工具组件的使用方法

- 参考文档：

  - OpenVINO notebook仓库：
    https://github.com/openvinotoolkit/openvino_notebooks
  - OpenVINO notebook仓库代码贡献规范：
    https://github.com/openvinotoolkit/openvino_notebooks/blob/main/CONTRIBUTING.md
  - PaddlePaddle OpenVINO notebook提交示例：
    https://github.com/openvinotoolkit/openvino_notebooks/pull/547
    https://github.com/openvinotoolkit/openvino_notebooks/pull/497


### No.206：为 OpenVINO 实现 Paddle 算子 flip 转换 <a name='task206'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：进阶

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子flip转换。该算子用于沿指定轴反转 n 维 Tensor，算子说明可参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip。该任务中的算子难度较高，Paddle2ONNX展示了如何将flip映射到ONNX的算子： https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/legacy/op_mapper/tensor.py#L2018。我们也可以用同样的方式将其映射到OpenVINO的算子。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*


### No.207：为 OpenVINO 实现 Paddle 算子 linspace 转换 <a name='task207'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：进阶

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子 linspace转换。该OP用于返回一个 Tensor，Tensor 的值为在区间 start 和 stop 上均匀间隔的 num 个值，输出 Tensor 的长度为 num。算子说明可参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linspace_cn.html#linspace
  该任务中的算子难度较高。Paddle2ONNX展示了如何将linspace映射到ONNX的算子： https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/legacy/op_mapper/tensor.py#L13840。我们也可以用同样的方式将其映射到OpenVINO的算子。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*

### No.208：为 OpenVINO 实现 Paddle 算子 set_value 转换 <a name='task208'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：进阶

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子 set_value转换，该OP用于将输入tensor中的指定片段的数据进行赋值， Paddle官方并未提供该OP的文档说明，算子复现方法可参考https://github.com/PaddlePaddle/Paddle3D/blob/develop/paddle3d/models/heads/dense_heads/petr_head.py#L663，此外PaddleStructure中的表格识别模型也会应用到该算子https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/models_list_en.md#22-table-recognition。该任务中的算子难度较高。Paddle2ONNX展示了如何将set_value映射到ONNX的算子： https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/paddle2onnx/legacy/op_mapper/tensor.py#L26。我们也可以用同样的方式将其映射到OpenVINO的算子。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*

### No.209：为 OpenVINO 实现 Paddle 算子 silu 转换 <a name='task209'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：基础

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子  silu 转换，该算子为激活层算子。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*

### No.210：为 OpenVINO 实现 Paddle 算子one_hot_v2 转换 <a name='task210'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：基础

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子 one_hot_v2 转换，该算子将输入'x'中的每个 id 转换为一个 one-hot 向量。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*


### No.211：为 OpenVINO 实现 Paddle 算子softshrink 转换 <a name='task211'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：基础

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子 softshrink 转换，该算子为激活层算子。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*

### No.212：为 OpenVINO 实现 Paddle 算子 mean 转换 <a name='task212'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：基础

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子 mean转换，该OP用于沿参数 axis 计算 x 的平均值。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*

### No.213：为 OpenVINO 实现 Paddle 算子index_select转换 <a name='task213'></a>

- 技术标签：深度学习框架，C++，Python，OpenVINO

- 任务难度：基础

- 详细描述：每个框架都有自己的模型和算子表达。OpenVINO对PaddlePaddle的支持需要从Paddle的算子映射转换到OpenVINO的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。

  在这个任务中，你需要为OpenVINO实现Paddle算子 index_select转换，该OP沿着指定轴 axis 对输入 x 进行索引，取 index 中指定的相应项，创建并返回到一个新的 Tensor。具体做法请参考https://github.com/openvinotoolkit/openvino/pull/11731

- 提交内容**：**

  - 提交地址：https://github.com/openvinotoolkit/openvino
    1. 在https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
    2. 在https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp中注册该算子映射
    3. 在https://github.com/openvinotoolkit/openvino/tree/master/src/core/tests/frontend/paddle/test_models/gen_scripts添加该算子的单测实例生成脚本
    4. 在https://github.com/openvinotoolkit/openvino/blob/master/src/core/tests/frontend/paddle/op_fuzzy.cpp注册单测实例
    5. PR中需附上该算子在Paddle中算子说明或者参考实现，例如：
       https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
    6. 提交时需附上单测结果的截图
    7. 提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样
    8. 官方将根据合格PR的提交顺序进行review，例如：A同学最先提交PR，并且测试用例通过，同时该测试用例已覆盖Paddle官方文档中所有支持的数据输入格式，那我们将优先review该份PR。但如果A同学没有在1周时间内根据官方给出的review意见进行反馈，我们将优先review第二位提交者的PR，以此类推。
    9. 如果该Paddle OP 无法被mapping到openvino现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

- 技术要求**：**

  - 熟练掌握 C++
  - 了解OpenVINO和PaddlePaddle相关深度学习计算算子
  - 了解OpenVINO推理引擎相关技术背景

- 参考文档：

  - OpenVINO算子库文档：
    https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/opset10.md

  - OpenVINO算子参考实现：
    https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/ngraph/runtime/reference

  - PaddlePaddle算子库文档：

    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html

    PaddlePaddle算子参考实现：
    https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests

    Paddle2ONNX算子映射参考代码
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/legacy/op_mapper
    https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

    可以先生成测试模型用Paddle VisualDL查看paddle算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph

    OpenVINO源码编译方法：
    参考：https://github.com/openvinotoolkit/openvino/wiki 
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

    单侧测试方法：
    $ cd bin/intel64/Release
    $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*


### No.214：Arm 虚拟硬件上完成 PP-OCR 文本检测模型的部署与优化 <a name='task214'></a>

- 任务难度：🌟 基础
- 详细描述：任务目标为将 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 模型库中的文本检测模型 (Text Detection Model) 部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
  提示，该任务会涉及以下几个关键步骤：
      1. 选择合适的模型
         - 可选模型库 [Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md#1-text-detection-model)(eg.en_PP-OCRv3_det)，但不局限于模型库模型，支持高性能自研文本检测模型。
         - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
      2. 编译模型
         - 编译模型前请对模型进行适当地量化处理, 确保算子能够尽可能多地调用 Arm CMSIS-NN 库 (不支持的算子可回调 C 标准库执行)。CMSIS-NN 文档请查看 [Arm-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)。
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证文本检测应用程序的运行结果。
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在RFC文档确认后提供。
- 技术要求：
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论(例如, 模型量化)并掌握基本的嵌入式软件开发知识。

### No.215：Arm 虚拟硬件上完成 PP-OCR 文本方向分类模型的部署与优化 <a name='task215'></a>

- - 任务难度：🌟 基础
  - 详细描述：任务目标为将 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 模型库中的文本方向分类模型 (Text Angle Classification Model) 部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
    提示，该任务会涉及以下几个关键步骤：
        1. 选择合适的模型
           - 可选模型库 [Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md#3-text-angle-classification-model)(eg.ch_ppocr_mobile_v2.0_cls)，但不局限于模型库模型，支持高性能自研文本方向分类模型。
           - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
        2. 编译模型
           - 编译模型前请对模型进行适当地量化处理, 确保算子能够尽可能多地调用 Arm CMSIS-NN 库 (不支持的算子可回调 C 标准库执行)。CMSIS-NN 文档请查看 [Arm-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)。
           - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
           - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
        3. 应用程序编写
           - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
           - 构建应用程序，生成可执行文件。
        4. 使用 Arm 虚拟硬件验证运行结果
           - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
           - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证文本方向分类应用程序的运行结果。
  - 提交内容：
    - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
    - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在RFC文档确认后提供。
  - 技术要求：
    - 熟练使用 c/c++，Python 进行工程项目开发。
    - 熟悉基础的 Linux 操作系统命令。
    - 熟悉深度学习工程开发流程，tinyML 相关知识理论(例如, 模型量化)并掌握基本的嵌入式软件开发知识。


### No.216：Arm 虚拟硬件上完成 PaddleClas 模型的部署 <a name='task216'></a>

- - 任务难度：🌟 基础
  - 详细描述：任务目标为将 [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 模型库中的模型部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
    提示，该任务会涉及以下几个关键步骤：
        1. 选择合适的模型
           - 可选模型库 [Model Zoo](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5/docs/zh_CN/models)(eg.PP-LCNet)，但不局限于模型库模型，支持高性能自研模型。
           - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
        2. 编译模型
           - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
           - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
        3. 应用程序编写
           - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
           - 构建应用程序，生成可执行文件。
        4. 使用 Arm 虚拟硬件验证运行结果
           - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
           - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证应用程序的运行结果。
  - 提交内容：
    - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
    - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在 RFC 文档确认后提供。
  - 技术要求：
    - 熟练使用 c/c++，Python 进行工程项目开发。
    - 熟悉基础的 Linux 操作系统命令。
    - 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。


### No.217：Arm 虚拟硬件上完成 PaddleClas 模型的部署与优化 <a name='task217'></a>

- - 任务难度：🌟🌟 进阶
  - 详细描述：任务目标为将 [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 模型库中的模型部署在 [Arm Cortex-M85](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m85) 处理器上并使用 Arm 虚拟硬件 [Corstone-310](https://www.arm.com/products/silicon-ip-subsystems/corstone-310) 平台进行验证 (本题目与 No.3 题的主要区别在于是否对模型进行量化等处理从而确保运行时可以最大程度地调用 Arm CMSIS-NN 库的算子，及对部署目标平台的调整)。  
    提示，该任务会涉及以下几个关键步骤：
        1. 选择合适的模型
           - 可选模型库 [Model Zoo](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5/docs/zh_CN/models)(eg.PP-LCNetv2)，但不局限于模型库模型，支持高性能自研模型。
           - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
        2. 编译模型
           - 编译模型前请对模型进行适当地量化处理, 确保算子能够尽可能多地调用 Arm CMSIS-NN 库 (不支持的算子可回调 C 标准库执行)。CMSIS-NN 文档请查看 [Arm-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)。
           - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
           - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
        3. 应用程序编写
           - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
           - 构建应用程序，生成可执行文件。
        4. 使用 Arm 虚拟硬件验证运行结果
           - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
           - 调用 Arm 虚拟硬件 Corstone-310 (`VHT_Corstone_SSE-310`) 平台验证应用程序的运行结果。
  - 提交内容：
    - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
    - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在 RFC 文档确认后提供。
  - 技术要求：
    - 熟练使用 c/c++，Python 进行工程项目开发。
    - 熟悉基础的 Linux 操作系统命令。
    - 熟悉深度学习工程开发流程，tinyML 相关知识理论 (例如, 模型量化) 并掌握基本的嵌入式软件开发知识。


### No.218：Arm 虚拟硬件上完成 PaddleSeg 模型的部署 <a name='task218'></a>

- 任务难度：🌟 基础
- 详细描述：任务目标为将 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 模型库中的模型部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
  提示，该任务会涉及以下几个关键步骤：
      1. 选择合适的模型
         - 可选模型库 [Model Zoo](https://github.com/PaddlePaddle/PaddleSeg#-overview)(eg.BiSeNet V2)，但不局限于模型库模型，支持高性能自研模型。
         - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
      2. 编译模型
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证应用程序的运行结果。
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在 RFC 文档确认后提供。
- 技术要求：
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。

### No.219：Arm 虚拟硬件上完成 PaddleSeg 模型的部署与优化 <a name='task219'></a>

- 任务难度：🌟🌟 进阶
- 详细描述：任务目标为将 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 模型库中的模型部署在 [Arm Cortex-M85](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m85) 处理器上并使用 Arm 虚拟硬件 [Corstone-310](https://www.arm.com/products/silicon-ip-subsystems/corstone-310) 平台进行验证 (本题目与 No.5 题的主要区别在于是否对模型进行量化等处理从而确保运行时可以最大程度地调用 Arm CMSIS-NN 库的算子，及对部署目标平台的调整)。  
  提示，该任务会涉及以下几个关键步骤：
      1. 选择合适的模型
         - 可选模型库 [Model Zoo](https://github.com/PaddlePaddle/PaddleSeg#-overview)(eg. PP-HumanSeg-Lite)，但不局限于模型库模型，支持高性能自研模型。
         - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
      2. 编译模型
         - 编译模型前请对模型进行适当地量化处理, 确保算子能够尽可能多地调用 Arm CMSIS-NN 库 (不支持的算子可回调 C 标准库执行)。CMSIS-NN 文档请查看 [Arm-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)。
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Corstone-310 (`VHT_Corstone_SSE-310`) 平台验证应用程序的运行结果。
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在 RFC 文档确认后提供。
- 技术要求：
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论 (例如, 模型量化) 并掌握基本的嵌入式软件开发知识。

### No.220：Arm 虚拟硬件上完成 PP-TinyPose 模型的部署与优化并在物理开发板上进行验证 <a name='task220'></a>

- 任务难度：🌟🌟 进阶
- 详细描述：任务目标为首先将 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 模型库中的 [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose) 模型部署在 [Arm Cortex-M33](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m33) 处理器上并使用 Arm 虚拟硬件 Cortex-M33 平台进行验证；然后将验证后的应用程序移植在物理开发板上进行二次验证。    
  提示，该任务会涉及以下几个关键步骤：
      1. 选择合适的模型
         - 可选模型 [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose#关键点检测模型)，但不局限于该模型，支持高性能自研模型。
         - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
      2. 编译模型
         - 编译模型前请对模型进行适当地量化处理, 确保算子能够尽可能多地调用 Arm CMSIS-NN 库 (不支持的算子可回调 C 标准库执行)。CMSIS-NN 文档请查看 [Arm-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)。
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Cortex-M33 (`VHT_MPS2_Cortex-M33`) 平台验证应用程序的运行结果。
      5. 使用物理开发板验证运行结果
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在RFC文档确认后提供。
- 技术要求：
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论(例如, 模型量化)并掌握基本的嵌入式软件开发知识。

### No.221：Arm 虚拟硬件上完成 PaddleSpeech 模型的部署与优化 <a name='task221'></a>

- 任务难度：🌟🌟 进阶
- 详细描述：任务目标为首先将 [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) 模型库中的 [PANN](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#audio-classification-models) 模型部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
  提示，该任务会涉及以下几个关键步骤：
      1. 选择合适的模型
         - 可选模型 [PANN](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#audio-classification-models)，但不局限于该模型，支持高性能自研模型。
         - 注意所选模型是否能成功地被`tvmc`编译(部分算子目前不支持 TVM 编译，tvmc 相关[帮助文档](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html?highlight=tvmc)请访问 [TVM 官网](https://tvm.apache.org/docs/)或 [GitHub 仓库](https://github.com/apache/tvm))。
      2. 编译模型
         - 编译模型前请对模型进行适当地量化处理, 确保算子能够尽可能多地调用 Arm CMSIS-NN 库 (不支持的算子可回调 C 标准库执行)。CMSIS-NN 文档请查看 [Arm-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)。
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/OCR-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证应用程序的运行结果。  
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [OCR-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在RFC文档确认后提供。
- 技术要求：
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论(例如, 模型量化)并掌握基本的嵌入式软件开发知识。


### No.222：为 TVM 增加单个 Paddle 算子 yolo_box 并在 Arm 虚拟硬件上完成 PP-Yolo 模型的部署 <a name='task222'></a>

- 任务难度：🌟 基础
- 详细描述：由于目前 TVM 前端 Paddle 算子 yolo_box 缺失，导致部分模型 (eg. [PP-Yolo](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyolo/README_cn.md) ) 无法通过 TVM 编译。本任务目标为首先为 TVM 增加单个 Paddle 算子 yolo_box，然后将算子补齐后的 PP-Yolo 模型部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
  提示，该任务会涉及以下几个关键步骤：
      1. 完成 TVM Paddle 前端算子 yolo_box 的开发
         - 完成 yolo_box 算子映射，并提交PR。算子代码添加在[相应文件](https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py)中。
         - 完成 yolo_box 算子实现代码，并添加单测且验证通过。单测添加在[相应文件](https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py)中。
         - 可参考：[paddle 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Overview_cn.html) 以及 [TVM Relay api文档](https://tvm.apache.org/docs/reference/api/python/relay/index.html)
      2. 编译模型
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/Object-Detection-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证应用程序的运行结果。
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [Object-Detection-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/Object-Detection-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在 RFC 文档确认后提供。
- 技术要求：
  - 熟练掌握 TVM 原理及 TVM 前端算子开发流程。
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。

### No.223：为 TVM 增加多个 Paddle 算子 stack 和 prior_box 并在 Arm 虚拟硬件上完成 SSD 模型的部署 <a name='task223'></a>

- 任务难度：🌟🌟 进阶
- 详细描述：由于目前 TVM 前端 Paddle 算子 stack 和 prior_box 缺失，导致部分模型 (eg. [SSD](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ssd/README.md) ) 无法通过 TVM 编译。本任务目标为首先为 TVM 增加 2 个 Paddle 算子 stack 和 prior_box，然后将算子补齐后的 SSD 模型部署在 [Arm Cortex-M55](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) 处理器上并使用 Arm 虚拟硬件 [Corstone-300](https://www.arm.com/products/silicon-ip-subsystems/corstone-300) 平台进行验证。  
  提示，该任务会涉及以下几个关键步骤：
      1. 完成 TVM Paddle 前端算子 stack 和 prior_box 的开发
         - 完成  stack 和 prior_box 算子映射，并提交PR。算子代码添加在[相应文件](https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py)中。
         - 完成 stack 和 prior_box 算子实现代码，并添加单测且验证通过。单测添加在[相应文件](https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py)中。
         - 可参考：[paddle 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Overview_cn.html) 以及 [TVM Relay api文档](https://tvm.apache.org/docs/reference/api/python/relay/index.html)
      2. 编译模型
         - 训练模型 (Trained model) 需导出为 Paddle inference 推理模型才可编译。
         - 可参考[示例代码](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/Object-Detection-example/run_demo.sh)中模型编译部分(`line152-167`)。
      3. 应用程序编写
         - 确保结果的可读性，请正确地完成应用程序的前后端数据处理。
         - 构建应用程序，生成可执行文件。
      4. 使用 Arm 虚拟硬件验证运行结果
         - 订阅 Arm 虚拟硬件产品并远程登入创建的实例 (可通过 `ssh` 命令)。RFC 文档确认后，会提供相关镜像订阅及使用指导。
         - 调用 Arm 虚拟硬件 Corstone-300 (`VHT_Corstone_SSE-300_Ethos-U55`) 平台验证应用程序的运行结果。
- 提交内容：
  - 项目启动前，请提交 RFC 文档（注意标明所选模型及来源）。
  - PR 示例工程代码至 GitHub 仓库，工程文件结构请参考 [Object-Detection-example](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/blob/main/Object-Detection-example)，注意同步更新 README.md 说明文档的内容。代码合入仓库地址及合入规范将在 RFC 文档确认后提供。
- 技术要求：
  - 熟练掌握 TVM 原理及 TVM 前端算子开发流程。
  - 熟练使用 c/c++，Python 进行工程项目开发。
  - 熟悉基础的 Linux 操作系统命令。
  - 熟悉深度学习工程开发流程，tinyML 相关知识理论并掌握基本的嵌入式软件开发知识。


### No.224：利用 Jina AI 来部署开放域聊天模型 PLATO-Mini <a name='task224'></a>

**题目简介：** 使用 [Jina](https://docs.jina.ai/) 框架部署预训练模型 [PLATO-Mini](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/dialogue/unified_transformer/README.md) 实现开放域对话服务。

- 技术标签：深度学习，Python
- 任务难度：基础 ⭐️
- 详细描述：
  - 1）学习 `Jina` 开源MLOps框架的使用，熟悉3个基本概念 `Document`, `Executor`, `Flow`；详细文档参见 [docs](https://docs.jina.ai/)
  - 2）封装实现一个新的 Executor `PlatoXLExecutor`，在 `__init__` 中实现paddle模型的加载; 实现一个新的 endpoint `generate` 完成对话任务；请参考文档 [doc](https://docs.jina.ai/concepts/executor/)
  - 3）定义一个 Jina Flow，Flow中使用 `PlatoXLExecutor`对外提供 `gRPC`, `HTTP` 和 `WebSocket` 三种服务类型；请参考文档 [doc](https://docs.jina.ai/concepts/flow/#why-should-you-use-a-flow)
  - 4）【进阶】 实现 半精度 fp16 的推理，减少模型对GPU显存的占用，同时加速模型推理速度；
  - 5）【进阶】完成一个简单的交互界面展示对话效果
- 提交内容：
  - 提交方案 rfc 至仓库 [rfcs](https://github.com/jina-ai/jina-paddle-hackathon/tree/main/rfcs) 目录下；
  - 上传 `PlatoXLExecutor` 到 [Jina Hub](https://cloud.jina.ai/executors)
  - 请将代码提交至仓库 [src](https://github.com/jina-ai/jina-paddle-hackathon/tree/main/src) 目录下，并创建自己的文件夹。
- 合入标准
  - 按 rfc 设计文档格式，提交设计思路，并保证后期提交的内容与该技术方案保持一致；如果实际开发过程中需要对设计实现思路有更改，提前和mentor沟通确认通过后才可以被接受。
  - 针对提交的内容，在自定义 repo 下建立API功能实现、单测、API 文档对应的文件夹，并完成功能实现、单测、功能测试。
- 技术要求：
  - 熟练掌握Python, 了解AI部署


### No.225：使用 Jina AI 和 UIE 搭建可视化信息抽取系统 <a name='task225'></a>

**题目简介：** 基于通用信息抽取 (UIE)(https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie) 在 [Jina](https://docs.jina.ai/) 中实现一个可视化的信息抽取系统

- 技术标签：深度学习，Python
- 任务难度：基础 ⭐️
- 详细描述：
  - 1）学习 `Jina` 开源MLOps框架的使用，熟悉3个基本概念 `Document`, `Executor`, `Flow`；详细文档参见 [docs](https://docs.jina.ai/)
  - 2）封装实现一个新的 Executor `UIEExecutor`，在 `__init__` 中实现UIE Taskflow 的加载; 请参考文档 [doc](https://docs.jina.ai/concepts/executor/)
  - 3）定义一个 Jina Flow，Flow中使用 `UIEExecutor`对外提供 `gRPC`, `HTTP` 和 `WebSocket` 三种服务类型；请参考文档 [doc](https://docs.jina.ai/concepts/flow/#why-should-you-use-a-flow)
  - 4）实现一个前端交互页面可视化信息抽取结果 (前端页面的实现可以使用 Dask, Gradio, Streamlit 或者 Vue, React等)
- 提交内容：
  - 提交方案 rfc 至仓库 [rfcs](https://github.com/jina-ai/jina-paddle-hackathon/tree/main/rfcs) 目录下；
  - 上传 `UIEExecutor` 到 [Jina Hub](https://cloud.jina.ai/executors)
  - 请将代码提交至仓库 [src](https://github.com/jina-ai/jina-paddle-hackathon/tree/main/src) 目录下，并创建自己的文件夹。
- 合入标准
  - 按 rfc 设计文档格式，提交设计思路，并保证后期提交的内容与该技术方案保持一致；如果实际开发过程中需要对设计实现思路有更改，提前和mentor沟通确认通过后才可以被接受。
  - 针对提交的内容，在自定义 repo 下建立API功能实现、单测、API 文档对应的文件夹，并完成功能实现、单测、功能测试。
- 技术要求：
  - 熟练掌握Python, 了解AI部署

### No.226：TVM项目1--为Paddle框架新增TVM算子（进阶题）<a name='task231'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend覆盖算子不够全面，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend补充算子。


**目标：**

为PaddlePaddle框架新增以下TVM算子，并通过单测测试；

**任务难度：进阶题**

将以下6个算子做适配并完成单算子测试。

| TVM                   | 队伍 |
| --------------------- | ---- |
| multiclass_nms3       |      |
| set_value             |      |
| pool3d                |      |
| max_pool2d_with_index |      |
| tanh_shrink           |      |
| max_pool3d_with_index |      |

【提示】：

1.在提交PR前需通过RFC提交自己要算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）

2.优先提交代码的团队会被优先review

3.优先通过算子单测的团队视为完成任务，且此算子不会被作为比赛任务继续适配

**算子任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

2. 完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3. 注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4. OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）

**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.227：TVM项目2-为Paddle框架新增TVM算子（基础题） <a name='task227'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend覆盖算子不够全面，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend补充算子。


**目标：**

为PaddlePaddle框架新增以下TVM算子，并通过单测测试；

**任务难度：新增TVM算子基础题**

从下表中选取题目类型为 **“基础题”** 的任意6个算子做适配并完成单算子测试。

| TVM                        | 队伍 |
| -------------------------- | ---- |
| affine_channel             |      |
| conv3d                     |      |
| data_norm                  |      |
| dist                       |      |
| eye                        |      |
| fill_zeros_like            |      |
| gaussian_random            |      |
| grid_sampler               |      |
| index_select               |      |
| mish                       |      |
| flip                       |      |
| p_norm                     |      |
| roi_align                  |      |
| share_data                 |      |
| silu                       |      |
| softmax_with_cross_entropy |      |
| softshrink                 |      |
| linspace                   |      |
| take_along_axis            |      |
| thresholded_relu           |      |
| tile                       |      |
| unique                     |      |
| unstack                    |      |
| where                      |      |

【提示】：

1.在提交PR前需通过RFC提交自己要算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）

2.优先提交代码的团队会被优先review

3.优先通过算子单测的团队视为完成任务，且此算子不会被作为比赛任务继续适配



**算子任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

 2.完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3.注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4.OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入



**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）



**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.228：TVM项目3 -为Paddle框架新增TVM算子（基础题）<a name='task228'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend覆盖算子不够全面，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend补充算子。


**目标：**

为PaddlePaddle框架新增以下TVM算子，并通过单测测试；

**任务难度：新增TVM算子基础题**

从下表中选取题目类型为 **“基础题”** 的任意6个算子做适配并完成单算子测试。

| TVM                        | 队伍 |
| -------------------------- | ---- |
| affine_channel             |      |
| conv3d                     |      |
| data_norm                  |      |
| dist                       |      |
| eye                        |      |
| fill_zeros_like            |      |
| gaussian_random            |      |
| grid_sampler               |      |
| index_select               |      |
| mish                       |      |
| flip                       |      |
| p_norm                     |      |
| roi_align                  |      |
| share_data                 |      |
| silu                       |      |
| softmax_with_cross_entropy |      |
| softshrink                 |      |
| linspace                   |      |
| take_along_axis            |      |
| thresholded_relu           |      |
| tile                       |      |
| unique                     |      |
| unstack                    |      |
| where                      |      |

【提示】：

1.在提交PR前需通过RFC提交自己要算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）

2.优先提交代码的团队会被优先review

3.优先通过算子单测的团队视为完成任务，且此算子不会被作为比赛任务继续适配



**算子任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

 2.完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3.注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4.OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入



**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）



**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.229：TVM项目4 -为Paddle框架新增TVM算子（基础题）<a name='task229'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend覆盖算子不够全面，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend补充算子。


**目标：**

为PaddlePaddle框架新增以下TVM算子，并通过单测测试；

**任务难度：新增TVM算子基础题**

从下表中选取题目类型为 **“基础题”** 的任意6个算子做适配并完成单算子测试。

| TVM                        | 队伍 |
| -------------------------- | ---- |
| affine_channel             |      |
| conv3d                     |      |
| data_norm                  |      |
| dist                       |      |
| eye                        |      |
| fill_zeros_like            |      |
| gaussian_random            |      |
| grid_sampler               |      |
| index_select               |      |
| mish                       |      |
| flip                       |      |
| p_norm                     |      |
| roi_align                  |      |
| share_data                 |      |
| silu                       |      |
| softmax_with_cross_entropy |      |
| softshrink                 |      |
| linspace                   |      |
| take_along_axis            |      |
| thresholded_relu           |      |
| tile                       |      |
| unique                     |      |
| unstack                    |      |
| where                      |      |

【提示】：

1.在提交PR前需通过RFC提交自己要算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）

2.优先提交代码的团队会被优先review

3.优先通过算子单测的团队视为完成任务，且此算子不会被作为比赛任务继续适配



**算子任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

 2.完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3.注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4.OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入



**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）



**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.230：TVM项目5-为Paddle框架新增TVM算子（基础题） <a name='task230'></a>

1. **背景：**

   TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

   目前TVM中的PaddlePaddle Frontend覆盖算子不够全面，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend补充算子。

   **目标：**

   为PaddlePaddle框架新增以下TVM算子，并通过单测测试；

   **任务难度：新增TVM算子基础题**

   从下表中选取题目类型为 **“基础题”** 的任意6个算子做适配并完成单算子测试。

   | TVM                        | 队伍 |
   | -------------------------- | ---- |
   | affine_channel             |      |
   | conv3d                     |      |
   | data_norm                  |      |
   | dist                       |      |
   | eye                        |      |
   | fill_zeros_like            |      |
   | gaussian_random            |      |
   | grid_sampler               |      |
   | index_select               |      |
   | mish                       |      |
   | flip                       |      |
   | p_norm                     |      |
   | roi_align                  |      |
   | share_data                 |      |
   | silu                       |      |
   | softmax_with_cross_entropy |      |
   | softshrink                 |      |
   | linspace                   |      |
   | take_along_axis            |      |
   | thresholded_relu           |      |
   | tile                       |      |
   | unique                     |      |
   | unstack                    |      |
   | where                      |      |

   【提示】：

   1.在提交PR前需通过RFC提交自己要算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）

   2.优先提交代码的团队会被优先review

   3.优先通过算子单测的团队视为完成任务，且此算子不会被作为比赛任务继续适配

   

   **算子任务提交：**

   1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

   整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

   ```bash
   def convert_leaky_relu(g, op, block):
       """Operator converter for leaky_relu."""
   
       alpha = op.attr("alpha")
       x = g.get_node(op.input("X")[0])
       out = _op.nn.leaky_relu(x, alpha=alpha)
       g.add_node(op.output("Out")[0], out)
   ```

   更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

    2.完成TVM算子实现代码，并添加单测且验证通过

   用户需要做两件事：

   - 定义组网结构
   - 制定输入shape以及输入类型

   通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

   ```bash
   @tvm.testing.uses_gpu
   def test_forward_leaky_relu():
       @paddle.jit.to_static
       def leaky_relu(inputs):
           return nn.functional.leaky_relu(inputs)
   
       input_shape = [1, 3, 10, 10]
       input_data = paddle.rand(input_shape, dtype="float32")
       verify_model(leaky_relu, input_data=input_data)
   ```

   更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

   3.注意代码风格问题

   TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

   4.OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

   

   **RFC内容提交（必须项）**

   开发者需要提供一份RFC，用来描述本次任务的设计方案；

   参考模板：

   Solution name 方案名称

   Description 方案描述

   Workflow 方案流程

   Results visualizing 方案运行效果 

   Project Timeline 项目提交时间计划

   Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）

   

   **算子参考文档：**

   1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
   2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.231：TVM项目6 -为TVM PaddlePaddle前端完善算子支持程度或修复问题（基础题） <a name='task232'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend已经支持100+算子，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，但由于在实现后由于Paddle框架/TVM框架的升级，部分算子可能存在没支持全（某些属性，或某些输入情况没覆盖）/TVM OP/API存在兼容问题。 我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend完善或修复现有Frontend中算子支持程度和问题。


**目标：（开放题）**

为TVM PaddlePaddle前端完善TVM算子支持程度或修复问题，并补充相应单测；对于TVM算子是否支持全（某些属性，或某些输入情况没覆盖），可参考Paddle对算子的定义，或者Paddle2ONNX的实现

1. TVM Paddle前端代码：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py
2. Paddle算子定义：https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators
3. Paddle2ONNX实现：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

**任务难度：基础题**

完善或修复4个以上算子支持程度或问题

【提示】：

1.优先提交代码的团队会被优先review

2.优先通过单测并合入的团队视为完成任务

3.在提交PR前需通过RFC提交自己要修复的算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）；如有选手重复，优先通过RFC的选手有效。

4.如提交的代码已经被合入的PR覆盖，例如修复的4个算子，其中已经有2个被合入的修复，则视为这2个无效

**任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

2. 完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3. 注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4. OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）

**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html



### No.232：TVM项目7-为TVM PaddlePaddle前端完善算子支持程度或修复问题（基础题） <a name='task232'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend已经支持100+算子，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，但由于在实现后由于Paddle框架/TVM框架的升级，部分算子可能存在没支持全（某些属性，或某些输入情况没覆盖）/TVM OP/API存在兼容问题。 我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend完善或修复现有Frontend中算子支持程度和问题。


**目标：（开放题）**

为TVM PaddlePaddle前端完善TVM算子支持程度或修复问题，并补充相应单测；对于TVM算子是否支持全（某些属性，或某些输入情况没覆盖），可参考Paddle对算子的定义，或者Paddle2ONNX的实现

1. TVM Paddle前端代码：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py
2. Paddle算子定义：https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators
3. Paddle2ONNX实现：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

**任务难度：基础题**

完善或修复4个以上算子支持程度或问题

【提示】：

1.优先提交代码的团队会被优先review

2.优先通过单测并合入的团队视为完成任务

3.在提交PR前需通过RFC提交自己要修复的算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）；如有选手重复，优先通过RFC的选手有效。

4.如提交的代码已经被合入的PR覆盖，例如修复的4个算子，其中已经有2个被合入的修复，则视为这2个无效

**任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

2. 完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3. 注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4. OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）

**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.233：TVM项目8-为TVM PaddlePaddle前端完善算子支持程度或修复问题（基础题） <a name='task233'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend已经支持100+算子，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，但由于在实现后由于Paddle框架/TVM框架的升级，部分算子可能存在没支持全（某些属性，或某些输入情况没覆盖）/TVM OP/API存在兼容问题。 我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend完善或修复现有Frontend中算子支持程度和问题。


**目标：（开放题）**

为TVM PaddlePaddle前端完善TVM算子支持程度或修复问题，并补充相应单测；对于TVM算子是否支持全（某些属性，或某些输入情况没覆盖），可参考Paddle对算子的定义，或者Paddle2ONNX的实现

1. TVM Paddle前端代码：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py
2. Paddle算子定义：https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators
3. Paddle2ONNX实现：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

**任务难度：基础题**

完善或修复4个以上算子支持程度或问题

【提示】：

1.优先提交代码的团队会被优先review

2.优先通过单测并合入的团队视为完成任务

3.在提交PR前需通过RFC提交自己要修复的算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）；如有选手重复，优先通过RFC的选手有效。

4.如提交的代码已经被合入的PR覆盖，例如修复的4个算子，其中已经有2个被合入的修复，则视为这2个无效

**任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

2. 完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3. 注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4. OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）

**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.234：TVM项目9 -为TVM PaddlePaddle前端完善算子支持程度或修复问题（进阶题）<a name='task234'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend已经支持100+算子，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，但由于在实现后由于Paddle框架/TVM框架的升级，部分算子可能存在没支持全（某些属性，或某些输入情况没覆盖）/TVM OP/API存在兼容问题。 我们希望通过此次黑客松活动为TVM中的PaddlePaddle Frontend完善或修复现有Frontend中算子支持程度和问题。


**目标：（开放题）**

为TVM PaddlePaddle前端完善TVM算子支持程度或修复问题，并补充相应单测；对于TVM算子是否支持全（某些属性，或某些输入情况没覆盖），可参考Paddle对算子的定义，或者Paddle2ONNX的实现

1. TVM Paddle前端代码：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py
2. Paddle算子定义：https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators
3. Paddle2ONNX实现：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper

**任务难度：进阶题**

完善或修复8个以上算子支持程度或问题

【提示】：

1.优先提交代码的团队会被优先review

2.优先通过单测并合入的团队视为完成任务

3.在提交PR前需通过RFC提交自己要修复的算子列表。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）；如有选手重复，优先通过RFC的选手有效。

4.如提交的代码已经被合入的PR覆盖，例如修复的8个算子，其中已经有2个被合入的修复，则视为这2个无效？

**任务提交：**

1. 完成TVM算子映射，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle OP，可通过一对一或者多对一去组合实现，以下面Leaky_relu为例，可直接通过_op.nn.leaky_relu去表示PaddlePaddle的leaky_relu op

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

2. 完成TVM算子实现代码，并添加单测且验证通过

用户需要做两件事：

- 定义组网结构
- 制定输入shape以及输入类型

通过定义PaddlePaddle 动态图 api 进行组网，指定输入数据的shape以及类型，调用verify_model函数自动进行Relay IR转换以及精度验证工作

```bash
@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)
```

更多单测实现以及添加在：https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py

3. 注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

4. OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）

**算子参考文档：**

1. Paddle算子文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
2. TVM Relay API文档：https://tvm.apache.org/docs/reference/api/python/relay/index.html

### No.235：TVM项目10-为TVM PaddlePaddle前端增加PaddleSlim量化模型的支持（进阶题） <a name='task235'></a>

**背景：**

TVM是一款开源的、端到端的深度学习模型编译框架，用于优化深度学习模型在CPU、GPU、ARM等任意目标环境下的推理运行速度，TVM采用编译器思想，前端接收各个深度学习框架保存的模型格式，通过中间件Relay IR翻译之后，通过一系列优化（子图融合等），编译为特定后端可以识别的机器码完成模型推理。目前TVM支持PaddlePaddle/Pytorch/TensorFlow/Caffe/MxNet 等，同时也支持一些模型的中间格式如 ONNX、CoreML。

目前TVM中的PaddlePaddle Frontend已经支持100+算子，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，但目前仍未支持PaddleSlim量化模型，此题期望开发者增加TVM对PaddleSlim模型的支持，并验证量化的性能提升情况。

**目标：**

为TVM PaddlePaddle前端增加PaddleSlim量化模型的支持。 PaddleSlim量化模型与普通Paddle模型差异在于，模型中各OP前后通过`linear_quantize`和`dequantize_linear`来表达量化信息(算子与ONNX的DequantizeLinear/QuantizeLinear类似)，与ONNX量化模型原理类似。开发者在开发时，可参考TVM中其它深度学习框架前端对量化模型的支持方式来实现。

开发者可使用下面的量化模型进行正确性验证和性能测试

1. ResNet50_vd: https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar
2. MobileNetV1: https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar
3. PP-LiteSeg: https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new.tar

**代码参考链接**

1. TVM Paddle前端代码：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py
2. TVM ONNX前端代码: https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/onnx.py
3. Paddle 量化算子定义：https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/quantize_linear_op.cc
4. ONNX 量化算子定义：https://github.com/onnx/onnx/blob/main/docs/Operators.md

【提示】：

1.优先提交代码的团队会被优先review

2.优先通过单测并合入的团队视为完成任务

3.在提交PR前需通过RFC提交自己方案。RFC提交后，不可随意修改（如确实必要，与运营人员沟通）

**任务提交：**

1. 完成TVM量化支持，与量化性能验证数据，并提交 PR；提交PR时需在PR标题加上【PaddlePaddle Hackathon 4】字样

整体思想通过TVM Relay IR去表示对应PaddlePaddle 量化OP，并进行量化的支持。具体实现方案可参考TVM中的onnx前端实现

```bash
def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)
```

更多算子代码实现以及添加在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py

2. 注意代码风格问题

TVM的PR会有代码风格检查，python代码检查工具基于black，更多细节可参考：[https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code%20style#python-code-styles](https://tvm.apache.org/docs/contribute/code_guide.html?highlight=code style#python-code-styles)

3. 代码实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**RFC内容提交（必须项）**

开发者需要提供一份RFC，用来描述本次任务的设计方案；

参考模板：

Solution name 方案名称

Description 方案描述

Workflow 方案流程

Results visualizing 方案运行效果 

Project Timeline 项目提交时间计划

Your experience in ML and DL (optional) 个人介绍及以往项目经历（可选）


～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～


### 合入标准

-  按 [API 设计规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) 完成 API设计文档；
- 按 [API 验收标准](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) 完成 API功能实现、单测、API文档；
- 稀疏 API 任务需符合稀疏 OP 的特殊开发规范（如有）：
  * 【yaml规则】：写到同一个yaml api，不要写多个，yaml需支持调度
  * 【kernel名规则】：[计算名] + 异构后缀，例如 matmul_csr_dense、softmax_csr、softmax_coo
  * 【文件名规则】：sparse/xx_kernel.cc，sparse/xx_kernel.cu，以目录区分，文件名与dense保持一致

### 参考内容

- [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)
- [新增 API 设计模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)
- [飞桨API Python 端开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html)
- [C++ 算子开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
- [飞桨API文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
- [API单测开发及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)


### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&QQ群的通知，及时参与。
