此文档展示 **PaddlePaddle Hackathon 第六期活动——开源贡献个人挑战赛科学计算方向任务** 详细介绍，更多详见 [PaddlePaddle Hackathon 说明](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/10_contribution/hackathon_cn.md)。

## 【开源贡献个人挑战赛-合作伙伴】任务详情

### No.43：为 OpenVINO 实现 Paddle 算子 tril/triu 转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO 对 PaddlePaddle 的支持需要从 Paddle 的算子映射转换到 OpenVINO 的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。
在这个任务中，你需要为 OpenVINO 实现 Paddle 算子 tril/triu 转换，该算子为激活层算子。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在 https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在 https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在 https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在 https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR 中需附上该算子在 Paddle 中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交 PR 时需在 PR 标题加上【PaddlePaddle Hackathon 6】字样
4. 官方将根据合格 PR 的提交顺序进行 review，例如：A 同学最先提交 PR，并且通过测试用例，同时该测试用例已覆盖 Paddle 官方文档中所有支持的数据输入格式，那我们将优先 review 该份 PR。但如果 A 同学没有在 1 周时间内根据官方给出的 review 意见进行反馈，我们将优先 review 第二位提交者的 PR，以此类推。
5. 如果该 Paddle OP 无法被 mapping 到 openvino 现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解 OpenVINO 和 PaddlePaddle 相关深度学习计算算子
- 了解 OpenVINO 推理引擎相关技术背景

**参考文档：**

- OpenVINO 算子库文档：https://docs.openvino.ai/2023.3/openvino_docs_ops_opset13.html
- OpenVINO 算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference
- PaddlePaddle 算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
- Paddle2ONNX 算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper
- 可以先生成测试模型用 Paddle VisualDL 查看 paddle 算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph
- OpenVINO 源码编译方法：
  1.  CMakeList 中开启 Paddle frontend 测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9
  2.  编译说明：https://github.com/openvinotoolkit/openvino/wiki

**Ubuntu 可参考：**

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

### No.44：为 OpenVINO 实现 Paddle 算子 rsqrt 转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO 对 PaddlePaddle 的支持需要从 Paddle 的算子映射转换到 OpenVINO 的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。
在这个任务中，你需要为 OpenVINO 实现 Paddle 算子 rsqrt 转换。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在 https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在 https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在 https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在 https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR 中需附上该算子在 Paddle 中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交 PR 时需在 PR 标题加上【PaddlePaddle Hackathon 6】字样
4. 官方将根据合格 PR 的提交顺序进行 review，例如：A 同学最先提交 PR，并且通过测试用例，同时该测试用例已覆盖 Paddle 官方文档中所有支持的数据输入格式，那我们将优先 review 该份 PR。但如果 A 同学没有在 1 周时间内根据官方给出的 review 意见进行反馈，我们将优先 review 第二位提交者的 PR，以此类推。
5. 如果该 Paddle OP 无法被 mapping 到 openvino 现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解 OpenVINO 和 PaddlePaddle 相关深度学习计算算子
- 了解 OpenVINO 推理引擎相关技术背景

**参考文档：**

- OpenVINO 算子库文档：https://docs.openvino.ai/2023.3/openvino_docs_ops_opset13.html
- OpenVINO 算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference
- PaddlePaddle 算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
- Paddle2ONNX 算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper
- 可以先生成测试模型用 Paddle VisualDL 查看 paddle 算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph
- OpenVINO 源码编译方法：
  1. CMakeList 中开启 Paddle frontend 测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9
  2. 编译说明：https://github.com/openvinotoolkit/openvino/wiki

**Ubuntu 可参考：**

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

### No.45：为 OpenVINO 实现 Paddle 算子 scaled_dot_product_attention 转换

**详细描述:**

每个框架都有自己的模型和算子表达。OpenVINO 对 PaddlePaddle 的支持需要从 Paddle 的算子映射转换到 OpenVINO 的算子。在这个过程中，我们将熟悉深度学习神经网络的算子表达和计算。
在这个任务中，你需要为 OpenVINO 实现 Paddle 算子 scaled_dot_product_attention 转换。

**提交地址：**

https://github.com/openvinotoolkit/openvino

**提交内容：**

1. 在 https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/src/op 添加算子映射的实现
2. 在 https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/src/op_table.cpp 中注册该算子映射
3. 在 https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle/tests/test_models/gen_scripts 添加该算子的单测实例生成脚本
4. 在 https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp 注册单测实例

**注意事项：**

1. PR 中需附上该算子在 Paddle 中算子说明或者参考实现，例如：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#topk
2. 提交时需附上单测结果的截图
3. 提交 PR 时需在 PR 标题加上【PaddlePaddle Hackathon 6】字样
4. 官方将根据合格 PR 的提交顺序进行 review，例如：A 同学最先提交 PR，并且通过测试用例，同时该测试用例已覆盖 Paddle 官方文档中所有支持的数据输入格式，那我们将优先 review 该份 PR。但如果 A 同学没有在 1 周时间内根据官方给出的 review 意见进行反馈，我们将优先 review 第二位提交者的 PR，以此类推。
5. 如果该 Paddle OP 无法被 mapping 到 openvino 现有算子中，需要开发者以文档的形式进行论证说明，并提出解决方案，一经证实，我们将挑选其中相对比较优秀的方案进行颁奖。

**示例参考：**

https://github.com/openvinotoolkit/openvino/issues?q=paddle+hackathon+is%3Amerged

**技术要求：**

- 熟练掌握 C++
- 了解 OpenVINO 和 PaddlePaddle 相关深度学习计算算子
- 了解 OpenVINO 推理引擎相关技术背景

**参考文档：**

- OpenVINO 算子库文档：https://docs.openvino.ai/2023.3/openvino_docs_ops_opset13.html
- OpenVINO 算子参考实现：https://github.com/openvinotoolkit/openvino/tree/master/src/core/reference/include/openvino/reference
- PaddlePaddle 算子库文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html
- Paddle2ONNX 算子映射参考代码：https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/paddle2onnx/mapper
- 可以先生成测试模型用 Paddle VisualDL 查看 paddle 算子的输入输出以及属性： https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph
- OpenVINO 源码编译方法： 5. CMakeList 中开启 Paddle frontend 测试：https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/CMakeLists.txt#L9 6. 编译说明：https://github.com/openvinotoolkit/openvino/wiki

**Ubuntu 可参考：**

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

### No.46：为 Openvino 支持 Paddle 2.6.0

**详细描述:**

Paddle2.6.0 中移除了很多 API 接口且部分 op 的输出大小产生了变化，本赛题的目标是为 Openvino 支持 Paddle 2.6.0。将 Paddle 升级到 2.6.0 后可以成功编译 Opennivo 且通过全部单测。

具体来说本赛提包含如下主要内容：

1. 修复因为 API 移除（主要是 fluid 相关 api）而导致的编译报错
2. 排查输出产生变化的 op（主要问题和 0-d tensor 有关），并对这些 op 进行适配使其可以通过单测
3. 确保修复后所有单测可以通过

该赛题已由 [@AndSonder](https://github.com/AndSonder) 提前开发并锁定： https://github.com/openvinotoolkit/openvino/pull/23010

### No.47：修复 OpenVINO 算子 set_value 问题

**描述:**

目前 OpenVINO 算子 set_value 完成了基本的功能，但在一些测试场景时会有报错。需要修复并完善 set_value 的功能。

**提交内容：**

- 启用测试”set_value8” https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/tests/op_fuzzy.cpp#L509
- 修复”set_value8”测试问题

**注意事项：**

- 启用”set_value8”测试后，编译运行会有‘StridedSlice doesn't have compiled executor’的错误。需修复问题并通过 CI 测试

**参考文档：**

- https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/paddle/docs/tests.md
- https://github.com/openvinotoolkit/openvino/pull/17536 解决了类似问题，可以此为参考扩展思路
- OpenVINO 源码编译方法（如上略）

**单测测试方法：**

```shell
  $ cd bin/intel64/Release
  $ ./paddle_tests --gtest_filter=PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/*set_value8*
```

### No.48（预留）：CPU 赛题，后续提供

...

### No.49（预留）：CPU 赛题，后续提供

...
