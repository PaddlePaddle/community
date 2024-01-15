# paddle_test推全
> 
> Tracking issue: [PaddlePaddle/Paddle#60793](https://github.com/PaddlePaddle/Paddle/issues/60793)

## 目的
现阶段Paddle编译c++单测有多种方式，如cc_test（windows下link静态库，linux下link paddle.so），paddle_test(windows和linux下都link动态库paddle.so)，现在推荐使用paddle_test，好处在于：
1. 使得编译单测时链接动态库，以减小编译单测过程中的产物体积。 
2. 提升至少两部分编译过程的速度，一是编译行为本身的速度，二是在CI上打包/下载编译产物的速度。

## 方案设计
以一个CMakeLists.txt文件为一个任务单元，共43个CMakeLists.txt文件包含至少一处cc_test。
将其中一个CMakeLists.txt文件的cc_test替换为paddle_test即为完成一个任务。

## 注意事项
1. 在 test阶段出现 Exit code 0xc000007b 问题，例如：
[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9766842/job/24813466)
解决办法：在与之对应的CMakeLists.txt中添加 copy_onnx(xxx_test),例如：
[示例PR](https://github.com/PaddlePaddle/Paddle/pull/60008)

2.在编译阶段出现unresolved external symbol TouchOpRegistrar_xxxx(void)问题，例如：
[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9750347/job/24789126)
解决办法：在相关文件里删除USE_OP_ITSELF(xxx)和 PD_DECLARE_KERNEL(xxx, OneDNN,ONEDNN),例如：
[示例PR](https://github.com/PaddlePaddle/Paddle/pull/60008)

3.在编译阶段出现ld的链接错误问题，例如：
[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9737328/job/24768552)
解决办法：在相关文件里添加REGISRE_FILE_SYMBOL(xxx), 而后在paddle/fluid/pybind/pybind.cc中添加DECLARE_FILE_SYMBOLS(xxx)
[示例PR](https://github.com/PaddlePaddle/Paddle/pull/59988)

4.在 编译阶段出现unresolved external symbol，例如：
   （1)[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9669394/job/24660474)
    解决办法：在cmake/generic.cmake中的cc_test_build
    [示例PR](https://github.com/PaddlePaddle/Paddle/pull/59477)

    (2) 在相应的方法或类前添加TEST_API
    [示例PR](https://github.com/PaddlePaddle/Paddle/pull/59477)



