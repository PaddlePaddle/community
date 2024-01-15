# paddle_test推全
> 
> Tracking issue: [PaddlePaddle/Paddle#60793](https://github.com/PaddlePaddle/Paddle/issues/60793)

## 目的
现阶段 Paddle 编译 c++ 单测有多种方式，如cc_test（windows 下 link 静态库，linux 下 link paddle.so），paddle_test ( windows 和 linux 下都 link 动态库paddle.so)，现在推荐使用 paddle_test，好处在于：
1. 使得编译单测时链接动态库，以减小编译单测过程中的产物体积。 
2. 提升至少两部分编译过程的速度，一是编译行为本身的速度，二是在CI上打包/下载编译产物的速度。

## 方案设计
以一个 CMakeLists.txt 文件为一个任务单元，共 43 个 CMakeLists.txt 文件包含至少一处 cc_test。
将其中一个 CMakeLists.txt 文件的 cc_test 替换为 paddle_test 即为完成一个任务。

## 注意事项
### 1. 在 test阶段出现 Exit code 0xc000007b 问题
   例如：[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9766842/job/24813466)
   <img width="503" alt="d07eaf4a7050bded70ae9e4b50650e63" src="https://github.com/PaddlePaddle/community/assets/55453380/97dc9384-587f-4188-8f36-21ae080d90f0">

   解决办法：在与之对应的 CMakeLists.txt 中添加 copy_onnx(xxx_test)，例如：[示例PR](https://github.com/PaddlePaddle/Paddle/pull/60008)
   <img width="570" alt="27f13a90665a3331d974fef600127089" src="https://github.com/PaddlePaddle/community/assets/55453380/a1ec8ae3-f755-488e-824d-a44c6bcc3853">


### 2. 在编译阶段出现unresolved external symbol TouchOpRegistrar_xxxx(void)问题
   例如：[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9750347/job/24789126)  
   <img width="770" alt="0cf89308ee7710c5ca5b4ade77824ec7" src="https://github.com/PaddlePaddle/community/assets/55453380/7bca56a4-915f-4dac-ab21-3c977e2b7e0f">

   解决办法：在相关文件里删除 USE_OP_ITSELF(xxx) 和 PD_DECLARE_KERNEL(xxx, OneDNN,ONEDNN)，例如：[示例PR](https://github.com/PaddlePaddle/Paddle/pull/60008)
   <img width="382" alt="75974c5a15c95320476a4c100296e83a" src="https://github.com/PaddlePaddle/community/assets/55453380/61a8b85b-9c82-413c-be78-22adc6b26c27">


### 3. 在编译阶段出现ld的链接错误问题
   例如：[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9737328/job/24768552)   
   <img width="783" alt="c3fcc220099ce87893c9ad2756969f8d" src="https://github.com/PaddlePaddle/community/assets/55453380/1afcc172-5903-4985-8f1e-1db874937574">

   解决办法：在相关文件里添加 REGISRE_FILE_SYMBOL(xxx), 而后在 paddle/fluid/pybind/pybind.cc 中添加 DECLARE_FILE_SYMBOLS(xxx)，[示例PR](https://github.com/PaddlePaddle/Paddle/pull/59988)   
   <img width="344" alt="f38841896eba5d82f40619d07bda6e69" src="https://github.com/PaddlePaddle/community/assets/55453380/49494c5a-f320-4716-88de-042c48ca471a">
   <img width="351" alt="b6665a0ce8d153252d016aeea9401606" src="https://github.com/PaddlePaddle/community/assets/55453380/4c13d717-c3fa-4d7c-80e4-d5177f5eb12d">

### 4. 在编译阶段出现unresolved external symbol  
（1) 例如：[示例CI日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/9669394/job/24660474)    
   <img width="779" alt="3c43f2e8b7427464ad93f241f2cad20c" src="https://github.com/PaddlePaddle/community/assets/55453380/81c4010d-84f1-4775-ac0c-0f0e709bbf9e">

   解决办法：在 cmake/generic.cmake 中的 cc_test_build，[示例PR](https://github.com/PaddlePaddle/Paddle/pull/59477)     
   ![9be606822f00744959fe6265fc2b41d4](https://github.com/PaddlePaddle/community/assets/55453380/5e89477b-7b92-4279-b8b4-6dd55a9f900a)


(2)例如：
   解决办法：在相应的方法或类前添加 TEST_API，[示例PR](https://github.com/PaddlePaddle/Paddle/pull/59477)
   ![ffc34f67e5271cf6ccc6de3688508250](https://github.com/PaddlePaddle/community/assets/55453380/a210466b-e40f-4dd6-b03d-aacde7d1d6c9)




