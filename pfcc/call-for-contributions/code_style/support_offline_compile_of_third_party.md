# 支持Paddle第三方库离线编译

> This project will be mentored by [@risemeup1](https://github.com/risemeup1) and [@luotao1](https://github.com/luotao1)
> 
> Tracking issue: [PaddlePaddle/Paddle#54305](https://github.com/PaddlePaddle/Paddle/issues/54305)
## 目的
现在Paddle编译第三方库通过ExternalProject_Add命令对第三方库下载，当编译到某个第三方库的时候才会下载第三方库，然后编译，这种方式会存在很多问题,具体如下：
1 一边编译一边下载，当编译到某个第三方库的时候开始下载，导致git clone的频率增加，如果网络或者代理不稳定，任何一次git clone失败就会导致编译出问题，本地编译和CI均太过于依赖代理；
2 外部用户需要翻墙才能访问github，经常会因为git clone而下载失败；
3 研发RD删除build目录后重新编译就需要重新下载这些第三方库，又要重新git clone第三方库，没有达到复用的效果，编译时间会增加很多，也会因为网络和代理问题影响研发效率。

## 方案设计
1 通过git submodule的方式将paddle依赖的所有第三方库放在根目录Paddle/third_party下,只需“git clone --recusrsive https://github.com/PaddlePaddle/Paddle.git”可以将Paddle的第三方库全部下载下来，后续在编译阶段将不再下载第三方库，直接编译。
2 若用户“git clone  https://github.com/PaddlePaddle/Paddle.git”，没有加--recusrsive则Paddle/third_party/目录存在但是为空，在setup.py和third_party.cmake会自动执行“git submodule update --init”，在cmake阶段也会把第三方库下载下来，在make阶段不需要网。
## 方案规划
1. 将不打开任何编译选项，即cmake ..需要下载的第三方库(zlib gflags glog eigen threadpool dlpack xxhash warpctc warprnnt utf8proc lapack protobuf gloo crypotopp pybind11 pocketfft xbyak)通过git submodule作为Paddle的子模块，编译的时候不再需要下载第三方库；
2. 将常用编译选项下(如WITH_GPU，WITH_DISTRIBUTE等)且比较小的第三方库作为Paddle的子模块；
3. 其它的第三方库根据不同的编译选项进行判断，在cmake时下载，在make -j时准备好所有的第三方库。

## 注意事项
1 注意所用第三方库的tag，有的第三方库在不同平台下使用的tag不同，如protobuf，需要进行判断；
2 需要了解增加删除修改submodule的常用命令以及cmake ExternalProject_Add各个参数的用法；
3 注意第三方库位置变换之后，include头文件的路径也需要变化；
4 注意不要在源码引入任何编译产物，第三方库的编译产物放在build目录下。参考PR：[https://github.com/PaddlePaddle/Paddle/pull/53744]

## 第三方库统计
可以作为submodule的第三方库，主要是放在github或者gitlab上的repo
### 可以作为submodule的第三方库，主要是放在github或者gitlab上的repo
brpc cinn cryptopp cub cudnn-frontend cutlass dirent dlpack eigen flashattn gflags glog gloo gtest leveldb libxsmm lite
mkldnn openblas pocketfft protobuf pybind11 rocksdb snappy threadpool utf8proc warpctc warpnnt xbyak xxhash zlib

### 不能作为Paddle submodule的第三方库(主要是一些二进制的文件)
arm_brpc box_ps concurrentqueue cusparselt dgc jemalloc lapack libmct mklml onnxruntime pslib_brpc pslib xpu paddle2onnx

### 编译必须用到(默认打开)的第三方库17个
zlib gflags glog eigen threadpool dlpack xxhash warpctc warprnnt utf8proc lapack protobuf gloo crypotopp pybind11 pocketfft xbyak

### 不同编译选项对应的第三方库
|依赖库名称|编译选项|paddle使用的安装包、repo及tag/commit|
|-------|----|-------|
|1|arm_brpc|WITH_ARM_BRPC|https://paddlerec.bj.bcebos.com/online_infer/arm_brpc_ubuntu18/output.tar.gz 1.1.0版本|
|2|box_ps|WITH_BOX_PS|http://box-ps.gz.bcebos.com/box_ps.tar.gz 0.1.1版本|
|3|brpc|WITH_PSLIB|repo: https://github.com/wangjiawei04/brpc tag: e203afb794caf027da0f1e0776443e7d20c0c28e|
|4|cinn|WITH_CINN|repo: PaddlePaddle/CINN.git tag: release/v0.2|
|5|cryptopp|WITH_CRYPTO(默认打开)|repo: weidai11/cryptopp.git tag: CRYPTOPP_8_2_0|
|6|cub|WITH_GPU（对cuda版本有要求）|repo：NVlabs/cub.git tag：1.16.0（win32）、1.18.0（others|
|7|cusparselt|WITH_CUSPARSELT|https://developer.download.nvidia.com/compute/libcusparse-lt/0.2.0/local_installers/libcusparse_lt-linux-x86_64-0.2.0.1.tar.gz|
|8|cutlass|WITH_GPU|https://github.com/NVIDIA/cutlass.git tag:v2.11.0|
||||
