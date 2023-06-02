# 支持Paddle第三方库离线编译

> This project will be mentored by [@risemeup1](https://github.com/risemeup1) and [@luotao1](https://github.com/luotao1)
> 
> Tracking issue: [PaddlePaddle/Paddle#54305](https://github.com/PaddlePaddle/Paddle/issues/54305)
## 目的
现在Paddle编译第三方库通过ExternalProject_Add命令对第三方库下载，当编译到某个第三方库的时候才会下载第三方库，然后编译，这种方式会存在很多问题,具体如下：
1. 一边编译一边下载，当编译到某个第三方库的时候开始下载，导致git clone的频率增加，如果网络不稳定，任何一次git clone失败就会导致编译出问题，本地编译和CI均太过于依赖网络；
2. 研发RD删除build目录后重新编译就需要重新下载这些第三方库，又要重新git clone第三方库，没有达到复用的效果，编译时间会增加很多，也会因为网络问题影响研发效率。

## 方案设计
1. 通过git submodule的方式将paddle依赖的所有第三方库放在根目录Paddle/third_party下,只需git clone --recusrsive 可以将Paddle的第三方库全部下载下来，后续在编译阶段将不再下载第三方库，直接编译。
2. 若用户git clone  xxx，没有加--recusrsive则Paddle/third_party/目录存在但是为空，在setup.py和third_party.cmake会自动执行git submodule update --init，在cmake阶段也会把第三方库下载下来，在make阶段不需要网。
## 方案规划
1. 将不打开任何编译选项，即cmake ..需要下载的第三方库(zlib gflags glog eigen threadpool dlpack xxhash warpctc warprnnt utf8proc lapack protobuf gloo crypotopp pybind11 pocketfft xbyak)通过git submodule作为Paddle的子模块，编译的时候不再需要下载第三方库；
2. 将常用编译选项下(如WITH_GPU，WITH_DISTRIBUTE等)且比较小的第三方库作为Paddle的子模块；
3. 其它的第三方库根据不同的编译选项进行判断，在cmake时下载，在make -j时准备好所有的第三方库。

## 注意事项
1. 注意所用第三方库的tag，有的第三方库在不同平台下使用的tag不同，如protobuf，需要进行判断；
2. 需要了解增加删除修改submodule的常用命令以及cmake ExternalProject_Add各个参数的用法；
3. 注意第三方库位置变换之后，include头文件的路径也需要变化；
4. 注意不要在源码引入任何编译产物，第三方库的编译产物放在build目录下。参考PR：[https://github.com/PaddlePaddle/Paddle/pull/53744]

## 第三方库统计
### 可以作为submodule的第三方库，主要是放在github或者gitlab上的repo
brpc cinn cryptopp cub cudnn-frontend cutlass dirent dlpack eigen flashattn gflags glog gloo gtest leveldb libxsmm lite
mkldnn openblas pocketfft protobuf pybind11 rocksdb snappy threadpool utf8proc warpctc warpnnt xbyak xxhash zlib

### 不能作为Paddle submodule的第三方库(主要是一些二进制的文件)
arm_brpc box_ps concurrentqueue cusparselt dgc jemalloc lapack libmct mklml onnxruntime pslib_brpc pslib xpu paddle2onnx

### 编译必须用到(默认打开)的第三方库17个
zlib gflags glog eigen threadpool dlpack xxhash warpctc warprnnt utf8proc lapack protobuf gloo crypotopp pybind11 pocketfft xbyak

### 不同编译选项对应的第三方库
|序号|依赖库名称|编译选项|paddle使用的安装包、repo及tag/commit|
|-------|----|-------|-------|
|1|arm_brpc|WITH_ARM_BRPC|https://paddlerec.bj.bcebos.com/online_infer/arm_brpc_ubuntu18/output.tar.gz 1.1.0版本|
|2|box_ps|WITH_BOX_PS|http://box-ps.gz.bcebos.com/box_ps.tar.gz 0.1.1版本|
|3|brpc|WITH_PSLIB|repo: https://github.com/wangjiawei04/brpc tag: e203afb794caf027da0f1e0776443e7d20c0c28e|
|4|cinn|WITH_CINN|repo: PaddlePaddle/CINN.git tag: release/v0.2|
|5|cryptopp|WITH_CRYPTO(默认打开)|repo: weidai11/cryptopp.git tag: CRYPTOPP_8_2_0|
|6|cub|WITH_GPU（对cuda版本有要求）|repo：NVlabs/cub.git tag：1.16.0（win32）、1.18.0（others|
|7|cusparselt|WITH_CUSPARSELT|https://developer.download.nvidia.com/compute/libcusparse-lt/0.2.0/local_installers/libcusparse_lt-linux-x86_64-0.2.0.1.tar.gz|
|8|cutlass|WITH_GPU|https://github.com/NVIDIA/cutlass.git tag:v2.11.0|
|9|dgc|WITH_DGC(受WITH_DISTRIBUTE控制)|https://fleet.bj.bcebos.com/dgc/collective_f66ef73.tgz|
|10|dirent|WIN32|repo：tronkko/dirent tag：1.23.2|
|11|dlpack|默认打开|repo：dmlc/dlpack.git tag：v0.4|
|12|eigen|默认打开|repo：https://gitlab.com/libeigen/eigen.git tag：f612df273689a19d25b45ca4f8269463207c4fee|
|13|flashattn|WITH_GPU|${GIT_URL}/PaddlePaddle/flash-attention.git tag:18106c1ba0ccee81b97ca947397c08a141815a47|
|14|gflags|默认打开|repo: gflags/gflags.git tag: v2.2.2|
|15|glog|默认打开|repo：google/glog.git tag：v0.4.0|
|16|gloo|NOT WIN32 AND NOT APPLE|repo：sandyhouse/gloo.git tag：v0.0.2|
|17|gtest|WITH_TESTING、WITH_DISTRIBUTE|repo：google/googletest.git tag：release-1.8.1|
|18|lapack|默认打开|repo：https://paddlepaddledeps.bj.bcebos.com/lapack_lnx_v3.10.0.20210628.tar.gz|
|19|leveldb|WITH_PSLIB|repo：https://github.com/google/leveldb tag：v1.18|
|20|libmct|WITH_PSLIB、WITH_LIBMCT|https://pslib.bj.bcebos.com/libmct/libmct.tar.gz 0.1.0版本|
|21|libxsmm|WITH_PSLIB|repo：hfp/libxsmm.git tag：7cc03b5b342fdbc6b6d990b190671c5dbb8489a2|
|22|lite|WITH_LITE|repo：PaddlePaddle/Paddle-Lite.git tag：81ef66554099800c143a0feff6e0a491b3b0d12e|
|23|mkldnn|WITH_MKLDNN、WITH_MKL、AVX2_FOUND|repo：oneapi-src/oneDNN.git tag：9b186765dded79066e0cd9c17eb70b680b76fb8e|
|24|mklml|WITH_MKLML|https://paddlepaddledeps.bj.bcebos.com/mklml_win_2019.0.5.20190502.zip|
|25|onnxruntime|WITH_ONNXRUNTIME|https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip|
|26|openblas|/|repo：xianyi/OpenBLAS.git tag：v0.3.7|
|27|paddle2onnx|WITH_ONNXRUNTIME|https://github.com/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-win-x64-${PADDLE2ONNX_VERSION}.zip|
|28|pocketfft|WITH_POCKETFFT（默认打开|repo：https://gitlab.mpcdf.mpg.de/mtr/pocketfft.git tag：release_for_eigen|
|29|poplar|WITH_IPU|/|
|30|protobuf|默认打开|repo：protocolbuffers/protobuf.git tag：9f75c5aa851cd877fb0d93ccc31b8567a6706546|
|31|pslib_brpc|WITH_PSLIB、WITH_PSLIB_BRPC|https://pslib.bj.bcebos.com/pslib_brpc.tar.gz 0.1.0版本|
|32|pslib|WITH_PSLIB|https://pslib.bj.bcebos.com/pslib_brpc.tar.gz 0.1.0版本|
|33|pybind11|WITH_PYTHON(默认打开)|repo: pybind/pybind11.git|tag: v2.4.3|
|34|python|WITH_PYTHON（默认打开）|/|
|35|rocksdb|WITH_PSCORE|repo: https://github.com/facebook/rocksdb tag: v6.10.1|
|36|snappy|WITH_PSLIB，WITH_DISTRIBUTE|repo：https://github.com/google/snappy tag：1.1.7|
|37|threadpool|默认打开|repo: progschj/ThreadPool.git tag: 9a42ec1e329f259a5f4881a291db1dcb8f2ad9040｜
|38|utf8proc|默认打开|repo: JuliaStrings/utf8proc.git tag: v2.6.1|
|39|warpctc|默认打开|repo：baidu-research/warp-ctc.git tag：37ece0e1bbe8a0019a63ac7e6462c36591c66a5b|
|40|xbyak|WITH_XBYAK(默认为ON)|repo：herumi/xbyak.git tag：v5.81|
|41|xxhash|默认|repo：Cyan4973/xxHash.git tag：v0.6.5|
|42|zlib|默认打开|repo: madler/zlib.git tag: v1.2.8|






