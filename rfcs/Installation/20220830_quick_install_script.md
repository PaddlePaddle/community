# 快速安装脚本功能补充文档

|   |  |
| --- | --- |
|提交作者 | gouzi | 
|提交时间 | 2022-08-30 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | 2.2.2 | 
|文件名 | 20220830_quick_install_script.md | 


# 一、概述
## 1、相关背景

提升安装体验, 降低新手用户入门门槛, 快速安装部署新环境。

## 2、功能目标

在原有[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)的基础上, 实现安装前 CUDA 、 cuDNN 和 docker 环境配置。

更新现有env收集[脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/summary_env.py)

## 3、意义
解决安装PaddlePaddle前的环境配置。

# 二、飞桨现状

飞桨目前支持[环境检测](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html#sanyanzhenganzhuang)和[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)。但是并不会帮助用户安装前期的环境配置。

示例:
```bash
# 安装cuda
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

sudo sh cuda_10.2.89_440.33.01_linux.run

# 安装cudnn
# 在官网注册下载cudnn-10.2-linux-x64-v7.6.5.32.tgz 
tar -zxvf cudnn-10.2-linux-x64-v7.6.5.32.tgz 
sudo cp cuda/lib64/* /usr/local/cuda-10.2/lib64/
sudo cp cuda/include/* /usr/local/cuda-10.2/include/

# 安装paddle
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# 确认安装
~$ python3
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import paddle
>>> paddle.utils.run_check()
Running verify PaddlePaddle program ... 
W0902 16:26:50.091753   111 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0902 16:26:50.094720   111 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
PaddlePaddle works well on 1 GPU.
PaddlePaddle works well on 1 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

```bash
# 硬件检测
~/Paddle-develop$ python3 tools/summary_env.py 
tools/summary_env.py:51: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  plat = distro.linux_distribution()[0]
tools/summary_env.py:52: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  ver = distro.linux_distribution()[1]
****************************************
Paddle version: 2.3.2
Paddle With CUDA: True

OS: Ubuntu 16.04
Python version: 3.7.4

CUDA version: 11.2.152
Build cuda_11.2.r11.2/compiler.29618528_0
cuDNN version: None.None.None
Nvidia driver version: 460.32.03
****************************************
```
# 三、业内方案调研
## 环境安装
### TensorFlow
通过发布 [arch包](https://archlinux.org/packages/community/x86_64/python-tensorflow-opt-cuda/) 实现 (仅支持 arch linux)。

示例: 
```bash
sudo pacman -S python-tensorflow-opt-cuda
...
```

通过 [conda](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/#cuda-versions) 实现自动安装 CUDA 和 cuDNN (支持windowns 和 linux)

示例:
```bash
conda create -n tf-gpu-cuda8 tensorflow-gpu cudatoolkit=9.0
conda activate tf-gpu-cuda8
```

### PyTorch
社区开发者通过发布 [arch包](https://aur.archlinux.org/packages/python-pytorch-opt-rocm?all_deps=1#pkgdeps) 实现 (仅支持 arch linux)

示例:
```bash
pacman -S python-pytorch-opt-rocm
...
```

社区开发者自行编写安装脚本 [csdn Ubuntu16.04 深度学习一键安装脚本](https://blog.csdn.net/hanlin_tan/article/details/77540128)

安装版本过老不做测试

## 环境检测

### PyTorch

[collect_env.py](https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py)
```bash
~/pytorch-master$ python3 torch/utils/collect_env.py 
Collecting environment information...
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 16.04.6 LTS (x86_64)
GCC version: (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
Clang version: Could not collect
CMake version: version 3.12.2
Libc version: glibc-2.10

Python version: 3.7.4 (default, Aug 13 2019, 20:35:49)  [GCC 7.3.0] (64-bit runtime)
Python platform: Linux-4.15.0-158-generic-x86_64-with-debian-stretch-sid
Is CUDA available: N/A
CUDA runtime version: 11.2.152
GPU models and configuration: GPU 0: A100-SXM4-40GB
Nvidia driver version: 460.32.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.2.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.2.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.2.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.2.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.2.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.2.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.2.0
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.5
[conda] numpy                     1.16.2                   pypi_0    pypi
```

### TensorFlow

TensorFlow采用的是[shell脚本](https://github.com/tensorflow/tensorflow/blob/master/tools/tf_env_collect.sh)读取环境
```bash
python build version: ('default', 'Aug 13 2019 20:35:49')
python compiler version: GCC 7.3.0
python implementation: CPython


== check os platform ===============================================
os: Linux
os kernel version: #166-Ubuntu SMP Fri Sep 17 19:37:52 UTC 2021
os release version: 4.15.0-158-generic
os platform: Linux-4.15.0-158-generic-x86_64-with-debian-stretch-sid
linux distribution: ('debian', 'stretch/sid', '')
linux os distribution: ('debian', 'stretch/sid', '')
mac version: ('', ('', '', ''), '')
uname: uname_result(system='Linux', node='jupyter-885527-4460262', release='4.15.0-158-generic', version='#166-Ubuntu SMP Fri Sep 17 19:37:52 UTC 2021', machine='x86_64', processor='x86_64')
architecture: ('64bit', '')
machine: x86_64


== are we in docker =============================================
Yes

== compiler =====================================================
c++ (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


== check pips ===================================================
numpy                                    1.20.3
protobuf                                 3.19.4
tensorflow                               2.9.1
tensorflow-estimator                     2.9.0
tensorflow-io-gcs-filesystem             0.26.0

== check for virtualenv =========================================
False

== tensorflow import ============================================
tf.version.VERSION = 2.9.1
tf.version.GIT_VERSION = v2.9.0-18-gd8ce9f9c301
tf.version.COMPILER_VERSION = 9.3.1 20200408
       984:	find library=libpthread.so.0 [0]; searching
       984:	 search path=/opt/conda/envs/python35-paddle120-env/bin/../lib/tls/x86_64:/opt/conda/envs/python35-paddle120-env/bin/../lib/tls:/opt/conda/envs/python35-paddle120-env/bin/../lib/x86_64:/opt/conda/envs/python35-paddle120-env/bin/../lib		(RPATH from file /opt/conda/envs/python35-paddle120-env/bin/python)
       984:	  trying file=/opt/conda/envs/python35-paddle120-env/bin/../lib/tls/x86_64/libpthread.so.0

..............

== env ==========================================================
LD_LIBRARY_PATH /usr/local/cuda-11.2/targets/x86_64-linux/lib/:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/targets/x86_64-linux/:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
DYLD_LIBRARY_PATH is unset

== nvidia-smi ===================================================
Fri Sep  2 17:25:48 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  A100-SXM4-40GB      On   | 00000000:67:00.0 Off |                    0 |
| N/A   35C    P0    53W / 400W |      3MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

== cuda libs  ===================================================
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudart_static.a
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudart.so.11.2.152

== tensorflow installed from info ==================
Name: tensorflow
Version: 2.9.1
Summary: TensorFlow is an open source machine learning framework for everyone.
Home-page: https://www.tensorflow.org/
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: /home/gouzi/.data/webide/pip/lib/python3.7/site-packages
Required-by: 

== python version  ==============================================
(major, minor, micro, releaselevel, serial)
(3, 7, 4, 'final', 0)

== bazel version  ===============================================

```

# 四、对比分析

通过 arch包 方式相对简单方便, 但支持的操作系统和环境并不多。

通过 conda 方式安装, 方便快捷, 但conda安装速度较慢且容易踩坑。

通过社区开发者自行编写脚本安装, 可控性较高可操作空间大, 但更新不及时。

# 五、设计思路与实现方案

## 流程

1. 判断用户环境, 如: windows、linux; x86_64、arm64; 是否有GPU
2. 用户选择安装方式, 如: docker、pip 安装
3. 用户选择安装CPU版本或GPU版本 (无GPU不提示)
4. CUDA，cuDNN 检测
5. 用户选择 CUDA 和 cuDNN 版本 (无GPU不提示)
6. 用户选择加速源 (可选的)
7. 安装 CUDA 和 cuDNN 驱动 (无GPU不安装)
8. 安装PaddlePaddle
9. 运行[环境检测](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html#sanyanzhenganzhuang)

## 实现方案

 - 通过 shell 脚本实现
 - 参考[pytorch](https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py)添加更多环境信息

# 六、测试和验收的考量

在不同的硬件环境下运行脚本, 并且环境检测通过

# 七、可行性分析和排期规划

对环境搭建和shell命令有一定了解, 需要多设备进行测试。

# 八、影响面

暂无

# 名词解释

 - arch: [Arch Linux](https://archlinux.org/)AUR用户[软件仓库](https://aur.archlinux.org/packages)

# 附件及参考资料

暂无
