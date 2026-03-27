FastDeploy编译打卡文档

## 🚀 FastDeploy 热身打卡活动
—— 从源码编译开始，解锁高性能推理框架开发之路

各位飞桨开发者大家好！
为了帮助更多小伙伴快速进入 **FastDeploy 二次开发生态**，熟悉大型框架的工程结构与编译流程，飞桨社区特推出本次 **FastDeploy 热身打卡活动**。
通过亲手完成一次完整的 FastDeploy 编译与打包流程，你将正式具备参与 FastDeploy 套件开发的基础能力。

参与热身打卡活动并按照邮件模板格式将截图发送至[ext_paddle_oss@baidu.com](mailto:ext_paddle_oss@baidu.com) ，还可获得社区认可与后续任务推荐资格。

## 🧩 活动目标
通过本次打卡，你将掌握：

FastDeploy 源码结构

Paddle 运行时与 FastDeploy 的依赖关系

自定义算子编译机制

wheel 构建与分发流程

二次编译优化与开发调试效率提升方法

注：本次热身打卡活动需要使用GPU A800硬件，赶快行动起来吧~也可 [申请AI Studio开发资源](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/README.md#%E9%A3%9E%E6%A1%A8%E7%BA%BF%E4%B8%8A%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83ai-studio)～

## 🧰 准备环境
### 1. 安装 PaddlePaddle
以 GPU 版本为例（CPU 同理）：

python -m pip install paddlepaddle-gpu==3.3.0 -i [https://www.paddlepaddle.org.cn/packages/stable/cu126/](https://www.paddlepaddle.org.cn/packages/stable/cu126/)

### 2. 克隆 FastDeploy 源码
git clone [https://github.com/PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
cd FastDeploy

## 🧪 编译打卡流程
> 所有关键步骤需加 time 记录耗时，并截图保存。
### Step 1：执行 FastDeploy 编译与打包
```
# 参数说明
# 第1个参数: 是否构建 wheel（1=构建，0=仅编译）
# 第2个参数: Python 解释器
# 第3个参数: 是否编译 CPU BF16 算子
# 第4个参数: GPU 架构（如 [80,90]）

time MAX_JOBS=24 bash build.sh 1 python false [80]
```
编译完成后，产物位于：

```
FastDeploy/dist/
```
### Step 2：初次编译/二次编译
初次编译时间较长，二次编译因为有编译缓存的存在，时间会缩短，对日常开发来说，二次编译时间才是影响开发效率的。让我们来感受下修改不同文件的二次编译时间。

* 修改kernel_traits的头文件：custom_ops/gpu_ops/flash_mask_attn/kernel_traits.h
* 修改transfer_output的cc文件：custom_ops/gpu_ops/transfer_output.cc
* 修改python文件：FastDeploy/custom_ops/setup_ops.py

二次编译方式：对应文件加一个空行/空格保存退出后，然后执行编译命令`time MAX_JOBS=24 bash build.sh 0 python false [80]`，二次编译不再需要执行cmake。

### Step 3：安装whl包
```
pip install FastDeploy/dist/xxx.whl
```
### Step 4：运行单元测试
python tests/layers/test_ffn.py

## 邮件格式
标题： [Hackathon-FastDeploy 热身打卡]

内容：

飞桨团队你好，

【GitHub ID】：XXX

【打卡内容】：初次编译/二次编译/安装whl包/运行单元测试

【打卡截图】：

如：

标题： [Hackathon-FastDeploy 热身打卡]

内容：

飞桨团队你好，

【GitHub ID】：XXX（例如：paddle-hack）

【打卡内容】：初次编译&安装whl包&运行单元测试

【打卡截图】：

|硬件|![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=c95e9b14881d4f5aa517e67044e9d1ca&docGuid=fJF6XajnDJ17aV)|
|-|-|
|编译方式|参考【编译】文档（[源码编译文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/nvidia_gpu.md#4-wheel包源码编译)）|
|初次编译命令和时间|命令：`time MAX_JOBS=24 bash build.sh 1 python false [80]` （写一下大家用几核哦）时间：以下时间仅作为示例，不代表真实的初次编译时间![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=fb2792afb085499bb0d7f32767469a62&docGuid=fJF6XajnDJ17aV)|
|二次编译时间|时间：以下时间仅作为示例，不代表真实的二次编译时间custom_ops/gpu_ops/flash_mask_attn/kernel_traits.hcustom_ops/gpu_ops/transfer_output.ccFastDeploy/custom_ops/setup_ops.py![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=63847c6842844c25a116068a6b7dcc80&docGuid=fJF6XajnDJ17aV)|
|安装whl包|编译完成后，产物位于 FastDeploy/dist/xxx.whl![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=de124a4dee7b43bf90810c7403cbc619&docGuid=fJF6XajnDJ17aV)![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0ea6c2ad992648d88b4cf0038e3e8018&docGuid=fJF6XajnDJ17aV)|
|运行单元测试|以 tests/layers/test_ffn.py 为例，由于aistudio环境的影响因素，可以把 quant_config = BlockWiseFP8Config(weight_block_size=[128, 128]) 移动至 self.fd_config 外。（不影响模型结构测试）`python tests/layers/test_ffn.py`![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=7a03c11826154552b815a60bf1ad9edc&docGuid=fJF6XajnDJ17aV)![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3236b91403334856ba508ba68e62ecd2&docGuid=fJF6XajnDJ17aV)|

## 优秀作品
* 期待你的跑通 “踩坑” 文档能成为大家的教程

## 参与飞桨框架贡献
如果你已经顺利完成了打卡，具备了基础的框架开发知识，你就可以参与飞桨社区丰富的开发任务，为一个大型开源项目做贡献，同时收获飞桨社区开发者的认可与各种福利。传送门：

- [ ] [https://github.com/orgs/PaddlePaddle/projects/7](https://github.com/orgs/PaddlePaddle/projects/7)






## 一、Fastdeploy编译
#### 1、A800环境

需要额外注意的是：

（1）编译过程中如果存在如下ld报错：

```python
/usr/bin/ld: cannot find -lnvidia-ml: No such file or directory collect2: error: ld returned 1 exit status error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1 [FAIL] 
```
那就需要设置软链避免报错

```python
sudo ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
```
（2）如果存在编译内存不够，需要根据以下命令查询实际可以使用的最大并发进程数

```python
# 查看机器配置
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us#查看容器实际可用核心数
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us#查看容器实际可用核心数 
# 计算公式为：cfs_quota_us/cfs_period_us
```
从而修改编译命令里面的MAX_JOBS：`time MAX_JOBS=24 bash build.sh 1 python false [80]`

（3）参考上面的文档里面跑通单测时需要根据文档中内容修改代码，因为 **FP8 量化需要再SM89+ (Ada Lovelace)，但是A800和V100都是89以下所以不支持**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f0eaf92338ac44c581dfa9b4c29b35ad&docGuid=wzzQACr1jCJLOR)
#### 2、天数环境
整体先参考这个文档：[https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/iluvatar_gpu.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/iluvatar_gpu.md)

在这个文档基础上，需要额外注意的是：

（1）设置环境变量，便于正常编译

```python
export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/corex/lib
```
（2）单卡编译时由于能够支持的线程数不高，因此需要自行设置暴露MAX_JOBS

修改脚本\FastDeploy/build.sh里面这部分新增下面这几行：

```python
# 控制编译并行度
export MAX_JOBS=${MAX_JOBS:-2}
echo "Compile parallel jobs: $MAX_JOBS"
```
修改代码后如下图：

<img width="960" height="573" alt="image" src="https://github.com/user-attachments/assets/7c3a567b-a5ff-4d6e-856a-137654295c99" />

然后再使用编译命令：MAX_JOBS=1 bash build.sh

## 二、Paddle编译
#### 1、A800环境
步骤参考：[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html)

#### 2、天数环境
步骤参考：[https://github.com/PaddlePaddle/Paddle-iluvatar/blob/develop/README_cn.md](https://github.com/PaddlePaddle/Paddle-iluvatar/blob/develop/README_cn.md)

详细测试步骤文档（两种环境）：[框架二次开发paddle编译文档](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/cyXSCTILGl/YNq-6H4qptZfgJ?t=mention&mt=doc&dt=doc)
