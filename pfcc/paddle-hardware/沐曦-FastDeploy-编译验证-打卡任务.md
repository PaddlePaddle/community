# 【热身打卡】Metax GPU+ PaddleOCR-VL-1.5 + FastDeploy 编译打卡

## 一、Metax GPU + FastDeploy 编译打卡

—— 从源码编译开始，解锁国产GPU高性能推理框架开发之路

**各位飞桨开发者大家好！**

为了帮助更多小伙伴快速进入Metax GPU FastDeploy 二次开发生态，熟悉大型框架的工程结构与编译流程，飞桨社区特推出本次 **FastDeploy 热身打卡活动**。
通过亲手完成一次完整的 FastDeploy 编译与打包流程，你将正式具备参与 FastDeploy 套件开发的基础能力。

## 二、活动目标

通过本次打卡，你将掌握：

- **FastDeploy 源码结构**
- **Paddle 运行时与 FastDeploy 的依赖关系**
- **自定义算子编译机制**
- **wheel 构建与分发流程**
- **二次编译优化与开发调试效率提升方法**
- **Metax GPU backend与paddle 框架的关系**
- **基于Metax GPU 运行FastDeploy 推理框架**

注：本次热身打卡活动需要使用 Metax GPU 硬件**，赶快行动起来吧。

## 三、准备环境

以MetaxGPU 版本为例：

#### 容器镜像获取

```
https://ai.gitee.com/compute/instances/new?id=3
Pytorch/2.6.0/Python 3.10/maca 3.2.1.3
```

#### paddle & custom backend 预安装

```
1）pip install paddlepaddle==3.4.0.dev20251223 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
2）pip install paddle-metax-gpu==3.3.0.dev20251224 -i https://www.paddlepaddle.org.cn/packages/nightly/maca/
```

#### FastDeploy代码下载并编译

```
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
```

## 四、编译打卡流程

1）熟悉并了解编译脚本，编译参数配置，完成fastdeploy编译，编译产物位于~/fastdeploy/dist；

2）完成fastdeploy 编译产物wheel包安装，了解安装路径；尝试直接修改python文件，重新运行推理程序；



