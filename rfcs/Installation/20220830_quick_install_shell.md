# 快速安装脚本设计文档

|   |  |
| --- | --- |
|提交作者 | gouzi | 
|提交时间 | 2022-08-30 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | 2.2.2 | 
|文件名 | 20220830_quick_install_shell.md | 


# 一、概述
## 1、相关背景

提升安装体验, 降低新手用户入门门槛, 快速安装部署新环境。

## 2、功能目标

在原有[pip 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html)的基础上, 实现安装前 CUDA 和 cuDNN 或 docker 环境配置

## 3、意义
解决安装PaddlePaddle前的环境配置。

# 二、飞桨现状

飞桨目前仅支持[环境检测](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html#sanyanzhenganzhuang)。

# 三、业内方案调研

TensorFlow 通过发布 [arch包](https://archlinux.org/packages/community/x86_64/python-tensorflow-opt-cuda/) 实现 (仅支持 arch linux), 通过 [conda](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/#cuda-versions) 实现自动安装 CUDA 和 cuDNN (支持windowns 和 linux)

PyTorch 社区开发者通过发布 [arch包](https://aur.archlinux.org/packages/python-pytorch-opt-rocm?all_deps=1#pkgdeps) 实现 (仅支持 arch linux)

社区开发者自行编写安装脚本 [csdn Ubuntu16.04 深度学习一键安装脚本](https://blog.csdn.net/hanlin_tan/article/details/77540128)

# 四、对比分析

通过 arch包 方式相对简单方便, 但支持的操作系统和环境并不多。

通过 conda 方式安装, 方便快捷, 但conda安装速度较慢且容易踩坑。

通过社区开发者自行编写脚本安装, 可控性较高可操作空间大, 但更新不及时。

# 五、设计思路与实现方案

## 流程

1. 判断用户环境, 如: windows、linux; x86_64、arm64; 是否有GPU
2. 用户选择安装方式, 如: docker、pip 安装
3. 用户选择安装CPU版本或GPU版本 (无GPU不提示)
4. 用户选择 CUDA 和 cuDNN 版本 (无GPU不提示)
5. 用户选择加速源 (可选的)
6. 安装 CUDA 和 cuDNN 驱动 (无GPU不安装)
7. 安装PaddlePaddle
8. 运行[环境检测](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html#sanyanzhenganzhuang)

## 实现方案

 - 通过 shell 脚本实现

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
