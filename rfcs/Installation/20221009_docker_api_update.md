# dockerApi更新

|   |  |
| --- | --- |
|提交作者 | gouzi | 
|提交时间 | 2022-10-09 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | paddle-docker-gpu | 
|文件名 | 20221009_docker_update.md | 


# 一、概述
## 1、相关背景

更新docker Api。

## 2、功能目标

更新 docker 快速部署文档

## 3、意义

使用较新的 docker Api

# 二、飞桨现状

需要从nvidia-docker启动容器[nvidia-docker](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html#huanjingzhunbei)

# 三、业内方案调研

### TensorFlow
目前已经更新为`--gpus all`参考: [TensorFlow-docker](https://tensorflow.google.cn/install/docker)

# 四、设计思路与实现方案

### 新增
```markdown
- 如需在Linux开启GPU支持, 需提前[安装nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker)
  * 请通过 docker -v 检查 Docker 版本。对于 19.03 之前的版本，您需要使用 nvidia-docker 和 nvidia-docker 命令；对于 19.03 及之后的版本，您将需要使用 nvidia-container-toolkit 软件包和 --gpus all 命令。这两个选项都记录在上面链接的网页上。

注nvidia-container-toolkit安装方法:
  * Ubuntu 系统可以参考以下命令
    * 添加存储库和密钥
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```
    * 安装nvidia-container-toolkit
    ```bash
    sudo apt update
    sudo apt install nvidia-container-toolkit
    ```
    * 重启docker
    ```bash
    sudo systemctl restart docker
    ```
  * centos 系统可以参考以下命令
    * 添加存储库和密钥 
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
    ```
    * 安装nvidia-container-toolkit
    ```bash
    sudo yun update
    sudo yum install -y nvidia-container-toolkit
    ```
    * 重启docker
    ```bash
    sudo systemctl restart docker
    ```
```

### 修改所有的`nvidia-docker`命令

# 五、测试和验收的考量

在不同的硬件环境和镜像下运行, 并且环境检测通过

# 六、影响面

仅为文档更新, 测试通过后可以无痛更新。

# 名词解释

暂无

# 附件及参考资料

* [安装nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
