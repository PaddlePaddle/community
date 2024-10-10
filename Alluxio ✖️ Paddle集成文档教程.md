# Alluxio ✖️ Paddle集成文档教程

# 背景

本教程展示了在Paddle Paddle中集成和使用Alluxio，为以阿里云等OSS对象存储为源的数据集进行缓存加速，提高训练效率的方法。在本教程中，采用了常见的Imagenet-mini数据集，训练Resnet50，并通过包括使用Alluxio在内不同方式从对象存储中加载数据，比较了各种方法的耗时及优劣。阅读本文档可帮助你了解Alluxio在Paddle数据加载、存储方面提供的价值，并上手使用测试，使得你的数据训练可以跨云、高效、低成本。

## Alluxio Fuse SDK的原理

Alluxio Fuse SDK是一个轻量化、非侵入式的SDK，允许将OSS、S3、HDFS等存储服务上的数据集挂载到本地，使之可以像本地文件系统般访问，并通过缓存提高频繁访问的数据的I/O速度。使用Alluxio Fuse，当应用程序通过SDK访问数据时，如果缓存命中，则直接返回数据，无需完整处理过程，从而显著提升读取性能。这使得读取远程存储数据就像读取本地文件一样快速；如果未命中，则会从挂载的存储中读取所需数据。

# 数据集和实验环境介绍

1. ResNet50
   1. ResNet50是一种深度残差网络（Deep Residual Network），属于卷积神经网络（CNN）的一种，专门用于图像识别和图像分类任务。
   2. 它由微软研究院提出，含有50层深的网络结构。
   3. ResNet的核心思想是引入了“残差学习”，通过跳过一些层来解决更深网络训练中的梯度消失问题。
   4. 在多个图像识别竞赛中，ResNet50表现出色，被广泛应用于图像处理和计算机视觉领域。
2. PaddlePaddle
   1. PaddlePaddle（Parallel Distributed Deep Learning）是百度开发的开源深度学习平台。
   2. 它提供了简单易用的API，支持多种深度学习模型，包括但不限于图像识别、语言理解、预测分析等领域。
   3. PaddlePaddle特别强调在大规模数据集上的高效训练和灵活的模型配置，适用于企业级和学术研究应用。
3. 阿里云OSS（Object Storage Service）
   1. 阿里云OSS是阿里巴巴提供的云存储服务，用于存储和处理大量数据。
   2. 它提供了稳定、安全、高效的对象存储解决方案，可以处理各种类型的数据（如图片、视频、日志文件等）。
   3. 阿里云OSS广泛应用于网站、移动应用、大数据分析等场景，特别是在数据备份、灾难恢复、数据共享方面具有重要作用。
4. ImageNet-mini
   1. ImageNet-mini是ImageNet数据集的一个缩小版本，包含了原始ImageNet数据集的一个子集。
   2. ImageNet是一个大规模的图像数据库，被广泛用于训练和测试图像识别、对象检测等计算机视觉模型。
   3. ImageNet-mini虽然规模较小，但依然包含了大量的图像和标签，适合用于测试和验证模型性能，尤其是在计算资源有限的情况下。

本实验选择imagenet-mini训练是为了能够在一定的数据集复杂度下，体现在图像识别领域，使用Alluxio Fuse SDK可以对Paddle的数据存储和训练带来的变化和价值。

本次选择的数据集来自https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html

实验机器配置：4 vCPU，15 GiB，1 * NVIDIA T4，1 * 16 GB，华北（青岛）节点

OSS配置：阿里云OSS，华北（北京）节点

# 实验方案

由于训练数据存储在OSS对象存储中，PaddlePaddle不能直接访问，因此需要通过适当的方式拉取数据，使得数据能够在训练节点被访问。本教程中采用了以下几种数据拉取模式：

* 方案一：本地训练

手动将对象存储中的数据集下载到训练节点，训练节点直接读取本地文件进行训练

* 方案二：ossfs挂载

使用ossfs(https://github.com/aliyun/ossfs)将对象存储的数据通过fuse挂载到训练节点，训练解读读取该挂载点内的数据进行训练，ossfs会向OSS请求对应的数据

* 方案三：Alluxio+kernel cache

使用Alluxio fuse sdk将对象存储的数据通过fuse挂载到训练节点，训练解读读取该挂载点内的数据进行训练。Alluxio fuse sdk会将部分热数据缓存在内存的Kernel cache中

* 方案四：Alluxio+local cache

使用Alluxio fuse sdk将对象存储的数据通过fuse挂载到训练节点，训练解读读取该挂载点内的数据进行训练。Alluxio fuse sdk会向OSS拉取数据，并在硬盘中缓存被访问过的数据

# 实验步骤

## 准备测试环境

### 测试环境

* 操作系统：Ubuntu22.04LTS
* JDK版本：JDK11
* libfuse版本：libfuse3
* Alluxio-fuse sdk版本：alluxio-305-bin
* Paddle Paddle版本：2.6.0

### 数据集上传到阿里云oss

参考https://help.aliyun.com/zh/oss/user-guide/simple-upload，将所下载的数据上传到你创建好的oss bucket中。

### 安装paddle paddle

```Shell
pip install paddlepaddle-gpu
```

## 方案一——本地训练

* 将数据集全部下载到本地文件夹中，数据集应该被组织成一个训练集和一个测试集的文件夹，每个类别的图片放在单独的子文件夹中。
* 编写训练程序

```Python
import paddle
import paddle.vision.transforms as T
from paddle.vision.models import resnet50
from paddle.io import DataLoader
from paddle.vision.datasets import DatasetFolder

# 转换函数
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = DatasetFolder("/path/to/imagenet-mini/train", transform=transform)
val_dataset = DatasetFolder("/path/to/imagenet-mini/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 初始化模型
model = resnet50(pretrained=False, num_classes=1000)
model = paddle.Model(model)

# 配置训练参数
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
model.prepare(optimizer, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())

# 训练和验证
model.fit(train_loader, val_loader, epochs=10, verbose=1)
```

注意：你需要替换`"/path/to/imagenet-mini/train"`和`"/path/to/imagenet-mini/val"`为你的ImageNet-mini数据集的实际路径。

* 运行脚本并记录时间: 运行脚本，并使用Python的`time`模块来记录训练时间。可以在脚本开始和结束时记录时间。

```Python
import time

start_time = time.time()
# 训练和验证代码
end_time = time.time()

print("Training time: ", end_time - start_time, " seconds")
```

## 方案二——ossfs挂载

* 安装 ossfs: 你需要在你的系统上安装`ossfs`。例如，在Ubuntu上，你可以使用以下命令安装：

```Shell
sudo apt-get install ossfs
```

* 配置 OSS 访问: 创建一个配置文件（`/etc/passwd-ossfs`），包含你的OSS访问密钥，格式如下：

```Shell
my-bucket-name:access-key-id:access-key-secret
```

* 确保文件的权限安全：

```Shell
sudo chmod 640 /etc/passwd-ossfs
```

* 挂载 OSS Bucket: 创建一个本地目录用于挂载OSS Bucket，然后使用`ossfs`命令挂载它，在挂载OSS Bucket时，使用`-ocache_size`和`-ouse_cache`参数来设置缓存大小和缓存目录。例如：

```Shell
mkdir /path/to/local/mount
mkdir /path/to/cache
ossfs my-bucket-name /path/to/local/mount -ourl=oss-cn-hangzhou.aliyuncs.com -ocache_size=<cache_size_in_MB> -ouse_cache=/path/to/cache
```

替换`my-bucket-name`、`/path/to/local/mount`和`oss-cn-hangzhou.aliyuncs.com`为实际的Bucket名称、本地挂载点和OSS Endpoint。

替换`<cache_size_in_MB>`为你希望的缓存大小（以MB为单位）。例如，如果你想要1000MB的缓存，就将其设置为`1000`。本实验采用的数据集相对适中，可以设置全量本地缓存，提高训练效率。指定缓存大小为10000MB，可以取得更好的实验结果。

* 性能考虑: 使用ossfs直接挂载OSS可能会有一些性能上的考虑，因为数据需要通过网络传输。这可能会增加数据加载的时间，特别是在大规模数据集和/或带宽受限的情况下。
* 缓存策略: 考虑到性能问题，可能需要实施适当的缓存策略，以减少对OSS的重复访问。

使用方案一中的训练程序和时间记录程序进行训练并记录时间。

## 方案三——alluxio+kernel cache

* JDK可以使用sdkman管理下载。

```Shell
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java 11.0.21-zulu
```

* libfuse使用libfuse3最新版的即可

```Shell
sudo apt-get install libfuse3
```

* 安装alluxio fuse sdk并解压，https://downloads.alluxio.io/downloads/files/

```Shell
wget https://downloads.alluxio.io/downloads/files/305/alluxio-305-bin.tar.gz
tar -xzf alluxio-305-bin.tar.gz
```

FUSE使用kernel cache时，具有以下I/O模式，用于控制是否缓存数据和缓存无效策略:

* `direct_io`:禁用内核数据缓存。libfuse 2和libfuse 3都支持，但尚未被Alluxio FUSE libfuse 3实现支持。
* `kernel_cache`：始终在内核中缓存数据，并且没有发生缓存失效。这应该只在文件系统上启用，因为文件数据永远不会从外部更改（不会通过当前的FUSE挂载点）
* `auto_cache`：在内核中缓存数据，如果修改时间或文件大小已更改，则使缓存无效

内核数据缓存将显著提高I/O性能，但容易消耗大量的节点内存。在普通的机器环境中，当节点内存不足时，内核内存会被自动回收，不会影响节点上的AlluxioFuse进程或其他应用程序的稳定性。然而，在容器化环境中，内核数据缓存将被计算为容器所使用的内存。当容器使用的内存超过配置的容器最大内存时，Kubernetes或其他容器管理工具可能会杀死容器中的一个进程，这将导致AlluxioFuse进程退出，并且在Alluxio FUSE挂载点上运行的应用程序失败。

本次集成测试采用`auto_cache`的方式。

### 前提条件

* 确保已安装Alluxio和FUSE。
* 拥有一个阿里云账户，并已创建OSS存储桶。
* 获取OSS的`AccessKeyId`和`AccessKeySecret`。

**配置Alluxio以使用OSS： 编辑Alluxio的配置文件** **`alluxio-site.properties`** **，添加OSS存储的配置。这通常包括以下配置项：**

```Shell
alluxio.underfs.oss.endpoint=<OSS_ENDPOINT>
alluxio.underfs.oss.accessKeyId=<ACCESS_KEY_ID>
alluxio.underfs.oss.accessKeySecret=<ACCESS_KEY_SECRET>
alluxio.underfs.oss.bucket.name=<BUCKET_NAME>
```

替换`<OSS_ENDPOINT>`、`<ACCESS_KEY_ID>`、`<ACCESS_KEY_SECRET>`和`<BUCKET_NAME>`为你的OSS配置信息。`OSS_ENDPOINT`是OSS服务的访问域名，例如`oss-cn-hangzhou.aliyuncs.com`。

**启动Alluxio服务： 如果还未启动Alluxio服务，需要先启动它。可以参考这个文档启动**

https://docs.alluxio.io/os/user/stable/cn/overview/Getting-Started.html

**挂载OSS存储桶到Alluxio： 使用Alluxio的****`mount`****命令将OSS存储桶挂载到Alluxio的命名空间中。例如：**

```Shell
./bin/alluxio fs mount --option alluxio.underfs.oss.endpoint=<OSS_ENDPOINT> --option alluxio.underfs.oss.accessKeyId=<ACCESS_KEY_ID> --option alluxio.underfs.oss.accessKeySecret=<ACCESS_KEY_SECRET> /mnt/oss oss://<BUCKET_NAME>/
```

这里`/mnt/oss`是Alluxio中的挂载点，`oss://<BUCKET_NAME>/`是OSS存储桶的路径。

**挂载Alluxio FUSE： 使用****`alluxio-fuse`** **命令挂载Alluxio到本地文件系统。启用内核缓存的关键是在挂载命令中使用** **`-o auto_cache`****选项，例如：**

```Shell
./bin/alluxio-fuse mount mount_point /alluxio_path -o auto_cache
```

这里`mount_point`是本地文件系统中的挂载点，`/alluxio_path`是Alluxio中的路径。使用`-o auto_cache`选项允许内核为读操作缓存数据。

使用方案一中的训练程序和时间记录程序进行训练并记录时间。

## 方案四：Alluxio+Local Cache

和方案三一样先配置相应的环境。

* 用户空间数据缓存可以通过

```Shell
$ bin/alluxio-fuse mount <under_storage_dataset> <mount_point> \    
-o local_data_cache=<local_cache_directory> \   
-o local_cache_size=<size>
```

`local_data_cache`（默认值=“”表示禁用）：用于本地数据缓存的本地文件夹 `local_cache_size`（默认= `512MB`）：本地数据缓存目录的最大缓存大小

数据可以根据缓存目录的类型缓存在ramdisk或磁盘上。

* 你也可以通过配置`alluxio-site.properties`文件来开启和配置本地缓存。

```Shell
alluxio.user.client.cache.enabled=true
alluxio.user.client.cache.dir=/mnt/alluxio-cache
alluxio.user.client.cache.size=10GB
```

上述配置启用了客户端本地缓存，设置了缓存目录以及缓存的大小。

### 使用Alluxio FUSE挂载

在完成Alluxio和OSS的配置之后，你可以使用Alluxio的FUSE功能将Alluxio挂载到本地文件系统中。首先，确定你的系统已经安装了FUSE。接着，可以使用以下命令挂载Alluxio：

```Shell
./bin/alluxio-fuse mount /path/to/mount/point /alluxio/path
```

这里`/path/to/mount/point`是你希望Alluxio挂载到的本地目录路径，`/alluxio/path`是Alluxio命名空间中的路径。如果你想将整个Alluxio命名空间挂载到本地目录，可以使用根目录`/`作为`/alluxio/path`。

# 实验对比分析

下表概述了使用不同存储和缓存方法训练 AI 模型的时间。这些时间是在训练 Resnet50/Imagenet-mini 数据集时记录的。

| Method                       | Duration (hours) |
|------------------------------|------------------|
| Alluxio fuse kernel cache    | 17.37            |
| Alluxio fuse userspace cache | 16.65            |
| Ossfs                        | 16.25            |
| Local training               | 15.47            |


方案一：从云存储中下载到本地训练

在列出的方法中，它是最快的。但需要注意的是，这个测量不包括将数据集下载到本地环境所需的时间。如果我们包括数据下载时间，那么总时间可能会根据网络速度和数据集大小显著增加。

优点：不需要依赖额外的组件

缺点：CPU、GPU空转时间长；无法训练大数据模型（本地存储不够）；多次IO，性能差；需要手动检验数据集完整

方案二：使用Ossfs挂载到本地，进行训练

OSSFS 略快可以归因于几个与阿里云的优化：OSSFS 针对阿里云的基础设施进行了优化，可能导致更好的网络性能和在访问存储在阿里云对象存储服务（OSS）中的数据时更低的延迟。

优点：适配阿里云服务器性能好；支持缓存设置；多客户端提供了一致性策略

缺点：无内核缓存策略；无法做到多存储平台数据迁移，只能挂载阿里云OSS存储

方案三：使用Alluxio fuse sdk+kernel cache

优点：适配多存储平台；内核缓存加速读写

缺点：内核空间有限，无法做到大数据集的全量缓存

方案四：使用alluxio fuse sdk+local cache

优点：适配多存储平台；用户空间全量缓存；支持本地、分布式缓存

缺点：用户空间一般使用磁盘开销代替网络开销，在小数据集不如内核缓存效果理想。

# 结论和总结

Alluxio Fuse 在 AI 模型训练中的作用主要体现在以下几个方面：

1. 适配多存储平台：Alluxio Fuse 能够连接到多种存储系统，如 HDFS、S3、GCS 等，这为 AI 模型训练提供了灵活的数据访问能力。
2. 缓存加速：Alluxio Fuse 支持本地和分布式缓存，能够将频繁访问的数据缓存到更靠近计算资源的位置，从而减少数据读取延迟，加速模型训练过程。
3. 内核缓存与用户空间缓存：Alluxio Fuse 支持内核缓存和用户空间缓存，其中内核缓存能够提供更快的数据访问速度，适合小数据集的场景；用户空间缓存能够实现更大规模的数据缓存，适合大数据集的场景。
4. 支持分布式训练：通过分布式缓存机制，Alluxio Fuse 能够支持在多个训练节点上进行并行训练，提高模型训练的效率。

总的来说，Alluxio Fuse 在 AI 模型训练中的作用是通过适配多存储平台、提供缓存加速以及支持分布式训练来提高数据访问效率和模型训练速度，从而加快 AI 模型的开发和迭代过程。

