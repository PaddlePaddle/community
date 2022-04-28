# 【PaddlePaddle Hackathon 2】87、PaddleSpatial添加路网表征学习模块

|   |  |
| --- | --- |
|提交作者 | aptx1231 |
|提交时间 | 2022-04-26 |
|版本号 | V1.0 |
|依赖飞桨版本 | 2.2.2 |
|文件名 | 【Hackathon No.87】RFC |


# 一、概述
## 1、相关背景
路网表征学习是时空大数据分析中的关键任务之一，主要是学习交通路网中每个路段的表征向量，是图表征任务的一种。
## 2、功能目标
在[PaddleSpatial](https://github.com/PaddlePaddle/PaddleSpatial)库的基础上，实现4种路网表征学习模块（Geom-GCN, ChebConv, DeepWalk, LINE) 。
## 3、意义
完善PaddleSpatial库在路网表征方向的空缺

# 二、飞桨现状
飞桨框架于 2.0 正式版之后正式发布了动静转换功能，并在2.1、2.2 两个大版本中不断新增了各项功能，以及详细的使用文档和最佳实践教程。


# 三、业内方案调研

图表征学习主要目的是将图的每个节点表示为低维向量。现有的研究可以根据各种标准进行分类。

一些方法专注于捕捉不同的图的属性，如接近性和同质性。特别是DeepWalk和Node2Vec采用图上的随机行走来获得被视为句子的节点序列，然后将最初为学习词嵌入而提出的skip-gram模型应用于学习节点表示。LINE通过明确优化相应的目标来保留了节点的一阶和二阶的接近性。图神经网络（GNN）也被用于图表示学习。基于GNN的模型通过交换和聚合邻域的特征来生成节点表示，并提出了不同的方法来探索不同的有效聚合操作。

路网表征学习：路网表示可以促进智能交通系统的应用，如交通推理和预测，区域功能发现等等。目前主要的研究方法一个是将图表示学习技术扩展到道路网络。另外，也可以考虑时空数据的独有特性，就是从轨迹数据中学习路网的表征等。

相关论文包括：

- [TIST2020]On Representation Learning for Road Networks
- [SIGSPATIAL2019]Learning Embeddings of Intersections on Road Networks
- [KDD2020]Learning Effective Road Network Representation with Hierarchical Graph Neural Networks
- [CIKM2021]Robust Road Network Representation Learning When Traffic Patterns Meet Traveling Semantics

目前，开源的路网表征算法仓库主要由[LibCity](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation)库，其中实现了多个路网表征方法。

# 四、对比分析
目前，开源的路网表征算法仓库主要由[LibCity](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation)库，其中实现了多个路网表征方法。其主要是基于Pytorch实现。

# 五、设计思路与实现方案
## 流程

1. 路网数据集构造路段向量和路网邻接矩阵
2. 在路网数据集上进行预处理，得到随机游走的路段（可选的）
3. 在上述数据集上，针对不同的模型以及不同的训练方案，搭建基于PaddlePaddle的模型进行训练，得到最终的路网表征向量。

### 数据

来自openstreetmap的交通路网信息，包括路段的长度、类别、限速等，以及路段之间的邻接关系（构成图结构）。

对路段的不同特征使用不同的方法处理（例如独热化、正则化等）转换成路段向量$N \times F$。F是特征维度，N是路段数量。

对路段的邻接关系进行处理，可以得到图邻接矩阵$N \times N$。

### 模型

复现4个路网表征模型

- ChebConv
  - 即使用图卷积（GCN）对路段的特性向量$N \times F$进行卷积得到隐层向量$N \times hid$，通过自回归的方式进行训练，即从隐层向量使用另一个图卷积将之恢复成路段特性向量的形状即$N \times F$，使用输入输出向量的重构loss进行模型的训练。以隐层向量$N \times hid$作为路网表征。
  - 参考文献：Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29, 3844-3852.
- GeomGCN
  - 类似于ChebConv，同样使用自回归的方案进行训练。GeomGCN模型将参照原始论文进行搭建。
  - 参考文献：Geom-gcn: Geometric graph convolutional networks
- DeepWalk
  - DeepWalk模型通过在邻接矩阵上进行随机游走，得到一系列游走出来的路径。在路径上训练skip-gram模型，得到路网表征。
  - 参考文献：DeepWalk: Online Learning of Social Representations
- LINE
  - LINE模型通过计算一阶相似性和二阶相似性，利用节点的链接关系来确定经验分布，通过对于分布的预测于经验分布的距离来作为最终的loss函数，最终对图中的节点进行编码，得到路网表征向量。
  - 参考文献：Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015, May). Line: Large-scale information network embedding. In Proceedings of the 24th international conference on world wide web (pp. 1067-1077).

### 实现方案

主要依赖的Python库有：

- paddlepaddle-gpu   2.2.2.post112
- pgl                2.2.3.post0
- gensim             4.1.2

- networkx           2.6.3

主要实现的代码模块有：

- 数据读取与预处理模块
- 模型模块（4个模型）
- 训练模块
- 训练日志模块

# 六、测试和验收的考量

4个路网表征模型都可以在某一城市的路网数据集上运行，并得到对应的路网表征向量。

# 七、可行性分析和排期规划
作者在路网表征学习和图表征学习方面有过研究，基于Pytorch开发过相关的模型。

# 八、影响面

暂无
# 名词解释

- 路网：即城市中若干条道路互相交错构成的网络。路网可以表示为一个$G=(V,E,A)$的图，其中V是路网中点的集合，E是路网中边的集合，A是G的邻接矩阵。
- 路段：即城市中的道路，也称为路段。
- 路网表征学习：对路网中的每条道路通过算法学习一个向量进行表示以支撑其他下游任务，是图表示学习的一种特例。

# 附件及参考资料

暂无
