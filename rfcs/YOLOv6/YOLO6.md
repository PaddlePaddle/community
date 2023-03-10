技术标签：Python、深度学习
任务难度：基础⭐️

|任务名称 | No.154：论文复现：YOLOv6 v3.0: A Full-Scale Reloading | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 德尔塔大雨淋| 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-3-10 | 
|版本号 | 此设计文档的版本号，V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | YOLO6.md<br> | 

# 一、概述
## 1、相关背景
自前两个YOLO6版本发布以来，YOLO 社区一直不断有新的突破。从2023年农历新年的到来开始，YOLOv6官方重新设计了 YOLOv6网络模型，在网络架构上进行了许多新颖的增强，并设计的新的训练策略，并将此版本命名为为 YOLOv6 v3.0。
在性能上， YOLOv6-N 在 COCO 数据集上的 AP 达到 37.5% 使用 NVIDIA Tesla T4 GPU 测试，FPS为 1187  。YOLOv6-S 模型的 AP 为 45.0 %，FPS为484，优于同规模的其他主流探测器 （YOLOv5-S， YOLOv8-S， YOLOX-S 和 PPYOLOE-S）.
同时，在相同的推理速度下，YOLOv6-M/L 的准确度性能优于其他探测器，分别达到了50.0%/52.8%的准确率。此外，凭借扩展的主干和颈部设计，YOLOv6-L6实现了最先进的实时精度。论文中进行了大量实验，以验证每个改进组件的有效性。

## 2、功能目标
1. 在paddle框架下，YOLOv6 s版本在COCO数据集精度达到37.5mAP
2. 完成复现后合入PaddleYOLO
3. 参考repo https://github.com/meituan/YOLOv6

## 3、意义

通过本次升级，可以丰富PaddleYOLO中的模型种类，使PaddleYOLO可以支持最前沿的推理模型

YOLOv6 v3.0的主要更新内容如下：
### 1.  表征能力更强的 RepBi-PAN Neck 网络
有效的多尺度特征融合网络对目标检测的效果尤为关键。特征金字塔网络 （FPN）通过自上而下的路径来融合来自骨干网络不同 Stage 的输出特征以弥补网络学习过程中目标位置信息的损失。鉴于单向信息流传输的局限性，PANet 在 FPN 之上添加了一个额外的自底向上路径。BiFPN 为不同的输入特征引入了可学习的权重，并简化了 PAN 以实现更好的性能和更高的效率。PRB-FPN 通过具有双向融合的并行残差 FPN 结构来保留高质量的特征，以进行准确定位。
受到上述工作的启发，论文提出了一个表征能力更强的可重参化双向融合 PAN（RepBi-PAN）Neck 网络。一般而言，骨干网络浅层特征分辨率高，具有丰富的空间信息，有利于目标检测中的定位任务。为了聚合浅层特征，常见的做法是在 FPN 中增加 P2 融合层以及一个额外的检测头，但这往往会带来较大的计算成本。

为了实现更好的精度和时延权衡，设计了一个双向联结（Birectional Concatenate,  BiC）模块，在自上而下的传输路径中引入自底向上的信息流，使得浅层特征能以更高效的方式参与多尺度特征融合，进一步增强融合特征的表达能力。此模块能够帮助保留更准确的定位信号，这对于小物体的定位具有重要意义。

此外，对上一版本的 SimSPPF 模块进行了特征增强优化，以丰富特征图的表示能力。实验发现 YOLOv7 使用的 SPPCSPC 模块能够提升检测精度，但对网络推理速度的影响较大。于是我们对其进行了简化设计，在检测精度影响不大的情况下，大大提升了推理效率。同时，我们引入了可重参数化思想并对 Neck 网络的通道宽度和深度进行了相应的调整。

### 2. 全新的锚点辅助训练（Anchor-Aided Training）策略
基于深度学习的目标检测技术从学习范式上主要可分为 Anchor-based 和 Anchor-free 两大类，这两类方法针对不同尺度的目标检测上分别存在不同的优势。作者使用 YOLOv6-N 作为基线，对 Anchor-based 和 Anchor-free 范式的异同点进行了相关的实验和分析。


当 YOLOv6-N 分别采用 Anchor-based 和 Anchor-free 训练范式时，模型的整体 mAP 几乎接近，但采用 Anchor-based 的模型在小、中、大目标上的 AP 指标会更高。从以上的实验可以得出结论：相比于 Anchor-free 范式，基于 Anchor-based 的模型存在额外的性能增益。

同时发现，YOLOv6 使用 TAL 进行标签分配时，其模型精度的稳定性与是否采用 ATSS 预热有较大关系。当不使用 ATSS 预热时，对同样参数配置的 YOLOv6-N 进行多次训练，模型精度最高可达35.9% mAP，最低至 35.3% mAP，相同模型会有 0.6% mAP 的差异。但当使用 ATSS 预热时，模型精度最高却只能到达 35.7% mAP。从实验结果可以分析得出，ATSS 的预热过程利用了 Anchor-based 的预设信息，进而达到稳定模型训练的目的，但也会在一定程度上限制网络的峰值能力，因此并不是一种最优的选择。

受到上述工作的启发，文章提出了基于锚点辅助训练（Anchor-Aided Training，AAT）策略。在网络训练过程中，同时融合 Anchor-based 和 Anchor-free 的两种训练范式，并对全阶段网络进行映射及优化，最终实现了Anchor 的统一，充分发挥了结合不同 Anchor 网络的各自优势，从而进一步提升了模型检测精度。

### 3. 无痛涨点的 DLD 解耦定位蒸馏策略
在目标检测的蒸馏任务中，LD 通过引入 DFL 分支，从而达到了在网络中对定位信息蒸馏的目的，使分类和定位信息得以同步回传，弥补了 Logit Mimicking 方法无法使用定位蒸馏信息的不足。但是，DFL 分支的添加，对于小模型速度的影响是很明显的。添加了 DFL 分支后，YOLOv6-N 的速度下降了 16.7%，YOLOv6-S 的速度下降了 5.2%。而在实际的工业应用当中，对于小模型速度的要求往往很高。因此，目前的蒸馏策略并不适合于工业落地。

针对这个问题，作者提出了基于解耦检测任务和蒸馏任务的 DLD（Decoupled Location Distillation）算法。DLD 算法会在网络每一层的回归头上分别添加了额外的强化回归分支，在训练阶段，该分支同样会参与 IoU 损失的计算，并将其累加到最终的 Loss 中。

通过增加的额外的强化回归分支，可以对网络添加更多的额外约束，从而对网络进行更全面细致的优化。并且，DLD算法在对强化回归分支进行训练时，引入了分支蒸馏学习策略。分支蒸馏学习策略会仅使用 DFL 分支参与网络标签分配的过程，并将标签分配的结果投入到强化回归分支进行引导学习，从而参与强化回归分支的损失函数计算和反向传播优化。

# 二、飞桨现状
PaddleYOLO中已有YOLOv6 前两个版本的模型，暂不支持最新的

# 三、业内方案调研
官方提交代码使用的框架为PyTorch，代码库为：https://github.com/meituan/YOLOv6
飞桨中与YOLOv6相关的项目有：
1. 【AI达人特训营】使用TensorRT加速 YOLOv6 ：https://aistudio.baidu.com/aistudio/projectdetail/4263301?channelType=0&channel=0
2. 基于YOLOv6实现野生动物检测 ：https://aistudio.baidu.com/aistudio/projectdetail/5150597?channelType=0&channel=0

# 四、对比分析
paddle接口和PyTorch接口有很多相似的地方，可以较为快速的完成代码移植

# 五、排期规划
依据导师安排和大赛要求，按时完成任务

