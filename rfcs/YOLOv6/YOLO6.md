技术标签：Python、深度学习
任务难度：基础⭐️

|任务名称 | No.154：论文复现：YOLOv6 v3.0: A Full-Scale Reloading | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 德尔塔大雨淋| 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-3-10 | 
|版本号 | 此设计文档的版本号，V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 提交的markdown设计文档文件名称，YOLO6.md<br> | 

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

# 二、飞桨现状
PaddleYOLO中已有YOLOv6 前两个版本的模型，暂不支持最新的YOLOv6 v3.0


# 三、业内方案调研
官方提交代码使用的框架为PyTorch，代码库为：https://github.com/meituan/YOLOv6

# 四、对比分析
paddle接口和PyTorch接口有很多相似的地方，可以较为快速的完成代码移植

# 五、排期规划
依据导师安排和大赛要求，按时完成任务

