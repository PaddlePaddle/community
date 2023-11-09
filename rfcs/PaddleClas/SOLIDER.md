# 分类大模型--人体视觉任务SOLIDER—— 设计文档

| 任务名称                                                     | 分类大模型--人体视觉任务SOLIDE |
|----------------------------------------------------------|----------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | Yang-Changhui |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-9-28            |
| 版本号                                                      | V1.0                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本            |
| 文件名                                                      | SOLIDER.md<br> |

# 一、概述
## 1、相关背景

该论文利用自监督训练方式，充分利用现有大量人体无标注数据，得到一个可以通用于下游各种人体视觉任务的预训练大模型，本任务的完成可以支持PaddleClas各种人体视觉任务。 


## 2、功能目标
需要前向对齐网络，需对齐的模型包括swin_tiny_patch4_window7_224、swin_small_patch4_window7_224以及swin_base_patch4_window7_224。

## 3、意义

为PaddleClas 增加分类大模型--人体视觉任务SOLIDE。


# 二、业内方案调研

SOLIDER 源码已经开源，地址：https://github.com/tinyvision/SOLIDER

性能表现如下

| Task                                              | Dataset     | Swin Tiny ([Link](https://drive.google.com/file/d/12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8/view?usp=share_link)) | Swin Small ([Link](https://drive.google.com/file/d/1oyEgASqDHc7YUPsQUMxuo2kBZyi2Tzfv/view?usp=share_link)) | Swin Base ([Link](https://drive.google.com/file/d/1uh7tO34tMf73MJfFqyFEGx42UBktTbZU/view?usp=share_link)) |
| ------------------------------------------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Person Re-identification (mAP/R1) w/o re-ranking  | Market1501  | 91.6/96.1                                                    | 93.3/96.6                                                    | 93.9/96.9                                                    |
|                                                   | MSMT17      | 67.4/85.9                                                    | 76.9/90.8                                                    | 77.1/90.7                                                    |
| Person Re-identification (mAP/R1) with re-ranking | Market1501  | 95.3/96.6                                                    | 95.4/96.4                                                    | 95.6/96.7                                                    |
|                                                   | MSMT17      | 81.5/89.2                                                    | 86.5/91.7                                                    | 86.5/91.7                                                    |
| Attribute Recognition (mA)                        | PETA_ZS     | 74.37                                                        | 76.21                                                        | 76.43                                                        |
|                                                   | RAP_ZS      | 74.23                                                        | 75.95                                                        | 76.42                                                        |
|                                                   | PA100K      | 84.14                                                        | 86.25                                                        | 86.37                                                        |
| Person Search (mAP/R1)                            | CUHK-SYSU   | 94.9/95.7                                                    | 95.5/95.8                                                    | 94.9/95.5                                                    |
|                                                   | PRW         | 56.8/86.8                                                    | 59.8/86.7                                                    | 59.7/86.8                                                    |
| Pedestrian Detection (MR-2)                       | CityPersons | 10.3/40.8                                                    | 10.0/39.2                                                    | 9.7/39.4                                                     |
| Human Parsing (mIOU)                              | LIP         | 57.52                                                        | 60.21                                                        | 60.50                                                        |
| Pose Estimation (AP/AR)                           | COCO        | 74.4/79.6                                                    | 76.3/81.3                                                    | 76.6/81.5                                                    |

# 三、对比分析

参考官方原码实现即可。

# 四、设计思路与实现方案

## 总体思路
### 在SwinTransformer中添加semantic_embed_w与semantic_embed_b处理模型

在对比swim_transformer模型与SOLIDER模型发现，前者缺少semantic_embed_w与semantic_embed_b，需要补充。

### 对齐SwinTransformerBlock

PaddleClas中现有的swim_transformer模型中的SwinTransformerBlock模块，与SOLIDER模型中的SwinBlock模块处理方式不一样，导致该部分输出结果不同。

### 模型转换

由于swim_transformer和SOLIDER网络结构参数名不同，以及paddle和torch的模型存储结构不同，需要进行模型转换。

# 五、测试和验收的考量

1. 增加介绍文档PaddleClas/docs/zh_CN/models/sodier.md
2. 对swin系列backbone进行必要的修改
3. 发送转化swin系列（swin_tiny_patch4_window7_224、swin_small_patch4_window7_224以及swin_base_patch4_window7_224）的权重和对齐日志

# 六、影响面

对其他模块没有影响。

# 七、排期规划

可以在活动时间内完成。
