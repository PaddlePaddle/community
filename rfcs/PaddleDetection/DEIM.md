# 目标检测网络DEIM —— 设计文档

|任务名称|目标检测网络DEIM|
|------|------|
|提交作者|[1111yiyiyiyi]|
|提交时间|2025-04-02|
|修改时间|2025-04-02|
|版本号|V1.0|
|依赖飞桨版本|develop版本|
|文件名|DEIM.md|

# 一、概述

## 1、相关背景

训练框架DEIM，通过密集O2O匹配策略和匹配感知损失函数加速实时对象检测收敛，显著提升模型性能。实验表明，DEIM在COCO数据集上集成RT-DETR和D-FINE后，训练效率大幅提高，单日内即可在GPU4090上完成训练，实现54.7%的检测精度，且推理延迟更低。与传统YOLO系列相比，DEIM在保持高效推理的同时，检测精度更胜一筹，为实时对象检测领域树立了新的性能基准，展现出框架的灵活性与扩展潜力。[^1]。

## 2、功能目标

为PaddleDetection添加DEIM系列模型，实现比同等规模的YOLO系列更优的精度和速度平衡，完成MS COCO等主流数据集的评测指标，并支持从训练到部署的全流程。DEIM系列在前代RT-DETR和D-FINE的基础上进一步优化，引入多项创新技术。

## 3、意义

为PaddleDetection增加最新的实时检测Transformer模型，丰富PaddleDetection模型库，推动DETR系列模型在实时目标检测领域的应用，满足用户对高性能实时目标检测模型的需求。DEIM的实现不仅提供了更高性能的检测选择，也为工业界的实时视觉应用提供了新的解决方案。

# 二、业内方案调研

目前目标检测领域的主流方案主要包括以下几类：

1. **基于CNN的单阶段检测器**：YOLO系列、SSD、RetinaNet等，以速度和部署便捷性著称
2. **基于CNN的两阶段检测器**：Faster R-CNN、Cascade R-CNN等，精度较高但速度相对较慢
3. **基于Transformer的检测器**：DETR系列、DINO、RT-DETR系列等，近年来发展迅速

在基于Transformer的检测器中，主要有：
1. **DETR**：首个端到端的基于Transformer的检测器，但收敛慢且推理速度不足以满足实时应用
2. **Deformable DETR**：通过可变形注意力机制改进DETR，加快收敛速度
3. **DINO**：通过改进的去噪策略和查询选择提高性能
4. **RT-DETR**：首个实时DETR模型，通过混合编码器和高效设计达到YOLO级别的速度
5. **RT-DETRv2**：RT-DETR的改进版本，引入更多训练优化策略
6. **RTDETRv3**：最新进展，主要创新点在于一对多分支策略和辅助检测头的结合使用
7. **DEIM**：通过密集O2O匹配策略和匹配感知损失函数加速实时对象检测收敛，显著提升模型性能。

## 性能表现如下

### DEIM-D-FINE
| Model   | Dataset   |   APD-FINE |   APDEIM |   #Params | Latency   |   GFLOPs |
|:--------|:----------|-----------:|---------:|----------:|:----------|---------:|
| N       | COCO      |       42.8 |     43   |         4 | 2.12ms    |        7 |
| S       | COCO      |       48.7 |     49   |        10 | 3.49ms    |       25 |
| M       | COCO      |       52.3 |     52.7 |        19 | 5.62ms    |       57 |
| L       | COCO      |       54   |     54.7 |        31 | 8.07ms    |       91 |
| X       | COCO      |       55.8 |     56.5 |        62 | 12.89ms   |      202 |

### DEIM-RT-DETRv2
| Model   | Dataset   |   APRT-DETRv2 |   APDEIM |   #Params | Latency   |   GFLOPs |
|:--------|:----------|--------------:|---------:|----------:|:----------|---------:|
| S       | COCO      |          47.9 |     49   |        20 | 4.59ms    |       60 |
| M       | COCO      |          49.9 |     50.9 |        31 | 6.40ms    |       92 |
| M*      | COCO      |          51.9 |     53.2 |        33 | 6.90ms    |      100 |
| L       | COCO      |          53.4 |     54.3 |        42 | 9.15ms    |      136 |
| X       | COCO      |          54.3 |     55.5 |        76 | 13.66ms   |      259 |

# 三、设计思路与实现方案

## 3.1 总体架构

DEIM框架旨在加速基于DETR（Detection Transformer）的实时目标检测器的收敛速度，通过改进匹配策略和优化损失函数实现。DEIM可以集成到现有的基于DETR的目标检测模型中，如RT-DETR和D-FINE。它主要包括两个部分：

1. Dense One-to-One (Dense O2O) 匹配策略：增加每张图像中的目标数量，生成更多正样本，增强监督信号。
2. Matchability-Aware Loss (MAL)：新型损失函数，优化不同质量级别的匹配，提升低质量匹配的有效性。

## 3.2 主要创新点

### 1. Dense O2O 匹配策略

通过经典的数据增强技术（如mosaic和mixup）实现，增加目标数量，生成更多正样本，无需额外计算开销。

```yaml
train_dataloader: 
  dataset: 
    transforms:
      ops:
        - {type: Mosaic, output_size: 320, rotation_range: 10, translation_range: [0.1, 0.1], scaling_range: [0.5, 1.5],
           probability: 1.0, fill_value: 0, use_cache: False, max_cached_images: 50, random_pop: True}
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640], }
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        epoch: [4, 29, 50]   # list 
        ops: ['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
      mosaic_prob: 0.5

  collate_fn:
    mixup_prob: 0.5
    mixup_epochs: [4, 29]
    stop_epoch: 50    # epoch in [72, ~) stop `multiscales`
```

### 2. Matchability-Aware Loss (MAL)：

扩展VariFocal Loss (VFL)的优点，解决其在处理低质量匹配时的不足，将匹配质量直接纳入损失函数，使模型对低质量匹配更加敏感。

```yaml
DEIMCriterion:
  weight_dict: {loss_mal: 1, loss_bbox: 5, loss_giou: 2}
  losses: ['mal', 'boxes', ]
  gamma: 1.5
```

## 3.3 网络结构

DEIM框架旨在加速基于DETR（Detection Transformer）的实时目标检测器的收敛速度，通过改进匹配策略和优化损失函数实现。
DEIM使用现有的基于DETR的目标检测模型结构，如RT-DETR和D-FINE。

## 3.4 实现细节

### 网络结构

根据RT-DETR和D-FINE网络结构，有多种大小不同的网络。

### 训练策略

学习率和数据增强调度器：提出DataAug Warmup调度器，在训练初始阶段简化注意力学习。使用FlatCosine学习率调度器。

```yaml
lrsheduler: flatcosine
lr_gamma: 0.5
warmup_iter: 2000
flat_epoch: 29    # 4 + epoch // 2, e.g., 40 = 4 + 72 / 2
no_aug_epoch: 8
```

# 四、对比分析

DEIM通过增加正样本数量和改进匹配质量，显著提高了模型的收敛速度和检测性能，同时保持了模型复杂度和推理延迟不变，提高了训练效率，并展现出良好的通用性和可扩展性。

# 五、测试和验收的考量

1. **精度指标**：
   - MS COCO验证集上mAP(IoU=0.5:0.95)
   - 小、中、大目标检测性能评估
   - 各类别AP评估

2. **速度指标**：
   - T4/V100 GPU上的FPS(帧率)
   - TensorRT FP16加速下的延迟时间
   - 不同输入尺寸下的性能变化

3. **部署测试**：
   - ONNX导出支持
   - TensorRT优化
   - 不同硬件平台兼容性测试(GPU/CPU/边缘设备)


# 六、影响面

DEIM的实现将丰富PaddleDetection的模型库，给用户提供更多高性能检测模型选择。其创新的架构设计(增加正样本数量和改进匹配质量)也为其他DETR系列检测模型提供了支持。

# 七、排期规划

1. **基础架构实现**：2周
   - DEIM主体架构实现

2. **训练与优化**：4周
   - RT-DETR模型训练与优化
   - D-FINE模型训练与优化

3. **性能评估与部署**：2周
   - MS COCO数据集评估
   - 速度基准测试
   - ONNX和TensorRT部署支持

4. **文档与示例**：1周
   - 使用文档编写
   - 模型训练部署示例编写

总计：约9周时间

[^1]: https://arxiv.org/pdf/2412.04234
