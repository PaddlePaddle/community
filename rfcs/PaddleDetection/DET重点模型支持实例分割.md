# DET重点模型支持实例分割 —— 设计文档

| 任务名称                                                     | DET重点模型支持实例分割        | 
|----------------------------------------------------------|----------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | MINGtoMING           | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-9-26            | 
| 版本号                                                      | V1.0                 | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本            | 
| 文件名                                                      | DET重点模型支持实例分割.md<br> | 

# 一、概述
## 1、相关背景
[【PaddlePaddle Hackathon 5th】开源贡献个人挑战赛 #57262](https://github.com/PaddlePaddle/Paddle/issues/57262)
## 2、功能目标
- 使PaddleDetection现阶段的重点模型(PP-YOLO-E+与RT-DETR)支持实例分割任务, 具体工作如下：
  - 新增PP-YOLO-E+与RT-DETR模型实例分割头
  - 其中PP-YOLO-E+_crn_l、RT-DETR-L模型在COCO数据集的实例分割任务上的精度优于同level的其他模型
  - 打通基于python的部署，并补全文档


## 3、意义
- 扩充PaddleDetection中支持实例分割任务的模型数量，给予用户更多选择
- 目前PaddleDetectin中缺少能同时权衡实时性与精度的实例分割模型，PP-YOLO-E+与RT-DETR模型的加入可以解决这一问题

# 二、飞桨现状
实例分割使用场景比较广泛，目前PaddleDetection支持的实例分割模型较老，不能满足用户需求，需要支持。


# 三、业内方案调研
- PP-YOLO-E+ 支持实例分割的调研
  - 目前YOLO类模型的实例分割是基于[YOLACT](https://github.com/dbolya/yolact)实现的，主要方法是在目标检测头之外再添加一个mask分割头，两者并行，从而使得网络结构仍为单阶段，适合YOLO系列，速度较快，参数量和计算量的增量较少。
  - [YOLOv5和YOLOv8](https://github.com/ultralytics/ultralytics)的实例分割任务也是基于YOLACT实现的，它们在COCO数据集的实例任务上取得了速度与精度的均衡, YOLOv8在COCO数据集的实例分割任务上的评测结果如下，更多细节见[Ultralytics](https://github.com/ultralytics/ultralytics).
 
| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

- RT-DETR 支持实例分割的调研
  - 目前较为先进的DETR模型的实例分割是基于[MaskDINO](https://github.com/idea-research/maskdino)实现的，它通过在DINO上扩展mask预测分支，并使用与MaskFormer等模型中类似的pixel embed点积query embed的方式来得到各个预测类别的二值mask
  - MaskDINO在COCO数据集的实例分割任务上曾经取得SOTA, 其精度是非常高的.

# 四、对比分析
PP-YOLO-E+和RT-DETR较于YOLOv8在目标检测任务上的精度、实时性和收敛速度更优，可以预期PP-YOLO-E+和RT-DETR在实例分割任务上有更优的表现。

# 五、设计思路与实现方案

## 总体思路
### PP-YOLO-E+ 分割头实现
在`ppdet/modeling/heads/ppyoloe_head.py`中添加分割头类，并对已有`PPYOLOEHead`类进行修改，使其可以通过`with_mask`参数来控制是否使用分割头。
### RT-DETR 分割头实现
参考已有的`ppdet/modeling/transformers/mask_dino_transformer.py`，复用maskdino中的分割模块实现，少量修改`ppdet/modeling/transformers/rtdetr_transformer.py`，使得`RTDETRTransformer`可以通过`with_mask`参数来控制是否使用分割头。

# 六、测试和验收的考量
1. PP-YOLO-E+_crn_l、RT-DETR-L模型在COCO数据集的实例分割任务上的精度优于YOLOv8-L
2. PP-YOLO-E+_crn_l、RT-DETR-L对应的实例分割模型能够进行动态图和静态图模式下的推理部署
3. 补全文档，其中包括PP-YOLO-E+_crn_l、RT-DETR-L在V100或T4上和其他模型的推理时间、FLOPs和参数量等的对比， 相关配置文件的使用， 如训练，评估和推理等等。

# 七、影响面
添加的分割头可以通过`with_mask`参数来控制是否使用，对其他模块没有影响。

# 八、排期规划
于2023年10月3日前，完成RT-DETR-L实例分割支持并在COCO上训练完成。

于2023年10月7日前，完成PP-YOLO-E+_crn_l实例分割支持并在COCO上训练完成。

总体可以在活动时间内完成。
