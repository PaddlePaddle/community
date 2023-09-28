# 轻量语义分割网络PIDNet —— 设计文档

| 任务名称                                                     | 轻量语义分割网络PIDNet | 
|----------------------------------------------------------|----------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll           | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-9-26            | 
| 版本号                                                      | V1.0                 | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本            | 
| 文件名                                                      | pidnet.md<br> | 

# 一、概述
## 1、相关背景

该模型为轻量化分割方向的前沿模型，超过自研模型ppliteseg精度和速度平衡，Cityscapes上精度直逼高精度OCRNet，数据和模型、代码均已经开源。


## 2、功能目标
为PaddleSeg 添加该模型，达到论文Table.6中的指标，进行TIPC验证lite train lite infer 链条，参考PR提交规范提交代码PR到ppseg中。

## 3、意义

为PaddleSeg 增加实时语义分割SOTA模型。


# 二、业内方案调研

PIDNet 源码已经开源，地址：https://github.com/XuJiacong/PIDNet

性能表现如下

| Model (Cityscapes) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| PIDNet-S | [78.8](https://drive.google.com/file/d/1JakgBam_GrzyUMp-NbEVVBPEIXLSCssH/view?usp=sharing) | [78.6](https://drive.google.com/file/d/1VcF3NXLQvz2qE3LXttpxWQSdxTbATslO/view?usp=sharing) | 93.2 |
| PIDNet-M | [79.9](https://drive.google.com/file/d/1q0i4fVWmO7tpBKq_eOyIXe-mRf_hIS7q/view?usp=sharing) | [79.8](https://drive.google.com/file/d/1wxdFBzMmkF5XDGc_LkvCOFJ-lAdb8trT/view?usp=sharing) | 42.2 |
| PIDNet-L | [80.9](https://drive.google.com/file/d/1AR8LHC3613EKwG23JdApfTGsyOAcH0_L/view?usp=sharing) | [80.6](https://drive.google.com/file/d/1Ftij_vhcd62WEBqGdamZUcklBcdtB1f3/view?usp=sharing) | 31.1 |

| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| PIDNet-S |-| [80.1](https://drive.google.com/file/d/1h3IaUpssCnTWHiPEUkv-VgFmj86FkY3J/view?usp=sharing) | 153.7 |
| PIDNet-M |-| [82.0](https://drive.google.com/file/d/1rNGTc8LD42h8G3HaedtqwS0un4_-gEbB/view?usp=sharing) | 85.6 |

# 四、对比分析

参考官方原码实现即可。

# 五、设计思路与实现方案

## 总体思路
### Edge label 实现

PIDNet 中需要额外生成 edge label 以对边缘部分进行监督，虽然 PaddleSeg 中已经内置了 edge label 的方式，但是具体实现细节差距较大，因此需要单独添加该生成方法。

另外 PIDNet 中 edge label 跟随 label 和 image 一起参与数据增强，而 PaddleSeg 中默认行为为使用增强过后的 label 生成 edge label，因此可以考虑单独增加一个 transform,代码如下

```python
@manager.TRANSFORMS.add_component
class AddEdgeLabel:
    def __call__(self, data):
        edge = cv2.Canny(data['label'], 0.1, 0.2)
        kernel = np.ones((4, 4), np.uint8)
        edge = edge[6:-6, 6:-6]
        edge = np.pad(edge, ((6,6),(6,6)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        data['gt_fields'].append('edge')
        data['edge'] = edge
        return data
```

### LOSS 对齐

PIDNet 中使用了多个loss，其中 sem_loss 为 cross_entropy 和 ohem 的组合，与 PaddleSeg 有以下冲突：
1. cross_entropy 的 reduction 参数为 False，即维持输入的形状；
2. ohem 中使用了 class weight。

考虑修改如下：
为 PaddleSeg 的CrossEntropyLoss 添加 use_post_process  的参数用于控制是否平均输出。
```python
    if self.use_post_process:
        return self._post_process_loss(logit, label, semantic_weights, loss)
    return loss
```
为 PaddleSeg 的 OhemCrossEntropyLoss 添加 weight 参数

```python
    if self.weight is not None:
        loss = F.cross_entropy(
            logit, label, weight=self.weight, ignore_index=self.ignore_index, axis=1)
    else:
        loss = F.softmax_with_cross_entropy(
            logit, label, ignore_index=self.ignore_index, axis=1)
```

# 六、测试和验收的考量

达到论文Table.6中的指标，进行TIPC验证lite train lite infer 链条，参考PR提交规范提交代码PR到ppseg中。

# 七、影响面

对其他模块没有影响。

# 八、排期规划

可以在活动时间内完成。
