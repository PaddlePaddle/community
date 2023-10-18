# NowcastNet 设计文档

|              |                            |
|--------------|----------------------------|
| 提交作者     | co63oc                     |
| 提交时间     | 2023-10-18                 |
| RFC 版本号   | v1.0                       |
| 依赖飞桨版本 | develop/release 2.5.0 版本 |
| 文件名       | 20231018_nowcastnet.md       |

## 1. 概述

### 1.1 相关背景

[No.61：Skillful nowcasting of extreme precipitation with NowcastNet](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/%E3%80%90PaddlePaddle%20Hackathon%205th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no61skillful-nowcasting-of-extreme-precipitation-with-nowcastnet)

气象雷达回波以低于 2 公里的空间分辨率和高达 5 分钟的时间分辨率提供云观测，非常适合降水临近预报。利用这些数据的自然选择是数值天气预报，它根据求解大气耦合原始方程来预测降水。然而，这些方法即使在超级计算平台上实施，也将数值天气预报的更新周期限制在几小时，空间分辨率限制在中尺度，而极端天气过程通常表现出数十分钟的寿命和对流尺度的个体特征。DARTS10 和 pySTEPS9 等替代方法基于仅受连续性方程启发的平流方案。这些方法分别求解运动场的未来状态和来自复合雷达观测的强度残差，并迭代地平流过去的雷达场以预测未来的场。平流方案部分遵守降水演化的物理守恒定律，能够在 1h，但超过该范围后，它很快就会退化，导致较高的位置误差，并失去小的对流特征。由于现有的平流实现未能纳入非线性演化模拟和端到端的预测误差优化，这些误差以不受控制的方式在自回归平流过程中累积。

### 1.2 功能目标

原仓库代码没有训练和评估过程，使用的为预训练权重，结果为生成预测图片，修改为转换预训练权重为 paddle 格式，使用 paddle 运行生成图片。
复现 mrms_case 和 mrms_large_case，并合入 PaddleScience

### 1.3 意义

复现 NowcastNet 模型，能够使用 NowcastNet 模型进行推理。

## 2. PaddleScience 现状

PaddleScience 套件暂无 NowcastNet 模型案例。

## 3. 目标调研

参考代码 https://codeocean.com/capsule/3935105/tree/v1
论文链接 https://www.nature.com/articles/s41586-023-06184-4#Abs1

原代码为 Pytorch 代码，需要在 PaddleScience 中复现，复现的主要问题是模型的转换，数据集读取的转换，使用 PaddleScience 的 API 调用。

## 4. 设计思路与实现方案

参考已有代码实现 NowcastNet
1. 模型构建
2. 读取预训练权重
3. 超参数设定
4. 生成预测图片

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

原代码使用脚本 mrms_case_test.sh，mrms_large_case_test.sh 运行，复现需要运行同样脚本生成相应图片，并使用 PaddleScience 复现

## 6. 可行性分析和排期规划

参考代码修改为 paddle 实现，使用 PaddleScience API，测试精度对齐
202309：调研
202310：基于 Paddle API 的复现，基于 PaddleScience 的复现
202311：整理项目产出，撰写案例文档

## 7. 影响面

在 ppsci.arch 下新增 NowcastNet 模型
