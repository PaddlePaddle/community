# dac-loss——设计文档

| 任务名   | PaddleSpeech--dac-loss |
| -------- | ------------------------ |
| 提交作者 | suzakuwcx                |
| 提交时间 | 2024-11-30               |
| 版本号   | v1.0                     |
| 依赖     | main 版本                |
| 文件名   | dac-loss.md    |

# 一、概述

## 1、相关背景

见 [Hackathon 7th Q.56](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E5%A5%97%E4%BB%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md)

为了在 Paddle 中复现论文 [High-Fidelity Audio Compression with Improved RVQGAN](http://arxiv.org/abs/2306.06546), 以及促进 paddle 框架在语音领域的发展, 需要实现为复现论文需要
用到的 Loss 函数

## 2、功能目标

1.在 PaddleSpeech 中的 paddlespeech/t2s/modules/losses.py 文件中放置对应的实现

## 3、意义

- 扩大 PaddleSpeech 中对于语音类模型的各类函数的支持

# 二、飞桨现状

目前 Paddle 框架尚无相关功能

# 三、业内方案调研

dac 代码地址如下：https://github.com/descriptinc/descript-audio-codec, 其 loss 函数的实现代码位于 https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py

# 四、设计思路与实现方案

## 总体思路

- 参考 paddlespeech/kws 的实现，利用 paddleaudio 实现好的 audiotools 中的功能，将 dac 中的损失函数重新进行实现，将内部的函数全部换成 paddleaudio 和 paddlespeech 内部的实现，保证以下接口可以访问

```
from paddlespeech.t2s.modules.losses import MultiScaleSTFTLoss，GANLoss，SISDRLoss
```

其中， AudioSignal 须使用 paddleaudio 内部的实现

# 五、测试和验收的考量

- 首先在原 dac 仓库进行一次完完整的训练流程，抓出其中的多次经过 Loss 函数之后的输入和输出作为测试用例，在 tests/units/ 下新增 t2s 文件夹，同时添加文件 test_losses.py 作为单测, 单测必须全部通过

- 安装 paddlespeech 后，以下代码须能够正常运行

```
from paddlespeech.t2s.modules.losses import MultiScaleSTFTLoss，GANLoss，SISDRLoss
```

# 六、可行性分析和排期规划

预计在 12 月完成 pr 合入

# 七、影响面

- 在 paddlespeech/t2s/modules/losses.py 中增加 MultiScaleSTFTLoss，GANLoss，SISDRLoss

- 新增文件夹 tests/units/

- 新增单测文件 tests/units/test_losses.py

- 修改文件 tests/unit/ci.sh
