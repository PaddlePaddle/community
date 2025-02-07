# audiotools——设计文档

| 任务名   | PaddleSpeech--audiotools |
| -------- | ------------------------ |
| 提交作者 | suzakuwcx                |
| 提交时间 | 2024-11-16               |
| 版本号   | v1.0                     |
| 依赖     | main 版本                |
| 文件名   | rfc0001-audiotools.md    |

# 一、概述

## 1、相关背景

见 [Hackathon 7th Q.55](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E5%A5%97%E4%BB%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md)

为了在 Paddle 中复现论文 [High-Fidelity Audio Compression with Improved RVQGAN](http://arxiv.org/abs/2306.06546), 以及促进 paddle 框架在语音领域的发展, 需要在 PaddleSpeech 中集成工具 [audiotools](https://github.com/descriptinc/audiotools) 的功能

## 2、功能目标

1.在 PaddleSpeech 中新增 audiotools 包，要求与原仓库接口兼容，且精度对标原仓库

## 3、意义

- 扩充PaddleOcr中的语音预处理能力

# 二、飞桨现状

目前 Paddle 框架尚无相关功能,

# 三、业内方案调研

audiotool 代码地址如下：https://github.com/descriptinc/audiotools

dac 代码地址如下：https://github.com/descriptinc/descript-audio-codec

# 四、对比分析

在原仓库的代码基础上进行移植是目前的最佳实践方式，同时需要将原仓库框架设计 torch 的部分转成 paddle, 且尽可能在保证 paddle 外部依赖不变的情况下实现其功能

# 五、设计思路与实现方案

## 总体思路

- audiotools 中需要将多种不同的语音源转换成 AudioSignal 格式，其内部使用 torch 进行转换操作，这里需要将 torch 的实现替换为 paddle 版本，然后保证其流程和输入输出保持一致

# 六、测试和验收的考量

相关 audiotools 的功能都需要实现

# 七、可行性分析和排期规划

预计在 12 月完成 pr 合入

# 八、影响面

- 在 PaddleSpeech/audio/paddleaudio 下新增 audiotools 目录

# 名词解释

# 附件及参考资料
