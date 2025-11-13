# FastDeploy投机解码功能性能优化设计文档

| 任务名称 | 为 FastDeploy 支持投机解码功能 |
|------|------|
| 提交作者 | fuzhenxin |
| 提交时间 | 2025-11-13 |
| 版本号 | V1.0 |
| 文件名 | 20251113_refine_speculate_decoding_ngram_for_fastdeploy.md |

# 一、概述

## 1、相关背景
投机解码（Speculative Decoding）在 FastDeploy 中被用于加速大模型推理，当前已集成 ngram_match 与 hybrid_mtp_ngram 两种算法。
两算法的核心匹配算子均仅提供 CPU 实现，推理时需将 Device Tensor 拷贝到 Host，匹配结束后再拷回 Device，形成同步瓶颈，导致端到端延迟增加。

## 2、功能目标
* 将 ngram_match.cc 与 ngram_match_mixed.cu 中的匹配逻辑重写为 统一、通用、可扩展的 GPU Kernel；
* Kernel 性能在长序列场景下不劣于 CPU 版本
* 保持 API 与现有 Python/C++ 接口 100% 兼容。

## 3、意义
优化后续大模型推理速度，增强投机解码的性能；打通 FastDeploy 在 GPU 推理全链路的零拷贝路径；

# 二、现状分析
- FastDeploy 当前仅提供 CPU 版投机解码核心算子。但是ngram只在 Host 端调用同一 CPU 逻辑，GPU 端仅做内存搬运。推理时必需把 Device 端的 history 与 draft Token 张量同步拷贝到 Host，匹配结束后再把结果拷回 Device，形成一次强制同步。导致 CUDA Stream 空等，GPU 计算资源闲置。随着序列长度增加，拷贝耗时线性增长，成为整体吞吐的硬性瓶颈。因此，亟需将匹配算子迁移至 GPU，实现零拷贝、高并行、无阻塞的端到端加速。

## 测试环境
| 项目 | 配置 |
|------|------|
| CPU | TBD |
| OS | TBD |
| RAM | TBD |
| GPU | TBD |

# 三、设计思路与实现方案

## 总体思路
## 1) 总体原则
- 零拷贝：输入、输出均留在显存；
- 单 Kernel 覆盖两业务：提取公共匹配逻辑，差异通过模板参数/分支常量控制；
- 可扩展：支持 n-gram的可变阈值；

## 2) Kernel设计和实现
- 提取ngram公共逻辑部分，在分别设计对应Kernel及其调用方式
- 完成基础功能并做算法功能对齐

## 3) CPU 退化路径
- 当 GPU 显存不足或序列长度 < 64 时，自动回退到原 CPU 实现；

# 四、可行性分析和排期规划
* 了解当前整体FastDeploy的pipeline和投机解码的功能原理，已基本完成。
* 分析当前ngram的两个实现文件，并设计kennel函数进行实现，预计一周。
* 功能对齐和性能优化&测试，预计一周。

# 五、影响面
在保证正确性的前提下，在较长的匹配下，Kernel 性能优于或基本不劣于目前的 CPU kernel。
