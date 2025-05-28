# paddle-sparse 设计文档

> RFC 文档相关记录信息

|              |                    |
| ------------ | ------------------ |
| 提交作者     | BeingGod          |
| 提交时间     | 2025-05-27         |
| RFC 版本号   | v1.0               |
| 依赖飞桨版本 | 3.0 版本 |
| 文件名       | 20250527_paddle_sparse|

## 1. 概述

### 1.1 相关背景

[Fundable Projects 科学计算方向开源工具组件 pytorch_sparse 适配](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_8th/%E3%80%90Hackathon_8th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E5%9B%9B%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E6%96%B9%E5%90%91%E5%BC%80%E6%BA%90%E5%B7%A5%E5%85%B7%E7%BB%84%E4%BB%B6-pytorch_sparse-%E9%80%82%E9%85%8D)

### 1.2 功能目标

1. 实现 SparseTensor 类及其成员函数，另外需要实现该类的 sharememory、is_shared、to、cpu、cuda、getitem、repr、from_scipy、to_scipy、index_select、index_select_nnz、masked_select、masked_select_nnz 等函数，实现精度与性能对齐；
2. 实现 SparseStorage 类及其成员函数，另外需要实现该类的 share_memor_、is_shared 等函数，实现精度与性能对齐；
3. 实现对应的单元测试；


### 1.3 意义

pytorch_sparse是一个专门为 PyTorch 框架设计的扩展库，它提供了对稀疏张量（SparseTensor）的高效操作和优化。稀疏张量在处理具有大量零值的数据时非常有用，能够显著减少内存占用和提高计算效率，在SOTA模型MatterGen、GemNet、DimeNet++等多个模型中均有使用。PaddlePaddle 支持稀疏 Tensor，但是与该库的功能仍有较大差距。因此为了扩充 PaddlePaddle 对稀疏矩阵的支持，对 pytorch_sparse 基于 PaddlePaddle 进行实现。。

## 2. PaddleScience 现状

torch-sparse 依赖 torch-scatter，其中 torch-scatter 已经完成了相应的适配 [paddle-scatter](https://github.com/PFCCLab/paddle_scatter)。

## 3. 目标调研

### 3.1 适配版本

|      组件     |  版本    |
| ------------ | ------------------ |
| paddle | 3.0 |
| pytorch-sparse | 6f86680 |
| paddle-scatter | f1af3ff |

### 3.2 跑通 torch-sparse

image: paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5

1. 安装

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install torch-scatter (torch-sparse 依赖 torch-scatter)
git clone https://github.com/rusty1s/pytorch_scatter.git
python setup.py install

# Install torch-sparse
git clone https://github.com/rusty1s/pytorch_sparse.git
git submodule update --init --recursive
export CUDA_HOME=/usr/local/cuda
python setup.py install
```

2. 单元测试

```bash
pip install pytest
pytest
```

### 3.3 相关单测梳理

本次适配**重点实现 SparseTensor 类对应接口**，部分无依赖接口暂时不实现，对应单测暂时跳过。

|      单测     |  是否需要验证    |
| ------------ | ------------------ |
| test_add.py                |Y|
| test_cat.py    |Y|
| test_coalesce.py|Y|
| test_convert.py|Y|
| test_diag.py|N|
| test_ego_sample.py|N|
| test_eye.py|Y|
| test_matmul.py|N|
| test_metis.py|N|
| test_mul.py|Y|
| test_neighbor_sample.py|N|
| test_overload.py|Y|
| test_permute.py|Y|
| test_saint.py|N|
| test_sample.py|N|
| test_spmm.py|N|
| test_spspmm.py|N|
| test_storage.py|Y|
| test_tensor.py|Y|
| test_transpose.py|Y|


## 4. 设计思路与实现方案

1. 梳理 SparseTensor 依赖算子及对应单测
2. 完成 ind2ptr/ptr2ind 算子接入，完成 SparseStorage 适配，并跑通对应单测
3. 完成 SparseTensor 类依赖底层 C++ 算子接入
4. 完成 SparseTensor 类 python 接口改写，跑通对应单测
5. 优化 SparseTensor 对应 API 性能，性能对齐 torch

### 4.1 补充说明[可选]

shared_memory 和 is_shared 接口依赖主框架，不影响功能实现，暂时绕过

## 5. 测试和验收的考量

1. 实现 SparseTensor 及其相关成员函数；
2. 实现 SparseStorage 类及其相关成员函数；
3. 性能与精度与 Torch 对齐，提供对齐代码及结果；
4. 实现其中对应的单元测试并通过；
5. 最终代码合入 PFCCLab 组织下；

## 6. 可行性分析和排期规划

- 2025.5：调研，复现代码并作调整
- 2025.6：整理项目产出，撰写案例文档

## 7. 影响面

1. 在 PFCC 下添加 paddle_sparse 仓库, 并作为第三方库集成到 PaddleScience 内(https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/install_setup/#143)
