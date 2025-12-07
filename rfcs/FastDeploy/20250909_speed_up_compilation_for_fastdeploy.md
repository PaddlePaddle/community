# FastDeploy编译加速设计文档

| 任务名称 | FastDeploy 编译加速 |
|------|------|
| 提交作者 | ccsuzzh |
| 提交时间 | 2025-09-09 |
| 版本号 | V1.0 |
| 文件名 | 20250909_speed_up_compilation_for_fastdeploy.md |

# 一、概述

## 1、相关背景
随着大语言模型推理部署需求的增加，FastDeploy 功能与自定义算子库快速扩张，带来显著的编译压力与 CI 用时上升。

## 2、功能目标
* 提升源码整体编译速度与增量编译效率

## 3、意义
提高开发效率，加快CI流水线的速度。

# 二、现状分析
- 当前自定义算子库 `fastdeploy_ops.so` 主要通过 setuptools 直接编译。少量目标时尚可，但随着算子与依赖的膨胀，整体编译速度显著下降。
- 采用 `nvcc --time` 统计各 `.cu` 文件编译耗时，定位编译瓶颈。

## 测试环境
| 项目 | 配置 |
|------|------|
| CPU | Intel® Core™ i5-10600KF @ 4.10GHz × 12 |
| OS | Ubuntu 20.04.6 LTS |
| RAM | 64 GB |
| GPU | GeForce RTX 4060 Ti 16 GB |

## 编译命令
```bash
bash build.sh 1 python false [80]
```

## 头部编译耗时单元统计
`build_and_install_ops` 总耗时：02:54:16s，而耗时最多的15个文件如下：

| 文件名 | 耗时占比 |
|--------|---------:|
| append_attention_c8_float16_fp8_kerne.cu | 2.88% |
| decode_attention_kernel.cu | 2.60% |
| append_attention_c8_bfloat16_fp8_kernel.cu | 2.58% |
| append_attention_c8_float16_int8_kerne.cu | 2.27% |
| append_attention_c8_bfloat16_int8_kernel.cu | 2.07% |
| append_attention_c8_float16_float16_kernel.cu | 2.02% |
| append_attention_c8_bfloat16_bfloat16_kernel.cu | 1.86% |
| append_attention_c4_bfloat16_fp8_kernel.cu | 0.87% |
| append_attention_c4_float16_fp8_kernel.cu | 0.86% |
| fast_hardamard_kernel.cu | 0.83% |
| append_attention_c4_bfloat16_int8_kernel.cu | 0.74% |
| append_attention_c4_float16_int8_kernel.cu | 0.68% |
| append_attention_c4_bfloat16_bfloat16_kernel.cu | 0.67% |
| append_attention_c4_float16_float16_kernel.cu | 0.65% |
| append_attention_c16_float16_fp8_kernel.cu | 0.47% |

上诉述头部耗时单元数量仅为总编译文件总数的 3.5%，而编译却占总耗时的 22.23%，这正是需要编译加速的重点文件。

# 三、业内方案调研
主流如 vLLM 采用 setuptools.extension 配合 CMake（实际由 CMake + Ninja 构建），具备更好的增量编译、依赖管理与并行能力；同时结合 `ccache/sccache` 复用编译产物，并通过 `NVCC_THREADS` 提升 nvcc 内部并行度，降低大型 CUDA 文件单次编译耗时。

# 四、设计思路与实现方案

## 总体思路
## 1) 构建系统与并行
- 从 setuptools 直接编译切换为 CMake + Ninja 驱动编译
- 开启 `NVCC_THREADS` 提升 nvcc 内部并行度（结合机器核心数做上限）

## 2) 编译缓存
- 如果不能继续复用Paddle主框架的setup，则还需要额外对`ccache/sccache`进行支持

## 3) 源码层面优化
- 降低头文件依赖与模板实例数量，控制编译单元体积
- 拆分超大 `.cu` 文件，按功能/数据类型切分为更细粒度目标
- 优化内核复杂度或编译选项

# 五、可行性分析和排期规划
* 学习业界主流框架的编译流程和优化方法，可靠性已经得到验证。
* 预计一周内完成编译流程的优化，在9月底完成任务收尾工作。

# 六、影响面
在保证正确性的前提下进行源码与构建组织优化，不改变功能；对开发者与 CI 的主要影响为构建速度与流程优化。
