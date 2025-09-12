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

## 总耗时
`build_and_install_ops` 总耗时：02:54:16

## 头部耗时（Top 片段）
```text
source file name,phase name,arch,tool,metric,unit
append_attention_c8_float16_fp8_kerne.cu,ptxas,sm_80,nvcc,2008004.2500,ms
append_attention_c8_bfloat16_fp8_kernel.cu,cicc,compute_80,nvcc,1752222.7500,ms
decode_attention_kernel.cu,ptxas,sm_80,nvcc,1746218.5000,ms
append_attention_c8_float16_int8_kerne.cu,cicc,compute_80,nvcc,1697108.6250,ms
append_attention_c8_float16_fp8_kerne.cu,cicc,compute_80,nvcc,1607405.0000,ms
append_attention_c8_bfloat16_int8_kernel.cu,cicc,compute_89,nvcc,1562781.7500,ms
decode_attention_kernel.cu,cicc,compute_80,nvcc,1520327.3750,ms
decode_attention_kernel.cu,cicc,compute_89,nvcc,1515748.6250,ms
append_attention_c8_bfloat16_fp8_kernel.cu,cicc,compute_89,nvcc,1502785.7500,ms
append_attention_c8_bfloat16_int8_kernel.cu,cicc,compute_80,nvcc,1500482.0000,ms
decode_attention_kernel.cu,ptxas,sm_89,nvcc,1497204.5000,ms
append_attention_c8_bfloat16_bfloat16_kernel.cu,cicc,compute_89,nvcc,1484485.7500,ms
append_attention_c8_bfloat16_fp8_kernel.cu,ptxas,sm_80,nvcc,1484273.0000,ms
append_attention_c8_bfloat16_bfloat16_kernel.cu,cicc,compute_80,nvcc,1400558.2500,ms
append_attention_c8_float16_int8_kerne.cu,cicc,compute_89,nvcc,1380351.1250,ms
append_attention_c8_float16_fp8_kerne.cu,cicc,compute_89,nvcc,1306988.3750,ms
append_attention_c8_bfloat16_int8_kernel.cu,ptxas,sm_89,nvcc,1302305.3750,ms
append_attention_c8_float16_int8_kerne.cu,ptxas,sm_89,nvcc,1298710.8750,ms
append_attention_c8_float16_float16_kernel.cu,ptxas,sm_80,nvcc,1291983.7500,ms
append_attention_c8_float16_float16_kernel.cu,cicc,compute_80,nvcc,1251027.0000,ms
append_attention_c8_float16_float16_kernel.cu,cicc,compute_89,nvcc,1247668.0000,ms
append_attention_c8_bfloat16_fp8_kernel.cu,ptxas,sm_89,nvcc,1174523.2500,ms
append_attention_c8_float16_fp8_kerne.cu,ptxas,sm_89,nvcc,1165719.5000,ms
append_attention_c8_float16_int8_kerne.cu,ptxas,sm_80,nvcc,1152774.1250,ms
append_attention_c8_bfloat16_int8_kernel.cu,ptxas,sm_80,nvcc,1096564.0000,ms
append_attention_c8_float16_float16_kernel.cu,ptxas,sm_89,nvcc,1020832.5625,ms
append_attention_c8_bfloat16_bfloat16_kernel.cu,ptxas,sm_89,nvcc,1019829.2500,ms
append_attention_c8_bfloat16_bfloat16_kernel.cu,ptxas,sm_80,nvcc,928371.9375,ms
fast_hardamard_kernel.cu,cicc,compute_80,nvcc,573600.1875,ms
append_attention_c4_bfloat16_fp8_kernel.cu,cicc,compute_80,nvcc,567793.9375,ms
append_attention_c4_float16_fp8_kernel.cu,cicc,compute_80,nvcc,564942.8125,ms
append_attention_c4_bfloat16_fp8_kernel.cu,ptxas,sm_80,nvcc,543922.3750,ms
append_attention_c4_float16_fp8_kernel.cu,ptxas,sm_80,nvcc,515630.1250,ms
scaled_mm_c2x.cu,cicc,compute_89,nvcc,502628.6250,ms
append_attention_c4_float16_int8_kernel.cu,cicc,compute_89,nvcc,502232.7812,ms
append_attention_c4_float16_int8_kernel.cu,cicc,compute_80,nvcc,501183.1875,ms
append_attention_c4_bfloat16_int8_kernel.cu,cicc,compute_89,nvcc,500279.3438,ms
append_attention_c4_bfloat16_int8_kernel.cu,cicc,compute_80,nvcc,497305.6250,ms
append_attention_c4_bfloat16_bfloat16_kernel.cu,cicc,compute_89,nvcc,474681.0000,ms
append_attention_c4_bfloat16_fp8_kernel.cu,cicc,compute_89,nvcc,474440.0625,ms
append_attention_c4_bfloat16_bfloat16_kernel.cu,cicc,compute_80,nvcc,472783.8750,ms
append_attention_c4_float16_fp8_kernel.cu,cicc,compute_89,nvcc,472267.0312,ms
fast_hardamard_kernel.cu,ptxas,sm_80,nvcc,472075.9375,ms
append_attention_c4_float16_float16_kernel.cu,cicc,compute_89,nvcc,470359.0938,ms
append_attention_c4_float16_float16_kernel.cu,cicc,compute_80,nvcc,464772.2500,ms
append_attention_c4_bfloat16_int8_kernel.cu,ptxas,sm_80,nvcc,430425.4688,ms
fast_hardamard_kernel.cu,ptxas,sm_89,nvcc,407123.0938,ms
append_attention_c4_float16_int8_kernel.cu,ptxas,sm_89,nvcc,397793.7500,ms
append_attention_c4_bfloat16_int8_kernel.cu,ptxas,sm_89,nvcc,394105.1562,ms
append_attention_c4_bfloat16_bfloat16_kernel.cu,ptxas,sm_80,nvcc,363122.6562,ms
append_attention_c4_float16_int8_kernel.cu,ptxas,sm_80,nvcc,355433.2812,ms
append_attention_c4_float16_float16_kernel.cu,ptxas,sm_80,nvcc,350369.5000,ms
append_attention_c4_bfloat16_fp8_kernel.cu,ptxas,sm_89,nvcc,346578.8438,ms
scaled_mm_c2x.cu,cicc,compute_80,nvcc,337077.0938,ms
append_attention_c4_float16_fp8_kernel.cu,ptxas,sm_89,nvcc,336544.1250,ms
append_attention_c4_bfloat16_bfloat16_kernel.cu,ptxas,sm_89,nvcc,330367.9688,ms
fused_moe_gemm_kernels_bf16_int2.cu,cicc,compute_89,nvcc,328741.7188,ms
fused_moe_gemm_kernels_bf16_int2.cu,cicc,compute_80,nvcc,328112.0938,ms
fused_moe_gemm_kernels_fp16_int2.cu,cicc,compute_80,nvcc,325256.0625,ms
append_attention_c4_float16_float16_kernel.cu,ptxas,sm_89,nvcc,323306.5000,ms
fused_moe_gemm_kernels_fp16_int2.cu,cicc,compute_89,nvcc,319977.8438,ms
append_attention_c16_float16_fp8_kernel.cu,cicc,compute_80,nvcc,312203.8125,ms
append_attention_c16_bfloat16_fp8_kernel.cu,cicc,compute_80,nvcc,312040.7812,ms
launch_dual_gemm_kernel_block128x128x64_warp64x32x64_mma16x8x32_stage7.cu,cicc,compute_80,nvcc,297165.3438,ms
append_attention_c16_float16_fp8_kernel.cu,ptxas,sm_80,nvcc,278560.8438,ms
append_attention_c16_bfloat16_fp8_kernel.cu,ptxas,sm_80,nvcc,272249.3438,ms
append_attention_c16_float16_int8_kernel.cu,cicc,compute_89,nvcc,261229.5938,ms
append_attention_c16_bfloat16_int8_kernel.cu,cicc,compute_89,nvcc,257425.1094,ms
w4a8_moe_cutlass_kernel_template.cu,cicc,compute_89,nvcc,252737.7031,ms
append_attention_c16_float16_fp8_kernel.cu,cicc,compute_89,nvcc,252025.4219,ms
append_attention_c16_bfloat16_fp8_kernel.cu,cicc,compute_89,nvcc,251701.8125,ms
w4a8_moe_cutlass_kernel_template.cu,cicc,compute_80,nvcc,251636.0625,ms
append_attention_c16_float16_int8_kernel.cu,cicc,compute_80,nvcc,247156.0938,ms
append_attention_c16_bfloat16_int8_kernel.cu,cicc,compute_80,nvcc,246625.9062,ms
launch_dual_gemm_kernel_block128x128x64_warp64x32x64_mma16x8x32_stage6.cu,cicc,compute_80,nvcc,233258.9219,ms
append_attention_c16_bfloat16_bfloat16_kernel.cu,cicc,compute_89,nvcc,228208.2500,ms
append_attention_c16_float16_float16_kernel.cu,cicc,compute_89,nvcc,228167.5312,ms
append_attention_c16_bfloat16_bfloat16_kernel.cu,cicc,compute_80,nvcc,226980.7656,ms
append_attention_c16_float16_float16_kernel.cu,cicc,compute_80,nvcc,226857.9688,ms
launch_dual_gemm_kernel_block128x128x64_warp64x32x64_mma16x8x32_stage5.cu,ptxas,sm_80,nvcc,199075.8906,ms
append_attention_c16_bfloat16_int8_kernel.cu,ptxas,sm_89,nvcc,190082.9375,ms
launch_dual_gemm_kernel_block128x128x64_warp128x32x64_mma16x8x32_stage8.cu,ptxas,sm_89,nvcc,188538.6406,ms
append_attention_c16_float16_int8_kernel.cu,ptxas,sm_89,nvcc,187910.3906,ms
scaled_mm_c2x.cu,ptxas,sm_89,nvcc,187419.3125,ms
launch_dual_gemm_kernel_block128x128x64_warp128x32x64_mma16x8x32_stage7.cu,cc (compiling),compute_89,nvcc,184058.9375,ms
fast_hardamard_kernel.cu,cicc,compute_89,nvcc,180076.5000,ms
fused_moe_gemm_kernels_bf16_int8.cu,cicc,compute_80,nvcc,179963.7969,ms
fused_moe_gemm_kernels_bf16_int8.cu,cicc,compute_89,nvcc,179362.7500,ms
fused_moe_gemm_kernels_fp16_int8.cu,cicc,compute_89,nvcc,176674.0312,ms
fused_moe_gemm_kernels_fp16_int4.cu,cicc,compute_89,nvcc,174325.0156,ms
fused_moe_gemm_kernels_fp16_int8.cu,cicc,compute_80,nvcc,174235.6406,ms
launch_dual_gemm_kernel_block128x128x64_warp64x32x64_mma16x8x32_stage8.cu,cicc,compute_80,nvcc,171859.5312,ms
fused_moe_gemm_kernels_bf16_int4.cu,cicc,compute_80,nvcc,170815.7344,ms
fused_moe_gemm_kernels_bf16_int4.cu,cicc,compute_89,nvcc,169237.5312,ms
fused_moe_gemm_kernels_fp16_int4.cu,cicc,compute_80,nvcc,168141.9219,ms
append_attention_c16_bfloat16_int8_kernel.cu,ptxas,sm_80,nvcc,164357.3594,ms
beam_search_softmax.cu,ptxas,sm_89,nvcc,163484.0625,ms
append_attention_c16_float16_int8_kernel.cu,ptxas,sm_80,nvcc,161471.2969,ms
launch_dual_gemm_kernel_block128x128x64_warp64x32x64_mma16x8x32_stage5.cu,cicc,compute_89,nvcc,159697.6250,ms
append_attention_c16_bfloat16_fp8_kernel.cu,ptxas,sm_89,nvcc,154251.7344,ms
```


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
