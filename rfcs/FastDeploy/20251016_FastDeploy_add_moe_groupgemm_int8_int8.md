# FastDeploy中的MoE GroupGEMM支持INT8*INT8实现

| 方案名称                         |  FastDeploy中的MoE GroupGEMM支持INT8*INT8实现  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | WanRui37                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2025-10-16                              | 
| 版本号                                                      | V1.1                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==3.2.0                     | 
| 文件名                                                      | 20251016_FastDeploy_add_moe_groupgemm_int8_int8.md<br> | 

# 一、概述
## 1、相关背景
大规模模型在自然语言处理、计算机视觉等领域取得了显著成果。其中，混合专家模型（Mixture of Experts，MoE）作为一种高效的模型架构，通过将输入数据分配给不同的专家子网络进行处理，能够有效提升模型的性能和计算效率。在MoE模型的训练和推理过程中，GroupGEMM（Group General Matrix Multiply）操作是核心计算步骤

## 2、功能目标
为FastDeploy 开发高性能 MoE算子(INT8*INT8)，将上述算子集成到EB、Qwen等开源模型中。

## 3、意义
支持INT8*INT8的MoE GroupGEMM实现能够充分利用硬件的整数计算单元，相较于高精度计算，大幅减少计算延迟，提高模型推理速度。

# 二、FastDeploy现状
- 目前`FastDeploy`中`MoE GroupGEMM`没有支持`INT8*INT8`的实现

# 三、业内方案调研
- 目前业内`MoE GroupGEMM`没有支持`INT8*INT8`的实现

# 四、设计思路与实现方案
1. 一些参考的代码路径
    1. `sm89`架构可以参考
        - `24_gemm_grouped`: `FastDeploy/custom_ops/third_party/cutlass/examples/24_gemm_grouped`
        - `64_ada_fp8_gemm_grouped`: `FastDeploy/custom_ops/third_party/cutlass/examples/64_ada_fp8_gemm_grouped`
            - 比较重要的就是`cutlass::gemm::device::GemmGrouped`
    1. `sm90`架构可以参考
        - `w4afp8_gemm`: `FastDeploy/custom_ops/gpu_ops/w4afp8_gemm`
            - 比较重要的是`cute::gemm`
        - `57_hopper_grouped_gemm`: `FastDeploy/custom_ops/third_party/cutlass/examples/57_hopper_grouped_gemm`
            - 比较重要的是`cutlass::gemm::device::GemmUniversalAdapter`

1. 需要修改的代码路径
    ```text
    custom_ops/                           
    └── gpu_ops/                            # GPU相关自定义算子
        ├── int8_gemm_with_cutlass/         # INT8*INT8 GEMM with Cutlass 算子实现
        │   ├── w8a8_group_gemm.cu          # Cutlass Kernel实现
        │   └── w8a8_group_gemm.h           # Cutlass Kernel头文件
        └── ...                             
    test/
    └── operators/
        ├── test_w8a8_group_gemm.py          # 测试INT8*INT8 GEMM with Cutlass 算子
        └── ...
    ```

# 五、测试和验收的考量
- 增加算子测试
- 在EB，Qwen开源模型上测试数据精度&性能

# 六、影响面
为`FastDeploy`集成`MoE GroupGEMM`，不影响其他部分

# 七、排期规划
* 2025-10-16 ~ 2025-11-16：完成集成代码开发
* 2025-11-16 ~ 2025-11-25：完成代码测试
* 2025-11-25 ~ 2025-12-01： 完成部署示例及文档

# 八、参考资料

[Accelerating MoE's with a Triton Persistent Cache-Aware Grouped GEMM Kernel](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)

[上述为 vllm 增加 BF16 Grouped GEMM Kernel 的 PR](https://github.com/vllm-project/vllm/pull/19443)