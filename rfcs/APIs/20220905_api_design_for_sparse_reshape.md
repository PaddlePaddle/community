# paddle.incubate.sparse.reshape 设计文档

| API名称                                                    | paddle.incubate.sparse.reshape                | 
|----------------------------------------------------------|-----------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | PeachML                                       | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-09-05                                    | 
| 版本号                                                      | V1.0                                          | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                       | 
| 文件名                                                      | 20220905_api_design_for_sparse_reshape.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 reshape 的计算逻辑，
一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

## 3、意义

支持稀疏tensor的reshape操作，丰富基础功能，提升稀疏tensor的API完整度。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中无相关实现

```c
Tensor reshape(const Tensor& self, IntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    AT_ERROR("reshape is not implemented for sparse tensors");
  }
```


## paddle DenseTensor

1. -1 表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
2. 0 表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过x的维度。
3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]，则将x变换为形状为[6,8]的2-D张量，且x的数据保持不变。
4. 给定一个形状为[2,4,6]的三维张量x，目标形状为[2,3,-1,2]，则将x变换为形状为[2,3,4,2]的4-D张量，且x的数据保持不变。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
5. 给定一个形状为[2,4,6]的三维张量x，目标形状为[-1,0,3,2]，则将x变换为形状为[2,4,3,2]的4-D张量，且x的数据保持不变。在这种情况下，0对应位置的维度值将从x的对应维数中复制,-1对应位置的维度值由x的元素总数和剩余维度推断出来。

### 实现方法

代码如下

```c
DenseTensor& DenseTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  return *this;
}
```
但是此处是Dense的，直接使用指针在Sparse中不可行

# 四、对比分析

为了适配paddle phi库的设计模式，需自行设计实现方式

# 五、方案设计

## 命名与参数设计

在 paddle/phi/kernels/sparse/cpu/sparse_utils_kernel.cc 中， kernel设计为

```    
template <typename T, typename Context>
void SparseCooResize(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const DDim& dims,
                    SparseCooTensor* out) {
```

```    
template <typename T, typename Context>
void SparseCsrResize(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const DDim& dims,
                    SparseCsrTensor* out) {
```

并在yaml中新增对应API

## 底层OP设计

对于Coo格式，实现对应的 CPU Kernel，使用 FlattenIndices 和 IndexToCoordinate 两个已有 func，将SparseTensor中的非零元素拍成一维后再重整形状

对于Csr格式，转换成Coo再进行处理

## API实现方案

对于SparseCsrTensor，将csr格式转换成coo格式再进行运算，然后转换回csr格式输出。

对于SparseCooTensor，直接进行运算。

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性
- 不同 `sparse_dim` 

# 七、可行性分析及规划排期

方案主要依赖paddle现有op组合而成，并自行实现核心算法

# 八、影响面

为独立新增op，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
