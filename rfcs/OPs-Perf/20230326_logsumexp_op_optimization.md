# Logsumexp OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-26                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20230326_logsumexp_op_optimization.md<br> |


# 1 背景与意义

目前 Paddle 内 logsumexp 算子 GPU 计算采用了Eigen库实现，性能仍有明显的提升空间。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

目前的实现有一定的性能优化空间，可以加入一些性能优化的技巧。当前forward性能如下表：

| Case No. | device | input_shape | input_type | Paddle Perf(ms) |
|---|---|---|---|---|
| 1 | RTX 2070s | [64L, 64L] | float32 | 0.0681 | 
| 2 | RTX 2070s | [1024L, 512L] | float32 | 0.67155 |
| 3 | RTX 2070s | [64L, 64L] | float16 | 0.06718 |
| 4 | RTX 2070s | [1024L, 512L] | float16 | 0.64455 |


API文档 https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logsumexp_cn.html

## 1.2 业内方案调研

Pytorch中API文档：　https://pytorch.org/docs/1.12/generated/torch.logsumexp.html?highlight=logsumexp#torch.logsumexp

Pytorch中sum_out可基于cuda计算, forward整体性能如下(基于pytorch　v1.12)：

| Case No. | device | input_shape | input_type | Pytorch Perf(ms) |
|---|---|---|---|---|
| 1 | RTX 2070s | [64L, 64L] | float32 | 0.03757 | 
| 2 | RTX 2070s | [1024L, 512L] | float32 | 0.05742 |
| 3 | RTX 2070s | [64L, 64L] | float16 | 0.04035 |
| 4 | RTX 2070s | [1024L, 512L] | float16 | 0.05294 |

## 1.3 对比分析

目前Paddle与Pytorch的API设计方案相似，4种case下测试Pytorch性能更优,
二者主要差别是Paddle采用的是Eigen方式计算，🕑然而Pytorch中基于cuda可明显提升性能。

pytorch中主要实现代码：

```c++
static Tensor& logsumexp_out_impl(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = at::amax(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    at::sum_out(result, (self - maxes).exp_(), dims, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    at::sum_out(result, at::exp(self), dims, keepdim);
    result.log_();
  }
  return result;
}
```
# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

基于Paddle中已封装好的Reduce及Elementwise，二者充分利用了向量化读写操作的优秀性能，已做了初步测试，优化的性能能够超出预期。

## 2.2 Host端计算流程

计算reduce的axis，keepdim等，调用reduce算子封装好的接口。

## 2.4 Device端计算流程

基于现有的kps算子进行组装即可。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-03-26 |
| 2 | 完成开发文档设计  | 2023-03-26 |
| 3 | logsumexp优化实现  | 2023-03-26 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-03-31 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)


