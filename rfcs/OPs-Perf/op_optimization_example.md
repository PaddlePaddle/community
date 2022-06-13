# IndexSample OP性能优化设计文档


| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | xxx   |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2021-01-21 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20210121_index_sample_fwd_op_optimization.md<br> |


# 1 背景与意义
目前，Paddle中暂时没有GPU版本的IndexSample OP，导致训练过程中采用CPU执行对应计算，性能明显存在瓶颈，影响了模型性能。

## 1.1 飞桨现状

Paddle中暂时没有IndexSample OP的GPU实现，需要实现一个GPU版本的IndexSample OP.

## 1.2 业内方案调研

Pytorch中对应`paddle.index_sample` 的Api为 `torch.gather`。调研发现Pytorch中采用的是`_scatter_gather_elementwise_kernel` Kernel完成该OP的GPU实现。PyTorch采用的方案是1维线程设置完成整体计算，整体性能如下：

| Case No. | index_shape | input_shape | Pytorch Perf(ms) |
|---|---|---|---|
| 1 | [5100,1] | [5100,38506] |1.7032 | 
| 2 | [100,64]  |  [100, 128] | 0.0083|
| 3 | [5100,96] | [5100,128]  |0.0377 |

## 1.3 对比分析

由于Paddle中没有GPU实现，考虑到IndexSample的计算多用于2维数据，因此决定采用2维线程设置，减少数据索引部分的计算量，2维线程设置方案如下：

```
  auto block_width = paddle::platform::RoundToPowerOfTwo(index_length);
  block_width = MIN(block_width, PREDEFINED_BLOCK_SIZE_X);
  
  int block_height = paddle::platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;
  block_height = MIN(block_height, PREDEFINED_BLOCK_SIZE / block_width);

  dim3 block_dim(block_width, block_height);
  dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                (batch_size + block_dim.y - 1) / block_dim.y);
  paddle::platform::LimitGridDim(ctx, &grid_dim);
```

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

根据 [paddle.index_sample](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_sample_cn.html#index-sample) 的api参数定义，输入参数`x`和 `index` 均为2维Tensor。那么实现优化的GPU 版IndexSample OP性能优化的核心是采用2维度的线程设置，降低索引计算的开销。预计OP整体性能预期提升约为10%左右。

## 2.2 Host端计算流程

Host端是主要是根据结合`index`参数的列数确定blockDim.x，根据`x` 参数的 第1个维度数据确定blockDim.y。这样能够便于Block能够快速得到`Index`中的数据元素。

```
  auto block_width = paddle::platform::RoundToPowerOfTwo(index_length);
  block_width = MIN(block_width, PREDEFINED_BLOCK_SIZE_X);
  
  int block_height = paddle::platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;
  block_height = MIN(block_height, PREDEFINED_BLOCK_SIZE / block_width);
  dim3 block_dim(block_width, block_height);
```

进一步结合Block的设置确定Grid设置：

```
  dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                (batch_size + block_dim.y - 1) / block_dim.y);
  paddle::platform::LimitGridDim(ctx, &grid_dim);
```

## 2.4 Device端计算流程

Device端则是完全按照2维的线程设置，快速在`Index`中搜索每行的元素，并确定`x`中的数据元素，导入至输出Tensor中。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

完成IndexSample OP 开发后，Paddle与Pytorch的性能对比效果如下，达到了预期性能提升效果：

| No. | index_shape | input_shape | Paddle Perf(ms) | Pytorch Perf(ms) | diff |
|---|---|---|---|---|---|
| 1 | [5100,1] | [5100,38506] |  0.7052 | 1.7032 | faster than 58.597% |
| 2 | [100,64] |  [100, 128]  | 0.0055  | 0.0083 | faster than 33.874% |
| 3 | [5100,96]| [5100,128]   | 0.0323  | 0.0377 | faster than 14.131% |


# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | xxxx-xx-xx |
| 2 | 完成开发文档设计  | xxxx-xx-xx |
| 3 | 完成代码开发工作，并通过线程CI测试 | xxxx-xx-xx |


# 5 影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响。



# 名词解释



# 附件及参考资料
[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)

