# 【Hackathon 5th No.57】Neural networks for topology optimization

|              |                    |
| ------------ | -----------------  |
| 提交作者      |       NKNaN        |
| 提交时间      |     2023-10-11     |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop版本         |
| 文件名        | 20231011_paper_topology_optimization.md             |

## 1. 概述

### 1.1 相关背景

复现论文：Neural networks for topology optimization  
[论文链接](https://arxiv.org/abs/1709.09578)  
[源码链接](https://github.com/ISosnovik/nn4topopt) 

### 1.2 目标

该论文为数据驱动，案例之间代码相同，数据集不同（案例代码生成），要求复现全部四个案例，并合入 PaddleScience。  

### 1.3 意义

增加使用 PaddleScience 现有 API 复现 AI4Science 论文的案例。  

## 2. PaddleScience 现状

1. PaddleScience 在 Arch 中已实现 Unet - 论文核心模型
2. 现有框架可以对 Data 进行 transformation 但不包括随机翻转和随机旋转 - 论文数据处理步骤
3. 现有框架文档中对 batched data 进行 transformation 的部分尚未展示 - 论文数据处理步骤

## 3. 目标调研

1. 论文解决的问题：
   拓扑优化问题: 如何在设计域内分布材料，使获得的结构具有最优性能同时满足一定的约束条件（例如：必须是01解，1-域内有材料，0-域内没有材料）
   对于具有连续设计变量的拓扑优化问题，最常用的方法是所谓的SIMP迭代算法，可以看到利用SIMP方法，求解器只需要进行第N_0次迭代就可以得到结构的初步视图。
   获得的最终图像I*可以看作是mask了的原图像I(从N_0此迭代开始)。
   
2. 提出的方法：
   采用SIMP方法进行初始迭代，得到非01值密度分布;利用Unet对得到的图像进行分割，并将分布收敛到01解。
   
3. 复现目标:
   利用泊松或均匀分布采样初始迭代次数从而生成输入数据，从而训练Unet预测SIMP迭代算法的最终解。
   
4. 可能存在的难点:
   需要完全利用现有PaddleScience API在准备数据集时对通道进行随机采样。


## 4. 设计思路与实现方案

1. Unet:
   ![image](https://github.com/NKNaN/community/assets/49900969/75a42972-10c5-4eab-be43-5d08e899ad0d)
  论文中的Unet为如图结构，根据已有API需要对 `ppsci.arch.UNetEx` 的结构做调整例如加入 `Dropout` 层和 `Upsampling` 层

2. 初始迭代步数采样器：
```python
  def uniform_sampler():
      return lambda: np.random.randint(1, 99)
  
  
  def poisson_sampler(lam):
      def func():
          iter_ = max(np.random.poisson(lam), 1)
          iter_ = min(iter_, 99)
          return iter_
      return func
```

3. 构建 dataset 需要对图像数据做随机翻转和随机旋转操作：

```python
  def augmentation(input, label):
      """Apply random transformation from D4 symmetry group
      # Arguments
          x_batch, y_batch: input tensors of size `(batch_size, any, height, width)`
      """
      X = paddle.to_tensor(input["input"])
      Y = paddle.to_tensor(label["output"])
      n_obj = len(X)
      indices = np.arange(n_obj)
      np.random.shuffle(indices)
  
      if len(X.shape) == 3:
          # random horizontal flip
          if np.random.random() > 0.5:
              X = paddle.flip(X, axis=2)
              Y = paddle.flip(Y, axis=2)
          # random vertical flip
          if np.random.random() > 0.5:
              X = paddle.flip(X, axis=1)
              Y = paddle.flip(Y, axis=1)
          # random 90* rotation
          if np.random.random() > 0.5:
              new_perm = list(range(len(X.shape)))
              new_perm[1], new_perm[2] = new_perm[2], new_perm[1]
              X = paddle.transpose(X, perm=new_perm)
              Y = paddle.transpose(Y, perm=new_perm)
          X = X.reshape([1] + X.shape)
          Y = Y.reshape([1] + Y.shape)
      else:
          # random horizontal flip
          batch_size = X.shape[0]
          mask = np.random.random(size=batch_size) > 0.5
          X[mask] = paddle.flip(X[mask], axis=3)
          Y[mask] = paddle.flip(Y[mask], axis=3)
          # random vertical flip
          mask = np.random.random(size=batch_size) > 0.5
          X[mask] = paddle.flip(X[mask], axis=2)
          Y[mask] = paddle.flip(Y[mask], axis=2)
          # random 90* rotation
          mask = np.random.random(size=batch_size) > 0.5
          new_perm = list(range(len(X.shape)))
          new_perm[2], new_perm[3] = new_perm[3], new_perm[2]
          X[mask] = paddle.transpose(X[mask], perm=new_perm)
          Y[mask] = paddle.transpose(Y[mask], perm=new_perm)
  
      return X, Y
```
   
4. 构建 dataloader 需要在每次生成一批 batch data 时对初始迭代步数（原始数据的通道从0-99表示SIMP算法的100次迭代步骤）进行采样。
```python
  def batch_transform_wrapper(sampler):
      def batch_transform_fun(batch):
          batch_input = paddle.to_tensor([])
          batch_label = paddle.to_tensor([])
          k = sampler()
          for i in range(len(batch)):
              x1 = batch[i][0][:, k, :, :]
              x2 = batch[i][0][:, k - 1, :, :]
              x = paddle.stack((x1, x1 - x2), axis=1)
              batch_input = paddle.concat((batch_input, x), axis=0)
              batch_label = paddle.concat((batch_label, batch[i][1]), axis=0)
          return ({"input": batch_input}, {"output": batch_label}, {})
  
      return batch_transform_fun
```

## 5. 测试和验收的考量

验收标准：
- 定性标准：得到与参考代码中<https://github.com/ISosnovik/nn4topopt/blob/master/results.ipynb>中的Binary Accuracy相近的指标图像
- 定量标准：计算与论文table 1与table 2结果的相对误差，不超过10%

## 6. 排期规划

10-01 ~ 10-07日完成复现

## 7. 影响面

增加使用 PaddleScience 现有API复现 AI4Science 论文的案例，同时发现现有API不足的地方。
