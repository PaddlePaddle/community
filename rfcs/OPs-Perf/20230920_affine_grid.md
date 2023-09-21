# Affine_grid OP性能优化设计文档


| 基本信息   | 内容      |
| --------------------- | -------------- |
| 提交作者   | yangguohao   |
| 提交时间  | 2023-09-20 |
| 版本号    | V1.0  |
| 依赖飞桨版本 | PaddleDevelop|
| 文件名                    | 20230920_affine_grid.md |


# 1 背景与意义
目前，Paddle中的 affine_grid 方法有提升空间

## 1.1 飞桨现状

Paddle中暂时没有IndexSample OP的GPU实现，需要实现一个GPU版本的IndexSample OP.

## 1.2 业内方案调研

Pytorch中对应的 affine_grid gpu 实现在 pytorch/aten/src/ATen/native/AffineGridGenerator.cpp 内
其主要使用的方法是通过矩阵乘法的方式进行计算

```
"4D method"
static Tensor make_base_grid_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = at::empty({N, H, W, 3}, theta.options());

  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  base_grid.select(-1, 2).fill_(1);

  return base_grid;
}
```

```
"5D method"
static Tensor make_base_grid_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = at::empty({N, D, H, W, 4}, theta.options());

  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D, align_corners).unsqueeze_(-1).unsqueeze_(-1));
  base_grid.select(-1, 3).fill_(1);

  return base_grid;
}
```
## 1.3 对比分析

Paddle 中的 gpu 版本则是自己手写 CUDA KERNEL 的形式实现了
```
template <typename T>
__global__ void affine_grid_kernel_4d(const int count,
                                      int n,
                                      int out_h,
                                      int out_w,
                                      T h_start,
                                      T w_start,
                                      T h_step,
                                      T w_step,
                                      const T* theta,  // N, 2, 3
                                      T* output) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int n = index / (out_w * out_h);

    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 6;  // 2 * 3;
    // affine from (h_coor, w_coor) to (x, y)
    output[index * 2] = theta[theta_offset] * w_coor +
                        theta[theta_offset + 1] * h_coor +
                        theta[theta_offset + 2];
    output[index * 2 + 1] = theta[theta_offset + 3] * w_coor +
                            theta[theta_offset + 4] * h_coor +
                            theta[theta_offset + 5];
  }
}
```
```
template <typename T>
__global__ void affine_grid_kernel_5d(const int count,
                                      int n,
                                      int out_d,
                                      int out_h,
                                      int out_w,
                                      T d_start,
                                      T h_start,
                                      T w_start,
                                      T d_step,
                                      T h_step,
                                      T w_step,
                                      const T* theta,  // N, 3, 4
                                      T* output) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int d = (index / (out_w * out_h)) % out_d;
    int n = index / (out_w * out_h * out_d);

    T d_coor = d_step * static_cast<T>(d) + static_cast<T>(d_start);
    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 12;  // 3 * 4
    // affine from (h_coor, w_coor) to (x, y)
    output[index * 3] =
        theta[theta_offset] * w_coor + theta[theta_offset + 1] * h_coor +
        theta[theta_offset + 2] * d_coor + theta[theta_offset + 3];
    output[index * 3 + 1] =
        theta[theta_offset + 4] * w_coor + theta[theta_offset + 5] * h_coor +
        theta[theta_offset + 6] * d_coor + theta[theta_offset + 7];
    output[index * 3 + 2] =
        theta[theta_offset + 8] * w_coor + theta[theta_offset + 9] * h_coor +
        theta[theta_offset + 10] * d_coor + theta[theta_offset + 11];
  }
}
```

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

目前的想法主要是以下两点:
1. 按照 pytorch 的实现，利用矩阵乘法(Paddle 是否可以以类似的方式实现)进行计算并测试。
2. 通过 KPS 给出的例如 elementwise_unary 之类的方法来重写 kernel 进行提升。


# 4 可行性分析和排期规划

可行性在讨论之后再给出详细的时间和规划。


# 5 影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响。



# 名词解释



# 附件及参考资料
[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
