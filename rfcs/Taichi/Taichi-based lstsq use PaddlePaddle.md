# Taichi 和 PaddlePaddle 高效结合的案例设计文档

|Paddle Friends | Taichi |
|---|---|
|提交作者 | 0xzhang |
|提交时间 | 2022-03-30 |
|版本号 | V0.1 |
|飞桨版本 | paddlepaddle-gpu 0.0.0 |
|太极版本 | taichi 1.0.2 |
|文件名 | Taichi-based lstsq use PaddlePaddle.md |

# 一、概述

深度学习框架如 PaddlePaddle 与 Taichi 的结合使用，可以使开发新的 op 更加高效快速。得益于 Paddle Tensor 的内部由一块连续内存实现，Taichi 和 Paddle 之间可以实现同硬件上的无拷贝交互，即 Taichi kernel 直接操作 PaddlePaddle 。Tensor 所在的内存地址，既实现了在同一个 Python 程序中 Taichi 和 PaddlePaddle 的数据交互，又避免了两个框架间的数据拷贝。

选取一个 PaddlePaddle 中暂不支持的 op，使用 Taichi 编写该 op 的并行实现，并在一个 PaddlePaddle 和Taichi的交互案例中展示效果。

这个题目为开放性题目，需要在另一个[题目](https://github.com/taichi-dev/hackathons/issues/3)的基础上完成。考虑实现最小二乘法（lstsq）作为新的 op。

最小二乘法，是一种数学优化建模方法，通过最小化误差的平方和寻找数据的最佳函数匹配。最小二乘法发展与天文学和大地测量学领域，最早由法国数学家勒让德（Legendre）于1806年发表，德国数学家高斯（Gauss）于1809年发表。

最小二乘法是对线性方程组，即方程个数比未知数更多的方程组，以回归分析求得近似解的标准方法。最小二乘法最重要的应用是曲线拟合，最佳拟合的内涵是：残差（残差为：观测值与模型提供的拟合值之间的差距）平方总和的最小化。

# 二、飞桨现状
参考 PaddlePaddle 的 API 文档，lstsq 是 PaddlePaddle 中暂不支持的 op。


# 三、业内方案调研
- torch.linalg.lstsq

  - CPU 可选 "gels"，"gelsy"，"gelsd"，"gelss"，默认使用 “gelsy"；CUDA 默认且仅支持使用 "gels"，假定 $$ A $$ 满秩。

- tf.linalg.lstsq

  - TensorFlow 默认使用 Cholesky 分解，可通过参数`fast = False`控制使用完全正交分解，慢 6~7 倍。

- numpy.linalg.lstsq

  - NumPy 使用 SVD 求解，默认基于 LAPACK 中的 GELSD ，采用分治 SVD 方法。

- scipy.linalg.lstsq

  - 使用 LAPACK 作为求解器驱动，默认使用 "gelsd"，"gelsy"在多数问题上更快，"gelss"已成为历史，通常更慢但占用更少的内存。

- LAPACK

  根据 LAPACK 提供的基准测试，在 Compaq AlphaServer DS-20 机器上，使用 100~1000 维度的数据。DGELS、DGELSY、DGELSX、DGELSD、DGELSS，速度依次递减。DGELSD 和 DGELSS 使用 SVD，是最可靠但也最昂贵的求解不满秩最小二乘问题的方法。DGELSY 的速度与 DGELS 非常接近。DGELSY 比 DGELSX 的实现更快，但是因为调用分块的算法会多占用一些空间。对于大矩阵，DGELSX 慢 2.5 倍左右。DGELSD 用来代替 DGELSS，DGELSD 比 DGELS 慢 3~5 倍， DGELSS 比 DGELS 慢 7~34 倍。

  xGELS 求解假定 $$ A $$ 是满秩的，xGELSX，xGELSY，xGELSS 和 xGELSD 用于求解 $$ A $$ 不满秩的情况。

| Operation                                         | Single precision |               | Double precision |               |
| ------------------------------------------------- | ---------------- | ------------- | ---------------- | ------------- |
|                                                   | real             | complex       | real             | complex       |
| solve LLS using *QR* or *LQ* factorization        | SGELS            | CGELS         | DGELS            | ZGELS         |
| solve LLS using complete orthogonal factorization | SGELSX/SGELSY    | CGELSX/CGELSY | DGELSX/DGELSY    | ZGELSX/ZGELSY |
| solve LLS using SVD                               | SGELSS           | CGELSS        | DGELSS           | ZGELSS        |
| solve LLS using divide-and-conquer SVD            | SGELSD           | CGELSD        | DGELSD           | ZGELSD        |

# 四、设计思路与实现方案

尝试使用 Taichi 实现需要的线性代数运算，支持正规方程、QR分解和SVD方法进行求解。

# 五、测试和验收的考量
- 公开 repo
- Repo 中包含详细的案例使用步骤，以及必要的代码讲解和背景知识
- Repo 中包含拟合示例、正确性比较和性能测试

# 六、可行性分析和排期规划

正在实现中。

# 七、附件及参考资料

1. [API 文档-API文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)
2. [Least squares - Wikipedia](https://en.wikipedia.org/wiki/Least_squares)
3. [torch.linalg.lstsq — PyTorch 1.11.0 documentation](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)
4. [tf.linalg.lstsq  | TensorFlow Core v2.8.0](https://www.tensorflow.org/api_docs/python/tf/linalg/lstsq)
5. [numpy.linalg.lstsq — NumPy v1.22 Manual](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
6. [scipy.linalg.lstsq — SciPy v1.8.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)
7. [Linear Least Squares (LLS) Problems (netlib.org)](https://www.netlib.org/lapack/lug/node27.html)
8. [LAPACK Benchmark (netlib.org)](https://www.netlib.org/lapack/lug/node71.html)
9. [Charles Jekel - jekel.me - Compare lstsq performance in Python](https://jekel.me/2019/Compare-lstsq-performance-in-Python/)

