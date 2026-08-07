# 新增 nn.FractionalMaxPool2d / nn.FractionalMaxPool3d API

## 详细描述

为 Paddle 新增分数阶最大池化算子与 API，参考论文 *Ben Graham, Fractional Max-Pooling, 2015*（http://arxiv.org/abs/1412.6071）。

需要实现：

- `paddle.nn.FractionalMaxPool2d` 层
- `paddle.nn.FractionalMaxPool3d` 层
- `paddle.nn.functional.fractional_max_pool2d` 函数
- `paddle.nn.functional.fractional_max_pool3d` 函数
- 对应算子的前向 kernel 与反向 kernel（CPU 与 GPU）
- 支持 `return_mask`，返回最大池化点的索引

API 说明（以 `fractional_max_pool2d` 为例）：

- `x`：输入，4-D Tensor，数据范围 `[N, C, H, W]`，数据类型支持 float16、bfloat16、float32、float64。
- `output_size`：输出尺寸，可为 int 或 (H, W) 的 list/tuple，元素可为 None（表示与输入相同）。
- `kernel_size`：可选，pool kernel 尺寸。为 None 时使用非重叠模式（disjoint），否则为重叠模式（overlapping）。
- `random_u`：可选，分数池化随机因子 u ∈ (0, 1)。为 None 时由框架随机生成，可用 `paddle.seed` 固定。
- `return_mask`：可选，为 True 时同时返回最大池化索引。

前向计算公式（每维）：

```
alpha = size_input / size_output
index_start = ceil(alpha * (i + u) - 1)
index_end = ceil(alpha * (i + 1 + u) - 1)
Output = max(Input[index_start:index_end])
```

其中 u ∈ (0, 1)，i = 0,1,2...size_output。

## 验收说明

- `fractional_max_pool2d` / `fractional_max_pool3d` 前向结果与公式一致
- 支持 disjoint（`kernel_size=None`）与 overlapping（给定 `kernel_size`）两种模式
- `return_mask=True` 时正确返回最大池化索引
- 反向梯度正确
- 支持 `nn.FractionalMaxPool2d` / `nn.FractionalMaxPool3d` 层封装

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
