# 为 MaxPool1D/2D/3D 添加 dilation 支持

## 详细描述

为 Paddle 的一维、二维和三维最大池化 API 增加空洞池化能力。以下函数式
API 和 Layer API 均应支持 `dilation` 参数：

- `paddle.nn.functional.max_pool1d/2d/3d`
- `paddle.nn.MaxPool1D/2D/3D`

`dilation` 用于控制池化窗口内相邻采样点的间隔。各维有效 kernel 大小应为：

```text
dilation * (kernel_size - 1) + 1
```

标量和与空间维数匹配的列表或元组均应得到正确处理。

## 验收说明

- 1D、2D、3D 的函数式 API 和 Layer API 均支持 `dilation`，默认值为 1。
- CPU 和 CUDA 的前向结果、输出形状及反向梯度符合空洞最大池化语义。
- `return_mask=True` 时结果和索引均正确，`ceil_mode`、padding、stride 与
  dilation 组合时形状推导正确。
- 动态图和静态图下的行为及输出形状保持一致
- 默认 `dilation=1` 时不得改变已有最大池化行为。
- XPU 暂不要求实现非默认 dilation；不支持时应给出明确错误，不得静默产生
  错误结果。
- 不允许通过删除测试、弱化断言或绕过参数校验来完成任务。

## 技术要求

- 熟悉 Python、C++ 和 CUDA
- 了解 Paddle CPU/CUDA kernel 开发机制
- 了解池化、空洞算子的形状推导、计算方法
