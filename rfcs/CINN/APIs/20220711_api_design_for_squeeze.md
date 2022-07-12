# CINN squeeze 设计文档

|API名称 | 新增API名称 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 六个骨头 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-11 | 
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220711_api_design_for_squeeze.md<br> | 

# 一、概述

## 1、相关背景

为了提升 CINN API 丰富度，需要扩充 API `squeeze`。

## 2、名词解释

无

## 3、功能目标
实现 squeeze 功能。

## 4、意义

为神经网络编译器 CINN 增加基础算子 squeeze 。

# 二、CINN现状

对CINN框架目前不支持此功能，可以使用 reshape API 替代，但使用 reshape API 需要明确的知道数据的尺寸，对开发者的精力消耗较大，因此有必要实现 squeeze API。

# 三、业内方案调研

- TVM：未实现该API，通常借用 numpy 等实现该功能。
- XLA：通过调用reshape相关API实现。
```cpp
xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  auto output_sizes =
      BuildSqueezedDimensions(input_shape.dimensions(), /*squeeze_dim=*/-1);
  return XlaHelpers::DynamicReshape(input, output_sizes);
}
```

# 四、对比分析

无

# 五、设计思路与实现方案

## 命名与参数设计
- A：输入张量
- name：输出名称

## 底层OP设计
1. 在 `cinn/hlir/pe/transform.cc` 里实现 `squeeze` 算子。
2. 在 `cinn/hlir/op/transform.h` 里声明相应的 `strategy`。
3. 在 `cinn/hlir/op/transform.cc` 里实现相应的 `strategy`。

## API实现方案
1. 在 `cinn/frontend/base_build.h` 里声明 `BaseBuilder::Squeeze`。
2. 在 `cinn/frontend/base_build.cc` 里实现 `BaseBuilder::Squeeze`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `BaseBuilder` 添加 `squeeze` 接口，并绑定到 `BaseBuilder::Squeeze`。
4. 上层 `net_builder` 调用提交到 `cinn/frontend/net_builder.h` 和 `.cc` 文件下。
5. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

# 六、测试和验收的考量
1. 提供基础的 demo 文件。
2. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：1-6已完成，7-9预计7月15日前完成

# 八、影响面

对其他模块无影响。

# 附件及参考资料

[CINN文档](https://paddlepaddle.github.io/CINN/)

