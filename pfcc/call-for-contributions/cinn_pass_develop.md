# CINN中端Pass开发

## 背景
[CINN](https://github.com/PaddlePaddle/CINN)是一种采用JIT技术加速飞桨模型运行速度的深度学习编译器。
CINN的整个Pass体系分为三层，即:前端Pass、中端Pass和c-ir Pass，这里主要介绍一下中端Pass的实现。
CINN中端Pass主要是对CINN中端表示的图进行优化，CINN前端接入模型之后，都会转换为中端的图表示。
`
    auto A = net_builder.CreateInput(Float(32), {32, 1, 32, 512}, "A");
    auto B = net_builder.CreateInput(Float(32), {32, 1, 32, 512}, "B");
    auto C = net_builder.Concat({A, B}, 3);
    auto D = net_builder.Reshape(C, {32, 32, 1024});
    auto E = net_builder.ReduceSum(D, {2}, false);
`
