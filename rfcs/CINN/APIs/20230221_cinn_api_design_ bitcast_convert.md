# CINN bitcast_convert设计文档

|API名称 | 新增API名称 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | XDUWQ| 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-02-21 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 20230221_cinn_api_design_bitcast_convert.md<br> | 


# 一、概述
## 1、相关背景
`bitcast_convert` 是神经网络中的算子，实现的功能是在不改变底层存储的情况下，强制转换数据类型。
若转换前后数据类型的字节大小不相同，则形状会改变。比如一个 shape=[10] 的 float32 类型数据被强制转换为 float16 类型后，其 shape 应为[10, 2]。

## 2、功能目标
在不复制数据的情况下，将张量从一种类型转换为另一种类型。若转换前后数据类型的字节大小不相同，则形状会改变。
输入是`inputs` 和 目标转换的类型`dtype`，输出是`outputs`。

## 3、意义
实现`bitcast_convert` 算子，将进一步完善CINN的基础算子库。

# 二、CINN现状
CINN框架暂不支持`bitcast_convert`算子，需要实现。

# 三、业内方案调研



# 四、对比分析



# 五、设计思路与实现方案



## 命名与参数设计
* `A`：Tensor类型，表示输入张量
* `dytpe`: string类型，表示转换输出类型

## 底层OP设计
1. 在 `cinn/hlir/op/contrib/bitcast_convert.h` 里声明`bitcast_convert`算子。
2. 在 `cinn/hlir/op/contrib/bitcast_convert.cc` 里实现`bitcast_convert`算子和 `strategy`。

## API实现方案
1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::Bitcast_convert`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `NetBuilder::Bitcast_convert`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `NetBuilder` 添加 `bitcast_convert` 接口，并绑定到 `NetBuilder::Bitcast_convert`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

python通过Builder类的方法调用`bitcast_convert`。
```python
builder = NetBuilder("test_basic")
b = builder.bitcast_convert([10], "float32")
```

# 六、测试和验收的考量
1. 提供基础的 demo 文件。
2. 在`cinn/hlir/op/contrib/bitcast_convert_test.cc`中添加对底层OP进行测试的代码。
3. 在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
4. 提交 API 说明到相应的文档中。

# 七、可行性分析和排期规划
- 可行性分析：CINN已实现Builder、Expr IR、算子注册等模块，在CINN已有的框架基础上能够很好地增加算子功能。
- 排期规划：预计3月1日前完成算子实现、功能测试以及文档

# 八、影响面
对其他模块无影响。

# 附件及参考资料
* [手把手教你为神经网络编译器CINN增加One-Hot算子](https://blog.csdn.net/PaddlePaddle/article/details/128509915)
* [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
* [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
* [CINN算子开发示例：pool2d_grad算子](https://github.com/PaddlePaddle/CINN/pull/858)  
* [CINN IR DSL在C++的matmul写法例子](https://github.com/PaddlePaddle/CINN/blob/develop/tutorials/matmul.cc)  
