## 方案名称
为TVM PaddlePaddle前端完善 dropout/gelu/hard_sigmoid/pixel_shuffle 算子支持程度设计文档

| API名称 | dropout/gelu/hard_sigmoid/pixel_shuffle | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-04-10 | 
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230410_paddle2tvm_design_fix_ops1.md<br> | 

# 方案描述
为TVM PaddlePaddle前端完善 dropout/gelu/hard_sigmoid/pixel_shuffle 算子支持程度，并完善算子测试。


# 方案流程

**1. dropout算子**

完善 dropout 算子的丢弃概率参数支持。

[Paddle的 dropout 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/dropout_cn.html#dropout)

**2. gelu**

完善 gelu 算子的 approximate 参数支持。

[Paddle的 gelu 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu)

**3. hard_sigmoid 算子**

完善 hard_sigmoid 算子的 offset 参数支持。

[Paddle的 hard_sigmoid 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/hardsigmoid_cn.html#hardsigmoid)

**4. pixel_shuffle 算子**

完善 pixel_shuffle 算子的 data_format 参数（"NCHW"或"NHWC"）支持。

[Paddle的 roi_align 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/pixel_shuffle_cn.html#pixel-shuffle)

# 方案运行效果
参考[单测代码](https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py)，设计并通过各个算子的单测。


# 项目提交时间计划
4月10日 ~ 4月15日完成算子适配

4月16日 ~ 4月20日完成算子适配的单测
