## 方案名称
为 PaddlePaddle 框架新增 grid_sampler/gaussian_random/flip/roi_align/fill_zeros_like/unique 的 TVM 算子映射设计文档。

| API名称 | grid_sampler/gaussian_random/flip/roi_align/fill_zeros_like/unique | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-03 | 
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230303_paddle2tvm_design_conv3d-data_norm-flip-roi_align_fill_zeros_like-unique.md<br> | 

# 方案描述
为 PaddlePaddle 框架的 conv3d/data_norm/flip/roi_align/fill_zeros_like/unique 算子做适配并完成单算子测试。


# 方案流程

**1. grid_sampler**

Paddle 的 grid_sampler 算子可用 tvm 的 transpose 和 grid_sampler 算子组合适配。

[Paddle的 grid_sampler 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/grid_sample_cn.html#grid-sample)

**2. gaussian_random**

Paddle 的 gaussian_random 算子可用 tvm 的 normal 算子适配。

[Paddle的 gaussian_random 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/normal_cn.html#normal)

**3. flip 算子**

Paddle 的 flip 算子可用 tvm 的 reverse 算子适配。

[Paddle的 flip 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/flip_en.html#flip)

**4. roi_align 算子**

Paddle 的 roi_align 算子可用 tvm 的 roi_align 算子适配。

[Paddle的 roi_align 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/vision/ops/roi_align_en.html#roi-align)

**5. fill_zeros_like 算子**

Paddle 的 fill_zeros_like 算子可用 tvm 的 fill_like 算子适配。

[Paddle的 fill_zeros_like 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/functional/softmax_with_cross_entropy_en.html#softmax-with-cross-entropy)

**6. unique算子**

<!-- | 类别 | Paddle | TVM |
| ------ | ------ | ------ |
| Input | x | data  |
| Attribute | return_index  | - |
| Attribute | return_inverse  | - |
| Attribute | return_counts  | - |
| Attribute | - | is_sorted  |
| Attribute | axis   | - |
| Attribute | dtype  | - |
| Output | out  | unique  |
| Output | index  | indices |
| Output | inverse   | inverse_indices  |
| Output | counts   | counts | -->

Paddle 的 unique 算子可用 tvm 的 reshape 和 unique 算子组合适配。

[Paddle 的 unique 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unique_cn.html#unique)


# 方案运行效果
完成上述算子的映射，并且参考[单测代码](https://github.com/apache/tvm/blob/main/tests/python/frontend/paddlepaddle/test_forward.py)，设计各个算子的单测。


# 项目提交时间计划
3月3日 ~ 3月10日完成算子适配

3月11日 ~ 3月18日完成算子适配的单测
