# paddle.corrcoef 设计文档

|API名称 | 新增API名称 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 张一乔 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-09 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0 | 
|文件名 | 20200309_api_design_for_corrcoef.md<br> | 


# 一、概述
## 1、相关背景
https://github.com/PaddlePaddle/Paddle/issues/40332
## 2、功能目标
为 Paddle 新增 corrcoef(皮尔逊积矩相关系数) API
## 3、意义
使paddle对tensor进行相关系数计算

# 二、飞桨现状
1. 当前develop版本中存在api：paddle.cov，可以按照相关系数和协方差的映射关系R{ij} = C{ij} / sqrt{ C{ii} * C{jj} }进行计算
2. 将tensor转换为numpy格式，通过numpy.corrcoef进行计算

# 三、业内方案调研
Numpy中包含函数corrcoef，介绍为

'''
numpy.corrcoef(x, y=None, rowvar=True, bias=<no value>, ddof=<no value>)
Return Pearson product-moment correlation coefficients.
Please refer to the documentation for cov for more detail. The relationship between the correlation coefficient matrix, R, and the covariance matrix, C, is
R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }
The values of R are between -1 and 1, inclusive.
'''
  
Numpy中实现方案见https://github.com/numpy/numpy/blob/v1.15.0/numpy/lib/function_base.py#L2330-L2410

pytorch中的corrcoef：https://pytorch.org/docs/master/generated/torch.corrcoef.html#torch-corrcoef
  函数定义为 torch.corrcoef(tensor)
  仅有一个输入参数，按行求对应的相关系数
  
tensorflow1.15的corrcoef:https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/contrib/metrics/streaming_pearson_correlation
  函数定义为 
  tf.contrib.metrics.streaming_pearson_correlation(
    predictions, labels, weights=None, metrics_collections=None,
    updates_collections=None, name=None)
  用于获得两个输入变量的相关系数矩阵。
  
# 四、对比分析
Numpy中允许输入两个矩阵以获取拼合后的相关系数，并且可以通过rowvar指定按行求解还是按列求解。
Numpy中的corrcoef效果较好，能满足大多数情况的使用需求。在实际使用中，可以在调用np.corrcoef之前，将这些变量进行拼合。即通常的调用方式为
  np.corrcoef(x)
或
  d=np.concatenate((a,b,c),axis=0)
  np.corrcoef(d)
  
pytorch中的corrcoef仅允许输入一个变量，即需要将一组用于求得相关系数的变量进行拼合。调用方式为：
  x = torch.tensor([[0, 1, 2], [2, 1, 0]])
  torch.corrcoef(x)

tensorflow中需要给定两个变量，将输出结果作为一种metric来进行使用，调用方式为：
  tf.contrib.metrics.streaming_pearson_correlation(predictions, labels)
  
综合考虑，允许输入两个变量x,y在函数内部进行拼合，有利于调用corrcoef并将结果用作metric。这种方式仅利于处理仅有一个返回值的回归问题。因此，没有必要特意保留输入参数y,仅允许用户将拼合后的矩阵传入函数求得对应的相关系数矩阵即可。
  
因此，本api拟实现在仅有输入x的情况下，与numpy.corrcoef效果相同的paddle.corrcoef。即paddle.corrcoef不接收参数y，仅接收参数x。

# 五、设计思路与实现方案
  
## 命名与参数设计
API设计为'paddle.corrcoef(x, rowvar=True, ddof=True, name=None)'
## 底层OP设计
使用已有API组合实现，不再单独设计OP。
## API实现方案
  
1. 使用'paddle.cov'得到协方差矩阵
2. 使用'paddle.diag'提取协方差矩阵的迹T
3. 使用'paddle.mm'获得矩阵{ C{ii} * C{jj} }
4. 对矩阵{ C{ii} * C{jj} }开平方获得sqrt{ C{ii} * C{jj} }
5. 使用'paddle.divide'获得相关系数矩阵
  
# 六、测试和验收的考量
测试考虑的case如下：
- 和numpy结果的数值的一致性, 'paddle.corrcoef','np.corrcoef'结果是否一致；
- 参数'x'为1-D Tensor和2-D Tensor时输出的正确性；
- 参数'rowvar'为True和False的输出正确性
- 参数'ddof'为True和False的输出正确性

# 七、可行性分析和排期规划
方案主要依赖现有paddle api组合而成。工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 九、评审意见
（由评审人填写，开发者无需填写）
|问题 | 提出人 | 处理说明 | 状态 | 
|---|---|---|---|
| |  |  |  | 

# 名词解释
# 附件及参考资料
