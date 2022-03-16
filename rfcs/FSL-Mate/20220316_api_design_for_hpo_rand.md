# paddlefsl.hpo.rand 设计文档
|API名称 | paddlefsl.hpo.rand | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | jinyouzhi | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-16 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0 | 
|文件名 | 20220316_api_design_for_hop_rand.md<br> | 

# 一、概述
## 1、相关背景
为提升飞桨社区活跃程度，促进生态繁荣，第二期Hackathon设置的任务。（https://github.com/tata1661/FSL-Mate/issues/19）
PaddleFSL是基于飞桨框架开发的小样本学习工具包，旨在降低小样本学习实践成本，提供了一系列常用小样本学习常用算法。
## 2、功能目标
随机搜索即在搜索空间随机的搜索超参数。它是一种不需要优化问题梯度的数值优化方法，也是常用的基线超参数搜索算法。

## 3、意义
超参数搜索（HPO，Hyper-Parameter Optimization）是。

# 二、飞桨现状
对飞桨框架目前支持此功能的现状调研，如果不支持此功能，如是否可以有替代实现的API，是否有其他可绕过的方式，或者用其他API组合实现的方式；


# 三、业内方案调研
NNI
https://github.com/microsoft/nni/blob/master/nni/algorithms/hpo/random_tuner.py

Optuna
https://optuna.readthedocs.io/zh_CN/latest/reference/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler
# 四、对比分析
对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的有劣势。

# 五、设计思路与实现方案

## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
## API实现方案

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
时间和开发排期规划，主要milestone

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料
