# paddle.nn.TripletMarginWithDistanceLoss 设计文档


|API名称 | 新增API名称                            | 
|---|------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | 高崧淇 李应钦                            | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-12                         | 
|版本号 | V1.0                               | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                            | 
|文件名 | paddle.nn.TripletMarginWithDistanceLoss.md<br> | 


# 一、概述
## 1、相关背景
paddle.nn.TripletMarginWithDistanceLoss 是三元损失函数，其针对 anchor 和正负对计算用户指定的距离函数下的三元损失，从而获得损失值。

## 3、意义
使paddle支持更多的损失函数

# 二、飞桨现状
无此功能，但可以在python层通过调用基础api的形式实现。

# 三、业内方案调研
torch有此api，是在c++层实现的，不过调用的基础op飞桨均有实现

# 四、对比分析
本方案计划在python层开发，不涉及底层c++，故减轻了日后的编译开销，计算全部在c++层，不会损失性能。
且与torch的api一致，方便用户无缝迁移到paddle。

# 五、设计思路与实现方案

## 命名与参数设计
新增两个api，调用路径为：
paddle.nn.TripletMarginWithDistanceLoss
和
paddle.nn.functional.triplet_margin_with_distance_loss

[//]: # (参考：[飞桨API 设计及命名规范]&#40;https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html&#41;)
## 底层OP设计
无需底层op
## API实现方案
核心功能放在functional中，在nn中暴露出来，方便用户调用，与其它api一致

# 六、测试和验收的考量
- 数值正确性及稳定性
- 支持的数据类型
- 异常检测

[//]: # (参考：[新增API 测试及验收规范]&#40;https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html&#41;)

# 七、可行性分析和排期规划
根据调研结果，已经开发了一个版本的functional.triplet_margin_loss，可以直接调用，数值正确，有待进一步测试。

预计整体工期在一周之内，完成文档、代码、测试、验收。

# 八、影响面
无

[//]: # (# 名词解释)

[//]: # (# 附件及参考资料)
