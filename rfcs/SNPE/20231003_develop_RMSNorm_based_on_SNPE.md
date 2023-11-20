
# 基于高通 AI 软件栈 SNPE SDK，开发算子 RMSNorm, 在高通 HTP 运行

|任务名称 | 基于 Qualcomm SNPE SDK 开发 RMSNorm 算子 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | UnseenMe | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-10-03 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | - | 
|文件名 | 20231003_develop_RMSNorm_based_on_SNPE.md<br> | 

# 一、概述
## 1、相关背景
[【PaddlePaddle Hackathon 5th】开源贡献个人挑战赛 任务99](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/%E3%80%90PaddlePaddle%20Hackathon%205th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E5%90%88%E4%BD%9C%E4%BC%99%E4%BC%B4%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no99%E5%9F%BA%E4%BA%8E-qualcomm-snpe-sdk-%E5%BC%80%E5%8F%91-rmsnorm--%E7%AE%97%E5%AD%90)

## 2、功能目标
利用 SNPE 提供的自定义算子能力，开发 RMSNorm 算子。在 QNN HTP-simulator 运行并验证精度。

## 3、意义
- 使 SNPE 支持带有 RMSNorm 算子的模型。

# 二、飞桨现状
飞桨框架目前也没有对 RMSNorm 算子的支持。

# 三、业内方案调研
目前业内主流深度学习框架也都没有原生支持 RMSNorm 算子。

# 四、对比分析
无。

# 五、设计思路与实现方案

## 1、主体设计思路与折衷
利用 SNPE 提供的 UDO 相关支持，定义、生成并编译 RMSNorm UDO 包，生成并运行包含 RMSNorm UDO 的模型。

## 2、关键技术点/子模块设计与实现方案
- 要清楚 RMSNorm 的原理。
- 要清楚 RMSNorm UDO 的各项参数如何配置。
- 要区分好在各运行时及各精度下的参数配置。

## 3、主要影响的模块接口变化
无。

# 六、测试和验收的考量
任务要求：在 QNN HTP-simulator 运行并验证精度。（这里的模拟器，目前的理解为 AndroidStuido 的模拟器）

# 七、影响面
无。

# 八、排期规划
2023/12/15 23:59 GMT+8 前完成任务提交。

# 名词解释
- **RMSNorm** Root Mean Square layer Normalization  
- **SNPE** Snapdragon Neural Processing Engine
- **UDO** User-Defined Operations

# 附件及参考资料
- [Root Mean Square Layer Normalization 论文](https://arxiv.org/pdf/1910.07467.pdf)
- [UDO Reference Guide](https://developer.qualcomm.com/sites/default/files/docs/snpe/udo_overview.html)
- [Android 模拟器](https://developer.android.com/studio/run/managing-avds?hl=zh-cn)