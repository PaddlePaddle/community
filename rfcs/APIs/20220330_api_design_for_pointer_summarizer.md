# 基于pointer_summarizer的中文文本摘要


|API名称 | 新增API名称 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 王骐昊 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-01 | 
|版本号 | V1.0
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220330_api_design_for_pointer_summarizer.md<br> | 


# 一、概述
## 1、相关背景
文本摘要是一类经典的NLP任务。本任务是为了实现基于pointer_summarizer方法的摘要任务，并且完成在LCSTS_new中文数据集上的开发例子。
## 2、功能目标
在paddlenlp repo中[text_summarization](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)下增加基于pointer_summarizer在中文数据集上LCSTS_new数据集上实现的文本摘要任务，并且达到ROUGE-1 在验证集0.3553，测试集0.3396的指标。
## 3、意义
为飞桨提供了基于pointer_summarizer的中文文本摘要的支持，丰富飞桨在text_summarization任务下的代码案例。

# 二、飞桨现状
飞桨框架中目前有pointer_summarizer方法在CNN-DailMail上的[实现](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_summarization/pointer_summarizer),但是没有对强化学习部分方法的实现。同时飞桨中没有针对LCSTS_new中文数据集的文本摘要案例。

# 三、业内方案调研
针对[论文](https://arxiv.org/abs/1704.04368)在Pytorch、TensorFlow等主流深度学习框架上都有实现代码但大多在CNN-DailMail数据集上实现，其中Pytorch中有该方法针对LCSTS_New数据集的实现，即该[参考repo](https://github.com/LowinLi/Text-Summarizer-Pytorch-Chinese)。但是需要注意的是该repo使用了强化学习方法进行优化，提高了文本的可读性和在长句上的表现，得到了在测试集上的指标。
另一种在LCSTS数据集上受到较多关注的方法是基于[论文](https://arxiv.org/pdf/1708.00625.pdf),在Pytorch上的实现参考[repo](https://github.com/lipiji/DRGD-LCSTS)




# 四、对比分析
考虑主要参考该[repo](https://github.com/LowinLi/Text-Summarizer-Pytorch-Chinese)进行对应方法的实验，并实验得到相应的指标。

# 五、设计思路与实现方案

## 命名与参数设计
无
## 底层OP设计
无
## API实现方案
pointer_summarizer部分参考飞桨框架中的实现，强化学习部分考虑参考Pytorch中实现，在飞桨框架中完成相应功能的编写。
# 六、测试和验收的考量
根据验证集和测试集上的ROUGE-1进行验收。

# 七、可行性分析和排期规划
对应方法在飞桨中都有实现案例，不存在可行性问题。数据集上的指标也被多次复现达到。

week1:数据处理和MLE部分训练，达到验证集上指标。

week2:编写RL部分代码以及进行对应训练和调优。

week3:完善实验以及编写说明文档。

# 八、影响面
无

# 名词解释
无
# 附件及参考资料
无