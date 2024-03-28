# PaddleCV 套件开发者第三次研讨会会议纪要

## 会议概况
- 会议时间：2023-07-19 19:00
- 会议地点：线上会议
- 参会人：本次会议共有9名成员参会，由来自全国各地的飞桨套件的贡献者组成。本次会议由本次轮值主席李昊轩[KevinLi43](https://github.com/KevinLi43)主持

## 会议分享与讨论

### 新人自我介绍
  * 活动经历，接触飞桨的经历，对我们活动的期待等等。
  * 新人DEYi进行了自我介绍（刚刚接触Paddle，主要对关键信息提取KIE比较感兴趣），欢迎加入研讨！
### 飞桨RD进行Seg方向科普分享
  * 飞桨RD [shiyutang](https://github.com/shiyutang) 进行了有关Seg代码框架，全流程以及三个垂类方向应用的分享
  * [会议录屏](https://meeting.tencent.com/user-center/shared-record-info?id=40109da3-47dc-4139-9a37-e1bdeea21aae&from=3)
### 需求和命题讨论
  * 本周提出的新需求讨论
    * [ToddBear](https://github.com/ToddBear) 分享了有关改进KIE以应对长文本输入的新需求
    * [livingbody](https://github.com/livingbody) 分享了有关改进介绍文档的新需求（现有的文档比较简单，有时不容易知道具体的做法）。[shiyutang](https://github.com/shiyutang) 认为该需求是合理的，可以起到文档的细化的改进的作用
    * [onecatcn](https://github.com/onecatcn) 提出有关新语言OCR模型的训练的需求，现在有2-3个模型需要重新训练，开发者可以根据现有的教程进行模型的训练，欢迎开发者开发新语言的OCR模型
  * 命题任务讨论
    * [GreatV](https://github.com/GreatV) 负责的是板面矫正网络DocTr++的复现，目前已经完成网络代码的转化，已经在下载数据集。由于数据集过大，可以先下一部分数据集，然后尝试把训练的代码跑通
    * [ToddBear](https://github.com/ToddBear) 负责的是文字识别返回单字识别坐标的任务，目前还在验证CTC方案的可行性，本周应该能确认基于CTC解码的结果是否能定位每个字符所在的位置。
    * [livingbody](https://github.com/livingbody) 负责的套件一致性计划的任务，目前已经完成了
    * [WilliamQf-AI](https://github.com/WilliamQf-AI)提出了C++版的板面恢复功能的任务。由于他今天没有来，所以就暂时没有讨论
    * [shiyutang](https://github.com/shiyutang) 展开了有关新增生僻字模型任务的讨论：是否能做一下古籍识别，以及是否需要扩充一下字典。[livingbody](https://github.com/livingbody) 提出古籍中的一些字在通用汉语表中是没有的
## 会议分享与讨论 自由发言，可以提需求给大家讨论、提问等，对活动内容的建议
  * DEYi提出可以在现有的PPOCRLabel标注工具中加入有关绑定实体间的QA关系的功能。 [shiyutang](https://github.com/shiyutang) 认为该功能可以通过修改前端代码的方式来完成。也可以不修改前端代码，而是在利用PPOCRLabel工具标注时就把QA关系额外记录，然后再通过后处理进行提取。
## 会议分享与讨论 确定下次的会议主席、call for tech presentation、召开时间
  * 确定下次的技术分享由开发者 [raoyutain](https://github.com/raoyutian) 进行，确定会议主席是开发者 [livingbody](https://github.com/livingbody) ，召开时间暂定为下周末，具体时间需要等待群投票

## 期待下次再相聚
下次会议我们会和进一步进行技术讨论，希望我们一起能通过代码开发、issue解答等方式让飞桨越来越好！
同时，我们也会在微信群里进行随时的交流。

