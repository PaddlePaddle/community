# PaddleCV 套件开发者第五次研讨会会议纪要

## 会议概况
- 会议时间：2023-08-06 19:00
- 会议地点：线上会议
- 参会人：本次会议共有8名成员参会，由来自全国各地的飞桨套件的贡献者组成。本次会议由飞桨实习生[ToddBear](https://github.com/ToddBear)主持

## 会议分享与讨论

### 新人自我介绍
  * 活动经历，接触飞桨的经历，对我们活动的期待等等。
  * 本次会议没有新开发者参与，因此跳过该环节
### 飞桨开发者进行OCR方向的开源工作分享
  * 飞桨开发者[GreatV](https://github.com/GreatV) 展开了有关版面矫正网络DocTr++论文复现经验分享
  * 飞桨RD[shiyutang](https://github.com/shiyutang)提问：
    * Q: 训练是否已经在进行? A：正在用小批量的数据集验证）
    * Q: 测试集是否能获取? A: 已经用尝试划分三种数据进行测试）
     * Q: 训练时间较长的原因 A: 数据集较大，主要是数据读取比较耗时
  * [会议录屏](https://meeting.tencent.com/user-center/shared-record-info?id=b3d1dcff-0b52-4467-88e4-52d2f11c16cf&from=3)
### 需求和命题讨论
  * 本周提出的新需求讨论
    * 支持小语种-藏文: 看看V4是否支持小语种，如果没有，后续调研相关数据集后可以考虑增加该需求
    * 给fastdeploy服务化部署 的方式提供修改参数: 文档没写，有一部分参数是可以修改，请开发者[EasyIsAllYouNeed](https://github.com/EasyIsAllYouNeed)在Issue#10562下分享具体的操作方法
    * fastdeploy ios版本编译: 更像是FD的需求，可以跟FD的RD询问一下
## 会议分享与讨论 自由发言，可以提需求给大家讨论、提问等，对活动内容的建议
  * 飞桨PM[Ligoml](https://github.com/Ligoml): 
    * 开会频次：可以两周开一次会，灵活一些
    * 拉新：大家可以推广宣传一下研讨会，争取更多开发者参与（可以在社交平台上宣传和分享，在Feature Request页面上贴上链接，或者直接Pin到PaddleOCR的Issue上）
## 会议分享与讨论 确定下次的会议主席、call for tech presentation、召开时间
  * 确定下次的技术分享由飞桨PM[Ligoml](https://github.com/Ligoml)进行，确定会议主席是开发者[GreatV](https://github.com/GreatV) ，召开时间暂定为8月16日19:00

## 期待下次再相聚
下次会议我们会和进一步进行技术讨论，希望我们一起能通过代码开发、issue解答等方式让飞桨越来越好！
同时，我们也会在微信群里进行随时的交流。

