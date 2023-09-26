# PaddleCV 套件开发者第七次研讨会会议纪要

## 会议概况

- 会议时间：2023-09-10 19:00
- 会议地点：线上会议
- 参会人：本次会议共有11名成员参会，由来自全国各地的飞桨套件的贡献者组成。本次会议由开发者[Ligoml](https://github.com/Ligoml)主持。

## 会议分享与讨论

### 飞桨实习生[ToddBear](https://github.com/ToddBear)分享论文复现经验

  * 论文复现1.0时期，需要参考[《论文复现指南》](https://github.com/PaddlePaddle/models/blob/8042c21b690ffc0162095e749a41b94dd38732da/tutorials/article-implementation/ArticleReproduction_CV.md)逐步打卡，完成全流程对齐
  * 现在进入论文复现2.0时期，借助自动化工具 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 与 [PaDiff](https://github.com/PaddlePaddle/PaDiff) ，可以提升复现效率
    * 需要重点注意：套件层面的代码迁移、模型训练、部署推理和文档撰写

  * [会议录屏](https://meeting.tencent.com/v2/cloud-record/share?id=d14d569c-850b-4e12-ab5d-65c08a67c243&from=3)

### 需求和命题讨论

  * 本周提出的新需求讨论
    * 两条部署相关需求，需要转FastDeply：
      * https://github.com/PaddlePaddle/PaddleOCR/issues/10334#issuecomment-1694977766 
      * https://github.com/PaddlePaddle/PaddleOCR/issues/10334#issuecomment-1696691945
    * 一条需求不清晰，需要更多说明
      * https://github.com/PaddlePaddle/PaddleOCR/issues/10334#issuecomment-1707495322
    * 一条需求在沟通后，可以通过社区任务doctr++的复现解决
      * https://github.com/PaddlePaddle/PaddleOCR/issues/10334#issuecomment-1707495322

## 会议分享与讨论 确定下次的会议主持人、call for tech presentation、召开时间

  * 确定下次的会议主持人[Lylinnnnn](https://github.com/Lylinnnnn)，议题待定。

## 期待下次再相聚

下次会议我们会和进一步进行技术讨论，希望我们一起能通过代码开发、issue解答等方式让飞桨越来越好！
同时，我们也会在微信群里进行随时的交流。