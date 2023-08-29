# PaddleCV 套件开发者第四次研讨会会议纪要

## 会议概况
- 会议时间：2023-07-30 19:00
- 会议地点：线上会议
- 参会人：本次会议共有13名成员参会，由来自全国各地的飞桨套件的贡献者组成。本次会议由开发者[livingbody](https://github.com/livingbody)主持

## 会议分享与讨论

### 新人自我介绍
  * 活动经历，接触飞桨的经历，对我们活动的期待等等。
  * 新人Ligoml，微澜，姚新元，刘元灵，sunny进行了自我介绍
### 飞桨开发者进行OCR方向的开源工作分享
  * 飞桨开发者[raoyutain](https://github.com/raoyutian) 展开了有关基于PaddleOCR开源工作的介绍，包括
    * 基于PaddleOCR开发的工具PaddleOCRSharp的部署，封装逻辑
    * PaddleOCR存在的问题
      * v2.4.1模型开启MKL加速会报错
      * v2.5预测会比v2.4和v3更耗时
      * v2.5的预测库的预测精度大幅降低
    * 部分问题的解决方案和原因分析
      * 通过对比前后的文件修改，增加两行代码即可解决预测精度较低的问题
      * 推理速度慢的可能原因是C++代码中使用的识别算法仍为CRNN，而非最新的LCNet_SVRT
  * 参会人员展开讨论
    * Intel工程师姚新元分析推理速度较慢原因：PPOCRv4的新模型可能大量用到了oneDNN中尚未优化的算子，进而导致其推理速度较慢。需要百度的工程师后续排查一下. [raoyutain](https://github.com/raoyutian) 后续会分享用PaddleOCR官方代码实现的存在上述问题的代码，以便于工程师排查问题。
    * [raoyutain](https://github.com/raoyutian) 提出不带avx512指令集的机器进行推理速度会慢很多。Intel工程师姚新元提出当前的加速主要是依赖avx512指令集，因此如果芯片不支持该指令集，则无法加速了，所以最终的推理速度就会慢慢很多。
    * [会议录屏](https://meeting.tencent.com/user-center/shared-record-info?id=b1ea1b29-8ba1-4685-90c8-401c866c28a3&from=3)
### 需求和命题讨论
  * 已认领的需求讨论：
    * **单字识别**：开发者秋闻语提出了基于CTC解码算法反推单字位置的方法是否适用于SVRT这类基于Transformer的方法。飞桨实习生[ToddBear](https://github.com/ToddBear)解答了疑问：虽然SVRT基于transformer提取特征，但其回归字符的方法仍然是基于CTC的解码算法的，所以上述方法仍然是使用的。ToddBear分享了当前在中英文文档上的识别结果，当前效果已经达到了可以应用的程度。开发者[EasyIsAllYouNeed](https://github.com/EasyIsAllYouNeed)提出，当CTC算法分割的单个框占据8个像素时效果会很好，但占据4个像素时效果就会比较差。
    * **古籍识别**: 开发者[EasyIsAllYouNeed](https://github.com/EasyIsAllYouNeed)询问Paddle的训练预料是否能开源。因为在自己数据上微调后模型在原始数据集上的精度会下降。飞桨RD [shiyutang](https://github.com/shiyutang)声明后续会询问相应的开发人员是否会开源。
    * **板面矫正网络复现**: [GreatV](https://github.com/GreatV)表示推理和训练的脚本都已经写好了，但遇到一个问题：pytorch和paddle的nn.MultiHeadAttention的api不一致，可能需要重新改一下。飞桨RD [shiyutang](https://github.com/shiyutang)后续将提供修改的指导。

  * 本周提出的新需求讨论
    * **数据增强的依赖库问题**：需要确认opencv版本以解决该bug
    * **识别数学公式**：这是一个比较普遍的需求，后续可考虑实现。开发者[GreatV](https://github.com/GreatV)提出现有的公式识别算法针对的是手写的公式识别，印刷体的公式识别还没有。
    * **Parseq论文复现**：Paresq识别算法比现有的SVRT算法更好一些，数据和训练代码都已开源，大家可以考虑复现
    * **复现MobileSAM和pidNet**：轻量化的前沿分割大模型，大家可以考虑复现
    * **大模型复现赛题**：NLP方向的大模型复现，对NLP感兴趣的开发者可以参加
## 会议分享与讨论 自由发言，可以提需求给大家讨论、提问等，对活动内容的建议
  * **开发者秋闻雨提出问题**：当单行文本内存在公式或者图片时，公式和图片旁边的文字容易被漏掉。单独截取出来是可以识别出来的。飞桨RD [shiyutang](https://github.com/shiyutang)提出可以通过数据增强的方式增加一些数据，然后在这些特殊的数据上微调一下。
## 会议分享与讨论 确定下次的会议主席、call for tech presentation、召开时间
  * 确定下次的技术分享由开发者 [GreatV](https://github.com/GreatV) 进行，确定会议主席是飞桨实习生 [ToddBear](https://github.com/ToddBear) ，召开时间为8月6日的19:00

## 期待下次再相聚
下次会议我们会和进一步进行技术讨论，希望我们一起能通过代码开发、issue解答等方式让飞桨越来越好！
同时，我们也会在微信群里进行随时的交流。

