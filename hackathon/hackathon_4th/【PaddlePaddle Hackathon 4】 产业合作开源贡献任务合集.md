# 【PaddlePaddle Hackathon 4】产业合作开源贡献任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

* 本赛道欢迎众多队伍踊跃报名，不锁定单支队伍，5.15截止提交。5.15-5.19期间评审公示，根据各队伍的综合成绩择优选择一支队伍获奖。

* Tips：AI Studio项目公开时间也是评奖时的考虑因素，如果两支队伍方案和效果相似，倾向选择优先完成（以AI Studio项目最新版本公开时间为准）的队伍。

### No.236：基于PaddleOCR的工频场强计读数识别 <a name='task236'></a>

* 技术标签：深度学习，Python，OCR
* 任务难度：基础⭐️
* 详细描述：
  * 任务：识别工频场图像中的工频电磁场数值和单位、以及下方X\Y\Z的数值，要求结构化输出结果: `[  {"Info_Probe":""}, {"Freq_Set":""}, {"Freq_Main":""}, {"Val_Total":""},{"Val_X":""}, {"Val_Y":""}, {"Val_Z":""}, {"Unit":""}, {"Field":""} ]`
 输出示意图片:<img src="https://user-images.githubusercontent.com/11793384/223640728-f174f542-0b69-434a-925a-1844a2b410f2.png" width="40%" height="40%" />  <img src="https://user-images.githubusercontent.com/11793384/223640767-243c36cc-ce01-43ba-a0f1-eac051d785f3.png" width="30%" height="30%" /> 


  * 数据集
     * 训练集：包含100张原始图片，数据集[下载链接](https://paddleocr.bj.bcebos.com/dataset/digital_rec_hackon_train.zip)；不提供标签信息，可以结合 PPOCRLabel等标注工具构建训练数据并进行模型微调；可以使用数据生成方法批量生成识别数据。
     * 测试集：评审前公开
  * 方案建议（不局限于此）
      * 基于PaddleOCR中的PP-OCR模型微调
* 评分方式（根据以下几项综合评估）
  * 测试集H-means值，需大于90%（分数占比80%）
  * 技术方案的可用性和可扩展性、产业落地价值和影响、项目作为教程的详细程度（分数占比20%）
* 提交内容：
  * AI Studio 项目（遵从[模板规范](https://aistudio.baidu.com/aistudio/projectdetail/5520056)），全部代码和使用数据需开源，效果完整可复现
* 技术要求：
  * 熟练掌握 Python 开发
  * 熟悉 OCR 算法
  
### No.237：基于PaddleNLP的跨模态文档信息抽取 <a name='task237'></a>

* 技术标签：深度学习，Python，NLP
* 任务难度：基础⭐️
* 详细描述：
  * 结合OCR及NLP技术实现机动车发票信息结构化
  * 目标抽取字段：
     *  共包含11种实体标签：购买方名称、车辆类型、厂牌型号、产地、发动机号、销售单位名称、纳税人识别号、开户银行、增值税税额、不含税价、主管税务机关及代码
  * 数据集
     *  可进行标注用来训练的数据，包含30张原始图片，[数据集下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/191867)
      * 数据集说明：不提供标签信息，可以结合 PPOCRLabel、Label-Studio 等标注工具构建训练数据并进行模型微调
     * 带标注的数据，仅可作为测试集使用，不可作为训练集。评审前公开
  * 方案建议（不局限于此）
     * 基于PaddleOCR PPStructure训练定制版面分析模型，优化发票场景的版面分析能力，提升UIE-X跨行抽取效果
     * 基于PaddleNLP UIE-X实现在小样本数据集下的文档抽取模型定制，可使用PaddleNLP 提供的数据协议基于 Label-Studio 对原始文档进行标注
* 评分方式（根据以下几项综合评估）
  * 测试集F1值（50%）
  * 技术方案的可用性和可扩展性、产业落地价值和影响、项目作为教程的详细程度（50%）
* 提交内容：
  * AI Studio 项目（遵从[模板规范](https://aistudio.baidu.com/aistudio/projectdetail/5520056)），全部代码和使用数据需开源，效果完整可复现
* 技术要求：
  * 熟练掌握 Python 开发
  * 熟悉 NLP 算法

### No.238：基于PaddleClas的中草药识别 <a name='task238'></a>

* 技术标签：深度学习，Python，图像分类
* 任务难度：基础⭐️
* 详细描述：
  * 任务：训练一个中草药分类模型
  * 数据集：按照[开源数据集](https://aistudio.baidu.com/aistudio/datasetdetail/105575)划分的训练集和测试集进行实验，可自行扩增训练集数目，不可将测试集用于训练
  * 方案建议：可以使用包括但不限于PaddleClas中的基础模型库，产业特色解决方案（PP-LCNet、PULC等）
* 评分方式（根据以下几项综合评估）
  * 测试集ACC大于0.842（分数占比80%）
  * 技术方案的可用性和可扩展性、产业落地价值和影响、项目作为教程的详细程度（分数占比20%）
* 提交内容：
  * AI Studio 项目（遵从[模板规范](https://aistudio.baidu.com/aistudio/projectdetail/5520056)），全部代码和使用数据需开源，效果完整可复现
* 技术要求：
  * 熟练掌握Python开发
  * 熟悉 PaddleClas或者相关图像分类模型算法。

### No.239：基于PaddleDetection的无人机航拍图像检测 <a name='task239'></a>


* 技术标签：深度学习，Python，Detection
* 任务难度：基础⭐️
* 详细描述：
  * 使用PaddleDetection套件提升无人机航拍图像的检测精度并实现跟踪功能
  * 数据集
     * VisDrone-DET是一个无人机航拍场景的小目标数据集，整理后的COCO格式VisDrone-DET数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone.zip)，切图后的COCO格式数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone_sliced.zip)，检测其中的10类，包括 pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)，原始数据集[下载链接](https://github.com/VisDrone/VisDrone-Dataset)。
  * 方案建议（不局限于此）
     * 基于PP-YOLOE-SOD进行小目标检测，https://github.com/PaddlePaddle/PaddleDetection/tree/v2.6.0/configs/smalldet/visdrone
     * 基于PP-Human/ PP-Vehicle/ PP-Tracking中已实现的追踪算法进行迁移，https://github.com/PaddlePaddle/PaddleDetection/tree/v2.6.0/deploy/pipeline
* 评分方式（根据以下几项综合评估）
  * 检测模型，使用原图评估，cocoapi mAP 0.5:0.95大于42.5
  * 跟踪模型，效果稳定
  * 算法实时性较好，可以在T4或同等算力硬件平台上大于20FPS
  * 技术方案的可用性和可扩展性、产业落地价值和影响、项目作为教程的详细程度（评分占比20%）
* 提交内容：
  * AI Studio 项目（遵从[模板规范](https://aistudio.baidu.com/aistudio/projectdetail/5520056)），全部代码和使用数据需开源，效果完整可复现
* 技术要求：
  * 熟练掌握 Python 开发
  * 熟悉 Detection, Tracking算法


### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&社群的通知，及时参与。
