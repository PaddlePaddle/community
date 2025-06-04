## 第二十一次 PaddleOCR 开源共建研讨会

* 会议时间：2025-06-04 19:30
* 本次会议接入方式：
    * 腾讯会议：383-5276-3788
    * [点击链接入会](https://meeting.tencent.com/dm/xVCAma7fohAo)，或添加至会议列表
* 本次拟参会 member list：
    * [cuicheng01](https://github.com/cuicheng01)
    * [jzhang533](https://github.com/jzhang533)
    * [Topdu](https://github.com/Topdu)
    * [greatv](https://github.com/greatv)
    * [Gmgge](https://github.com/Gmgge)
    * [SWHL](https://github.com/SWHL)
    * [Liyulingyue](https://github.com/Liyulingyue)
    * [luotao1](https://github.com/luotao1)
    * [mattheliu](https://github.com/mattheliu)
    * [UserWangZz](https://github.com/UserWangZz)
    * [jingsongliujing](https://github.com/jingsongliujing)
    * [Alex Zhang](https://github.com/openvino-book)

## 会议议程

### 0. 新人介绍

- @E-Pudding @lxw112190 @TingquanGao

### 1.PaddleOCR 3.0 发版后的问题的专项讨论

| 问题                                                         | 是否已经解决                                            |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| 转换ONNX模型时遇到Unsupported IR model IR version: 11, max supported IR version: 10错误 | 是，在3.0.1体现                                         |
| 在CPU环境中仍然会尝试获取GPU信息，并导致报错。               | 是，在3.0.1体现                                         |
| PP-OCRv5_server_det模型GPU推理，当输入张量尺寸较大时报错     | 框架未解决，但套件已修改默认行为绕过此问题，在3.0.1体现 |
| 构造PaddleOCR对象时，如果指定了lang或者ocr_version, 即便设置text_detection_model_name='PP-OCRv5_server_det', text_recognition_model_name='PP-OCRv5_server_rec'，实际使用的模型也会是PP-OCRv5_mobile_det, PP-OCRv5_mobile_rec. | 是，在3.0.1体现                                         |
| PP-OCRv5、PP-StructureV3等执行GPU推理后，对象析构时报错;     | 是，在3.0.1体现                                         |
| PPStructureV3对象缺少concatenate_markdown_pages方法，导致文档中的示例跑不通 | 是，在3.0.1体现                                         |
| 部分图像上的预测结果较PP-OCRv4更差                           | 是，默认推理配置不合理，大部分已经解决，在3.0.1体现     |
| 因numpy、pandas等依赖的限制，Python 3.12安装PaddleOCR 3.0失败 | 否，解决中                                              |

### 2. 其他议题

- PaddleOCR 社区运营进展 @E-Pudding
- PaddleOCR on HuggingFace
    - https://huggingface.co/spaces/PaddlePaddle/PP-OCRv5_Online_Demo
    - https://huggingface.co/spaces?category=ocr
    - Planned to release a joint blog with HuggingFace team.


### 3. 自由发言，可以提需求给大家讨论、提问等

### 4. 确定下次的部分议题及召开时间

### 轮值值班安排

* 职责：处理 issue 区和 Discussion 区的新增问题，召集，主持例会并记录。
* 期间：每隔两周轮换。
* 排班：@greatv, @SWHL, @jingsongliujing, @cuicheng01， @jzhang533, @mattheliu， @Topdu

序号|GITHUB ID|值班开始|值班结束
:------:|:------|:------:|:------:
1|@greatv|2025-03-06|2025-03-19
2|@SWHL |2025-03-20|2025-04-02
3|@jingsongliujing |2025-04-03|2025-04-23
4|@cuicheng01 |2025-04-17|2025-05-16
5|@jzhang533 |2025-05-17|2025-06-03
6|@mattheliu |2025-06-04|2025-06-18
7|@Topdu |2025-06-19|2025-07-02
