## Objectives

围绕飞桨文字识别模型套件PaddleOCR进行讨论和协作，鼓励社区开发者开发新的辅助工具和插件来改进PaddleOCR，以满足更多开发者和用户的需求。

## Recent updates

①PPOCRLabel功能完善，提供更便捷的标注体验

②垂类模型建设，提供各个场景优质模型，未来共建方向有：手写体识别、单字检测器、手写公式模型

③丰富的部署能力，覆盖所有部署方式（目前已有：C++/C#/.Net/Java/WebIO/Vue/React/Streamlit)

④学术模型测试：对PaddleOCR现有学术模型完成中文数据集训练与测试、并视情况调优

⑤移动端部署app打磨：设计与开发成熟实用的OCR IOS与安卓app

⑥多语种快递单识别

### Membership

Active Contributors: 

* [RangeKing](https://github.com/RangeKing)
* [mymagicpower](https://github.com/mymagicpower)
* [raoyutian](https://github.com/raoyutian)
* [edencfc](https://github.com/edencfc)
* [xiaxianlei](https://github.com/xiaxianlei)
* [Wei-JL](https://github.com/Wei-JL)
Contributors:

* [hao6699](https://github.com/hao6699)
* [sdcb (Zhou Jie)](https://github.com/sdcb)
* [zhiminzhang0830 ](https://github.com/zhiminzhang0830)
* [Lovely-Pig](https://github.com/Lovely-Pig)
* [livingbody](https://github.com/livingbody)
* [fanruinet (Rui Fan)](https://github.com/fanruinet)
* [bupt906](https://github.com/bupt906)
## Contribution

|Name|Introduce |link|
|:----|:----|:----|
|PPOCRLabel|适用于OCR领域的半自动化图形标注工具，内置PP-OCR模型对数据自动标注和重新识别。使用Python3和PyQT5编写，支持矩形框标注和四点标注模式，导出格式可直接用于PaddleOCR检测和识别模型的训练。|[https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/PPOCRLabel/README_ch.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/PPOCRLabel/README_ch.md)|
|FastOCRLabel|完整的C#版本标注GUI|[https://gitee.com/BaoJianQiang/FastOCRLabel](https://gitee.com/BaoJianQiang/FastOCRLabel)|
|PaddleSharp|基于PaddleOCR的C++代码修改并封装的.NET的类库。包含文本识别、文本检测、基于文本检测结果的统计分析的表格识别功能，同时针对小图识别不准的情况下，做了优化，提高识别准确率。相较于之前的项目，封装极其简化，实际调用仅一行代码.|[PaddleOCRSharp](https://gitee.com/raoyutian/paddle-ocrsharp)|
|车牌识别|支持汽车多种车牌(黄牌，双层黄牌，蓝牌，绿牌)检测和识别<br>|[https://aistudio.baidu.com/aistudio/projectdetail/2768559](https://aistudio.baidu.com/aistudio/projectdetail/2768559)|
|电表识别|在PPOCRLabel半自动标注的基础上，使用PPOCR的检测模型finetune，实现电表读数和编号的检测。|[https://aistudio.baidu.com/aistudio/projectdetail/3429765?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/3429765?channelType=0&channel=0)|
|版面恢复与文本纠错应用程序|具备识别文本纠错和恢复识别区域内文字原本的版面两大核心功能，后续会支持摄像头实时识别和在线编辑的功能|    |
|DangoOCR离线版|通用型桌面级即时翻译GUI|[PantsuDango](https://github.com/PantsuDango)|
|scr2txt|截屏转文字GUI|[lstwzd](https://github.com/lstwzd)|
|ocr_sdk|OCR java SDK工具箱|[Calvin](https://github.com/mymagicpower)|
|iocr|IOCR 自定义模板识别(支持表格识别)|[Calvin](https://github.com/mymagicpower)|
|Lmdb   Dataset Format Conversion Tool|文本识别任务中lmdb数据格式转换工具|[OneYearIsEnough](https://github.com/OneYearIsEnough)|
|用paddleocr打造一款“盗幕笔记”|用PaddleOCR记笔记|[kjf4096](https://github.com/kjf4096)|
|AI   Studio项目|英文视频自动生成字幕|[叶月水狐](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/322052)|
|id_card_ocr|身份证复印件识别|[baseli](https://github.com/baseli)|
|Paddle_Table_Image_Reader|能看懂表格图片的数据助手|[thunder95](https://github.com/thunder95%5D)|
|AI   Studio项目|OCR流程中对手写体进行过滤|[daassh](https://github.com/daassh)|
|AI   Studio项目|电表读数和编号识别|[深渊上的坑](https://github.com/edencfc)|
|AI   Studio项目|LCD液晶字符检测|[Dream拒杰](https://github.com/zhangyingying520)|
|paddleOCRCorrectOutputs|获取OCR识别结果的key-value|[yuranusduke](https://github.com/yuranusduke)|
|optlab|OCR前处理工具箱，基于Qt和Leptonica。|[GreatV](https://github.com/GreatV)|
|PaddleOCRSharp|PaddleOCR的.NET封装与应用部署。|[raoyutian](https://github.com/raoyutian/PaddleOCRSharp)|
|PaddleSharp|PaddleOCR的.NET封装与应用部署，支持跨平台、GPU|[sdcb](https://github.com/sdcb)|
|PaddleOCR-Streamlit-Demo|使用Streamlit部署PaddleOCR|[Lovely-Pig](https://github.com/Lovely-Pig)|
|PaddleOCR-PyWebIO-Demo|使用PyWebIO部署PaddleOCR|[Lovely-Pig](https://github.com/Lovely-Pig)|
|PaddleOCR-Paddlejs-Vue-Demo|使用Paddle.js和Vue部署PaddleOCR|[Lovely-Pig](https://github.com/Lovely-Pig)|
|PaddleOCR-Paddlejs-React-Demo|使用Paddle.js和React部署PaddleOCR|[Lovely-Pig](https://github.com/Lovely-Pig)|
|AI   Studio项目|StarNet-MobileNetV3算法–中文训练|[xiaoyangyang2](https://github.com/xiaoyangyang2)|
|ABINet-paddle|ABINet算法前向运算的paddle实现以及模型各部分的实现细节分析|[Huntersdeng](https://github.com/Huntersdeng)|

### Meetings 

How to join the meeting ：每双周周三晚19:00-20:00

### Contacts

填写申请表：[https://paddle.wjx.cn/vj/hoZnW83.aspx](https://paddle.wjx.cn/vj/hoZnW83.aspx)

**CONTRIBUTING**

[PaddleOCR贡献指南](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/code_and_doc.md/#%E9%99%84%E5%BD%953)

