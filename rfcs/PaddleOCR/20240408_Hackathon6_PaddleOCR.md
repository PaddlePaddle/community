# 解决 PaddleOCR 中的长期存在的 issue

|任务名称|解决 PaddleOCR 中的长期存在的 issue|
|------|------|
|提交作者|Liyulingyue、GreatV|
|提交时间|2024-04-08|
|版本号|v0.1|
|依赖飞桨版本|Unknow|
|文件名| 20240408_Hackathon6_PaddleOCR.md|

## 一、概述
飞桨套件曾经凭借其出色的性能吸引了诸多的开发者，但随着社区发展，这些套件中的一些历史问题也暴露出来，例如代码Bug没有及时处理、功能缺失、兼容性不足等。
以PaddleOCR 项目为例，该套件有非常多的使用者，在 issue 区的讨论也很多。甚至有不少 issue 已经是长期存在的 issue。这些 issue 缺少诊断，复现，以及修复。因此，期望能够挑选部分长期存在的，讨论较多的issue，能够进行分析、复现并解决。

## 二、现状
我们对前70条Issue进行了统计和分类，部分统计结果见附件。根据这些Issue，当前的PaddleOCR Issue中存在的问题主要由以下几部分组成：
1. **多语言OCR开发**:当前有很多用户希望基于他们工 作所使用到的语言、他们的母语环境训练特定语言的OCR识别模型，但由于文档缺失/依赖不明确等问题，导致训练工作并不顺利、训练后的模型准确性较低。
2. **兼容性问题**：由于平台、GPU环境、文档缺失、依赖不明确的问题，导致部分用户无法在自己的电脑上运行PaddleOCR，因此需要针对该问题进行完善，例如明确依赖关系、指出不同平台的依赖性。
3. **OCR结果的准确性和数据缺失**：在不同的环境下，训练、导出、推理都可能带来一定的精度损失，部分用户的Issue中表明OCR推理有时候会存在丢失特定区域文字的现象。
4. **参考文档缺失或不明确**：在代码的开发过程中，文档工作没有被很好的跟进，从而导致用户无法获得完整的代码功能说明，并给用户的推理、训练工作带来较大的阻碍。
5. **其他未解决的BUG**
   


## 三、任务列表
根据统计结果，我们挑选了部分长期存在的issue，进行复现和解决。该列表如下所示：

|Issue|说明|建议|
|---|---|---|
|https://github.com/PaddlePaddle/PaddleOCR/issues/10760|关于模型微调和cuda版本不匹配的问题||
|https://github.com/PaddlePaddle/PaddleOCR/issues/10685|更新Backbone后无法运行，解决了一部分|跑通此逻辑，梳理相关文档|
|https://github.com/PaddlePaddle/PaddleOCR/issues/10288|打包后GPU模式下无法运行|需要查验|
|https://github.com/PaddlePaddle/PaddleOCR/issues/10197|关于模型自动下载的问题，和解码问题||
|https://github.com/PaddlePaddle/PaddleOCR/issues/6559|内存泄露||
|https://github.com/PaddlePaddle/PaddleOCR/issues/11149|训练问题|可以针对此问题增加文档说明|
|https://github.com/PaddlePaddle/PaddleOCR/issues/11551|导出报错||
|https://github.com/PaddlePaddle/PaddleOCR/issues/10499|文本检测训练完进行单张图片预测时发现漏检内容||
|https://github.com/PaddlePaddle/PaddleOCR/issues/8743|Code doesn't work with numpy>=1.24|升级一下对基础依赖的适配|
|https://github.com/PaddlePaddle/PaddleOCR/issues/11441|训练时的shape问题，可以增加一下README||

## 四、预期时间和验收标准
预期工作计划如下：
- 4月：完成较为简单的Issue的处理
- 5月：开启难度较高的Issue的迭代更新工作
- 6月：完成所有Issue的处理

验收标准：
- 功能需求类的Issue：完成相关功能，并提交代码到PaddleOCR仓库
- Bug修复类Issue：完成相关Bug修复，并提交代码到PaddleOCR仓库，合入后，无Bug产生
- 文档类Issue：完成相关文档，并提交代码到PaddleOCR仓库，合入后，其他开发者能够正常查阅文档并根据文档进行工作

## 五、影响面
有助于括扩大语言支持、提高模型训练和准确性、确保跨不同平台的兼容性，以及提供调试和性能优化的明确指导。

## 六、附件

OCR Issue列表与描述：

|序号|isssue|描述|分类|进度|难度|
|:------:|:------:|------|:------:|:------:|:------:|
|1|https://github.com/PaddlePaddle/PaddleOCR/issues/11031|阿拉伯语数据集训练出现问题|多语言训练|已基本解决|普通|
|2|https://github.com/PaddlePaddle/PaddleOCR/issues/11079|Mac M2 上无法运行|硬件兼容性问题|未解决|具有挑战性|
|3|https://github.com/PaddlePaddle/PaddleOCR/issues/10270|PPStructure版面分析得到的结果，bbox里OCR的结果缺失最后一行|结果的准确性|已解决|-|
|4|https://github.com/PaddlePaddle/PaddleOCR/issues/11815|pubtab_dataset 加载模型图片时会堵塞|疑似bug|未解决|适中|
|5|https://github.com/PaddlePaddle/PaddleOCR/issues/8761|表格识别模型微调训练效果比较差|结果的准确性|待确认|适中|
|6|https://github.com/PaddlePaddle/PaddleOCR/issues/10265|PaddleOCR无法和yolov8共同安装|兼容性|基本解决&待验证|普通|
|7|https://github.com/PaddlePaddle/PaddleOCR/issues/11639|PaddleOCR内存泄露|内存泄露|进行中|具有挑战性|
|8|https://github.com/PaddlePaddle/PaddleOCR/issues/11056|PPOCRLabel运行出bug|bug|未解决|适中|
|9|https://github.com/PaddlePaddle/PaddleOCR/issues/10760|cuda报错:Hint: 'CUDNN_STATUS_NOT_SUPPORTED'|兼容性问题|基本解决|适中|
|10|https://github.com/PaddlePaddle/PaddleOCR/issues/11775|PaddleOCRv2在Android里面识别不了整行文字|模型库更新|基本解决|适中|
|11|https://github.com/PaddlePaddle/PaddleOCR/issues/11530|PaddleOCR无法在docker中运行|兼容性|未解决&待复现|适中|
|12|https://github.com/PaddlePaddle/PaddleOCR/issues/10700|训练ch_PP-OCRv4_rec_distill.yml，各种报错|更新文档|未解决|适中|
|13|https://github.com/PaddlePaddle/PaddleOCR/issues/10685|PPLCNetNew找不到对应模型|更新文档|未解决|适中|
|14|https://github.com/PaddlePaddle/PaddleOCR/issues/10476|使用预训练模型进行版面分析没有输出|版面分析|未解决&待复现|适中|
|15|https://github.com/PaddlePaddle/PaddleOCR/issues/10422|PPOCR-V3模型识别经常漏行|结果准确性|基本解决|普通|
|16|https://github.com/PaddlePaddle/PaddleOCR/issues/10669|PyMuPDF无法安装|第三方库|已解决|适中|
|17|https://github.com/PaddlePaddle/PaddleOCR/issues/10652|训练ser任务报错|兼容性|基本解决|普通|
|18|https://github.com/PaddlePaddle/PaddleOCR/issues/10358|提升阿拉伯语识别精度|多语言训练|基本解决|适中|
|19|https://github.com/PaddlePaddle/PaddleOCR/issues/10288|识别结果异常|兼容性|未解决&待复现|适中|
|20|https://github.com/PaddlePaddle/PaddleOCR/issues/10197|不断下载ch_PP-OCRv3_det_infer.tar|网络问题|基本解决|适中|
|21|https://github.com/PaddlePaddle/PaddleOCR/issues/9830|训练图和label 里如何让每个字符的出现频率类似|数据合成|基本解决|普通|
|22|https://github.com/PaddlePaddle/PaddleOCR/issues/9761|MAC M1 PRO无法安装PaddleOCR|兼容性|已解决|适中|
|23|https://github.com/PaddlePaddle/PaddleOCR/issues/11749|无法在Ubuntu 20上编译libpaddle_inference|编译|未解决&待复现|适中|
|24|https://github.com/PaddlePaddle/PaddleOCR/issues/11706|无法在mac上运行paddleocr|兼容性|进行中&待复现|适中|
|25|https://github.com/PaddlePaddle/PaddleOCR/issues/10444|如何切换不同版面分析模型|版面分析|待复现|适中|
|26|https://github.com/PaddlePaddle/PaddleOCR/issues/10378|模块计算机类型“x64”与目标计算机类型“x86”冲突|编译|已解决|普通|
|27|https://github.com/PaddlePaddle/PaddleOCR/issues/10346|PaddleOCR v4模型使用Mkldnn在非AVX512 CPU上变得非常慢|模型性能|未解决&待复现|适中|
|28|https://github.com/PaddlePaddle/PaddleOCR/issues/9821|crnn训练，怎么合并多种字体的模型比较好？|基本解决|适中|
|29|https://github.com/PaddlePaddle/PaddleOCR/issues/8938|在mac M1上运行卡住|兼容性|未解决&待复现|适中|
|30|https://github.com/PaddlePaddle/PaddleOCR/issues/6559|单机多卡使用多进程内存不释放|内存泄露|未解决&待复现|具有挑战性|
|31|https://github.com/PaddlePaddle/PaddleOCR/issues/11849|ch_PP-OCRv4_rec_hgnet.yml 用这个模型转换成onnx进行识别，速度超级慢|模型性能|基本解决|适中|
|32|https://github.com/PaddlePaddle/PaddleOCR/issues/11763|从哪下载 ubuntu20.04 libpaddle_inference|安装下载|基本解决|普通|
|33|https://github.com/PaddlePaddle/PaddleOCR/issues/11149|ppocrv4训练，配置文件有误|更新文档|未解决&待验证|适中|
|34|https://github.com/PaddlePaddle/PaddleOCR/issues/6559|使用gpu=True没有预测结果|兼容性|进行中|适中|
|35|https://github.com/PaddlePaddle/PaddleOCR/issues/10438|使用命令行没有预测结果|硬件兼容性|未解决|适中|
|36|https://github.com/PaddlePaddle/PaddleOCR/issues/11551|导出报错|文档|未解决|普通|
|37|https://github.com/PaddlePaddle/PaddleOCR/issues/10499|文本检测训练完进行单张图片预测时发现漏检内容|未解决|适中|
|38|https://github.com/PaddlePaddle/PaddleOCR/issues/8743|Code doesn't work with numpy>=1.24|兼容性|未解决|困难|
|39|https://github.com/PaddlePaddle/PaddleOCR/issues/11441|训练时的shape问题，可以增加一下README|文档缺失|未解决|适中|
