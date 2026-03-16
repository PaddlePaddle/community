> 模版：进阶任务：为 OpenVINO notebook 新增 demo 示例
> * 技术标签：深度学习框架，Python，OpenVINO
> * 任务难度：进阶
> * 详细描述：作为深度学习工具套件，OpenVINO 可以被广泛应用于不同的应用场景，实现 AI 模型的推理部署，为此我们也想收集更多基于 PaddlePaddle 模型所打造的优秀应用案例，丰富示例仓库。 在这个任务中，你需要在 OpenVINO notebook 仓库新增一个 notebook 示例。本次任务评估将分为两个阶段，在第一阶段中，开发者需要提供一份 RFC，用来描述本次任务的设计方案； 在第二阶段中，我们将从第一阶段提交的结果中，挑选出 2 份比较优秀的方案，并请相对应的开发者根据自己的方案提交 PR。
> * 提交内容：
>    * 第一阶段：RFC 方案提交
>      1. 提交方式：1）以 issue 的形式进行提交到[这儿](https://github.com/openvinotoolkit/openvino_notebooks/issues)，2）标题处打上【PaddlePaddle Hackathon 10】，3）RFC 语言不做强制要求
>      2. 基本要求：1）应用场景与现有 notebook demo 不重复，2）该示例中需要使用最新版本的 openvino 完成所有模型的推理部署
>      3. 筛选依据：1）该示例在真实场景下是否具有实际应用价值，2）该示例的流程逻辑是否清晰，3）运行结果是否符合预期
>         
>    * 第二阶段：PR代码提交
>      1. 提交地址： https://github.com/openvinotoolkit/openvino_notebooks ，标题加上【PaddlePaddle Hackathon 10】字样，并在描述处链接之前的 RFC 地址
>      2. 该 PR 需满足 notebook 贡献规范，开发者需要及时根据 review 的结果进行 PR 修改
>      3. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排
> * 参考示例：考虑到通用性，选取的应用场景尽量以英文为主，推荐方案场景有：
>   * PaddleDetection 1）行为识别（打架，抽烟，接打电话 ) 2）车辆追踪 3）高空抛物 4）产线上包装盒内产品计数
>   * PaddleOCR 1）古籍电子化保存
> * 技术要求：熟练掌握OpenVINO python API与其他工具组件的使用方法
> * 参考文档：[OpenVINO notebook仓库](https://github.com/openvinotoolkit/openvino_notebooks)、[OpenVINO notebook仓库代码贡献规范](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/CONTRIBUTING.md)

## 赛题详情（厂商排名不分先后）
### 请 Intel 填写

### 请 AMD 填写

### 请 Arm 填写

### 天数智芯：进阶任务：基于天数智芯硬件与文心多模态模型的创新应用
* 技术标签：深度学习框架，Python，文心大模型，多模态
* 详细描述：本任务旨在利用天数智芯硬件(BI-150S)的算力优势，结合文心系列多模态模型，打造具有真实落地价值、逻辑闭环且体验优秀的创新案例。开发者可**任选其一**或组合使用以下模型进行应用开发：**ERNIE-4.5-VL-28B-A3B-Thinking** 与 **PaddleOCR-VL-1.5**，参考 [飞桨 AI Studio 应用案例库](https://aistudio.baidu.com/topic/applications)。本次任务评估将分为两个阶段，在第一阶段中，开发者需要提供一份 RFC，用来描述本次任务的设计方案及预期性能指标；在第二阶段中，我们将从第一阶段提交的结果中，挑选出 2 份比较优秀的方案，并请相对应的开发者根据自己的方案提交 PR。
* 提交内容：
   * 第一阶段：RFC 方案提交
     1. 提交方式：1）以 markdown 文件的形式提交到 https://aistudio.baidu.com/projectoverview ，2）标题处打上【PaddlePaddle Hackathon 10】，3）RFC 语言不做强制要求。
     2. 基本要求：1）应用场景避免与现有 Demo 重复，2）方案需明确说明选用哪个/哪些模型（ERNIE-4.5-VL-28B-A3B-Thinking 或 PaddleOCR-VL-1.5）及使用方式。
     3. 筛选依据：1）该示例在真实场景下是否具有实际应用价值，2）所选模型的使用是否合理、流程逻辑是否清晰，3）预期效果与业务指标是否匹配。

   * 第二阶段：PR 代码提交
     1. 提交地址：以 Notebook (ipynb) 格式提交完整代码到 https://aistudio.baidu.com/projectoverview 里自己的 project 项目，标题加上【PaddlePaddle Hackathon 10】字样，并在描述处链接之前的 RFC 地址。
     2. 该提交需满足 notebook 贡献规范，包含完整训推代码、依赖环境说明及运行脚本，必须提供在天数智芯硬件上运行的成功截图或录屏证明；开发者需及时根据 review 结果进行修改。
     3. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排。
* 参考示例：推荐参赛者基于所选模型实现以下类型场景（可扩展），推荐方案方向有：
  * 文档智能：合同/票据关键信息抽取、表格理解与问答、多页文档摘要（OCR + 推理）。
  * 多模态理解：图文问答、图表解析与结论生成、说明书/手册理解与问答。
  * 垂直场景：古籍/档案数字化与知识问答、证照识别与信息核验、教育/试卷批改与解析。
  * 参考 Demo：
    * [基于 PaddleOCR-VL 构建论文格式规范器](https://aistudio.baidu.com/projectdetail/9469300?searchKeyword=paddle-ocr-vl&searchTab=PROJECT)
    * [基于 ERNIE-4.5-VL-28B-A3B-Thinking 的目标检测器](https://aistudio.baidu.com/projectdetail/9726489?searchKeyword=ERNIE-4.5-VL-28B-A3B-Thinking&searchTab=PROJECT)
* 技术要求：熟练掌握 Python、文心系列模型与 PaddleOCR-VL 的调用与部署方式，以及在天数智芯硬件上的运行环境配置。
* 参考文档：[飞桨 AI Studio](https://aistudio.baidu.com/modelsoverview)、[ERNIE-4.5-VL-28B-A3B-Thinking 模型](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking)、[PaddleOCR-VL-1.5 模型](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

### 请 沐曦 填写

### 请 燧原 填写

### 请 海光 填写

### 请 瀚博 填写

### 请 飞腾 填写

### 请 龙芯 填写

### 请 高通 填写

### 请 联发科技 填写

### 请 紫光展锐 填写

### 请 此芯 填写

### 请 瑞芯微 填写

### 请 地瓜机器人 填写

### 请 麒麟 填写

### 请 统信 填写
