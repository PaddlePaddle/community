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
### Intel：基于 OpenVINO 的多模态文档理解与智能应用开发
* 技术标签：OpenVINO、PaddleOCR-VL、Python、GenAI/LLM、Agent（可选）、OpenVINO Model Server（可选）

* 详细描述：在完成打卡任务的基础上，可参考以下场景之一，利用OpenVINO的优化部署，完成基于PaddleOCR-VL系列模型的多模态文档理解与智能应用开发，即利用OpenVINO优化部署运行PaddleOCR-VL系列模型的推理、完成文档解析后，参考以下场景描述的下游任务，完成多模态文档理解与智能应用开发。
    * 解析设计图/流程图/技术文档，将结构化内容交给 Coder 模型完成程序设计或代码生成。
    * 理解海报/版面设计稿/宣传材料，结合生成式模型完成改写、重构或多模态创作。
    * 解析论文/报告/说明书，实现摘要、问答、知识提炼或解读等下游任务。
    * 需体现“文档/视觉理解”到“下游智能处理”的完整流程，并突出 OpenVINO 的部署价值。

* 提交内容：
    1. 提交地址： 请将 PR 提交到 [openvino_build_deploy](https://github.com/openvinotoolkit/openvino_build_deploy)  仓库（demos 目录下新增 Demo，结构与现有示例一致），标题加上【PaddlePaddle Hackathon 10】字样。
    2. 必备：
       * 源代码 + README + 依赖/模型说明 + 效果展示（截图/录屏/演示文稿）。
       * PR 需满足 notebook 以及 openvino_build_deploy 仓库贡献规范，开发者需要及时根据 review 的结果进行 PR 修改。
       * 使用 OpenVINO 完成全部推理部署；可复现、尽量一键运行。
    3. 加分（可选）：支持 OpenVINO Model Server；多设备（CPU/GPU/NPU）切换；性能对比/优化说明。
    4. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结遇到的问题与挑战、并介绍后半段比赛的计划安排。
* 参考示例：
   * openvino_build_deploy demos 的各 demo：[https://github.com/openvinotoolkit/openvino_build_deploy/demos](https://github.com/openvinotoolkit/openvino_build_deploy/demos)
   * OpenVINO notebooks：[https://github.com/openvinotoolkit/openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
   * Paddleocr-vl OpenVINO Notebook: [https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl)
* 技术要求：
   * 熟练使用 OpenVINO Python API 完成推理、后处理与可视化。
   * 将结构化输出对接到下游 LLM/Agent 流程（可用任意开源框架/模型，需说明）。
   * 保证可复现：环境说明、依赖安装、模型获取方式与一键运行命令。
* 参考文档：[OpenVINO notebook仓库](https://github.com/openvinotoolkit/openvino_notebooks)、[OpenVINO notebook仓库代码贡献规范](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/CONTRIBUTING.md)、[openvino_build_deploy仓库](https://github.com/openvinotoolkit/openvino_build_deploy)、[OpenVINO Model Server仓库](https://github.com/openvinotoolkit/model_server)


### 请 AMD 填写

### 请 Arm 填写

### 请 天数智芯 填写

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
