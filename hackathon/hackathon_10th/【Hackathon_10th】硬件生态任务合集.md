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
进阶任务：PaddleOCR-VL-1.5 在此芯 P1 芯片上的端侧部署与优化
* 技术标签
- **模型移植与优化**：将 PaddleOCR-VL-1.5 模型（0.9B 参数）移植到此芯 P1 芯片，实现高效模型推理。
- **异构算力调度**：基于此芯 P1 的异构架构（Armv9 CPU + Arm Immortalis GPU + 周易 NPU），实现模型算子的最优分配与调度。

* 任务难度：级别：进阶

* 详细描述
本任务旨在将 **PaddleOCR-VL-1.5** 模型移植到 **此芯 P1** 芯片平台，充分利用其 **CPU+GPU+NPU** 异构算力，实现文档解析的端侧高效推理，推动国产 AI 芯片在文档智能领域的应用落地。

### 获胜规则
优先同时完成“核心任务”和“挑战任务”的前两位开发者获胜。

#### 1. 核心任务：实现基于此芯 P1 的 CPU / GPU 的异构推理
- **实现 PaddleOCR-VL-1.5 的 Pipeline**：实现 PaddleOCR-VL-1.5 在 CPU/GPU 上的基础推理。
- **量化加速**：使用 llama.cpp 或者 MNN 等推理框架，对 PaddleOCR-VL-1.5 在 CPU/GPU 上完成加速推理。

#### 2. 可选挑战任务：实现基于此芯 P1 的 CPU + NPU 异构推理

## 环境准备

### 1. 开发环境配置
- 搭建此芯 P1 开发环境，配置交叉编译工具链。
- 安装此芯科技推理 NOE SDK。

### 2. 模型格式转换
- 将 Hugging Face 的 PaddleOCR-VL-1.5 转换为 Paddle/ONNX 格式。
  > **重要**：若需完成挑战任务，必须执行此步骤。

## 具体实施步骤与挑战

### 1. 实现 PaddleOCR-VL-1.5 的 Pipeline
- 实现 Layout（版面分析）推理。
- 实现 PaddleOCR-VL-1.5 模型在 CPU/GPU 上的推理。

### 2. 量化加速
- 对 PaddleOCR-VL-1.5 模型进行 Q4_0 的量化和推理。

### 3. 进阶挑战
- 对 Layout 模块使用 NPU SDK 进行量化，以降低内存占用。

* 提交内容
需包含如下内容：
1. PaddleOCR-VL-1.5 在此芯 P1 上的详细部署步骤。
2. 此芯 P1 推理引擎的使用说明。
3. 示例应用（命令行工具或 GUI 演示）。

* 提交方式
1. **项目提交**：提交使用案例到 [AI Studio](https://aistudio.baidu.com/projectoverview) 的项目并公开，请提交全部源码。
2. **标题规范**：标题处打上【PaddlePaddle Hackathon 10】。
3. **基本要求**：需包含 PaddleOCR-VL-1.5 在此芯 P1 上的部署详细步骤、此芯 P1 推理引擎的示例应用步骤（包括命令行工具 / GUI 演示）。
4. **筛选依据**：
   a) 该示例在真实场景下是否具有实际应用价值。
   b) 该示例的流程逻辑是否清晰。
   c) 运行结果是否符合预期。

* 参考示例
考虑到通用性，选取的应用场景为 **实时解析文档**（发票、合同、表格等）。

* 技术要求
- 模型架构理解
- 此芯 P1 硬件特性了解
- 开发工具链（推理框架，量化工具，编程语言 Python & C++）

* 参考文档
- [CIX AI Model Hub](https://modelscope.cn/models/cix/ai_model_hub/files?version=25_Q4)

- [CIX NOE SDK](https://developer.cixtech.com/) 找到 NeuralONE AI SDK，在注册后并下载。

- [PaddleOCR-VL-1.5 模型](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

- [Paddle 主仓库](https://github.com/PaddlePaddle/Paddle)

- [PaddleX 仓库](https://github.com/PaddlePaddle/PaddleX)
### 请 瑞芯微 填写

### 请 地瓜机器人 填写

### 请 麒麟 填写

### 请 统信 填写
