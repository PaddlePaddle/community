## 赛题详情（厂商排名不分先后）
### Intel：基于 OpenVINO 的多模态文档理解与智能应用开发
* 技术标签：OpenVINO、PaddleOCR-VL、Python、GenAI/LLM、Agent（可选）、OpenVINO Model Server（可选）

* 详细描述：在完成打卡任务的基础上，可参考以下场景之一，利用OpenVINO的优化部署，完成基于PaddleOCR-VL系列模型的多模态文档理解与智能应用开发，即利用OpenVINO优化部署运行PaddleOCR-VL系列模型的推理、完成文档解析后，参考以下场景描述的下游任务，完成多模态文档理解与智能应用开发。
    * 解析设计图/流程图/技术文档，将结构化内容交给 Coder 模型完成程序设计或代码生成。
    * 理解海报/版面设计稿/宣传材料，结合生成式模型完成改写、重构或多模态创作。
    * 解析论文/报告/说明书，实现摘要、问答、知识提炼或解读等下游任务。
    * 需体现“文档/视觉理解”到“下游智能处理”的完整流程，并突出 OpenVINO 的部署价值。

* 算力支持：提供基于Intel酷睿Ultra处理器的迷你电脑用于任务开发。

* 提交内容：
    * 第一阶段：RFC 方案提交
      1. 提交方式：1）将方案说明提交到厂商邮件组 zhuo.wu@intel.com 及 ethan.yang@intel.com ，2）标题处打上【PaddlePaddle Hackathon 10方案说明】，3）RFC 语言不做强制要求
      2. 基本要求：1）应用场景与现有 [openvino_notebooks/notebooks](https://github.com/openvinotoolkit/openvino_notebooks/notebooks) 中以及 [openvino_build_deploy/demos](https://github.com/openvinotoolkit/openvino_build_deploy/demos) 中的内容不重复，2）该方案说明中需要使用openvino 完成模型的推理部署
      3. 筛选依据：1）应用价值；2）逻辑清晰度；3）可复现性与完成可行性。

    * 第二阶段：PR代码提交。请将 PR 提交到 [openvino_build_deploy](https://github.com/openvinotoolkit/openvino_build_deploy)  仓库（demos 目录下新增 Demo，结构与现有示例一致），标题加上【PaddlePaddle Hackathon 10】字样。必备：
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


### AMD：为 Paddle 框架适配 HIP BF16 精度类型
* 技术标签：深度学习框架，C++/HIP，ROCm，MIOpen，PaddlePaddle
* 详细描述：
  * **背景**：当前 Paddle 框架在 ROCm 平台上未适配 HIP BF16 精度类型，导致该精度类型下的相关算子不可用。在 AMD GPU 上运行 PaddleOCR-VL 等包含卷积视觉编码器的模型时，由于 BF16 算子缺失，不得不通过 `_keep_in_fp32_modules` 等方式将视觉编码器强制回退到 FP32 精度运行。
    HIP BF16 精度类型不可用对 Paddle 框架在 AMD GPU 上的推理能力有显著制约：
    1. 显存开销倍增：FP32 相比 BF16 显存占用翻倍，制约了可部署的模型规模与批处理容量
    2. 推理性能受损：无法利用 BF16 更高的计算吞吐，推理效率大幅降低
    3. 框架生态受限：不仅影响 PaddleOCR-VL，也制约了 Paddle 框架上其他 LLM 及多模态模型在 AMD GPU 上的推理部署能力
  * **现有 Workaround 参考**：目前 PaddleX 中通过以下方式临时绕过此问题：
    1. 在 `paddlex/inference/utils/misc.py` 的 `is_bfloat16_available()` 中，对 ROCm 平台默认返回 False，强制使用 FP32 精度
    2. 在 `PaddleOCRVLForConditionalGeneration` 模型类中设置 `_keep_in_fp32_modules = ["visual", "mlp_AR"]`，将视觉编码器保持在 FP32
    3. 在静态推理配置中禁用 `conv2d_add_act_fuse_pass` 和 `conv2d_add_fuse_pass`（因依赖 cuDNN，HIP 上不可用）
    4. 详细参考：https://github.com/PaddlePaddle/PaddleX/compare/release/3.3...vivienfanghuagood:PaddleX:dev_rocm70
  * **任务目标**：在 [Paddle 框架](https://github.com/PaddlePaddle/Paddle) 中适配 HIP BF16 精度类型，使得：
    1. PaddleOCR-VL 等模型在 ROCm 上可以原生使用 BF16 精度进行推理，无需将视觉编码器强制回退到 FP32
    2. Paddle 框架的 ROCm BF16 算子能力得到完善，有利于框架上其他 LLM/多模态模型的 AMD GPU 推理
  * **验收标准**：PaddleOCR-VL-1.5 能在 AMD GPU + ROCm 环境下以 BF16 精度完整运行并输出正确结果。
* 提交内容：
  1. 向 [Paddle 主仓库](https://github.com/PaddlePaddle/Paddle) develop 分支提交 Issue 描述问题，并提交 PR 实现 HIP BF16 精度类型适配
  2. 向 [PaddleX 仓库](https://github.com/PaddlePaddle/PaddleX) develop 分支提交 Issue 和 PR，移除现有 ROCm BF16 的 workaround 代码
  3. PR 中需包含测试用例和在 AMD GPU 上的验证结果截图
* 提交方式：GitHub Issue + PR，并将验证结果截图发送邮件至 ext_paddle_oss@baidu.com，抄送 Zijun.Wei@amd.com, Huaqiang.Fang@amd.com, bingqing.guo@amd.com
* 技术要求：
  * 熟悉 PaddlePaddle 框架的算子注册与编译机制
  * 了解 ROCm/HIP 编程模型与 MIOpen 库的使用
  * 熟悉 BF16 精度计算的基本原理
  * 具备 C++ 和 Python 开发能力
* 参考文档：
  * [Paddle 主仓库](https://github.com/PaddlePaddle/Paddle)
  * [PaddleX 仓库](https://github.com/PaddlePaddle/PaddleX)
  * [PaddleX ROCm BF16 现有 workaround](https://github.com/PaddlePaddle/PaddleX/compare/release/3.3...vivienfanghuagood:PaddleX:dev_rocm70)
  * [PaddleOCR-VL-1.5 模型](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
  * [MIOpen 文档](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
  * [PaddleOCR-VL AMD GPU 部署教程](../../pfcc/paddle-hardware/AMD-PaddleOCR-VL-GPU打卡任务.md)
* 额外奖励说明：AMD Radeon 9070 XT 16GB 显卡 × 1，PN54 AI 5 340 AMD 锐龙处理器 Mini PC × 1。最终完成代码合入的参赛者将成为本次赛事 AMD 赛道冠军，并可优先选择一项进阶礼品；第二名为除冠军外其余参赛选手中 PR 审核通过时间最早的提交者，可获得剩余的进阶礼品。

### 天数智芯：基于天数智芯硬件与文心多模态模型的创新应用
* 技术标签：深度学习框架，Python，文心大模型，多模态
* 详细描述：本任务基于天数智芯 BI-150S，围绕 **PaddleOCR-VL-1.5**、**ERNIE-4.5-VL-28B-A3B-Thinking** 两类文心多模态模型完成选型与开发（选型规则见下「模型范围」），搭建可复现的创新应用 Demo 并形成业务闭环。参考 [飞桨 AI Studio 应用案例库](https://aistudio.baidu.com/topic/applications)。

  * **模型范围**：**PaddleOCR-VL-1.5** 与 **ERNIE-4.5-VL-28B-A3B-Thinking** 中任选其一作为主能力即可完成赛题。

  * **端侧与模型分工**
    * 所选模型的接入方式：可在 BI-150S 上 本地推理，也可通过 文心大模型 API 调用云端能力。RFC 中需写清选用哪一种模型及其接入方式（「本地推理」或「API」），并围绕该模型形成完整应用闭环。
    * 加分项：在 BI-150S 上对所选模型做本地推理，可在 RFC 与最终报告中单独说明。
    * 必须在 BI-150S 侧体现的部分：Notebook 主流程须在可接入 BI-150S 的环境中完成运行与证明材料；所选模型能力可走 本地推理 或 文心 API，但该环境下须能复现端到端执行（含鉴权、请求与结果处理）。

* 算力支持：可申请使用星河平台 BI-150S 算力。

* 提交内容：
   * **第一阶段：RFC 方案提交**
     1. 提交方式：1）以 Markdown 文件提交到 https://aistudio.baidu.com/projectoverview ，2）标题含【PaddlePaddle Hackathon 10】，3）RFC 语言不做强制要求。
     2. 基本要求：
        1）应用场景避免与现有 Demo 简单重复；
        2）任选其一：明确本方案选用 PaddleOCR-VL-1.5 与 ERNIE-4.5-VL-28B-A3B-Thinking 中哪一种；并写明接入方式（本地推理 或 文心 API）；
        3）说明在 BI-150S 环境中具体执行哪些环节（本地推理模块、API 编排、前后处理等）；
        4）列出预期业务指标（如端到端延迟、准确率等），与场景匹配即可。
     3. 筛选依据：应用价值；技术方案逻辑是否清晰；可复现性与完成可行性；在 BI-150S 环境中的验证路径是否清晰。

   * **第二阶段：Notebook 与材料提交**（由第一阶段入选者完成）
     1. 提交地址：以 Notebook（ipynb）为主，提交到 https://aistudio.baidu.com/projectoverview 个人公开项目，标题含【PaddlePaddle Hackathon 10】，描述中附上 RFC 链接。
     2. 必备交付物：
        * 完整可运行源码（Notebook + 必要脚本/模块）；
        * README：环境（含 BI-150S 驱动/镜像或星河任务说明）、依赖安装、模型与 API 配置方式、一键或分步运行命令；
        * 依赖与模型说明：针对 RFC 中声明选用的模型——若走本地推理，说明权重或模型获取方式；若走 文心 API，说明所用接口与鉴权方式；
        * 效果展示：截图或录屏，须能体现应用在 BI-150S 环境（含星河 BI-150S 任务）下主流程已成功执行；
        * 满足 AI Studio Notebook 贡献与评审习惯，并根据 review 及时修改。
     3. 在比赛过半时设置中期检查会：汇报进度、已完成功能、问题与后半程计划。

* **验收要求**
  1. 天数环境参与：提交材料能证明端到端主流程在 BI-150S 环境中成功执行，并附截图/录屏。
  2. 端到端闭环：从用户输入（如文档/图片/问题）到可展示的输出（结构化字段、摘要、问答答案等）链路完整，非仅单接口调用演示。
  3. 可复现：他人按 README 可在同类 BI-150S / 星河任务环境中复现。
  4. 稳定性：对主流程给出基本说明；录屏或文档中体现一次完整成功运行即可。

* 参考示例：推荐参赛者实现以下类型场景：
  * 文档智能：合同/票据关键信息抽取、表格理解与问答、多页文档摘要（OCR + 推理）。
  * 多模态理解：图文问答、图表解析与结论生成、说明书/手册理解与问答。
  * 垂直场景：古籍/档案数字化与知识问答、证照识别与信息核验、教育/试卷批改与解析。
  * 参考 Demo：
    * [基于 PaddleOCR-VL 构建论文格式规范器](https://aistudio.baidu.com/projectdetail/9469300?searchKeyword=paddle-ocr-vl&searchTab=PROJECT)
    * [基于 ERNIE-4.5-VL-28B-A3B-Thinking 的目标检测器](https://aistudio.baidu.com/projectdetail/9726489?searchKeyword=ERNIE-4.5-VL-28B-A3B-Thinking&searchTab=PROJECT)

* 技术要求：熟练掌握 Python；能在 BI-150S 环境中完成应用联调与运行；对所选模型须掌握 文心 API 或本地部署与推理之一。
* 参考文档：[飞桨 AI Studio](https://aistudio.baidu.com/modelsoverview)、[文心大模型 API 说明](https://ai.baidu.com/ai-doc/AISTUDIO/rm344erns)、[ERNIE-4.5-VL-28B-A3B-Thinking 模型](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking)、[PaddleOCR-VL-1.5 模型](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

### 沐曦：优化 PaddleOCR-VL-1.5+Metax GPU

- 技术标签：深度学习框架，Python，PaddleOCR-VL-1.5，Metax GPU

- 详细描述：*PaddleOCR* 是智能文档解析与文字识别工具，支持多语言识别与手写体识别，轻松处理PDF、图片等格式，高效提取文字信息。为此我们也想基于PaddlePaddle + FastDeploy + Metax GPU实现更优的推理性能。在这个任务中，你需要基于：

  ```
  paddlepaddle==3.4.0.dev20251223
  paddle-metax-gpu==3.3.0.dev20251224
  https://github.com/PaddlePaddle/FastDeploy/tree/release/2.4
  ```

- 本次任务评估将分为两个阶段：
  - 第一阶段，开发者需要提供一份性能瓶颈分析评估报告(包含但不限于推理框架调度，GPU 利用率，5个以上kernel函数分析)，按照 profiling trace 文件+分析报告形式提交；
  - 第二阶段，我们将从第一阶段提交的结果中，review 并 comment 需要进一步优化的算子，并请相对应的开发者根据确定的性能瓶颈点提交优化 PR，预期性能提升目标 20%+。

- 提交内容：
  - 第一阶段：[PR 提交地址](https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy)
  - 第二阶段：[PR 提交地址](https://github.com/PaddlePaddle/FastDeploy/tree/develop)


### 燧原：基于燧原卡为`FastDeploy`新增应用
* 技术标签：PaddlePaddle，FastDeploy，Python

* 详细描述：本任务旨在利用 燧原 S60 加速卡 (GCU) 的算力优势，结合 FastDeploy 高性能推理框架，对 ERNIE-4.5-0.3B-Paddle 模型进行二次开发与应用。我们鼓励开发者打造具有真实落地价值、逻辑闭环且体验优秀的创新案例。参考 [飞桨 AI Studio 应用案例库](https://aistudio.baidu.com/topic/applications) 。
* 提交内容：
    * 第一阶段：RFC 方案提交
      1. 提交方式：1）以markdown文件的形式提交到 https://aistudio.baidu.com/projectoverview, 2）标题处打上【PaddlePaddle Hackathon 10】。
      2. 基本要求：1）应用场景避免与现有 Demo（如简单的情感分析）重复，2）方案需充分挖掘 `ERNIE-4.5-0.3B-Paddle` 轻量且高效的特点。
      3. 筛选依据：1）该示例在真实场景下是否具有实际应用价值，2）该示例的流程逻辑是否清晰，3）预期的推理效果与业务指标是否匹配。

    * 第二阶段：PR代码提交
      1. 提交地址：以 Notebook (ipynb) 格式提交完整代码到 https://aistudio.baidu.com/projectoverview 里自己的project项目，标题加上【PaddlePaddle Hackathon 10】字样，并在描述处链接之前的 RFC 地址
      2. 该 PR 需满足 notebook 贡献规范，开发者需要及时根据 review 的结果进行 PR 修改
      3. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排
* 参考示例：考虑到通用性，选取的应用场景尽量以英文为主，推荐方案场景有：
   * 智能文本处理：长文摘要、垂直领域翻译。
   * 语义理解应用：行业知识库问答、高级情感倾向挖掘。
   * 参考Demo：
     * [ERINE-4.5-0.3B老北京风格微调](https://aistudio.baidu.com/projectdetail/10000880?channelType=0&channel=0)
     * [基于ERNIE-4.5-0.3B 中文情感分析实战教程](https://aistudio.baidu.com/projectdetail/9385231)

* 技术要求：熟练掌握 python 和 FastDeploy 部署流程与其他工具组件的使用方法
* 参考文档：[FastDeploy](https://paddlepaddle.github.io/FastDeploy/zh/) 、[飞桨AI Studio](https://aistudio.baidu.com/overview)



### 海光：PaddleOCR-VL-1.5 应用性能分析与调优

- 技术标签：PaddleX，PaddleOCR-VL，性能分析，性能调优，海光DCU

- 详细描述：参赛选手在我们提供的DCU环境(scnet.cn)中完成了前置打卡任务后，可以挑战本任务。在本任务中，选手将继续使用scnet提供的开发环境（我们会加充200卡时时长），针对测试程序进行性能分析，增加对PaddleX/PaddleOCR框架的理解，并且进行对模型推理部分进行性能调优的尝试。最终我们对选手提交的 **性能分析报告** 和 **性能优化效果** 进行综合考量，选出优胜者。

- 提交内容：
    - 第一阶段：**性能分析报告**提交，报告中要回答以下问题：
        - Paddleocr-vl-1.5-0.9b模型推理过程中，有几个步骤？分别是做什么的？VLLM加速的是哪个步骤？
        - 使用VLLM加速推理时，调用的modeling程序是PaddleX中的哪个代码文件？
        - 请在代码中添加一行，打印输入tensor的尺寸，把输出截图附在报告中。
        - 启动vllm是加上profile参数，用（ui.perfetto.dev）工具打开prof报告，截图。
        - 描述模型中一层推理的过程，例如包含几个norm，linear，attention算子？
        - 在prof报告中整理出encoder层中执行的算子，查看耗时，把算子与耗时整理到表格里。

    - 第二阶段：我们根据性能分析报告，选出2名理解比较深刻的同学，进行推理优化。
        - 开发代码可以从这里fork: https://github.com/Yun1Liu/paddlex ，完成后向这个项目里提PR，不要直接向paddle社区提交PR。（针对DCU平台特殊的优化不一定有普适行，我们会推动优化成果向光源社区OpenDAS融合）
        - 辅导老师会参与优化方案的讨论，可尝试的优化方法包含但不限于：
            - 对VLLM参数进行优化，获得比默认参数更好的性能。
            - 使用DCU优化算子，代替torch算子或python代码实现的算子。
            - 手写triton算子，做算子融合。
            - 混合精度的尝试，并做精度验证。
        - 优化后，需要提供性能提升的数据和精度验证的结果。保证精度不下降。

- 验收标准：优化后的性能比优化前有较明显提升(>=2%),精度无明显下降(<1%)，即可通过验收。

- 参考文档：[海光DCU-PaddleOCR应用-打卡任务](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-hardware/%E6%B5%B7%E5%85%89DCU-PaddleOCR%E5%BA%94%E7%94%A8-%E6%89%93%E5%8D%A1%E4%BB%BB%E5%8A%A1.md)



### 瀚博：基于瀚博载天系列加速卡部署文心 ERNIE-4.5 / PP-OCRv4 模型

* 技术标签：ERNIE-4.5, PP-OCRv4, 瀚博载天(VA1/VA10/VA16/VE1), VACC, Python, 大模型推理, 多模态

* 详细描述：瀚博半导体（Vastai）载天系列 AI 加速卡覆盖云端推理（VA1、VA10、VA16）、边缘计算（VE1S、VE1M）等场景，基于自研 VUCA 统一计算架构，在视频处理、大模型推理、智能视觉等领域已有广泛部署。本任务面向**已有瀚博硬件资源的开发者和企业用户**，征集基于瀚博载天系列加速卡部署**文心 ERNIE-4.5 系列开源模型**或**PP-OCRv4 多模态文档理解模型**的应用 Demo，展示模型在瀚博硬件上的推理效果与性能表现。**说明**：本赛题瀚博提供免费算力资源。

* 可选模型范围：认领者须基于 ERNIE-4.5 系列模型或者 PP-OCRv4（任选其一或组合）完成部署。推荐应用场景（非强制，仅供参考）：

| 场景 | 说明 | 推荐模型 |
|------|------|---------|
| **智能文档处理** | 合同/票据/报表的端到端解析，输出结构化数据 | PP-OCRv4 |
| **多模态问答** | 基于图片+文本的视觉问答、图表解读、说明书理解 | ERNIE-4.5-VL-28B-A3B-Thinking |
| **轻量推理服务** | 在边缘设备上提供大模型对话/摘要/翻译服务 | ERNIE-4.5-0.3B |
| **文档智能审核** | 论文格式校验、合规性检查、关键信息提取 | PP-OCRv4 + ERNIE-4.5 |
| **其他创新场景** | 认领者可自行提出场景，不限于以上列表 | 任一可选模型 |

* 提交内容
  * 第一阶段：方案认领（RFC）。以 Markdown 文件形式提交至 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 `rfcs/hardware` 目录，内容需包含如下四方面。组委会将从报名者中挑选 **1-2 名**合适的认领者，确认后进入第二阶段。
    1. **简历**：开发者个人信息简介
    2. **过往项目介绍**：列举开发者参与过的项目概要与贡献

 * 第二阶段：代码与 Demo 提交。认领者提交完整的可复现代码和 Demo，以 Notebook (ipynb) 形式提交至 [AI Studio](https://aistudio.baidu.com)。提交物必须包含：

```
submission/
├── README.md # 环境配置、复现步骤、运行说明
├── src/ # 源代码（模型加载、推理、应用逻辑）
├── configs/ # 配置文件（如有）
├── docs/
│ ├── hardware_env.md # 硬件环境详细说明
│ ├── performance_report.md # 性能测试报告
│ └── screenshots/ # 运行截图或录屏
└── requirements.txt # 依赖列表
```

* 验收标准：认领者的提交需满足以下**全部验收项**，方可通过验收并获得奖金。

| 编号 | 验收项 | 验收标准 | 验证方式 |
|------|--------|---------|---------|
| V1 | **模型成功运行** | 所选的 ERNIE-4.5 系列或 PP-OCRv4 模型在瀚博硬件上成功完成推理，输出结果正确合理 | 运行截图/录屏 + 输出日志 |
| V2 | **硬件运行证据** | 提供在瀚博硬件上实际运行的截图或录屏，须包含硬件设备信息（如 `vastai-smi` 或等效命令输出），证明确实在瀚博设备上运行 | 截图/录屏审查 |
| V3 | **环境可复现** | 提供完整的环境配置文档，包括瀚博驱动版本、VACC SDK 版本、Python 依赖列表及安装步骤，使另一台同型号瀚博设备可据此文档复现 | 文档审查 |
| V4 | **代码完整可运行** | 代码包含从模型下载/加载到推理输出的完整流程，结构清晰，无硬编码绝对路径，有必要的注释和说明 | 代码审查 |
| V5 | **性能数据** | 提供至少包含**推理延迟**（单条/批量）和**吞吐量**的定量性能测试数据，附测试方法说明 | 性能报告审查 |
| V6 | **应用 Demo 完整** | 不仅是裸推理脚本，需包含至少一个完整的应用场景演示（含输入处理→模型推理→结果展示的闭环） | Demo 审查 |
| V7 | **文档质量** | README 包含：(1) 一键运行的快速开始指南；(2) 适配过程中的关键步骤说明；(3) 遇到的问题及解决方案记录 | 文档审查 |

* 参考示例（非强制，仅供启发）：
  1. **智能文档解析服务**：在 VA1/VA10/VA16 上部署 PP-OCRv4，输入合同/票据图片，输出结构化 JSON（含文字内容、表格、版面结构）。提供 REST API 封装和前端可视化页面。
  2. **多模态视觉问答助手**：在 VA1/VA10/VA16 上部署 ERNIE-4.5-VL-28B-A3B-Thinking，实现"上传图片 + 提问"的交互式视觉问答 Demo，支持图表解读、场景描述等。
  3. **边缘端轻量对话服务**：在 VE1M/VE1S 上部署 ERNIE-4.5-0.3B，提供低延迟的文本摘要/翻译/问答推理服务，展示边缘场景下的实用性。

* 技术要求：
  * 硬件：瀚博载天系列任一型号（VA1 / VA10 /VA16/ VE1S / VE1M ）
  * 模型：ERNIE-4.5 系列开源模型 或 PP-OCRv4（见可选模型范围）
  * 编程语言：Python（主），C++ 可选
  * 操作系统：Linux（Ubuntu 20.04/22.04 推荐）

* 参考文档：[ERNIE-4.5 模型仓库](https://huggingface.co/baidu)，[PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR)，[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)，[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)，[瀚博半导体官网](https://www.vastaitech.com)
* 其他说明：
  1. **算力资源**：本赛题瀚博提供免费算力资源。
  2. **认领制**：组委会从第一阶段报名者中挑选 1-2 名认领者，确认后进入开发阶段。满足全部验收标准即可获得奖金。
  3. **硬件真实性**：仅在模拟器或其他硬件上运行的提交不予通过，必须提供瀚博设备上的真实运行证据。
  4. **成果开源**：鼓励认领者将成果以 Apache 2.0 协议开源，优秀方案将有机会合入官方仓库作为社区参考。


### 飞腾：基于飞腾 ARM64 的 OpenClaw 智能体协作系统

**技术标签**：飞腾、ARM64、FastDeploy、ERNIE-4.5-21B-A3B、OpenClaw、多智能体协作

**详细描述**：在完成飞腾平台 FastDeploy 打卡任务的基础上，本任务旨在利用已攻克的 ARM64 编译成果（已解决`R_AARCH64_CALL26`、CUTLASS 路径、MoE 内核缺失等问题），基于开源智能体框架 OpenClaw，打造一个可重复利用的**通用智能体协作系统**。

该系统以 ERNIE-4.5-21B-A3B 为 “大脑”，通过 OpenClaw 的多智能体机制，实现复杂任务的自动化拆解与协同执行。系统具备高可复用性，可灵活适配不同场景——如文献综述、报告撰写、数据分析、信息聚合等知识密集型工作。核心在于体现从 “国产硬件适配” 到 “通用智能体能力” 的完整落地价值，突出 FastDeploy 在飞腾 ARM64 平台上的部署优势。

**提交内容**：

* **PR 提交地址**：提交完整代码至[仓库](https://github.com/zongwave/pixelcraft/tree/main/ai)，标题标注【PaddlePaddle Hackathon 10】，并关联本 RFC。

* **必备**：
  1. **源代码**：完整的 OpenClaw 多智能体配置代码及 ERNIE 模型调用示例。
  2. **部署脚本**：飞腾平台 FastDeploy 一键环境脚本 (`phytium_install.sh`) 及 OpenClaw 配置指南。
  3. **模型与依赖说明**：ERNIE-4.5-21B-A3B 模型获取方式及 Python 依赖清单。
  4. **效果展示**：在飞腾 + L20 硬件上成功运行至少 2 个不同场景（如文献综述+报告撰写）的演示截图/录屏。
  5. **可复现性**：所有步骤需确保能在同类飞腾环境下一键运行。

* **中期检查**：汇报项目进度，展示已完成的核心智能体功能，并介绍后续多场景适配计划。

**参考示例**：
* OpenClaw 官方文档：https://docs.openclaw.ai
* FastDeploy 仓库：https://github.com/PaddlePaddle/FastDeploy

**技术要求**：

1. **环境配置**：需在飞腾 S5000C + NVIDIA L20 上完成部署，提供已验证的编译参数。
2. **模型支持**：核心调用 **ERNIE-4.5-21B-A3B-Thinking** 模型。
3. **核心能力**：
   * 实现 OpenClaw 多智能体（至少3个角色）的配置与协作机制。
   * 对接 FastDeploy OpenAI API Server 完成模型推理。
   * 确保智能体工作区隔离与权限控制，支持任务灵活编排。
   * 至少适配 2 个不同场景，验证系统的可复用性。
4. **可复现性**：提供完整的环境配置、依赖安装与一键运行命令。

**参考文档**：
* [飞桨 AI Studio](https://aistudio.baidu.com)
* [ERNIE-4.5-21B-A3B 模型](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)
* [FastDeploy 仓库](https://github.com/PaddlePaddle/FastDeploy)
* [OpenClaw 官方文档](https://docs.openclaw.ai)

### 龙芯中科：基于OpenClaw与AI算力驱动的龙芯LoongArch架构软件包自动化移植应用
* 技术标签：二次开发、龙芯LoongArch架构、软件包自动化移植、AI算力集成

- 详细描述：本次终极任务是基于打卡任务基础能力的进阶考察，核心围绕智能体、龙芯LoongArch架构软件包适配、AI算力融合三大核心方向，基于开源智能体框架OpenClaw ，打造LoongArch架构软件包的自动化移植，构建可复用、标准化的工具链体系。

* 提交内容

  1. 阶段1（需求分析与方案设计）：《龙芯LoongArch架构软件包自动化移植方案文档》（初版）
  2. 阶段2（OpenClaw二次开发与AI算力集成）：OpenClaw二次开发代码、工具链主流程脚本（初版）
  3. 阶段3（软件包自动化移植测试）：优化后的自动化移植工具链全套代码、《龙芯LoongArch架构软件包自动化移植测试报告》（初版）、LoongArch移植问题库
  4. 阶段4（文档完善与PR提交）：工具链全套最终代码、所有终版技术文档（方案文档、测试报告、使用手册）、符合规范的PR提交记录

* 提交方式：提 PR 到 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 `rfcs/hardware` 目录
* 本阶段不提供算力/资源支持，需选手自备相应算力

- 技术要求

  1. OpenClaw开发能力：熟练掌握OpenClaw的核心实现原理、源码结构与可扩展接口，能基于其进行二次开发，新增LoongArch架构移植适配模块，代码符合开源开发规范。
  2. LoongArch架构能力：精通龙芯LoongArch架构的系统操作、编译原理、依赖库适配、软件包移植流程，能独立解决该架构下软件包移植的各类典型问题。
  3. AI集成能力：熟练使用文心API，能根据移植流程需求，设计合理的AI融合方案，实现移植自动化、报错智能排查及修复、适配方案智能推荐等核心AI能力。

### 高通：基于 Qualcomm AI Engine Direct (QNN) 部署 PaddleOCR-VL 模型，实现端侧页面级文档解析

* 技术标签：PaddleOCR-VL，高通 QNN SDK，Hexagon NPU，Paddle2ONNX，Python，C/C++

* 详细描述：PaddleOCR-VL 模型基于视觉语言大模型（PP-DocBee2-3B）实现了文本块、表格、公式及图表的统一识别，是 PaddleOCR 3.x 的旗舰文档解析方案。本赛题要求选手通过高通 QNN（Qualcomm Neural Networks）工具链，将 PaddleOCR-VL pipeline 中的关键子模型（布局检测模型、VL 识别模型等）转换并部署至高通 Hexagon NPU 进行端侧推理。参考 PaddleOCR CLI 中的 `doc_parser`，构建依赖端侧推理服务的完整页面级文档解析 pipeline，可使用 Python 或 C/C++ 语言进行开发。

  **整体流程**：
  1. 使用 Paddle2ONNX 将 PaddleOCR-VL 的子模型（布局检测模型、VL 识别模型）导出为 ONNX 格式
  2. 使用 QNN SDK 工具链（`qnn-onnx-converter` → `qnn-model-lib-generator` → `qnn-context-binary-generator`）将 ONNX 模型转换为 QNN 格式并针对 HTP 后端进行优化
  3. 对模型进行量化（INT8/INT16/FP16），以适配 HTP 高效推理
  4. 搭建端侧推理服务，串联各子模型构建完整的文档解析 pipeline

* 提交内容：
   1. 模型转换脚本及说明文档（Paddle → ONNX → QNN 全链路）
   2. 基于高通 QNN 部署的端侧推理服务代码
   3. 基于端侧推理服务的完整页面级文档解析 pipeline 代码（参考 `doc_parser`）
   4. 精度对比报告（端侧推理结果 vs 原始 PaddlePaddle 推理结果）
   5. 单页文档解析性能及耗时测试报告

* 验收要求（需全部满足）：
   1. **模型转换完整**：完成布局检测模型和 VL 识别模型从 Paddle → ONNX → QNN 的全链路转换，提供可复现的转换脚本
   2. **端侧推理可运行**：转换后的 QNN 模型可在高通 HTP-simulator 或实际高通设备上成功加载并执行推理
   3. **文档解析 pipeline 可用**：参考 `doc_parser` 实现完整的页面级文档解析功能，输入单页文档图片，输出结构化 Markdown 解析结果，覆盖文本块与表格两种以上版面元素
   4. **精度可接受**：端侧推理的文档解析结果与原始 PaddlePaddle 推理结果对比，文本识别准确率损失不超过 5%

* 技术要求：
   * 熟练掌握 Paddle2ONNX 模型导出工具的使用
   * 熟悉高通 QNN SDK 工具链（qnn-onnx-converter、qnn-model-lib-generator、qnn-context-binary-generator）
   * 具备在高通 Hexagon NPU 上进行模型量化与推理部署的经验
   * 了解 PaddleOCR-VL 的模型结构与 pipeline 设计

* 参考文档：
   * [PaddleOCR-VL GitHub 仓库](https://github.com/PaddlePaddle/PaddleOCR)
   * [Paddle2ONNX 文档](https://github.com/PaddlePaddle/Paddle2ONNX)
   * [Qualcomm QNN SDK 文档](https://developer.qualcomm.com/software/qualcomm-neural-network-sdk)
   * [高通 HTP 后端优化指南](https://developer.qualcomm.com/software/qualcomm-neural-network-sdk/getting-started)

### 联发科技：在天玑9500 手机上运行 OpenClaw —— 基于文心大模型的移动端个人 AI 助手

* 技术标签：天玑9500，MediaTek NPU，OpenClaw，文心大模型 API，Android，移动端 AI Agent，端云协同

* 详细描述：
  * **背景**：OpenClaw（openclaw.ai）是当下最火的开源个人 AI 助手平台，社区俗称"养龙虾"。它支持通过 WhatsApp、Telegram、Discord 等聊天应用与 AI 助手交互，能够执行邮件处理、日程管理、网页浏览、文件操作、Shell 命令等实际任务，并具备持久记忆、技能扩展（Skills）、主动行为（Heartbeats）等核心能力。联发科技天玑9500 是新一代旗舰移动平台，搭载强大的 APU（AI 处理单元），为端侧 AI 应用提供卓越的算力支持。
  * **任务目标**：本任务旨在将 OpenClaw 移植并适配到天玑9500 Android 手机上运行，接入文心大模型 API 作为底层 LLM 能力，让用户可以在手机上"养龙虾"——拥有一个随身的、能真正做事的、更安全的个人 AI 助手。开发者需完成以下核心工作并可选择扩展方向：

    **核心目标（必选）：**
    1. **OpenClaw 移动端适配** —— 将 OpenClaw 移植到天玑9500 Android 设备上运行，确保 Gateway 核心服务、技能系统、持久记忆等基础功能正常工作
    2. **接入文心大模型 API** —— 将 OpenClaw 的 LLM 后端对接到文心大模型 API（ERNIE-4.5 系列），使其成为 OpenClaw 的"大脑"

    **扩展方向（可选加分项）：**
    3. **NPU 加速端侧能力** —— 利用天玑9500 NPU 在端侧运行轻量模型，增强 OpenClaw 的本地能力（如语音识别、图像理解、OCR 等），减少对云端的依赖，提升响应速度
    4. **移动端专属 Skills 开发** —— 针对手机场景开发 OpenClaw 专属技能，如：拍照识物、通讯录管理、手机设置控制、位置感知服务、健康数据分析等
    5. **移动端交互优化** —— 适配移动端的用户交互体验，如通知栏快捷交互、悬浮窗助手、语音唤醒、Widget 小组件等

  * **技术架构**：
    * **云侧**：调用文心大模型 API（ERNIE-4.5 系列）作为 OpenClaw 的 LLM 后端，完成自然语言理解、任务规划、文本生成等核心 AI 能力
    * **端侧**：在天玑9500 Android 设备上运行 OpenClaw Gateway 核心服务，管理 Sessions、Channels、Tools 和 Skills
    * **端侧 NPU（加分项）**：利用天玑9500 APU 运行轻量端侧模型，提供低延迟的本地 AI 感知能力（语音、视觉等）

  * **资源支持**：
    * 联发科技将提供 1-2 台天玑9500 工程设备及 NPU 开发文档
    * 百度将提供文心大模型 API 调用额度，支持开发者完成应用开发

* 提交内容：
  * **第一阶段：RFC 方案提交**
    1. 提交方式：
       - 以 markdown 文档形式提交到 [PaddlePaddle/community](https://github.com/PaddlePaddle/community) 的 `rfcs/hardware` 目录
       - 标题格式：【PaddlePaddle Hackathon 10 方案说明】- 联发科技 OpenClaw 移动端 - 开发者姓名
    2. 基本要求：
       - 需说明 OpenClaw 移动端适配的技术方案（Node.js 运行环境、服务架构、资源限制应对等）
       - 需说明文心大模型 API 的接入方案及选用的模型类型
       - 如涉及 NPU 端侧能力，需说明端侧模型选择及加速方案
       - 如涉及移动端专属 Skills，需描述技能设计及应用场景
       - 需提供初步的技术可行性分析（内存/存储需求、功耗评估、网络依赖分析等）
    3. 筛选依据：
       - 可行性：移动端适配方案的合理性与完整度
       - 创新性：移动端场景下的独特价值与创意
       - 技术深度：对 OpenClaw 架构的理解深度，以及文心大模型 API 的合理运用
       - 加分项：NPU 端侧能力利用、移动端专属 Skills 设计
    4. 通过第一阶段筛选的开发者将获得天玑9500 工程设备使用权及文心大模型 API 调用额度

  * **第二阶段：代码提交与应用演示**
    1. 提交方式：
       - 将完整代码提交至 GitHub 仓库（开发者自建）
       - 以邮件形式提交演示视频及部署文档
       - 标题标注【PaddlePaddle Hackathon 10】
    2. 提交内容：
       - 完整的移动端适配源码（含 OpenClaw 移植代码、文心大模型 API 接入代码）
       - 部署文档：包括 Android 环境配置、依赖安装、文心 API 配置、运行说明
       - 演示视频：展示 OpenClaw 在天玑9500 手机上的实际运行效果（时长不少于 3 分钟），需演示至少 3 个完整的 AI 任务执行流程
       - 性能报告：包括内存占用、功耗表现、API 调用延迟、任务完成成功率等指标
       - 如有移动端专属 Skills，需提供 Skills 代码及使用说明
    3. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排

* 验收要求（需全部满足）：
  1. **OpenClaw 核心运行正常**：OpenClaw Gateway 在天玑9500 Android 设备上稳定运行，核心功能（会话管理、工具调用、持久记忆）正常工作
  2. **文心大模型接入成功**：文心大模型 API 作为 LLM 后端正确接入，自然语言理解与生成功能正常
  3. **任务执行能力**：至少能成功完成 3 类实际任务（如信息查询、文件操作、日程/提醒管理等）
  4. **运行稳定性**：连续运行 30 分钟以上无崩溃，内存占用合理（< 2GB）
  5. **代码可复现**：提交的代码与文档可在同类 Android 设备上完整复现部署和运行过程（API 密钥除外）

* 参考示例：推荐参赛者实现以下场景（可扩展）：
  * **随身信息助手**：通过聊天界面与 OpenClaw 对话，查询天气、新闻、翻译，或让它帮你搜索本地文件
  * **移动端开发助手**：在手机上通过 OpenClaw 执行简单的 Shell 命令、查看日志、管理文件
  * **智能日程管家**：OpenClaw 主动通过 Heartbeat 提醒待办事项，结合日历 API 管理日程
  * **拍照识物助手**（NPU 加分项）：调用手机摄像头 + NPU 端侧视觉模型识别物体，再通过文心大模型 API 生成详细说明
  * **语音交互模式**（NPU 加分项）：利用 NPU 进行端侧语音识别，实现免打字的语音控制体验

* 技术要求：
  * 熟悉 Node.js / TypeScript 开发
  * 熟悉 Android 应用开发（了解 Termux 或 Android Node.js 运行环境方案）
  * 了解 OpenClaw 的架构设计（Gateway、Sessions、Channels、Skills）
  * 了解文心大模型 API 的调用方式（千帆平台 / ERNIE Bot SDK）
  * 了解 MediaTek NeuroPilot SDK 的基本使用（加分项）

* 参考文档：
  * [OpenClaw 官网](https://openclaw.ai/)
  * [OpenClaw GitHub 仓库](https://github.com/openclaw/openclaw)
  * [OpenClaw 技能市场 ClawHub](https://clawhub.ai/)
  * [文心大模型 API 支持](https://ai.baidu.com/ai-doc/AISTUDIO/rm344erns)
  * [MediaTek NeuroPilot SDK](https://mediatek.gitlab.io/aiot/doc/neuropilot/)

* 附加说明：
  * 天玑9500 工程设备需在项目结束后归还联发科技
  * 开发者需签署相关保密协议（如涉及未公开的技术资料）
  * 联发科技将提供技术支持邮箱及工程师答疑渠道

### 此芯 & Arm：PaddleOCR-VL-1.5 在此芯 P1 芯片上的端侧部署与优化

* 技术标签：CIX P1，Armv9 CPU，CIX NOE SDK，PaddleOCR-VL-1.5，模型移植优化，异构算力调度
* 详细描述：本任务旨在将 PaddleOCR-VL-1.5 模型移植到此芯 P1 芯片平台，充分利用其 CPU+GPU+NPU 异构算力，实现文档解析的端侧高效推理，推动国产 AI 芯片在文档智能领域的应用落地。开发者只要完成任意一项任务，即视为成功。
  * **任务1：实现基于此芯 P1 的 CPU / GPU 的异构推理**
    * 实现 PaddleOCR-VL-1.5 的 Pipeline：实现 PaddleOCR-VL-1.5 在 CPU/GPU 上的基础推理。
    * 量化加速：使用 llama.cpp 或者 MNN 等推理框架，建议使用 int4 量化，对 PaddleOCR-VL-1.5 在 CPU/GPU 上完成加速推理。
  * **任务2 ：实现基于此芯 P1 的 CPU + NPU 异构推理**
    * 实现 PaddleOCR-VL-1.5 的 Pipeline：1）实现 Layout（版面分析）推理。 2）实现 PaddleOCR-VL-1.5 模型在 CPU + NPU 上的推理。
    * 量化加速：对 PaddleOCR-VL-1.5 模型进行 Q4_0 的量化和推理。
* 硬件支持：
  * 资源支持：提供硬件算力。
  * 详情：提供此芯 P1 硬件（瑞莎星睿 O6 开发板或美高迷你 PC）供选手使用。
* 提交内容：
    1. PaddleOCR-VL-1.5 在此芯 P1 上的详细部署步骤。
    2. 此芯 P1 推理引擎的使用说明。
    3. 示例应用（命令行工具或 GUI 演示）。
* 提交方式：
  1. 项目提交：提交使用案例到 [AI Studio](https://aistudio.baidu.com/projectoverview) 的项目并公开，请提交全部源码。
  2. 标题规范：标题处打上【PaddlePaddle Hackathon 10】。
  3. 基本要求：需包含 PaddleOCR-VL-1.5 在此芯 P1 上的部署详细步骤、此芯 P1 推理引擎的示例应用步骤（包括命令行工具 / GUI 演示）。
  4. 筛选依据：
       1. 该示例在真实场景下是否具有实际应用价值。
       2. 该示例的流程逻辑是否清晰。
       3. 运行结果是否符合预期。
  5. 验收标准（需全部满足）：部署流程清晰, 推理结果准确。具体包括：
      1. 完善相关代码, 成功地将模型运行在目标硬件平台上。请提供一步一步的完整的模型部署过程技术文档/报告，同时提供配套的完整测试代码（需符合开源代码的代码规范）。确保他人可根据该技术文档和代码复现该部署流程。
      2. 确保模型推理结果正确且稳定：请使用至少 3 张 PaddleOCR-VL 代码库中提供的官方 OCR 测试图片进行结果测试验证, 确保其结构化输出结果正确。请将相关测试图片和结果截图添加至上述技术文档/报告中。
          * 精度可接受：端侧推理的文档解析结果与原始文档图像，推理结果对比，文本识别准确率损失不超过 8 %。
          * 推理性能稳定：请使用此芯提供的指定文档图像，要求 batch=1 的完整 pipeline 推理时间应小于 60 s。
* 参考示例：考虑到通用性，选取的应用场景需要严格符合实时解析文档（发票、合同、表格等）的要求。
* 技术要求：模型架构理解、此芯 P1 硬件特性了解、开发工具链（推理框架，量化工具，编程语言 Python & C++）。
* 参考文档：[CIX AI Model Hub](https://modelscope.cn/models/cix/ai_model_hub/files?version=25_Q4)，[CIX NOE SDK](https://developer.cixtech.com/) (在此芯开发者中心找到 NeuralONE AI SDK，注册并下载)，[PaddleOCR-VL-1.5 模型](https://huggingface.com/PaddlePaddle/PaddleOCR-VL-1.5)，[Paddle 主仓库](https://github.com/PaddlePaddle/Paddle)，[PaddleX 仓库](https://github.com/PaddlePaddle/PaddleX)，[Paddle OCR使用教程](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html)

### 请 瑞芯微 填写

进阶任务：基于RK1820部署的PaddleOCR-VL模型，实现完整页面级文档解析功能

 - 技术标签：PaddleOCR-VL，RK3588+RK1820部署，RKNN2、RKNN3工具链，Python，C/C++

 - 详细描述：PaddleOCR-VL模型实现了文本块、表格、公式及图表的识别。通过RKNN3工具链，使用RK3588+RK1820进行模型端侧推理部署，搭建文本识别推理服务。参考PaddleOCR CLI中的doc_parser，构建依赖端侧识别推理服务的完整页面级文档解析服务，可使用Python或C/C++语言进行开发。本次任务评估将分为两个阶段，在第一阶段中，开发者需要实现基于RKNN3工具链的RK3588+RK1820部署的端侧识别推理服务，提供相应的解决方案报告。在第二阶段中，我们将从第一阶段提交的结果中，挑选出2份比较优秀的方案，并请相对应的开发者基于第一阶段中搭建的端侧识别推理服务，完成完整页面级文档解析功能。

 - 提交内容：
     - 第一阶段：
         - 提交内容：基于RK3588+RK1820部署搭建的端侧识别推理API服务的解决方案
         - 提交要求：解决方案报告（需包含技术思路，方案，测试结果及性能评估）；可复现的代码及环境
         - 筛选依据：1）完整的端侧识别推理API服务方案报告，2）识别结果正确（忽略模型量化引入的个别识别错误），3）推理性能为应用端调用API服务的单次推理耗时，4）筛选推理性能top2的队伍进入第二阶段
     - 第二阶段：
         - 提交内容：1）基于端侧文本识别推理服务的完整页面级文档解析服务代码；2）单页文档解析性能及耗时测试报告
         - 提交要求：代码需包含完整页面级文档解析功能，测试结果需包含单页文档解析性能及耗时测试报告
         - 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排

 - 提交方式：Github

 - 算力/资源支持：提供RK3588+RK1820开发板

 - 技术要求：

     - 熟练掌握RK开发板模型部署及工具链使用技巧
     - 熟悉Python或C/C++语言
     - 熟悉文档解析流程

### 地瓜机器人：openclaw-机器人快速方案验证

* 活动目标：本活动由地瓜机器人社区发起，围绕文心大模型 API 能力与机器人开发场景开展联合探索，验证 AI 机器人应用的可行落地方案，并筛选优秀开发者进入后续 RDK X3 硬件实践。具体目标如下
  1. 推动“机器人 + AI”联合开发模式在社区落地
  2. 通过文心大模型 API 快速构建机器人应用 Demo
  3. 在社区中筛选优秀项目与开发者
  4. 促进 RDK X3 在机器人应用中的实践与扩展

* 活动分为两个阶段：
  - 阶段 1：地瓜机器人社区技术分享 + 开发实践
    -  目标：引导开发者基于文心大模型 API 完成机器人相关案例分享，形成可复用的落地路径。
    -  形式：在地瓜机器人社区进行技术分享，主题包括但不限于：机器人 + AI 的应用模式与落地路径、文心大模型 API 的调用方式与集成要点、机器人 Demo 实战方法论
    -  分享内容重点：文心大模型 API 的调用方式、机器人开发流程、Demo 示例
    -  案例演示：AI Agent 机器人，其流程示例：用户指令 → LLM 理解 → 任务规划 → 机器人执行
  - 阶段 2：RDK X3 硬件试用（从阶段 1 参与开发者中筛选优秀开发者，开展进一步联合实践）
    -  支持与激励：提供 RDK X3 开发板、官方技术支持、社区宣传
    -  开发者任务：使用 RDK X3 对项目进行升级或重新实现，完成一个完整的机器人应用案例。
    -  示例方向：AI 语音机器人、视觉识别机器人、Agent 控制机器人
* 最终产出：
  * 沉淀 1–2 个机器人应用案例，包括 技术文章、Demo 视频、示例代码
  * 输出渠道：发布到地瓜机器人社区。
* 验收标准
  1. 至少提交 1 篇技术文章（含方案说明、系统架构、关键实现步骤）。
  2. 至少提交 1 个可运行 Demo（代码仓库可访问，含运行说明）。
  3. 至少提交 1 段演示视频（展示“指令输入 -> 模型理解 -> 机器人执行”的完整流程）。
  4. 方案中需明确文心大模型 API 的调用位置、输入输出与关键参数。
  5. 需说明机器人控制链路（感知/决策/执行）及与大模型的交互方式。
  6. Demo 需可复现，评审可按文档步骤在本地或指定环境完成运行。
  7. Demo 能稳定完成至少 3 组不同用户指令的任务执行演示。
  8. 对失败场景有基本处理说明（如异常输入、执行失败回退或提示机制）。
  9. 综合技术完整度、可复现性、演示效果进行评审。
  10. 通过阶段 1 验收的开发者，可进入阶段 2 的 RDK X3 硬件试用与深度实践。
* 社区入口：地瓜机器人社区 https://developer.d-robotics.cc/
* 联系邮箱：guosheng.xu@d-robotics.cc
* 地瓜机器人入群邀请

<img src="images/hackthon.png" width="300px">



### 麒麟：在openKylin桌面上运行的图形化AI助手demo应用
* 任务描述：开发一个可在openKylin桌面上运行的图形化AI助手demo应用，不限技术栈（如 Qt、GTK 等框架），基于ERNIE大模型的文本处理能力，实现基础对话功能即可。
* 作品提交：
  * 通过提交PR的方式，在如下 [仓库地址](https://gitee.com/openkylin/community-management/tree/master/2026%E9%BB%91%E5%AE%A2%E6%9D%BE%E5%A4%A7%E8%B5%9B-openKylin%E8%B5%9B%E9%81%93) 中，新建个人目录（以真实姓名为目录名），并将代码和运行效果视频上传到该目录下。
  * 作品提交成功后，请发送一封邮件附上 PR 链接到邮箱：：contact@openkylin.top + ext_paddle_oss@baidu.com
* 验收标准：
  * 在openKylin操作系统实现并运行
  * 需基于在openKylin本地运行的ERNIE-4.5-0.3B-PT大模型能力
  * 可以实现基础的文本对话功能

### 统信：deepin Agent Teams 智能体团队协作系统

**技术标签**：环境感知，意图识别，多智能体， MCP， Skills

**详细描述**：设计并实现一个运行在deepin操作系统上的智能体应用——**deepin Agent Teams**，该应用具备“环境感知”能力，通过分析用户的实时操作行为，如窗口标题、屏幕内容、交互动作等，主动理解用户意图并调用相应智能体提供辅助。

**第一阶段：环境感知与意图理解**

智能体需具备更高级别的环境感知能力，不仅能获取原始系统数据，还能进行多模态融合分析，并对用户意图进行更深层次的理解和预测。本阶段的核心挑战在于如何有效整合多源异构信息，使大模型能够精准识别用户意图。

1.  **多模态环境感知与融合**
    *   **视觉感知增强：** 不仅仅是识别屏幕上的文本，还需要结合图像识别来识别屏幕上的UI元素、应用、用户交互的上下文等。智能体应能根据用户鼠标移动、点击、键盘输入等行为，动态调整屏幕截图的关注区域，减少不必要的计算资源消耗，并提高意图识别的准确性。
    *   **多模态信息融合：** 将OCR识别的文本、图像识别的UI元素、窗口元数据（标题、类名等）、剪贴板内容等进行融合，并结合系统 API、D-Bus 信号、`wmctrl`、 `/proc` 信息等，构建更全面的用户操作上下文。例如，用户在代码编辑器中复制了一段错误信息，智能体不仅能识别错误文本，还能识别出这是“代码编辑器”中的“错误提示”，从而更准确地推断用户意图是“寻求代码调试帮助”。
    *   **系统行为预测：** 智能体应能记录并分析用户一系列操作行为，如打开应用、切换窗口、输入文本、点击按钮等，从中学习用户的工作模式和习惯，并预测用户下一步可能的操作。基于行为序列分析和多模态感知，智能体应能主动预测用户意图，并在用户明确发出指令之前，提前准备好相关信息或工具。例如，用户连续打开多个与项目相关的文档，智能体可以预测用户可能需要“项目总结”或“信息汇总”服务。

2.  **复杂意图识别与上下文管理**
    *   **多轮对话与意图澄清：** 智能体应能支持多轮对话，在用户意图不明确时，主动进行提问和澄清，逐步缩小意图范围。
    *   **跨应用上下文理解：** 智能体应能理解用户在不同应用之间切换时的上下文关联。例如，用户在浏览器中搜索某个技术问题，然后切换到代码编辑器，智能体应能将浏览器中的搜索内容与代码编辑器中的代码关联起来，提供更精准的帮助。
    *   **情感与语气分析（可选）：** 智能体可以尝试分析用户输入的文本或语音中的情感和语气，以便在提供帮助时调整回复的风格和优先级。

**第二阶段：多智能体协作与任务执行**

在第一阶段所构建的强大环境感知和意图理解能力的基础上，本阶段旨在构建一个高效、智能的多智能体团队，使其能够自动拆解复杂任务并协同完成。

1.  **智能体团队构建与调度**
	-   **智能体创建：** 构建至少包含3个不同职能的智能体团队（例如：系统操作员、信息收集员、内容创作员等）。
    *   **动态智能体编排：** 智能体团队应具备动态编排能力，能够根据任务需求灵活选择、组合和调度智能体，并支持智能体间高效的任务交接与信息共享。
    *   **任务拆解与子任务分配：** 智能体团队应能将复杂任务自动拆解成更小的子任务，并根据每个智能体的专长和当前状态，智能地分配子任务。
    *   **冲突解决与协商机制：** 当多个智能体在执行任务过程中出现冲突或需要共享资源时，应具备冲突解决和协商机制，确保任务顺利进行。

2.  **工具使用与技能扩展**
    *   **智能工具选择与参数填充：** 智能体应能根据用户意图和当前上下文，主动弹出交互界面，并智能地选择合适的工具。
    *   **自适应技能学习：** 智能体应能通过学习用户操作和任务执行结果，不断优化自身的技能库，甚至可以从用户那里学习新的技能。
    *   **MCP、Skills 工具的深度集成：** 深入利用 MCP 协议和 Skills 规范，实现与更多第三方服务和自定义插件的无缝集成，从而有效扩展智能体的能力边界。
	以下是工具能力的举例：

| 工具类型         | 具体要求                              | 示例场景                         |
| ---------------- | ------------------------------------- | -------------------------------- |
| **系统工具**     | Bash命令执行、文件搜索、应用启动/停止 | 查找文件、安装软件、启动应用     |
| **系统配置工具** | 修改系统设置（对标deepin控制中心）    | 修改网络设置、调整显示、更改主题 |
| **MCP工具**      | 支持MCP协议的工具接入                 | 接入第三方服务、自定义插件       |
| **SKILLS**       | 预定义的技能模块                      | 邮件撰写、日程安排、信息汇总     |

**场景实现要求**

参赛作品必须能够演示以下两个核心场景，并充分体现智能体在**多模态环境感知、复杂意图理解、系统行为预测以及多智能体协同**方面的能力：

**场景一：智能邮件助手**

当系统通过多模态环境感知（如识别邮件客户端界面、用户输入关键词、剪贴板内容等）预测用户有发送邮件的意图时，智能体团队需：
1.  **深度意图识别：** 结合用户当前操作上下文（如正在查看的项目文档、会议日程等），智能识别邮件主题、收件人、以及邮件的核心目的。
2.  **多源信息智能聚合：**
    *   从文件系统中智能搜索与邮件主题或收件人相关的项目文档、报告。
    *   从剪贴板中理解并提取用户最近复制的关键内容。
    *   将上述多源信息进行融合分析，提炼出邮件所需的关键点。
3.  **智能体协同撰写：** 由“信息收集员”智能体负责信息聚合，“内容创作员”智能体根据聚合信息和用户意图自动生成结构清晰、内容完整的邮件正文，并可根据用户反馈进行迭代优化。
4.  **智能呈现与发送：** 生成邮件主题、收件人及正文，并提供发送前的预览和修改选项，或在获得用户明确授权后自动发送。

**场景二：系统问题智能诊断与修复**

当用户通过自然语言输入系统相关问题（如“打印机连不上”、“没有声音了”、“帮我安装微信”等），或系统通过环境感知（如检测到系统错误日志、硬件状态异常、应用崩溃等）主动发现潜在问题时，智能体团队需：
1.  **多模态问题分析与意图澄清：**
    *   **结合系统感知：** 利用屏幕图像识别（如错误弹窗、设备管理器界面）、系统日志分析、硬件状态监控等多模态信息，对用户描述的问题进行深度分析，识别问题类型、受影响的组件及可能的根本原因。
    *   **智能体交互澄清：** 当问题描述不明确时，由“系统操作员”智能体主动与用户进行**多轮对话**，提问关键信息，澄清用户意图，直至准确理解问题。
2.  **智能体协同诊断与方案生成：**
    *   “信息收集员”智能体实时检查当前系统状态：包括打印机服务状态/驱动情况、音频设备与音量设置、软件源与应用安装状态、网络连接状况等。
    *   “系统操作员”智能体根据诊断结果，智能生成一套或多套可行的修复方案，并评估其风险和效果。
3.  **自动化修复与用户确认：**
    *   在获得用户明确授权后，由“系统操作员”智能体自动执行修复操作：如重启服务、更新驱动、调整系统配置、执行安装命令等。
    *   对于复杂或有风险的操作，智能体应提前告知用户，并等待用户确认。
4.  **智能反馈与效果验证：** 向用户清晰反馈处理结果，并验证问题是否已解决。如果问题未完全解决，智能体团队应能继续诊断并提供替代方案。

**验收标准：**

| **验收维度**       | **验收标准详情**                                             |
| ------------------ | ------------------------------------------------------------ |
| **系统感知能力**   | 集成多模态模型，能够识别屏幕特定区域的文字或图像内容。系统能基于感知到的上下文（如剪贴板变化、窗口切换、输入内容）主动弹出辅助建议，且意图识别准确率在演示中表现稳定。 |
| **多智能体协同**   | 智能体具备基础对话、系统工具调用（Bash、文件、设置等）以及多智能体协同能力。 |
| **场景演示成功**   | 完整演示“智能邮件助手”与“系统问题修复”两个场景，逻辑闭环，无崩溃现象。 |
| **文档与报告**     | 提供完整的源码、详尽的部署说明（确保环境可复现）以及技术报告（需包含系统架构图及意图识别原理说明）。 |
| **代码规范与体验** | 代码结构清晰，注释规范；GUI 交互流畅，无明显的性能卡顿，资源占用在合理范围内。 |

- **提交内容：**
	1. **源代码**：完整的项目代码，包含清晰的目录结构
	2. **部署文档**：环境配置、依赖安装、运行说明
	3. **演示视频**：录制视频，展示核心功能与两个场景
	4. **技术报告**：包含系统架构、关键技术、创新点说明，**重点阐述多模态融合意图识别原理和智能体动态编排与任务调度机制**

- **提交⽅式：** 代码托管于 GitHub，仓库由参赛者自行创建。

- **参考示例：** 无。

- **技术要求：**
	1. **系统环境**：deepin 25。
	2. **模型支持**：至少调用两款飞桨文心大模型 API（erniebot SDK）。
	3. **开发语言**：推荐使用 Python 或 C++/Qt。
	4. **隐私安全**：系统感知功能需注意用户隐私保护，敏感操作需获得用户明确授权。
	5. **交互入口：**

        - 提供唯一的用户交互界面（可以是悬浮球、侧边栏或独立窗口）
        - 支持用户通过自然语言输入任务指令
        - 系统需具备**智能决策调度**能力，根据用户指令自动选择合适的智能体执行

	6. 鼓励使用 deepin 操作系统的特色功能（如控制中心API、DDE桌面环境特性），环境感知需注意对 deepin 系统资源的占用的控制。

- **参考⽂档：**
	- **飞桨文心大模型SDK**：[ERNIE Bot SDK & 文档](https://github.com/PaddlePaddle/ERNIE-Bot-SDK)
	- **PaddleOCR**：[PaddleOCR GitHub仓库](https://github.com/PaddlePaddle/PaddleOCR)
	- **deepin 开发者社区**：[deepin 社区论坛](https://bbs.deepin.org/) （可用于获取 deepin 系统相关的开发帮助）
	- **飞桨星河社区：**[飞桨星河社区链接](https://aistudio.baidu.com/)
	- **赛题咨询：**[deepin 当前赛题相关咨询、公告等](https://github.com/deepin-mozart/Hackathon-deepin/issues)

## 💎 开源贡献任务

部分合作伙伴特别设立「开源贡献任务」，鼓励开发者在参赛过程中积极发现问题、提出建议（ISSUE）、贡献代码（PR）。您的每一次高质量贡献都可能获得额外奖励！

### 参与厂商

| 厂商 | 奖励内容 | 认定标准 |
|:--:|---|---|
| 沐曦 | 额外好礼，价值不低于100元 | 对沐曦 GPU 平台适配有实质性帮助的 Issue/PR |

> 💡 更多厂商持续加入中，欢迎关注更新！

### 贡献类型示例

| 类型 | 说明 | 示例 |
|:--:|---|---|
| 🐛 Bug 反馈 | 发现平台/工具链中的 Bug，提供复现步骤 | 编译报错、运行时异常、性能问题等 |
| 💡 功能建议 | 提出有价值的功能改进建议 | 新算子支持、API 改进、文档完善等 |
| 🔧 代码贡献 | 提交 PR 修复问题或新增功能 | Bug 修复、性能优化、功能实现等 |
| 📖 文档贡献 | 完善文档、教程、示例代码 | 补充缺失文档、修正错误、添加用例等 |


### 评选流程

1. **厂商评审**：厂商工程师对提交的贡献进行评审
2. **认定标准**：是否对平台/工具有实质性改进价值
3. **榜单公示**：每两周随动态公示同步更新本榜单
4. **奖励发放**：活动结束后统一发放
