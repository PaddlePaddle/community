# 基于高通Qualcomm AI Engine Direct (QNN) 部署 PaddleOCR-VL 模型，实现端侧页面级文档解析功能

- **技术标签**：PaddleOCR-VL，高通 QNN SDK，Hexagon NPU，Paddle2ONNX，Python，C/C++

- **任务难度**：进阶

- **详细描述**：

  PaddleOCR-VL 模型基于视觉语言大模型（PP-DocBee2-3B）实现了文本块、表格、公式及图表的统一识别，是 PaddleOCR 3.x 的旗舰文档解析方案。本赛题要求选手通过高通 QNN（Qualcomm Neural Networks）工具链，将 PaddleOCR-VL pipeline 中的关键子模型（布局检测模型、VL 识别模型等）转换并部署至高通 Hexagon NPU进行端侧推理。参考 PaddleOCR CLI 中的 `doc_parser`，构建依赖端侧推理服务的完整页面级文档解析 pipeline，可使用 Python 或 C/C++ 语言进行开发。

  **整体流程**：
  1. 使用 Paddle2ONNX 将 PaddleOCR-VL 的子模型（布局检测模型、VL 识别模型）导出为 ONNX 格式
  2. 使用 QNN SDK 工具链（`qnn-onnx-converter` → `qnn-model-lib-generator` → `qnn-context-binary-generator`）将 ONNX 模型转换为 QNN 格式并针对 HTP 后端进行优化
  3. 对模型进行量化（INT8/INT16/FP16），以适配 HTP 高效推理
  4. 搭建端侧推理服务，串联各子模型构建完整的文档解析 pipeline

- **提交内容**：

  1. 模型转换脚本及说明文档（Paddle → ONNX → QNN 全链路）
  2. 基于高通 QNN 部署的端侧推理服务代码
  3. 基于端侧推理服务的完整页面级文档解析 pipeline 代码（参考 `doc_parser`）
  4. 精度对比报告（端侧推理结果 vs 原始 PaddlePaddle 推理结果）
  5. 单页文档解析性能及耗时测试报告

- **验收要求**（需全部满足）：

  1. **模型转换完整**：完成布局检测模型和 VL 识别模型从 Paddle → ONNX → QNN 的全链路转换，提供可复现的转换脚本
  2. **端侧推理可运行**：转换后的 QNN 模型可在高通 HTP-simulator 或实际高通设备上成功加载并执行推理
  3. **文档解析 pipeline 可用**：参考 `doc_parser` 实现完整的页面级文档解析功能，输入单页文档图片，输出结构化 Markdown 解析结果，覆盖文本块与表格两种以上版面元素
  4. **精度可接受**：端侧推理的文档解析结果与原始 PaddlePaddle 推理结果对比，文本识别准确率损失不超过 5%
  5. **提供完整的工程代码、运行步骤文档及性能测试报告**

- **技术要求**：

  - 熟练掌握高通 QNN SDK 工具链使用（模型转换、量化、HTP 后端部署）
  - 熟悉 Paddle2ONNX 模型转换流程
  - 熟悉 Python 或 C/C++ 语言
  - 熟悉文档解析流程及 PaddleOCR-VL pipeline 架构
  - 了解模型量化技术（INT8/INT16/FP16）

- **参考资料**：

  - PaddleOCR-VL 文档：https://github.com/PaddlePaddle/PaddleOCR
  - Paddle2ONNX：https://github.com/PaddlePaddle/Paddle2ONNX
  - Qualcomm QNN SDK：https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
  - PaddleOCR CLI `doc_parser` 使用说明