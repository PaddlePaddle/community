此文档展示 **PaddlePaddle Hackathon 第九期活动——飞桨护航计划集训营（正式批）** 项目的详细介绍

## 赛题详情

### 项目一：模型迁移工具链建设

#### 项目介绍：

为了实现高效的将PyTorch代码自动化的转写成Paddle代码，从而提升模型迁移的效率，我们建设了[PaConvert代码自动转换工具](https://github.com/PaddlePaddle/PaConvert): PaddlePaddle Code Convert Toolkits。目前已支持约1800个Pytorch API的自动转换与95+%的代码转换率，但在 新增转换策略、转换机制优化、转换策略与Paddle主框架对齐、转换策略与映射文档一致性检查 等方面，仍然有很多可持续完善的地方。同时在Paddle框架API、[PaDiff精度对齐工具](https://github.com/PaddlePaddle/PaDiff)方面也有可继续完善的地方。
本课题的工作任务包括转换工具建设的以下内容：

1. 新增API转写策略，例如大模型相关的API转写策略
2. 优化转换机制，例如自定义算子转写机制方面的问题
3. 转换策略与Paddle主框架对齐，与框架API的增强项目保持同步
4. 转换策略与映射文档一致性校正，优化已有的自动校验工具，增强CI拦截手段
5. 转换工具CI日常维护
6. 视情况也可开展一些Paddle框架API增强或[PaDiff精度对齐工具](https://github.com/PaddlePaddle/PaDiff)增强的工作

#### 营员要求（1 人）：

- 熟悉 Python
- 熟悉Pytorch和Paddle框架使用
- 熟悉大模型相关的代码，HuggingFace transformers库及paddlenlp/paddleformers库
- 有论文复现与大模型迁移经验优先
