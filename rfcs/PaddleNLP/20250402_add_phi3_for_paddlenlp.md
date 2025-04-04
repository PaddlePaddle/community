

# 在 PaddleNLP 中实现 Phi-3 模型

| 任务名称 | 在 PaddleNLP 中实现 Phi-3 模型 |
| --- | --- |
| 提交作者 | robinbg |
| 提交时间 | 2025-04-02 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20250402_add_phi3_for_paddlenlp.md |

# 一、概述

## 1、相关背景
Phi-3是Microsoft最新发布的大规模语言模型，在推理和数学能力方面表现优异。将Phi-3引入PaddleNLP生态系统，可为用户提供更多高性能模型选择，丰富飞桨的模型库。

## 2、功能目标
在PaddleNLP中实现Phi-3模型，确保与原始实现的性能对齐，并提供完整的文档和示例。

## 3、意义
丰富PaddleNLP的模型库，为用户提供更多高性能模型选择。

# 二、飞桨现状
PaddleNLP目前尚未支持Phi-3模型。

# 三、业内方案调研
Phi-3模型已在Hugging Face Transformers等主流开源项目中得到支持。

# 四、对比分析
对标原始实现，确保性能完全对齐。

# 五、设计思路与实现方案

## 1. 精度对齐
- FP32 精度下误差 < 1e-5
- BF16 精度下误差 < 1e-3

## 2. 核心组件实现
- PhiConfig: 模型配置类
- PhiTokenizer: 分词器实现
- PhiPreTrainedModel: 预训练模型基类
- PhiModel: 主干网络实现
- PhiForCausalLM: 生成任务模型实现
- _get_name_mappings方法: 参数转换实现，参考LLaMA代码实现，用于将Hugging Face权重映射到PaddleNLP格式

## 3. 模型并行实现
- 实现模型并行相关代码，参考LLaMA实现
- 暂不实现sequence_parallel相关逻辑
- 添加`if config.tensor_parallel_degree > 1`相关代码，包括但不限于：
  - 张量并行化处理
  - 权重分片
  - 通信原语封装
- 优先完成单卡模型组网，后续再支持模型并行

## 4. 测试验证
- 模型结构测试
- 前向计算测试
- 精度对齐测试
- Tokenizer测试
- 模型并行功能测试（单独阶段）

# 六、测试验收的考量

## 1. 精度对齐
- [ ] FP32: 误差 < 1e-5
- [ ] BF16: 误差 < 1e-3

## 2. 代码完备性
- [ ] 模型组网完整
- [ ] Tokenizer完整
- [ ] 接口规范
- [ ] _get_name_mappings方法实现完整
- [ ] 单卡模型功能完整

## 3. 测试完备性
- [ ] 单元测试覆盖
- [ ] 精度测试完整
- [ ] CI测试通过
- [ ] 单卡模型测试通过

## 4. 文档完备性
- [ ] API文档
- [ ] 示例代码
- [ ] 代码注释

## 5. 模型并行（后续阶段）
- [ ] 模型并行代码实现
- [ ] 模型并行测试通过
- [ ] 模型并行文档完善

# 七、可行性分析和排期规划

## Day 1-3
- 完成模型核心代码
- 实现tokenizer
- 搭建验证框架
- 实现_get_name_mappings方法

## Day 4-7
- 编写单元测试
- 进行精度验证
- 性能优化
- 完成单卡模型功能测试

## Day 8-10
- 完善文档
- 添加示例
- 完善注释
- 规划模型并行实现方案

## Day 11-15（后续阶段）
- 实现模型并行相关代码
- 模型并行功能测试
- 模型并行文档完善

# 八、影响面
PaddleNLP新增Phi-3模型支持，包括单卡模型和模型并行能力。

# 名词解释
无

# 附件及参考资料
- Phi-3模型论文
- Microsoft Phi-3官方实现
- [LLaMA _get_name_mappings方法实现](https://github.com/PaddlePaddle/PaddleNLP/blob/bfd053db0897943f5d4d116dde755dbf21d18b23/paddlenlp/transformers/llama/modeling.py#L1334-L1366)
