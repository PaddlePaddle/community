
# 在 PaddleNLP 中复现 Apollo 精调算法

|任务名称 | 在 PaddleNLP 中复现 Apollo 精调算法 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 1111yiyiyiyi | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2025-03-05 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20250305_add_apollo_for_paddlenlp.md<br> |


# 一、概述
## 1、相关背景

训练 LLM 时，不仅需要存储亿级参数，还必须额外保存梯度与优化器状态（例如 Adam 中的动量和方差）。
例如，预训练 LLaMA 7B 的一个批次就消耗高达 58GB 内存：14GB 用于模型参数，42GB 用于优化器状态和梯度。
这巨大的 “隐形” 内存开销迫使研究者不得不选用显存更大的高端 GPU、增加 GPU 数量，甚至牺牲训练速度调低批量大小。
Apollo 精调算法的优点：
- 极低内存消耗：首次以类 SGD 内存成本完成大模型训练，达到甚至超越 AdamW 的性能。
- 无需 SVD 计算：首次实现仅需轻量级随机投影进行大模型预训练，甚至在 7B 模型上优化速度超越 Adam。
- 卓越系统性能：3 倍预训练加速：在 8 块 A100 GPU 上，Apollo 预训练 LLaMA 7B 模型实现了 3 倍的加速。

## 2、功能目标

在PaddleNLP中实现Apollo优化器，复现论文中 Apollo 与 LoRA 在 Llama3-8B 上的实验结果，优化器状态显存占用相比 AdamW 降低 50% 以上。

## 3、意义

在PaddleNLP支持Apollo优化器，带来训练性能上的提升。

# 二、飞桨现状

[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP),只支持adamw，adamw_mini两种优化器。

# 三、业内方案调研

Apollo 已在 Hugging Face Transformers、LLaMA-Factory 等主流开源项目中被支持。

# 四、对比分析

对标论文中的实现，让paddle支持Apollo优化器。

# 五、设计思路与实现方案

- paddlenlp/utils/optimizer.py中实现Apollo的主要逻辑。
  
- 修改paddlenlp/trainer/trainer.py以支持Apollo优化器。

- 修改对应的文档与添加对应的测试。

- 测试Apollo优化器在 Llama3-8B 上的实验结果，达到优化器状态显存占用相比 AdamW 降低 50% 以上。

# 六、测试验收的考量

对应的Apollo优化器的代码，一个对比实验的报告。

# 七、可行性分析和排期规划

- paddlenlp/utils/optimizer.py中实现Apollo的主要逻辑。一周
  
- 修改paddlenlp/trainer/trainer.py以支持Apollo优化器。一天

- 修改对应的文档与添加对应的测试。三天

- 测试Apollo优化器在 Llama3-8B 上的实验结果，达到优化器状态显存占用相比 AdamW 降低 50% 以上。两周

# 八、影响面

PaddleNLP新增Apollo优化器。

# 名词解释
# 附件及参考资料

参考PR https://github.com/PaddlePaddle/PaddleNLP/pull/9542


