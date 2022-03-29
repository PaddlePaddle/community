# 基于 XLNet 的阅读理解设计文档

|       |              |
| ------------ | ------------------------------------------------- |
| 提交作者     | Tukta                                             |
| 提交时间     | 2022-03-23                                        |
| 版本号       | V1.0                                              |
| 依赖飞桨版本 | develop                                            |
| 文件名       | 20220323_design_for_xlnet_readingcomprehension.md |

# 一、概述

## 1、相关背景
为了实现基于sentencepiece tokenier类型的PLM的阅读理解，以及增加中文阅读理解CMRC任务，PaddleNLP需要实现基于xlnet模型的中文阅读理解任务的实现。
## 2、功能目标
在paddlenlp repo的阅读理解示例目录之下增加CMRC任务，并添加使用xlnet在CMRC数据集上进行微调的脚本。

## 3、意义
为飞桨提供了基于sentencepeice tokenizer中文阅读理解的支持。

# 二、飞桨现状
目前paddle缺少相关功能实现。paddlenlp中阅读理解任务目前没有基于sentence piece的tokenizer的实现方案，但许多基于sentence piece tokenizer的模型都在阅读理解任务中取得很好的效果，比如xlnet，t5等等，此外，目前飞桨官方的示例中没有关于中文阅读理解任务的示例，所以需要添加一个sentence piece based tokenizer的模型进行中文阅读理解微调任务的示例代码。

# 三、业内方案调研
## Pytorch
Pytorch中提供了基于xlnet的阅读理解的示例，其提供的是xlnet（sentencepeice based tokenizer）在英文阅读理解任务squad以及squadv2的微调实现代码，数据处理核心代码如下：
```python
def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
        special_tokens = tokenized_examples.pop("special_tokens_mask")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossible"] = []
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples["cls_index"].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples["token_type_ids"][i]
            for k, s in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            context_idx = 1 if pad_on_right else 0

            # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
            # The cls token gets 1.0 too (for predictions of empty answers).
            tokenized_examples["p_mask"].append(
                [
                    0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)
                ]
            )

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1.0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_idx:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_idx:
                    token_end_index -= 1
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossible"].append(1.0)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["is_impossible"].append(0.0)

        return tokenized_examples
```

- 根据观察，我们得知，在Pytorch中，对于sentence piece based tokenizer类型的模型，实现阅读理解任务，非常重要的一点是对齐token与原始的文本，offsetmap至关重要。
- huggingface transformers 的xlnet tokenizer中已经包含offsetmap的生成，所以paddle中，在任务中生成合适的offsetmap是至关重要的一步。
- 中文阅读理解任务与英文相同，所以可以考虑数据处理，评估方式等有差异的方面进行更改。

## CMRC 微调示例代码 
参考链接：https://github.com/ewrfcas/bert_cn_finetune

* 该仓库包含许多中文任务的微调代码，可以参考xlnet在CMRC中的微调代码进行处理。


# 四、方案设计
1. 在paddlenlp的examples中，为machine_reading_comprehension目录增加CMRC任务目录。
2. 模仿SQuAD目录结构，增加微调脚本，预测脚本，模型导出脚本，readme说明等文件。
3. 在阅读理解任务中针对xlnet进行offsetmap的计算，对数据预处理进行改动。
4. 在结果评估方法中，针对中文任务进行调整，比如标点符号处理等。

# 五、测试和验收的考量
测试考虑的case如下：

- 提供在CMRC数据集上微调xlnet的脚本
- 脚本具有通用性，可以迁移到其它中英文问答任务，比如squad
- 能够解决偏移量映射图的问题，准确找出坐标
- 能够实现train，eval以及predict功能
- 代码匹配最新的paddlenlp，解决兼容性问题（squad代码中dataset数据类型是huggingface的dataset，新版本paddlenlp不兼容）
- 根据CMRC的数据，精度达到预期标准；

# 六、可行性分析及规划排期

方案主要依赖现有paddlenlp的squad方法，huggingface的examples以及github中对xlnet在各个问答数据集微调的repo进行实现，如果合理参考，实现可行。

# 七、影响面
独立实现examples，必要时可能会为xlnet增加额外的数据处理方法。



# 名词解释
无
# 附件及参考资料
无
