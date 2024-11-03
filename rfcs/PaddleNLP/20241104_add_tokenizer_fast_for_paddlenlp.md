
# PaddleNLP 支持 TokenizerFast

|任务名称 | 为PaddleNLP 支持 TokenizerFast | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yinfan98 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-11-04 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20241104_add_tokenizer_fast_for_paddlenlp.md<br> | 


# 一、概述
## 1、相关背景
TokenizerFast是Hugging Face的Transformers库中的一个类，用于快速分词。它基于SentencePiece或WordPiece模型，可以将文本转换为模型所需的输入格式。TokenizerFast支持多种语言和模型，可以用于自然语言处理任务，如文本分类、命名实体识别、机器翻译等。
TokenizerFast和Tokenizer都是Hugging Face的Transformers库中的类，用于将文本转换为模型所需的输入格式。它们的主要区别在于实现方式和性能。
- Tokenizer是基于Python的实现，而TokenizerFast是基于C++的实现，因此TokenizerFast通常比Tokenizer更快。
- TokenizerFast支持SentencePiece和WordPiece模型，而Tokenizer只支持WordPiece模型。
- TokenizerFast支持更多的语言和模型，因此在处理多语言和多模型时更灵活。
- TokenizerFast支持更多的功能，如添加特殊标记、添加位置信息等，因此在处理复杂任务时更强大。
目前，PaddleNLP里已有基础的TokenizerFast基建，但缺少对TokenizerFast用于存量模型的过程上。但当前大部分模型不包含 tokenizer_fast 实现，因此无法享受到 TokenizerFast 带来的性能提升。


## 2、功能目标
完善 TokenizerFast 功能支持，编写单测并验证大规模数据集。


## 3、意义

全量支持TokenizerFast，能带来性能上提升。

# 二、任务内容

- 实现 bert、bloom、chatglm、ernie、gemma、gpt、qwen、qwen2 对应的 toekizer_fast.py 文件
  
- 撰写理论上精度对齐报告和实验证明，在上述模型下TokenizerFast能和Tokenizer结果对齐

- 预计对上述8个模型每个模型选取2种大规模数据集，在16个单测上对齐性能

- 产出自动化单测脚本，方便对齐后续模型
