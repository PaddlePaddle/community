# PaddleNLP 待支持动转静模型列表文档

|任务名称 | 为PaddleNLP模型接入动转静训练功能 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-11-19 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20231119_add_to_static_for_paddlenlp.md<br> | 


# 一、概述
## 1、相关背景
目前飞桨的开源套件如 PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR 等，都支持了动转静训练功能，但是并非所有的模型都接入了 `--to_static` 策略。

随着 PaddleSOT 功能的完善和上线，动转静训练成功率大幅度提升，故可将动转静训练策略在开源套件中的所有模型进行推全。

## 2、功能目标
本次任务对 PaddleNLP 中所有模型进行动转静训练策略推全。


## 3、意义

推全动转静训练策略，可提高 PaddleNLP 模型推理部署的运行性能。

# 二、任务内容

目前 PaddleNLP 中待支持动转静模型列表如下:


| 模型名称 | 模型位置 | 
|---|---|
|electra| model_zoo | 
|ernie-1.0| model_zoo | 
|ernie-3.0| model_zoo | 
|ernie-code| model_zoo | 
|ernie-doc| model_zoo | 
|ernie-gen| model_zoo | 
|ernie-health| model_zoo | 
|ernie-layout| model_zoo | 
|ernie-m| model_zoo | 
|ernie-tiny| model_zoo | 
|ernie-vil2.0| model_zoo | 
|gpt| model_zoo | 
|gpt-3| model_zoo | 
|plato-xl| model_zoo | 
|tinybert| model_zoo | 
|uie| model_zoo | 

