# 微调 PaddleOCR-VL 新姿势 -- Prompt 与 信息抽取

> AI Studio 项目地址：[微调 PaddleOCR-VL 新姿势 -- Prompt 与 信息抽取](https://aistudio.baidu.com/projectdetail/9857242) ，可在 AI Studio 的 A100 环境中直接运行（V100 环境只能进行模型推理，无法进行微调）

## 引言

当使用 PaddleOCR-VL 时，会使用到如下的代码:

```python
CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
```

因此，PaddleOCR-VL 可以识别文本、公式、表格和图表元素。

PaddleOCR-VL 作为一款专为文档理解设计的视觉-语言模型（Vision-Language Model, VLM），是通过 `提示词` 完成不同任务的。

目前对于 PaddleOCR-VL 的微调 [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 也是围绕这四类任务展开的。

本文从微调 PaddleOCR-VL 的 `提示词` 入手，介绍如何通过微调 PaddleOCR-VL 用于 `信息抽取`。

### 微调结果对比

这里以识别与抽取一张发票内的信息为例：

**微调之前**

![raw](images/raw.png)

<details>

<summary> 点击查看原始输出 </summary>


```json

{
    "res": {
        "input_path": "/home/aistudio/paddleocr_vl/data/test.jpg",
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "parsing_res_list": [{
            "block_label": "ocr",
            "block_content": "购买方信息 | 名称 | 中青旅联科 | 杭州 | 公关顾问有限公司 | 销售方信息 | 名称 | 杭州万力酒店管理有限公司 | 统一社会信用代码/纳税人识别号 | 纳税人识别号 | 统一社会信用代码/纳税人识别号 | 税额 | 税额/征收率 | 税额/征收率\n**项目名称** | 规格型号 |   |   |   |   |   |   |   |   |   |   |   |  \n**住宿服务** | 住宿费 |   |   |   |   |   |   |   |   |   |   |   |  \n**合计** |   |   |   |   |   |   |   |   |   |   |   |   |  \n**价税合计（大写）** |   | 壹仟叁佰玖拾柒圆整 |   |   |   |   |   |   |   |   |   |   |  \n备注 | 销售方地址：浙江省杭州市西湖区转塘街道霞鸣街199号万美商务中心3号楼；电话：0571-85220222；销方开户银行：农行上泗支行；入住人：柳顺；入住日期：9月23日入住-9月26日退房；入住天数：3天；金额：1397元 |   |   |   |   |   |   |   |   |   |   |   |   |  \n开票人：祝营营",
            "block_bbox": [0, 0, 1260, 838]
        }]
    }
}

```

</details>

**微调之后**

![after](images/sft.png)

<details>

<summary> 点击查看微调之后的输出 </summary>

```json
{
    "res": {
        "input_path": "/home/aistudio/paddleocr_vl/data/test.jpg",
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "parsing_res_list": [{
            "block_label": "OCR:{}",
            "block_content": "{"发票信息": {"发票名称": "电子发票", "发票号码": "25332000000426443187", "开票日期": "2025年09月26日"}, "销售方信息": {"名称": "杭州万力酒店管理有限公司", "统一社会信用代码": "91330105MA2H2DUJ92", "纳税人识别号": "91330106MA2B1C4UXN"}, "项目名称": "规格型号", "单位": "个", "数量": "3 461.056105610561", "单价": "1383.17", "金额": "税率/征收率", "税额": "13.83"}, "合计": {"金额": "1383.17", "税额": "13.83"}, "价税合计（大写）": "壹仟叁佰玖拾柒圆整", "价税合计（小写）": "1397.00"}, "销售方地址": "浙江省杭州市西湖区转塘街道霞鸣街199号万美商务中心3号楼", "电话": "0571-85220222", "销方开户银行": "农行上泗支行", "入住人": "柳顺", "入住日期": "9月23日 入住-9月26日 退房", "入住天数": "3天", "金额": "1397元"},",
            "block_bbox": [0, 0, 1260, 838]
        }]
    }
}

```

或者指定字段抽取特定信息：

``` json
{
    "res": {
        "input_path": "/home/aistudio/paddleocr_vl/data/test.jpg",
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "parsing_res_list": [{
            "block_label": "OCR:{"发票号码": "", "开票日期": ""}",
            "block_content": "{"发票号码": "25332000000426443187", "开票日期": "2025年09月26日"}",
            "block_bbox": [0, 0, 1260, 838]
        }]
    }
}

```

</details>

微调之后可以输出 `JSON` 格式的数据，并且可以根据不同的 `prompt`（这里的 `block_label`）输出对应的信息。

> 由于此次微调的数据量很少，因此微调结果并不好，此处仅做参考。

关于 PaddleOCR-VL 的微调，[PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 中已经有很详细的介绍，由于本文微调针对的是 `prompt`，因此在：

- 数据准备
- 模型推理

这两部分与原文略有不同。

## 数据准备

使用 ERNIE 对 PaddleOCR-VL 进行微调，需要准备 `JSON` 格式的数据与对应的图片数据：

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, “এইই; পযছ! পযছ!'\nওপর", "tag": "no_mask"},
    ]
}
```

其中，

- `image_url` 是图片的路径
- `tag` 是 `mask` 的 `text_info` 对应 `prompt` 部分，也就是 PaddleOCR-VL 的 `TASK` 类型
- `tag` 是 `no_mask` 的 `text_info` 对应 `completion` 部分，也就是模型的输出

原始模型中只有

```json
{
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
```

这四类 `prompt`，而我们希望，模型能够根据我们的指令抽取对应的信息，因此需要自定义 `prompt`：

```json
{
    "image_info": [
        {
            "matched_text_index": 0,
            "image_url": "/home/aistudio/paddleocr_vl/data/zzsptfp/zzsptfp/b175.jpg"
        }
    ],
    "text_info": [
        {
            "text": "OCR:{\"发票名称\": \"\"}",
            "tag": "mask"
        },
        {
            "text": "{\"发票名称\": \"广东增值税专用发票\"}",
            "tag": "no_mask"
        }
    ]
}
```

这里 `tag` 为 `mask` 的 `text` 不是 `OCR:` 而是 `OCR:{\"发票名称\": \"\"}`，也就是说，我们希望模型抽取，且仅输出 `发票名称` 字段。

保留原始的 `OCR:` 部分，是为了保证模型能够识别 `OCR:` 部分，而仅对 `{\"发票名称\": \"\"}` 部分进行微调。

`tag` 为 `no_mask` 的 `text` 部分直接输出 `JSON` 格式的数据，并且与 `prompt` 对应。

最后，我们这里设计 `prompt` 为：

``` text
# 特定值为字符串，如 `{"发票编码":"123456"}`
"OCR:{\"xxx\":\"\"}"

# 特定值为字典，如 `{"购买方":{"名称":"A公司"}}`
"OCR:{\"xxx\":{}}"

# 特定值为列表，如 `{"货物或应税劳务、服务名称":[{"名称":"A产品"},{"名称":"B产品"}]}`
"OCR:{\"xxx\":[]}"
```

具体如何构建数据集，可以参考后续的附录部分。

## 模型微调

微调的过程与 此 类似，首先安装 ERNIE：

```bash
cd paddleocr_vl
git clone https://gitee.com/PaddlePaddle/ERNIE -b release/v1.4
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e .
python -m pip install tensorboard
python -m pip install opencv-python-headless
python -m pip install numpy==1.26.4
```

然后，修改配置文件并复制覆盖原有配置文件：

```bash
cp paddleocr_vl/sft_config/run_ocr_vl_sft_16k.yaml \
  paddleocr_vl/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

下载 PaddleOCR-VL 模型，这里使用 modelscope 的 SDK：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('PaddlePaddle/PaddleOCR-VL', local_dir='paddleocr_vl/paddleocr_vl_model')
```

最后，就是执行微调命令即可，在 AI Studio 的 A100 环境中进行微调，大约需要不到 1.5 小时。

> V100 环境无法执行微调，但是可以进行模型推理

```bash
cd paddleocr_vl/ERNIE; CUDA_VISIBLE_DEVICES=0 \
 erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

以下是训练的日志：

![logs](images/logs.png)

可以看到，`loss` 在稳定的下降，说明微调应该有效果。


## 模型推理

微调完成后，可以使用微调后的模型进行推理。模型可以：

1. 输出 `JSON` 格式的完整信息
2. 根据不同的输入字段，输出对应的 `JSON` 格式的信息

这为信息抽取任务提供了灵活的接口。

按照 [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 进行推理，首先需要安装必要的环境

```bash
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
python -m pip install --force-reinstall opencv-python-headless
python -m pip install numpy==1.26.4
```

此时，还不能直接进行模型的推理，因为，PaddleOCR 依赖的 PaddleX 中，目前对于 PaddleOCR-VL 仅支持 `['ocr', 'formula', 'table', 'chart']` 这四类 `prompt_label`，而我们的 `prompt` 显然无法通过代码的验证：

参考 `paddlex/inference/pipelines/paddleocr_vl/pipeline.py` 文件

``` python
assert prompt_label.lower() in [
    "ocr",
    "formula",
    "table",
    "chart",
], f"Layout detection is disabled (use_layout_detection=False). 'prompt_label' must be one of ['ocr', 'formula', 'table', 'chart'], but got '{prompt_label}'."

```

这里写了一个 patch 脚本，可以绕过以上限制：

```bash
python paddleocr_vl/patch/patch_assert_to_warning.py
```

然后，将以下文件拷贝到 PaddleOCR-VL-SFT 目录下，就可以愉快的进行推理验证了。

```bash
cp paddleocr_vl/paddleocr_vl_model/chat_template.jinja paddleocr_vl/PaddleOCR-VL-SFT
cp paddleocr_vl/paddleocr_vl_model/inference.yml paddleocr_vl/PaddleOCR-VL-SFT
```

这里使用一张新的发票数据来进行模型的验证。

```bash
python -m paddleocr doc_parser -i paddleocr_vl/data/test.jpg \
    --vl_rec_model_name "PaddleOCR-VL-0.9B" \
    --vl_rec_model_dir "paddleocr_vl/PaddleOCR-VL-SFT" \
    --save_path="paddleocr_vl/PaddleOCR-VL-SFT_response" \
    --use_layout_detection=False \
    --prompt_label="OCR:{}"
```

输出完整的信息：

```json
{
    "res": {
        "input_path": "/home/aistudio/paddleocr_vl/data/test.jpg",
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "parsing_res_list": [{
            "block_label": "OCR:{}",
            "block_content": "{
              "发票信息": {
                  "发票名称": "电子发票",
                  "发票号码": "25332000000426443187",
                  "开票日期": "2025年09月26日"
              },
              "销售方信息": {
                  "名称": "杭州万力酒店管理有限公司",
                  "统一社会信用代码": "91330105MA2H2DUJ92",
                  "纳税人识别号": "91330106MA2B1C4UXN"
              },
              "项目名称": "规格型号",
              "单位": "个",
              "数量": "3 461.056105610561",
              "单价": "1383.17",
              "金额": "税率/征收率",
              "税额": "13.83"
          }, "合计": {
              "金额": "1383.17",
              "税额": "13.83"
          }, "价税合计（大写）": "壹仟叁佰玖拾柒圆整", "价税合计（小写）": "1397.00"
          }, "销售方地址": "浙江省杭州市西湖区转塘街道霞鸣街199号万美商务中心3号楼", "电话": "0571-85220222", "销方开户银行": "农行上泗支行", "入住人": "柳顺", "入住日期": "9月23日 入住-9月26日 退房", "入住天数": "3天", "金额": "1397元"
          }",
            "block_bbox": [0, 0, 1260, 838]
        }]
    }
}
```

注意两点：

- `use_layout_detection=False`，不通过 layout 模型，而是直接将图片送入 `PaddleOCR-VL-0.9B`
- `prompt_label="OCR:{}"`，这里使用我们微调的 `prompt` ，希望模型输出完整的 json 格式的信息

> 注意，这里模型最终输出的数据实际上不完整，比如，缺少 `购买方` 信息，应该是微调数据较少导致的。

再来看看微调之前的模型，只能输出 table 样式的数据:

```bash
python -m paddleocr doc_parser -i /home/aistudio/paddleocr_vl/data/test.jpg \
    --vl_rec_model_name "PaddleOCR-VL-0.9B" \
    --vl_rec_model_dir "/home/aistudio/paddleocr_vl/paddleocr_vl_model" \
    --save_path="/home/aistudio/paddleocr_vl/paddleocr_vl_model_response" \
    --use_layout_detection=False \
    --prompt_label="ocr"
```

输出：

```json
{
    "res": {
        "input_path": "/home/aistudio/paddleocr_vl/data/test.jpg",
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "parsing_res_list": [{
            "block_label": "ocr",
            "block_content": "购买方信息 | 名称 | 中青旅联科 | 杭州 | 公关顾问有限公司 | 销售方信息 | 名称 | 杭州万力酒店管理有限公司 | 统一社会信用代码/纳税人识别号 | 纳税人识别号 | 统一社会信用代码/纳税人识别号 | 税额 | 税额/征收率 | 税额/征收率\n**项目名称** | 规格型号 |   |   |   |   |   |   |   |   |   |   |   |  \n**住宿服务** | 住宿费 |   |   |   |   |   |   |   |   |   |   |   |  \n**合计** |   |   |   |   |   |   |   |   |   |   |   |   |  \n**价税合计（大写）** |   | 壹仟叁佰玖拾柒圆整 |   |   |   |   |   |   |   |   |   |   |  \n备注 | 销售方地址：浙江省杭州市西湖区转塘街道霞鸣街199号万美商务中心3号楼；电话：0571-85220222；销方开户银行：农行上泗支行；入住人：柳顺；入住日期：9月23日入住-9月26日退房；入住天数：3天；金额：1397元 |   |   |   |   |   |   |   |   |   |   |   |   |  \n开票人：祝营营",
            "block_bbox": [0, 0, 1260, 838]
        }]
    }
}
```

然后，测试一下只抽取部分信息：

```bash
python -m paddleocr doc_parser -i /home/aistudio/paddleocr_vl/data/test.jpg \
    --vl_rec_model_name "PaddleOCR-VL-0.9B" \
    --vl_rec_model_dir "/home/aistudio/paddleocr_vl/PaddleOCR-VL-SFT" \
    --save_path="/home/aistudio/paddleocr_vl/PaddleOCR-VL-SFT_response" \
    --use_layout_detection=False \
    --prompt_label="OCR:{\"购买方名称\": {}, \"销售方名称\": {}}"
```

输出：

```json
{
    "res": {
        "input_path": "/home/aistudio/paddleocr_vl/data/test.jpg",
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "parsing_res_list": [{
            "block_label": "OCR:{"购买方名称": {}, "销售方名称": {}}",
            "block_content": "{
                "购买方名称": {
                    "名称": "中青旅联科（杭州）公关顾问有限公司",
                    "统一社会信用代码": "91330105MA2H2DUJ92"
                },
                "销售方名称": {
                    "名称": "杭州万力酒店管理有限公司",
                    "统一社会信用代码": "91330106MA2B1C4UXN"
                }
            }",
            "block_bbox": [0, 0, 1260, 838]
        }]
    }
}
```

可以看到，模型基本上可以跟随我们的指令抽取对应的信息。

## 使用 transformers 库进行信息抽取

可以使用 transformers 库进行信息抽取，参考 [[Model] Add PaddleOCR-VL Model Support by zhang-prog](https://github.com/huggingface/transformers/pull/42178)

> 注意，目前微调后生成的模型目录还没有同步更新，在使用 transformers 库进行信息抽取时，需要先下载 [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main) 中最新的模型，然后，将微调后的模型文件 `model-00001-of-00001.safetensors` 重命名为 `model.safetensors`，并放到（并覆盖）下载的模型目录下。

```python
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text", 
    model="./PaddleOCR_VL_SFT/PaddleOCR-VL", # 下载的模型目录
    dtype="bfloat16")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://ai-studio-static-online.cdn.bcebos.com/dc31c334d4664ca4955aa47d8e202a53a276fd0aab0840b09abe953fe51207d0"},
            {"type": "text", "text": "OCR:{}"},
        ]
    }
]
result = pipe(text=messages)
print(result)

```

如果显存不足，可以尝试以下量化方法：

```python
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

path = "./PaddleOCR_VL_SFT/PaddleOCR-VL", # 下载的模型目录
processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)

# 4-bit 量化配置，大幅减少显存占用
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    quantization_config=quantization_config,
    # device_map="auto",
    local_files_only=True
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://ai-studio-static-online.cdn.bcebos.com/dc31c334d4664ca4955aa47d8e202a53a276fd0aab0840b09abe953fe51207d0"},
            {"type": "text", "text": "OCR:{\"发票日期\": \"\"}"},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
print(result)

```

## 使用 PaddleOCR-VL-REC 进行信息抽取

可以使用 [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC) 进行信息抽取：

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# 初始化识别器
recognizer = PaddleOCRVLRec(
    model_dir="path/to/your/model"
)

# 使用 dict 作为 query（会被转化为 JSON 字符串）
# 返回 JSON 格式（使用 json_repair 解析结果）
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query={"NAME":"", "ITEMS":[]},
    return_json=True
)
# result_json 是一个字典对象
print(type(result_json))  # <class 'dict'>
print(result_json)

# 使用 list 作为 query（会被转化为 {"item1":"", "item2":""} 的形式）
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query=["item1", "item2"],
    return_json=True
)
print(result_json)

recognizer.close()

```

## 总结

本文介绍了如何通过微调 PaddleOCR-VL 的提示词（prompt）来实现信息抽取任务。主要方法包括：

1. **数据准备**：使用 VLM 模型生成结构化的训练数据，相比于传统标注方式更加高效。
2. **提示词设计**：通过精心设计的提示词模板，让模型能够灵活地输出不同字段的 `JSON` 格式信息。
3. **模型微调**：利用 PaddleOCR-VL 的微调能力，使其学会根据不同的提示词生成对应的输出。

这种方法相比于传统的信息抽取方法（如 NER + 关系抽取），具有更好的集成度和灵活性。

## 附录

### 1. 数据集

信息抽取的应用场景有很多，这里以 [增值税普通发票](https://aistudio.baidu.com/datasetdetail/125158) 数据为例。

> 可以参考 [基于VI-LayoutXLM的发票关键信息抽取](https://bbs.huaweicloud.com/blogs/383854) 这篇文章，对于微调 PaddleOCR 模型进行信息抽取做了比较完整的讲解。

但是，数据集对于 `关系抽取（Relation Extraction）` 的标注还是比较简陋的，比如:

![增值税普通发票](images/re.jpg)

这里只标注了 `名称`，而没有标注说明是 `购买方名称` 还是 `销售方名称`。

前面提到，我们可以把 PaddleOCR-VL 当作 VLM 模型来使用，那么，我们可以让能力更强的 VLM 模型来 `教` PaddleOCR-VL 去识别 `购买方名称` 和 `销售方名称`。

数据可以通过 `ernie-4.5-turbo-vl-preview` 模型来生成，参考脚本 `paddleocr_vl/tools/extract_ner/extract_ner.py`。

``` python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态图像识别脚本
通过调用OpenAI接口识别图片信息并返回JSON格式数据
支持本地图片和多模态大模型处理
"""
...

class MultimodalImageRecognizer:
    """多模态图像识别器"""
    ...

    def recognize_image(
        self,
        image_input: Union[str, bytes],
        prompt: str,
        system_prompt: str,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        识别图片信息

        Args:
            image_input: 图片路径、URL或base64编码
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_tokens: 最大令牌数

        Returns:
            识别结果的JSON格式数据
        """
        try:
            # 创建多模态消息
            content = self.create_multimodal_message(prompt, image_input)

            # 构建消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]

            logger.info(f"开始调用API识别图片，模型: {self.model}")

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )

    ...

    def analyze_image(
        self,
        image_input: Union[str, bytes],
        analysis_type: str = "document"
    ) -> Dict[str, Any]:
        """
        分析图片内容（简化版本）

        Args:
            image_input: 图片路径、URL或base64编码
            analysis_type: 分析类型，固定为 "document"

        Returns:
            分析结果的JSON格式数据
        """
        # 统一使用文档分析提示词
        prompt = "请分析这张文档图片中的所有信息，并返回完整的JSON格式数据。如果有的字段没有值，那么保留此字段，值为空。注意：所有的值都以string的形式返回，不要使用数字类型等。"
        system_prompt = '''
你是一个专业的文档分析助手，能够准确分析文档内容并返回结构化的JSON数据。

注意：数据的语言与文档的语言保持一致。
注意：需要保留完整的字段层级关系，不要把所有字段都放到一级字段中。
注意：JSON数据中不要包含注释，也不需要任何解释或说明。
注意：对于特殊字符需要进行转义。

注意：对于选项字段，只保留所选择的字段值，如果没有选择，则置为空。
比如，`业务类型` 包括 `账户开户、账户登记` 等选项，文档中`账户登记`是选中状态，则，返回 `{"业务类型"："账户登记"}`，不返回`账户开户`等其他选项。
再比如，`业务类型` 包括 `账户开户、账户登记` 等选项，文档中没有标记选中的选项，则，返回 `{"业务类型"：""}`，也就是说，只保留键，不需要有值。
...
'''

        return self.recognize_image(
            image_input=image_input,
            prompt=prompt,
            system_prompt=system_prompt
        )

...
```

使用 `paddleocr_vl/tools/extract_ner/batch_extract_ner.py` 脚本可以批量生成数据，最终生成的数据参考如下：

``` json

{
  "image": "/media/shun/bigdata/Dataset/增值税普通发票/zzsptfp/b0.jpg",
  "data": {
    "发票名称": "广东增值税专用发票",
    "发票编号": "12271524",
    "发票代码": "4400154130",
    "开票日期": "2016年06月12日",
    "购买方": {
      "名称": "深圳市购机汇网络有限公司",
      "纳税人识别号": "440300083885931",
      "地址、电话": "深圳市龙华新区民治街道民治大道展滔科技大厦A12070755-23806606",
      "开户行及账号": "中国工商银行股份有限公司深圳园岭支行4000024709200172809"
    },
    "密码区": "<<1<//3*26-++936-9<9*575>39 -<5//81>84974<00+7>2*0*53-+ +125*++9+-///5-7+/-0>8<9815 5<3/8*+//81/84+>6>4*36>4538",
    "货物或应税劳务、服务名称": [
      {
        "名称": "小米 红米3 全网通版 时尚金色",
        "规格型号": "红米3",
        "单位": "个",
        "数量": "5",
        "单价": "597.43589744",
        "金额": "2987.18",
        "税率": "17%",
        "税额": "507.82"
      },
      {
        "名称": "移动联通电信4G手机 双卡双待",
        "规格型号": "",
        "单位": "",
        "数量": "",
        "单价": "",
        "金额": "",
        "税率": "",
        "税额": ""
      }
    ],
    "合计": {
      "金额": "￥2987.18",
      "税额": "￥507.82"
    },
    "价税合计（大写）": "叁仟肆佰玖拾伍圆整",
    "价税合计（小写）": "￥3495.00",
    "销售方": {
      "名称": "广州晶东贸易有限公司",
      "纳税人识别号": "91440101664041243T",
      "地址、电话": "广州市黄埔区九龙镇九龙工业园凤凰三横路99号 66215500",
      "开户行及账号": "工行北京路支行3602000919200384952"
    },
    "备注": "dd42982413947(00001,1952)7996有限",
    "收款人": "王梅",
    "复核": "张雪",
    "开票人": "陈秋燕",
    "销售方（章）": "广州晶东贸易有限公司 发票专用章"
  }
}

```

这里生成的数据信息比原有的标注信息丰富很多，虽然有一些瑕疵 (比如 `货物或应税劳务、服务名称` 中应该只有一条记录)，但是不妨碍进行微调实验的进行。

> 处理后的数据已经上传至 [增值税普通发票与JSON格式信息](https://aistudio.baidu.com/dataset/detail/363136/intro)。

### 2. 提示词

这里的 `信息抽取` 任务，目标是：

- 模型可以输出 `JSON` 格式的完整信息
- 模型可以根据不同的输入字段，输出对应的 `JSON` 格式的信息

针对以上目标，这里设计了对应的提示词：

**完整信息**

```
"OCR:{}"
```

**特定信息**

```
# 特定值为字符串，如 `{"发票编码":"123456"}`
"OCR:{\"xxx\":\"\"}"

# 特定值为字典，如 `{"购买方":{"名称":"A公司"}}`
"OCR:{\"xxx\":{}}"

# 特定值为列表，如 `{"货物或应税劳务、服务名称":[{"名称":"A产品"},{"名称":"B产品"}]}`
"OCR:{\"xxx\":[]}"
```

可以使用 `paddleocr_vl/tools/process_ner_dataset.py` 生成完整的训练数据，包括随机生成的提示词：

```bash
python paddleocr_vl/tools/process_ner_dataset.py paddleocr_vl/data/zzsptfp \
  -o paddleocr_vl/output.jsonl \
  -n 10 \
  -p /media/shun/bigdata/Dataset/增值税普通发票 \
  -u /home/aistudio/paddleocr_vl/data/zzsptfp
```

之后，拆分训练数据集与验证数据集：

```bash
python paddleocr_vl/tools/split_jsonl.py paddleocr_vl/output.jsonl \
  paddleocr_vl/output \
  --train_ratio 0.9 \
  --seed 123
```

最终生成的数据参考如下：

```json
{
    "image_info": [
        {
            "matched_text_index": 0,
            "image_url": "/home/aistudio/paddleocr_vl/data/zzsptfp/zzsptfp/b175.jpg"
        }
    ],
    "text_info": [
        {
            "text": "OCR:{\"发票名称\": \"\"}",
            "tag": "mask"
        },
        {
            "text": "{\"发票名称\": \"广东增值税专用发票\"}",
            "tag": "no_mask"
        }
    ]
}
```

生成的训练数据与 [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 不同处有：

- `mask` 的 `text` 不仅仅是 `OCR:` ，还包括之后需要抽取的字段信息
- `no_mask` 的 `text` 是完整的 `JSON` 格式信息，而不是一段纯文本

### 3. 配置文件示例

```yaml
### data
train_dataset_type: "erniekit"
eval_dataset_type: "erniekit"
train_dataset_path: "/home/aistudio/paddleocr_vl/output_train.jsonl"
train_dataset_prob: "1.0"
eval_dataset_path: "/home/aistudio/paddleocr_vl/output_val.jsonl"
eval_dataset_prob: "1.0"
max_seq_len: 16384
num_samples_each_epoch: 6000000
use_pic_id: False
sft_replace_ids: True
sft_image_normalize: True
sft_image_rescale: True
image_dtype: "float32"

### model
model_name_or_path: "/home/aistudio/paddleocr_vl/paddleocr_vl_model"
fine_tuning: Full
multimodal: True
use_flash_attention: True
use_sparse_flash_attn: True

### finetuning
# base
stage: OCR-VL-SFT
seed: 23
do_train: True
# do_eval: True
distributed_dataloader: False
dataloader_num_workers: 8
prefetch_factor: 10
batch_size: 1
packing_size: 8
packing: True
padding: False
num_train_epochs: 2
max_steps: 80
# eval_batch_size: 1
# eval_iters: 50
# eval_steps: 100
# evaluation_strategy: steps
save_steps: 20
save_total_limit: 5
save_strategy: steps
logging_steps: 1
release_grads: True
gradient_accumulation_steps: 8
logging_dir: /home/aistudio/paddleocr_vl/PaddleOCR-VL-SFT/tensorboard_logs/
output_dir: /home/aistudio/paddleocr_vl/PaddleOCR-VL-SFT
disable_tqdm: True

# train
warmup_steps: 1
learning_rate: 5.0e-6
lr_scheduler_type: cosine
min_lr: 5.0e-7
layerwise_lr_decay_bound: 1.0
from_scratch: 0

# optimizer
weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

# performance
tensor_parallel_degree: 1
pipeline_parallel_degree: 1
sharding_parallel_degree: 1
sharding: stage1
sequence_parallel: False
pipeline_parallel_config: enable_delay_scale_loss enable_release_grads disable_partial_send_recv
recompute: True
recompute_granularity: "full"
recompute_use_reentrant: True
compute_type: bf16
fp16_opt_level: O2
disable_ckpt_quant: True
# amp_master_grad: True
amp_custom_white_list:
  - lookup_table
  - lookup_table_v2
  - flash_attn
  - matmul
  - matmul_v2
  - fused_gemm_epilogue
amp_custom_black_list:
  - reduce_sum
  - softmax_with_cross_entropy
  - c_softmax_with_cross_entropy
  - elementwise_div
  - sin
  - cos
unified_checkpoint: True
# unified_checkpoint_config: async_save
convert_from_hf: True
save_to_hf: True
```
