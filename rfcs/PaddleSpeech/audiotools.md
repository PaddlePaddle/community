【Hackathon 7th No 55】在 PaddleSpeech 中实现 DAC 的训练中使用的第三方库 audiotools

# audiotools ——设计文档

| 任务名  | 在 PaddleSpeech 中实现 DAC 的训练中使用的第三方库 audiotools |
| ---- | ----------------- |
| 提交作者 | DrRyanhuang |
| 提交时间 | 2024-11-17 |
| 版本号  | v1.0 |
| 依赖   ||
| 文件名  | audiotools.md|

# 一、概述

## 1、相关背景

audiotools 是一个基于 Torch 的面向对象的音频信号处理库，具有快速增强和批处理和等性能优势。

## 2、功能目标

在 PaddleSpeech 套件中实现并对齐 Descript-Audio-Codec 中使用到的第三方库 audiotools 的接口，并完成相应单测
用 paddle 完成全部 DAC 中的内容，非单测中不出现 PyTorch

## 3、意义

利于基于 PadddleSpeech 实现 Descript-Audio-Codec 训练

# 二、飞桨现状

PadddleSpeech 暂无关于 audiotools 的相关实现

# 三、业内方案调研

无

# 四、对比分析

无

# 五、设计思路与实现方案

依次实现以下文件用到的API，并完成相应单测, 与原 audiotools 对齐

```python
dac/__init__.py:
  5  
  6: import audiotools
  7  
  8: audiotools.ml.BaseModel.INTERN += ["dac.**"]
  9: audiotools.ml.BaseModel.EXTERN += ["einops"]
  10  

dac/compare/encodec.py:
  1  import torch
  2: from audiotools import AudioSignal
  3: from audiotools.ml import BaseModel
  4  from encodec import EncodecModel

dac/model/base.py:
  8  import tqdm
  9: from audiotools import AudioSignal
  10  from torch import nn

dac/model/dac.py:
  6  import torch
  7: from audiotools import AudioSignal
  8: from audiotools.ml import BaseModel
  9  from torch import nn

dac/model/discriminator.py:
  3  import torch.nn.functional as F
  4: from audiotools import AudioSignal
  5: from audiotools import ml
  6: from audiotools import STFTParams
  7  from einops import rearrange

dac/nn/loss.py:
    5  import torch.nn.functional as F
    6: from audiotools import AudioSignal
    7: from audiotools import STFTParams
    8  from torch import nn

dac/utils/__init__.py:
   3  import argbind
   4: from audiotools import ml
   5  

dac/utils/decode.py:
  6  import torch
  7: from audiotools import AudioSignal
  8  from tqdm import tqdm

dac/utils/encode.py:
  7  import torch
  8: from audiotools import AudioSignal
  9: from audiotools.core import util
  10  from tqdm import tqdm

scripts/compute_entropy.py:
  1  import argbind
  2: import audiotools as at
  3  import numpy as np

scripts/evaluate.py:
   8  import torch
   9: from audiotools import AudioSignal
  10: from audiotools import metrics
  11: from audiotools.core import util
  12: from audiotools.ml.decorators import Tracker
  13  from train import losses

scripts/get_samples.py:
  4  import torch
  5: from audiotools import AudioSignal
  6: from audiotools.core import util
  7: from audiotools.ml.decorators import Tracker
  8  from train import Accelerator

scripts/mushra.py:
  7  import gradio as gr
  8: from audiotools import preference as pr
  9  

scripts/organize_daps.py:
   9  import tqdm
  10: from audiotools import util
  11  

scripts/save_test_set.py:
  5  import torch
  6: from audiotools.core import util
  7: from audiotools.ml.decorators import Tracker
  8  from train import Accelerator

scripts/train_no_adv.py:
   9  import torch
  10: from audiotools import AudioSignal
  11: from audiotools import ml
  12: from audiotools.core import util
  13: from audiotools.data import transforms
  14: from audiotools.data.datasets import AudioDataset
  15: from audiotools.data.datasets import AudioLoader
  16: from audiotools.data.datasets import ConcatDataset
  17: from audiotools.ml.decorators import timer
  18: from audiotools.ml.decorators import Tracker
  19: from audiotools.ml.decorators import when
  20  from torch.utils.tensorboard import SummaryWriter

scripts/train.py:
   8  import torch
   9: from audiotools import AudioSignal
  10: from audiotools import ml
  11: from audiotools.core import util
  12: from audiotools.data import transforms
  13: from audiotools.data.datasets import AudioDataset
  14: from audiotools.data.datasets import AudioLoader
  15: from audiotools.data.datasets import ConcatDataset
  16: from audiotools.ml.decorators import timer
  17: from audiotools.ml.decorators import Tracker
  18: from audiotools.ml.decorators import when
  19  from torch.utils.tensorboard import SummaryWriter

tests/test_cli.py:
  10  import torch
  11: from audiotools import AudioSignal
  12  

tests/test_train.py:
  10  import numpy as np
  11: from audiotools import AudioSignal
```

除了 AudioSignal 需要单独实现外，audiotools.ml.decorators 目录下的这几个部分也要实现

```
ml.BaseModel
ml.Accelerator
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]
```

audiotools.data 目录下需要改写的内容
```
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
```

除单测外，其余小组件
```
from audiotools import metrics
from audiotools.core import util
from audiotools import preference as pr
```
# 六、测试和验收的考量

- 编写的单测与原 repo 保持一致

# 七、可行性分析和排期规划

预计能在活动期内完成。

# 八、影响面

- PaddleSpeech 实现相应的功能

# 名词解释

# 附件及参考资料
