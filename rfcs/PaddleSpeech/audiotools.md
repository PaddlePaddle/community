【Hackathon 7th No 55】在 PaddleSpeech 中实现 DAC 的训练中使用的第三方库 audiotools

# audiotools ——设计文档

| 任务名  | 在 PaddleSpeech 中实现 DAC 的训练中使用的第三方库 audiotools |
| ---- | ----------------- |
| 提交作者 | DrRyanhuang |
| 提交时间 | 2024-11-17 |
| 版本号  | v1.1 |
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

依次实现以下文件用到的API，并完成相应单测, 与原 audiotools 对齐, 各个API与原始 audiotools 的目录相同

```python

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

即除了 `AudioSignal` 需要单独实现外，audiotools.ml.decorators 目录下的这几个部分也要实现

```
ml.BaseModel
ml.Accelerator
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
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

目录结构在如下, 单测 `tests` 放置于 audio 下, 与 `audiotools` 处于同级目录
```
.
├── audiotools
│   ├── README.md --- 用于介绍 audiotools
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── _julius.py
│   │   ├── audio_signal.py
│   │   ├── dsp.py
│   │   ├── effects.py
│   │   ├── ffmpeg.py
│   │   ├── loudness.py
│   │   ├── resample.py
│   │   └── util.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── preprocess.py
│   │   └── transforms.py
│   ├── metrics
│   │   ├── __init__.py
│   │   └── quality.py
│   ├── ml
│   │   ├── __init__.py
│   │   ├── accelerator.py
│   │   ├── basemodel.py
│   │   └── decorators.py
│   ├── requirements.txt
│   └── post.py
├── tests
│   └── audiotools
│       ├── core
│       │   ├── test_audio_signal.py
│       │   ├── test_bands.py
│       │   ├── test_fftconv.py
│       │   ├── test_highpass.py
│       │   ├── test_loudness.py
│       │   ├── test_lowpass.py
│       │   └── test_util.py
│       ├── data
│       │   ├── test_datasets.py
│       │   ├── test_preprocess.py
│       │   └── test_transforms.py
│       ├── ml
│       │   ├── test_decorators.py
│       │   └── test_model.py
│       └── test_post.py
```

由于 audiotools 使用到了 julius, 而 julius 依赖于 torch, 所以也需要实现 julius 中的函数, 并写相关单测:
```
fft_conv1d
FFTConv1d
LowPassFilters
LowPassFilter
lowpass_filters
lowpass_filter
HighPassFilters
HighPassFilter
highpass_filters
highpass_filter
SplitBands
split_bands
```

相关单测放置在:
```
tests/audiotools/core/test_bands.py
tests/audiotools/core/test_fftconv.py
tests/audiotools/core/test_highpass.py
tests/audiotools/core/test_lowpass.py
```

在测试时会自动下载相应的音频文件，会放置在这两个位置：
```
.
├── audiotools
├── tests
│   └── audiotools
│       ├── audio
│       │   ├── * 放置测试所用到的 wav / mp3 音频文件
│       ├── regression
│       │   └── transforms
│       │       └── *.wav -- 放置测试 transforms 所用到的 wav / mp3 音频文件
```


# 六、测试和验收的考量

- 编写的单测与原 repo 保持一致, test 位置放到 audiotools 同级目录, 使用 pytest 全部通过即可

# 七、可行性分析和排期规划

预计能在活动期内完成。

# 八、影响面

- PaddleSpeech 实现相应的功能

# 名词解释

# 附件及参考资料
