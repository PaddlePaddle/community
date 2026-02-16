# paddle.audio.functional.resample 设计文档

| API 名称 | `paddle.audio.functional.resample` |
|---|---|
| 提交作者 | PlumBlossomMaid |
| 提交时间 | 2026-02-05 |
| 依赖飞桨版本 | v3.3.0 |
| 文件名 | 20260205_api_design_for_resample.md |

---

## 一、概述

### 1、相关背景
为了丰富飞桨在音频处理领域的 API 支持，提升科学计算与信号处理的能力，Paddle 需要新增音频重采样 API `paddle.audio.functional.resample`。

### 2、功能目标
实现基于 sinc 插值（支持 Hann 窗与 Kaiser 窗）的高质量音频重采样功能，支持从原始采样率转换到目标采样率。
支持硬件加速（如GPU或Custom Device），加速框架音频重采样的速度。

### 3、意义
为 Paddle 音频处理提供标准的、高效的重采样能力，支持语音识别、音频增强、信号分析等场景。

---

## 二、飞桨现状
目前 Paddle 暂无内置的音频重采样 API。相关信号处理功能（如 STFT、频谱图）已在 `paddle.signal`或`paddle.audio` 模块中提供，但缺乏直接的重采样实现。

---

## 三、业内方案调研

### PyTorch Audio
PyTorch 提供了 `torchaudio.functional.resample`，支持基于 sinc 插值的重采样，可选 Hann 窗或 Kaiser 窗。其实现方式为：
- 计算重采样卷积核（sinc 核加窗）
- 通过卷积实现插值
- 支持整数采样率比例，保证计算精度

### Librosa
Librosa 提供 `librosa.resample`，基于多相滤波实现，支持任意浮点采样率比例。

### 对比分析
- PyTorch Audio 的重采样质量较高，适合语音和音乐处理；
- Librosa 更灵活，但计算复杂度较高；
- 本设计参考 PyTorch Audio 实现，采用 sinc 插值方法，支持高质量重采样。

---

## 四、方案设计

### 命名与参数设计
API 设计为：

```python
paddle.audio.functional.resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    name: Optional[str] = None
) -> Tensor
```

参数说明：
- `waveform`：输入音频信号，形状为 `(..., time)`
- `orig_freq`：原始采样率（Hz），必须为正整数
- `new_freq`：目标采样率（Hz），必须为正整数
- `lowpass_filter_width`：低通滤波器宽度，控制滤波器锐度
- `rolloff`：截止频率相对于 Nyquist 频率的比例
- `resampling_method`：重采样方法，支持 `"sinc_interp_hann"` 和 `"sinc_interp_kaiser"`
- `beta`：Kaiser 窗的形状参数，仅在 `resampling_method="sinc_interp_kaiser"` 时有效
- `name`：可选名称

### 底层 OP 设计
使用现有 API 组合实现，不单独设计 OP。核心依赖：
- `paddle.nn.functional.conv1d`
- `paddle.arange`、`paddle.clip`
- `paddle.where`、`paddle.sin`、`paddle.cos`
- `paddle.i0`（第一类修正 Bessel 函数）

### API 实现方案
实现步骤：
1. 计算原始采样率与目标采样率的最大公约数，进行约分
2. 根据所选窗口函数（Hann 或 Kaiser）生成 sinc 插值核
3. 对输入信号进行边缘填充
4. 使用卷积实现插值
5. 裁剪输出长度，返回重采样后的信号

实现位置：
```
paddle/audio/functional.py
```

---

## 五、测试和验收的考量

测试用例包括：
- 数值准确性：与 PyTorch Audio 结果一致
- 采样率转换：整数倍上采样/下采样
- 数据类型：支持 `float32`、`float64`
- 设备支持：CPU/GPU
- 运行模式：动态图/静态图
- 边界情况：输入为单通道/多通道/批量数据
- 错误检查：无效采样率、非浮点输入、无效方法参数等
- 在paddle仓库中，`Paddle/test/legacy_test/test_audio_functions.py`是对音频相关API进行测试的地方，所以应该在此处增加测试用例。

---

## 六、可行性分析及规划排期

实现依赖的 Paddle API 均已稳定支持，无外部依赖。预计可在当前版本月内完成开发与测试。

---

## 七、影响面
为独立新增 API，不影响现有模块功能。

---

## 八、附件及参考资料
- PyTorch Audio `resample` 实现
- 数字信号处理相关文献（如《Discrete-Time Signal Processing》）
- Paddle API 设计规范


