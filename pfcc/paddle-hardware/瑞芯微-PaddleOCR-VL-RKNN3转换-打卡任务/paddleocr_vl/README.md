# PaddleOCR-VL 模型部署说明

## 1. 部署环境

由于 PaddleOCR-VL 模型对依赖库版本有特殊要求，与本仓库 `requirements.txt` 中的版本不兼容，请手动安装以下依赖：

```
transformers == 4.55.0
```

> ⚠️ 使用默认依赖会导致模型转换失败，请务必按上述版本安装。

## 2. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时可进行适当裁剪。

### 2.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 2.2 MLP-AR 模型拆分

由于 MLP-AR 层涉及动态形状变换，将其拆分为单独子图。形状变化操作在 CPU 上运行。

### 2.3 LLM 模型裁剪

将 LLM Head 独立出来，在主控设备上单独运行，从而减少协处理器的内存占用。（可选）

### 2.4 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型，导出时添加 `--no_prune_mode` 参数：

```bash
python export_rknn.py --no_prune_mode
```

## 3. 导出模型流程

```bash
# 导出 LLM ONNX 模型
python export_llm.py \
    --model_path PaddleOCR-VL/PaddleOCR-VL-0.9B \
    --export_llm_path ../../model/llm/PaddleOCR-llm.onnx

# 导出 LLM RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/PaddleOCR-llm.onnx \
    --config ../../model/llm/PaddleOCR-llm.config.pkl \
    --rknn_path ../../model/llm/PaddleOCR-llm.rknn

# 导出 Vision ONNX 模型 和 MLP-AR ONNX 模型
python export_vision.py \
    --model_path PaddleOCR-VL/PaddleOCR-VL-0.9B \
    --export_vision_path ../../model/vision/PaddleOCR-vision.onnx \
    --export_mlp_AR_path ../../model/vision/PaddleOCR-vision-mlp_AR.onnx

# 导出 Vision RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/PaddleOCR-vision.onnx \
    --rknn_path ../../model/vision/PaddleOCR-vision.rknn \
    --mlpar_onnx_path ../../model/vision/PaddleOCR-vision-mlp_AR.onnx \
    --mlpar_rknn_path ../../model/vision/PaddleOCR-vision-mlp_AR.rknn
```

## 4. KV Cache INT4 量化
在大规模语言模型推理过程中，KV Cache（Key/Value Cache）用于存储历史的注意力键值，以避免重
复计算，从而提高推理速度。随着序列长度增长，KV Cache 的内存占用会快速增加。为了减少 KV
Cache 的存储带宽与内存访问开销，可以采用量化方式将其从 FP16/FP32 转换为 INT8 或更低位宽表
示。但由于 KV Cache 数值分布随时间逐 token 动态变化，如果对整段 KV 使用统一的量化参数，会导致
量化误差累积从而影响推理精度。因此通常采用分组量化（Group Quantization）来降低精度损失。
目前，RKNN 的LLM支持两种 KV Cache 量化模式：
Int8_to_F16（默认）：以 INT8 格式存储，计算时转换回 FP16；
Int4_to_F16（适用于更长上下文场景,有一定精度损失）：以 INT4 格式存储，计算时转换回 FP16。
若需支持更长的上下文长度并进一步压缩 KV Cache 内存，建议启用 Int4_to_F16 模式。
启用 Int4_to_F16 的 RKNN 模型转换配置如下：
```python
rknn.config(target_platform='rk1820', 
          quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
          max_ctx_len           =2048,
          max_position_embeddings=2048,
          kvcache_store_method='GroupQuant', kvcache_dtype='Int4_to_F16', 
          kvcache_group_size=16, kvcache_residual_depth=64,
          )
```
- 注意：上述配置位于 python/llm/export_rknn.py 文件中，请根据实际需求调整相关参数。


## 5. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 28 的倍数）：

```bash
python export_vision.py --img_h 504 --img_w 504
```

> ⚠️ **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

## 6. C++ 部署说明

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。

若修改了 Vision 模型的分辨率，需同步调整 `rknn_paddleocr_vl_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```
