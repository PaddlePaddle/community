# PaddleOCR-VL Model Deployment Guide

## 1. Environment Requirements

PaddleOCR-VL has specific dependency requirements that are incompatible with the `requirements.txt` in this repository. Please install the following dependencies manually:

```
transformers == 4.55.0
```

> ⚠️ Using the default dependencies will cause model conversion to fail. Please install the versions specified above.

## 2. Model Pruning Strategy

To support larger context lengths, appropriate pruning is required when deploying large-scale multimodal models.

### 2.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 2.2 MLP-AR Model Partitioning

Due to dynamic shape transformations involved in the MLP-AR layer, it is partitioned into a separate subgraph. Shape manipulation operations are executed on the CPU.

### 2.3 LLM Model Pruning

The LLM Head is separated and runs independently on the host device, reducing memory usage on the coprocessor. (Optional)

### 2.4 Full Model Mode (No Pruning)

Devices with larger memory (e.g., RK1828) can use the full model directly. Add the `--no_prune_mode` parameter when exporting:

```bash
python export_rknn.py --no_prune_mode
```

## 3. Model Export

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

## 4. KV Cache INT4 Quantization

In large-scale language model inference, a KV (Key/Value Cache) is used to store historical attention keys to avoid redundant calculations and thus improve inference speed. As sequence length increases, the memory footprint of the KV Cache increases rapidly. To reduce the storage bandwidth and memory access overhead of the KV Cache, quantization can be used to convert it from FP16/FP32 to INT8 or a lower bit width representation. However, since the KV Cache value distribution changes dynamically token by token over time, using a uniform quantization parameter for the entire KV segment can lead to accumulated quantization errors, affecting inference accuracy. Therefore, group quantization is typically used to reduce accuracy loss.

Currently, RKNN's LLM supports two KV cache quantization modes:

Int8_to_F16 (default): Stores in INT8 format, converts back to FP16 during computation;

Int4_to_F16 (suitable for longer contexts, with some precision loss): Stores in INT4 format, converts back to FP16 during computation.

For support of longer context lengths and further compression of KV cache memory, it is recommended to enable the Int4_to_F16 mode.

The configuration for enabling Int4_to_F16 RKNN model transformation is as follows:

```python
rknn.config(target_platform='rk1820', 
          quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
          max_ctx_len           =2048,
          max_position_embeddings=2048,
          kvcache_store_method='GroupQuant', kvcache_dtype='Int4_to_F16', 
          kvcache_group_size=16, kvcache_residual_depth=64,
          )
```
- Note: The above configuration is located in the python/llm/export_rknn.py file. Please adjust the relevant parameters according to your actual needs.

## 5. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 28):

```bash
python export_vision.py --img_h 504 --img_w 504
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur

## 6. C++ Deployment

The C++ inference code automatically detects the model format and is compatible with both pruned and full models without code modification.

If you modified the Vision model resolution, update the parameters in `rknn_paddleocr_vl_vision.h`:

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```
