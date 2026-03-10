// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef _RKNN_DEMO_PADDLEOCR_VL_VISION_UTILS_H_
#define _RKNN_DEMO_PADDLEOCR_VL_VISION_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

#define MODEL_WIDTH 504 // 由于vision模型被裁剪了，需要手动添加这些参数配置
#define MODEL_HEIGHT 504
#define PATCH_SIZE 14
#define POSITION_EMBED_DIM 1152
#define POSITION_EMBED_GRID 27
#define ROPE_DIM 36

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_height;
    int model_width;
    int patch_num;

    float* patch_pos_embed;
    float* inv_freq;

    uint32_t* embeds_shape;
    uint32_t embeds_ndims;
    int pruned_version_flag;
} rknn_paddleocr_vl_vision_context;

typedef struct {
    float* virt_addr;
    int size;
} embed_t;

int init_paddleocr_vl_vision(rknn_paddleocr_vl_vision_context* vision_ctx, const char* model_path, const char* weight_path, const char* position_embedding_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height);

int release_paddleocr_vl_vision(rknn_paddleocr_vl_vision_context* vision_ctx);

int inference_paddleocr_vl_vision(rknn_paddleocr_vl_vision_context* vision_ctx, image_buffer_t* img, float16* vision_embeds, int* gridh, int* gridw);

#endif //_RKNN_DEMO_PADDLEOCR_VL_VISION_UTILS_H_