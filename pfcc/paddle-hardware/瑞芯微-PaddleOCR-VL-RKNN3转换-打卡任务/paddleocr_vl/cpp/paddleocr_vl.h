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


#ifndef _RKNN_DEMO_PADDLEOCR_VL_H_
#define _RKNN_DEMO_PADDLEOCR_VL_H_

#include "rknn3_api.h"
#include "common.h"
#include "rknn_paddleocr_vl_vision.h"
#include "rknn_paddleocr_vl_llm.h"
#include "rknn_paddleocr_vl_mlpar.h"
#include "time_utils.h"

// llm
extern const rknn3_sampling_params SAMPLE_PARAMS;

typedef struct {
    rknn_paddleocr_vl_llm_context llm;
    rknn_paddleocr_vl_vision_context vision;
    rknn_paddleocr_vl_mlpar_context mlpar;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
    uint32_t model_width;
    uint32_t model_height;
} rknn_app_context_t;

int init_paddleocr_vl_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, 
                        const char* vision_model_path, const char* vision_weight_path, const char* position_embedding_path, const char* mlpar_model_path, const char* mlpar_weight_path,
                        rknn3_llm_param* params, int n_params, RKLLMCallback callback, uint32_t vision_core_mask, uint32_t mlpar_core_mask, uint32_t llm_core_mask);

int release_paddleocr_vl_model(rknn_app_context_t* app_ctx);

int inference_paddleocr_vl_model(rknn_app_context_t* app_ctx, image_buffer_t* img, float16* vision_embeds, float16* img_embeds, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf);

#endif //_RKNN_DEMO_PADDLEOCR_VL_H_