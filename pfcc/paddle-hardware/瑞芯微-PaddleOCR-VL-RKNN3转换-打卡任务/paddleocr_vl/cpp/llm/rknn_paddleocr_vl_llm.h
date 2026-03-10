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


#ifndef _RKNN_DEMO_PADDLEOCR_VL_LLM_UTILS_H_
#define _RKNN_DEMO_PADDLEOCR_VL_LLM_UTILS_H_

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"

extern const char* system_prompt;
extern const char* prompt_prefix;
extern const char* prompt_postfix;

#define MAX_NEW_TOKENS 1024
#define MAX_CONTEXT_LEN 1024

typedef struct {
    rknn3_context   rknn_ctx;
    rknn3_session* rknn_sess;

} rknn_paddleocr_vl_llm_context;

int init_paddleocr_vl_llm(rknn_paddleocr_vl_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback callback, uint32_t core_mask);

int release_paddleocr_vl_llm(rknn_paddleocr_vl_llm_context* llm_ctx);

int inference_paddleocr_vl_llm(rknn_paddleocr_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf);

#endif //_RKNN_DEMO_paddleocr_VL_LLM_UTILS_H_