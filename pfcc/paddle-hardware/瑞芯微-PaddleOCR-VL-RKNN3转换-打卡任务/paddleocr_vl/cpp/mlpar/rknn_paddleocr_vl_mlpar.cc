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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <float.h>

#include "rknn_paddleocr_vl_mlpar.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"


static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
      shape_str += std::to_string(attrs->shape[j]);
      if (j < attrs->n_dims - 1) {
        shape_str += ", ";
      }
    }
  
    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
      stride_str += std::to_string(attrs->stride[j]);
      if (j < attrs->n_stride - 1) {
        stride_str += ", ";
      }
    }
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

int init_paddleocr_vl_mlpar(rknn_paddleocr_vl_mlpar_context* mlpar_ctx, const char* model_path, const char* weight_path, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 0;

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    mlpar_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    mlpar_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    mlpar_ctx->rknn_ctx = ctx;
    mlpar_ctx->io_num = io_num;
    for (int i = 0; i < mlpar_ctx->io_num.n_input; i++) {
        mlpar_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        mlpar_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(mlpar_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < mlpar_ctx->io_num.n_output; i++) {
        mlpar_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        mlpar_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(mlpar_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED)
    {
        printf("model is UNDEFINED output layout\n");
        mlpar_ctx->embeds_shape = mlpar_ctx->outputs[0].attr->shape;
        mlpar_ctx->embeds_ndims = mlpar_ctx->outputs[0].attr->n_dims;
    }
    else
    {
        printf("model is not UNDEFINED output layout, model output error!\n");
        return -1;
    }

    return ret;
}

int release_paddleocr_vl_mlpar(rknn_paddleocr_vl_mlpar_context* mlpar_ctx)
{
    for (int i = 0; i < mlpar_ctx->io_num.n_input; i++) {
        if (mlpar_ctx->inputs[i].mem) {
            rknn3_destroy_mem(mlpar_ctx->rknn_ctx, mlpar_ctx->inputs[i].mem);
        }
        if (mlpar_ctx->inputs[i].attr != NULL) {
            free(mlpar_ctx->inputs[i].attr);
            mlpar_ctx->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < mlpar_ctx->io_num.n_output; i++) {
        if (mlpar_ctx->outputs[i].mem) {
            rknn3_destroy_mem(mlpar_ctx->rknn_ctx, mlpar_ctx->outputs[i].mem);
        }
        if (mlpar_ctx->outputs[i].attr != NULL) {
            free(mlpar_ctx->outputs[i].attr);
            mlpar_ctx->outputs[i].attr = NULL;
        }
    }
    if (mlpar_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(mlpar_ctx->rknn_ctx);
        mlpar_ctx->rknn_ctx = 0;
    }
    return 0;
}

/**
 * @brief 重排图像特征
 * @param vision_feature 输入图像特征，形状为 (t, h, w, d)
 * @param t 时间维度大小
 * @param h 图像高度
 * @param w 图像宽度
 * @param d 特征维度大小
 * @param m1 高度方向合并因子
 * @param m2 宽度方向合并因子
 * @param output 输出重排后的特征，形状为 (t, h/m1, w/m2, m1*m2*d)
*/
int RearrangeImageFeature(float16* vision_feature, int t, int h, int w, int d, int m1, int m2, float16 *output)
{
    // 参数校验
    if (!vision_feature || !output || t <= 0 || h <= 0 || w <= 0 || d <= 0 || m1 <= 0 || m2 <= 0 || h % m1 != 0 || w % m2 != 0) {
        return -1;
    }

    // 计算新维度
    int h_new = h / m1;
    int w_new = w / m2;
    int total_elements = t * h * w * d;  // float16元素总数

    // 输出维度
    int out_token_num = t * h_new * w_new;
    int out_dim = m1 * m2 * d;  // 每个输出token的维度

    // 重排主循环
    for (int t_idx = 0; t_idx < t; ++t_idx) {
        for (int h_idx = 0; h_idx < h_new; ++h_idx) {
            for (int w_idx = 0; w_idx < w_new; ++w_idx) {
                // 计算输出token起始位置
                int out_token_idx = t_idx * h_new * w_new + h_idx * w_new + w_idx;
                float16 *out_ptr = output + out_token_idx * out_dim;

                // 遍历当前块内的 m1 x m2 个原始patch
                int local_offset = 0;
                for (int p1 = 0; p1 < m1; ++p1) {
                    for (int p2 = 0; p2 < m2; ++p2) {
                        // 计算原始patch位置
                        int orig_h = h_idx * m1 + p1;
                        int orig_w = w_idx * m2 + p2;
                        int orig_token_idx = t_idx * h * w + orig_h * w + orig_w;
                        
                        // 指向原始特征的指针（d维）
                        const float16 *src = vision_feature + orig_token_idx * d;
                        
                        // 拷贝d个float16元素（2字节/元素）
                        memcpy(out_ptr + local_offset, src, (size_t)d * sizeof(float16));
                        local_offset += d;
                    }
                }
            }
        }
    }

    return 0;
}

int inference_paddleocr_vl_mlpar(rknn_paddleocr_vl_mlpar_context* mlpar_ctx, float16* vision_embeds, int grid_h, int grid_w, float16* img_embeds)
{
    if ((!mlpar_ctx) || (!vision_embeds))
    {
        printf("mlpar_ctx or vision_embeds is NULL");
        return -1;
    }
    int ret;

    // Rearrange Vision Embeds
    int t = 1;
    int m1 = 2;
    int m2 = 2;
    float16* rearranged_vision_embeds = (float16*)mlpar_ctx->inputs[0].mem->virt_addr;
    if (!rearranged_vision_embeds)
    {
        printf("rearranged_vision_embeds malloc failed");
        return -1;
    }
    ret = RearrangeImageFeature(vision_embeds, t, grid_h, grid_w, MLPAR_EMBED_DIM, m1, m2, rearranged_vision_embeds);
    if (ret != 0)
    {
        printf("RearrangeImageFeature failed! ret=%d\n", ret);
        return ret;
    }

    // Sync Inputs
    for (int i = 0; i < mlpar_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(mlpar_ctx->rknn_ctx, mlpar_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }
    
    // Run
    ret = rknn3_run(mlpar_ctx->rknn_ctx, mlpar_ctx->inputs, mlpar_ctx->io_num.n_input, mlpar_ctx->outputs, mlpar_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync Outputs
    for (int i = 0; i < mlpar_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(mlpar_ctx->rknn_ctx, mlpar_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Get Output
    memcpy(img_embeds, (float16*)mlpar_ctx->outputs[0].mem->virt_addr, mlpar_ctx->outputs[0].mem->size);

out:

    return ret;
}