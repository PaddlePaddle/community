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

#include "rknn_paddleocr_vl_vision.h"
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


int init_paddleocr_vl_vision(rknn_paddleocr_vl_vision_context* vision_ctx, const char* model_path, const char* weight_path, const char* position_embedding_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

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
    vision_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    vision_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        vision_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        vision_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    int patch_num = MODEL_WIDTH * MODEL_HEIGHT / 14 / 14;
    if(model_width != 0 && model_height != 0) {
        patch_num = model_width * model_height / 14 / 14;
    }
    if(input_attrs[0].shape[1] != patch_num){
        printf("Please note: model_width=%d and model_height=%d do not match the model's expected input dimensions. Please either update the default parameters or explicitly specify the correct model_width and model_height arguments in the command line.\n", model_width == 0 ? MODEL_WIDTH : model_width, model_height == 0 ? MODEL_HEIGHT : model_height);
        return -1;
    }
    vision_ctx->patch_num = patch_num;
    vision_ctx->pruned_version_flag = 1;
    vision_ctx->model_channel = 3;
    // 由于vision模型被裁剪了，需要手动添加这些参数配置
    if(model_width != 0 && model_height != 0) {
        vision_ctx->model_width = (int)model_width;
        vision_ctx->model_height = (int)model_height;
    } else {
        vision_ctx->model_width = (int)MODEL_WIDTH;
        vision_ctx->model_height = (int)MODEL_HEIGHT;
    }
    printf("model_width=%d model_height=%d \n", vision_ctx->model_width,vision_ctx->model_height);

    if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED)
    {
        printf("model is UNDEFINED output layout\n");
        vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
        vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;
    }
    else
    {
        printf("model is not UNDEFINED output layout, model output error!\n");
        return -1;
    }

    // Load patch_pos_embed from bin file
    vision_ctx->patch_pos_embed = (float*)malloc(POSITION_EMBED_DIM * POSITION_EMBED_GRID * POSITION_EMBED_GRID * sizeof(float));
    if (!vision_ctx->patch_pos_embed) {
        printf("Failed to allocate memory for patch_pos_embed\n");
        return -1;
    }
    FILE* f = fopen(position_embedding_path, "rb");
    if (!f) {
        printf("Failed to open position embedding file: %s\n", position_embedding_path);
        free(vision_ctx->patch_pos_embed);
        return -1;
    }
    size_t read_size = fread(vision_ctx->patch_pos_embed, sizeof(float), POSITION_EMBED_DIM * POSITION_EMBED_GRID * POSITION_EMBED_GRID, f);
    if (read_size != POSITION_EMBED_DIM * POSITION_EMBED_GRID * POSITION_EMBED_GRID) {
        printf("Failed to read position embedding data from file: %s\n", position_embedding_path);
        printf("Expected size: %lu, Read size: %lu\n", POSITION_EMBED_DIM * POSITION_EMBED_GRID * POSITION_EMBED_GRID, read_size);
        free(vision_ctx->patch_pos_embed);
        fclose(f);
        return -1;
    }
    fclose(f);

    // Init inv_freq
    const float theta = 10000.0f;
    int rope_dim_ = ROPE_DIM / 2;
    vision_ctx->inv_freq = (float*)malloc(rope_dim_ * sizeof(float));
    if (!vision_ctx->inv_freq) {
        printf("Failed to allocate memory for inv_freq\n");
        free(vision_ctx->patch_pos_embed);
        fclose(f);
        return -1;
    }
    float* inv_freq = vision_ctx->inv_freq;
    // 计算逆频率: inv_freq[i] = 1 / (theta ^ (2*i / dim))
    for (int i = 0; i < rope_dim_; ++i) {
        float exponent = (2.0f * (float)i) / (float)ROPE_DIM;
        inv_freq[i] = 1.0f / powf(theta, exponent);
    }

    return ret;
}

int release_paddleocr_vl_vision(rknn_paddleocr_vl_vision_context* vision_ctx)
{
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        if (vision_ctx->inputs[i].mem) {
            rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
        }
        if (vision_ctx->inputs[i].attr != NULL) {
            free(vision_ctx->inputs[i].attr);
            vision_ctx->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        if (vision_ctx->outputs[i].mem) {
            rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem);
        }
        if (vision_ctx->outputs[i].attr != NULL) {
            free(vision_ctx->outputs[i].attr);
            vision_ctx->outputs[i].attr = NULL;
        }
    }
    if (vision_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    if (vision_ctx->patch_pos_embed) {
        free(vision_ctx->patch_pos_embed);
        vision_ctx->patch_pos_embed = NULL;
    }
    if (vision_ctx->inv_freq) {
        free(vision_ctx->inv_freq);
        vision_ctx->inv_freq = NULL;
    }
    return 0;
}

/**
 * 根据图像宽高和 patch 数量，寻找最优的网格划分 (grid_w, grid_h)
 * @param width      图像宽度（像素）
 * @param height     图像高度（像素）
 * @param patch_num  patch 数量
 * @param gw         输出：最优网格宽度
 * @param gh         输出：最优网格高度
*/
void FindBestGrid(int width, int height, int patch_num, int *gw, int *gh) {
    // 参数校验：确保输出指针有效
    if (gw == NULL || gh == NULL) {
        printf("Error: Output pointers gw or gh are NULL.\n");
        return;
    }

    // 无效 patch_num 处理
    if (patch_num <= 0 || patch_num < 4) {
        *gw = 0;
        *gh = 0;
        return;
    }

    // 计算图像宽高比
    float ratio = (height > 0) ? (float)width / (float)height : 1.0f;

    // 初始化最优解
    float best_diff = FLT_MAX;
    int best_gw = 0;
    int best_gh = 0;
    int found = 0;  // 标记是否找到有效网格

    // 遍历所有因子对
    int max_factor = (int)sqrt((double)patch_num);
    for (int f = 2; f <= max_factor; ++f) {
        if (patch_num % f != 0) {
            continue;
        }
        int other = patch_num / f;

        // 检查 (f, other) 组合
        if (f >= 2 && other >= 2 && (f % 2 == 0) && (other % 2 == 0)) {
            float cand_ratio = (float)f / (float)other;
            float diff = (float)fabs(cand_ratio - ratio);
            if (diff < best_diff) {
                best_diff = diff;
                best_gw = f;
                best_gh = other;
                found = 1;
            }
        }

        // 检查 (other, f) 组合（避免重复）
        if (f != other && other >= 2 && f >= 2 && (other % 2 == 0) && (f % 2 == 0)) {
            float cand_ratio = (float)other / (float)f;
            float diff = (float)fabs(cand_ratio - ratio);
            if (diff < best_diff) {
                best_diff = diff;
                best_gw = other;
                best_gh = f;
                found = 1;
            }
        }
    }

    // 设置输出结果
    if (found) {
        *gw = best_gw;
        *gh = best_gh;
    } else {
        *gw = 0;
        *gh = 0;
    }

    // 调试输出
    printf("Best grid for image %dx%d is %dx%d\n", width, height, *gw, *gh);
}

/**
 * 从 NHWC 格式图像直接归一化并打包为 flatten patches
 *
 * @param src             输入图像数据，NHWC 格式（HWC），大小为 height * width * 3，像素值范围 [0, 255]
 * @param width           图像宽度（像素）
 * @param height          图像高度（像素）
 * @param grid_w          水平方向网格数
 * @param grid_h          垂直方向网格数
 * @param flatten_patches 输出缓冲区，调用者需预先分配至少 grid_h * grid_w * 3 * PATCH_SIZE * PATCH_SIZE 个 float 空间
 *
 * 要求：width == grid_w * PATCH_SIZE 且 height == grid_h * PATCH_SIZE
 */
void NormalizeAndPackPatches(const unsigned char *src,
                             int width,
                             int height,
                             int grid_w,
                             int grid_h,
                             float *flatten_patches) {
    // 参数校验
    if (src == NULL || flatten_patches == NULL) {
        return;
    }

    // 验证图像尺寸与网格划分匹配（可选，生产环境建议保留）
    if (width != grid_w * PATCH_SIZE || height != grid_h * PATCH_SIZE) {
        return;
    }

    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float std[3]  = {127.5f, 127.5f, 127.5f};
    size_t out_idx = 0;

    // 五重循环：网格行 -> 网格列 -> 通道 -> patch 行 -> patch 列
    for (int py = 0; py < grid_h; ++py) {
        for (int px = 0; px < grid_w; ++px) {
            for (int c = 0; c < 3; ++c) {
                for (int ph = 0; ph < PATCH_SIZE; ++ph) {
                    for (int pw = 0; pw < PATCH_SIZE; ++pw) {
                        // 计算源图像中的绝对坐标
                        int src_y = py * PATCH_SIZE + ph;
                        int src_x = px * PATCH_SIZE + pw;

                        // NHWC 索引: (y * width + x) * 3 + channel
                        size_t nhwc_idx = ((size_t)src_y * width + src_x) * 3 + c;

                        // 归一化: (pixel - mean) / std
                        float normalized = ((float)src[nhwc_idx] - mean[c]) / std[c];

                        // 写入输出缓冲区
                        flatten_patches[out_idx++] = normalized;
                    }
                }
            }
        }
    }
}

/**
 * 构建图像的 flatten patches
 * @param src_img         输入图像缓冲区
 * @param patch_num       期望的 patch 数量
 * @param flatten_patches 输出的 flatten patches 缓冲区，调用者需预先分配足够空间
 * @param grid_h          输出网格高度
 * @param grid_w          输出网格宽度
*/
int BuildFlattenPatches(image_buffer_t* src_img, int patch_num, embed_t* flatten_patches, int* grid_h, int* grid_w) {
    if (!src_img || !flatten_patches || !grid_h || !grid_w || patch_num <= 0) {
        printf("invalid parameters for BuildFlattenPatches\n");
        return -1;
    }
    
    int src_width  = src_img->width;
    int src_height = src_img->height;
    FindBestGrid(src_width, src_height, patch_num, grid_w, grid_h);
    if (*grid_w <= 0 || *grid_h <= 0) {
        printf("failed to find best grid for image %dx%d with patch_num=%d\n", src_width, src_height, patch_num);
        return -1;
    }

    int new_width  = *grid_w * PATCH_SIZE;
    int new_height = *grid_h * PATCH_SIZE;
    printf("Resizing image from %dx%d to %dx%d for patch_num=%d with grid %dx%d\n", src_width, src_height, new_width, new_height, patch_num, *grid_w, *grid_h);
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));
    dst_img.width         = new_width;
    dst_img.height        = new_height;
    dst_img.width_stride  = new_width;
    dst_img.height_stride = new_height;
    dst_img.format        = IMAGE_FORMAT_RGB888;
    dst_img.size          = get_image_size(&dst_img);
    dst_img.virt_addr     = (unsigned char *)malloc(dst_img.size);
    if (!dst_img.virt_addr) {
        printf("malloc dst_img buffer failed for size=%d\n", dst_img.size);
        return -1;
    }

    if (convert_image(src_img, &dst_img, NULL, NULL, 0) != 0) {
        printf("convert_image failed for resize %dx%d -> %dx%d\n", src_width, src_height, new_width, new_height);
        return -1;
    }

    NormalizeAndPackPatches(dst_img.virt_addr, new_width, new_height, *grid_w, *grid_h, flatten_patches->virt_addr);
    
    printf("flattened patches size=%zu for grid %dx%d\n", flatten_patches->size, *grid_w, *grid_h);
    free(dst_img.virt_addr);
    return 0;
}

// 辅助函数：边界裁剪
static inline int clip(int x, int min_val, int max_val) {
    return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
}

// 双线性插值采样单个通道值
static inline float bilinear_sample(const float *input,
                                    int c, int h, int w,
                                    int in_h, int in_w,
                                    float y, float x) {
    // 边界处理：PyTorch使用边界反射（实际是clamp）
    int y0 = (int)floorf(y);
    int x0 = (int)floorf(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    // clamp到有效范围 [0, size-1]
    y0 = clip(y0, 0, in_h - 1);
    y1 = clip(y1, 0, in_h - 1);
    x0 = clip(x0, 0, in_w - 1);
    x1 = clip(x1, 0, in_w - 1);

    float dy = y - y0;
    float dx = x - x0;

    // 输入内存布局: (1, dim, H, W) -> [c][h][w]
    int base_offset = c * in_h * in_w;
    float v00 = input[base_offset + y0 * in_w + x0];
    float v01 = input[base_offset + y0 * in_w + x1];
    float v10 = input[base_offset + y1 * in_w + x0];
    float v11 = input[base_offset + y1 * in_w + x1];

    // 双线性插值公式
    float top = v00 * (1.0f - dx) + v01 * dx;
    float bottom = v10 * (1.0f - dx) + v11 * dx;
    return top * (1.0f - dy) + bottom * dy;
}

/**
 * 生成位置嵌入
 * @param grid_h          输出网格高度
 * @param grid_w          输出网格宽度
 * @param patch_pos_embed 输入的 patch 位置嵌入，形状为 (1, POSITION_EMBED_DIM, POSITION_EMBED_GRID, POSITION_EMBED_GRID)
 * @param position        输出位置嵌入，形状为 (grid_h * grid_w, POSITION_EMBED_DIM)
*/
int GeneratePositionEmbedding(int grid_h,
                              int grid_w,
                              const float *patch_pos_embed,
                              float *position) {
    if (!patch_pos_embed || !position || grid_h <= 0 || grid_w <= 0) {
        return -1; // 参数校验失败
    }

    // 处理特殊情况：当输出尺寸为1时，PyTorch的scale计算需特殊处理
    float scale_h = (grid_h > 1) ? ((float)POSITION_EMBED_GRID / grid_h) : 1.0f;
    float scale_w = (grid_w > 1) ? ((float)POSITION_EMBED_GRID / grid_w) : 1.0f;

    // 直接输出到 (grid_h * grid_w, dim) 格式，等价于 interpolate + permute + view
    for (int h_out = 0; h_out < grid_h; ++h_out) {
        for (int w_out = 0; w_out < grid_w; ++w_out) {
            // 计算源坐标 (align_corners=False)
            float y_src = h_out * scale_h;
            float x_src = w_out * scale_w;

            // 为每个通道插值
            for (int c = 0; c < POSITION_EMBED_DIM; ++c) {
                float val = bilinear_sample(
                    patch_pos_embed, 
                    c, h_out, w_out, 
                    POSITION_EMBED_GRID, POSITION_EMBED_GRID,
                    y_src, x_src
                );
                // 直接按 (h, w, c) 顺序写入输出: 等价于 permute(0,2,3,1).view(1,-1,dim)
                int out_idx = (h_out * grid_w + w_out) * POSITION_EMBED_DIM + c;
                position[out_idx] = val;
            }
        }
    }

    return 0; // 成功
}

/**
 * 生成 RoPE 位置嵌入
 * @param grid_h     网格高度
 * @param grid_w     网格宽度
 * @param inv_freq   逆频率数组，长度为 rope_dim
 * @param rope_dim   RoPE 维度
 * @param rope_out   输出缓冲区，调用者需预先分配至少 grid_h * grid_w * rope_dim * 4 个 float 空间
*/
int GenerateRopeEmbedding(int grid_h, int grid_w,
                          const float *inv_freq, int rope_dim,
                          float *rope_out) {
    // 参数校验
    if (grid_h <= 0 || grid_w <= 0 || !inv_freq || rope_dim <= 0 || !rope_out) {
        return -1;
    }

    // 直接计算输出，避免临时table分配
    size_t idx = 0;
    const int final_dim = rope_dim * 4;  // 每个位置的总维度

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            // 第1组：高度频率 (h * inv_freq[j])
            for (int j = 0; j < rope_dim; ++j) {
                rope_out[idx++] = (float)h * inv_freq[j];
            }
            // 第2组：宽度频率 (w * inv_freq[j])
            for (int j = 0; j < rope_dim; ++j) {
                rope_out[idx++] = (float)w * inv_freq[j];
            }
            // 第3组：高度频率（重复）
            for (int j = 0; j < rope_dim; ++j) {
                rope_out[idx++] = (float)h * inv_freq[j];
            }
            // 第4组：宽度频率（重复）
            for (int j = 0; j < rope_dim; ++j) {
                rope_out[idx++] = (float)w * inv_freq[j];
            }
        }
    }

    // 安全检查：确保写入元素数量正确
    if (idx != (size_t)grid_h * grid_w * final_dim) {
        return -1; // 理论上不应发生
    }

    return 0;
}

int inference_paddleocr_vl_vision(rknn_paddleocr_vl_vision_context* vision_ctx, image_buffer_t* img, float16* vision_embeds, int* gridh, int* gridw)
{
    if ((!vision_ctx) || (!img))
    {
        printf("vision_ctx or img is NULL");
        return -1;
    }
    int ret;

    embed_t flatten_patches, position_embeds, rope_embeds;
    float16* fp_fp16 = (float16*)vision_ctx->inputs[0].mem->virt_addr;
    float16* pe_fp16 = (float16*)vision_ctx->inputs[1].mem->virt_addr;
    float16* rope_fp16 = (float16*)vision_ctx->inputs[2].mem->virt_addr;
    // Build flatten patches
    flatten_patches.size = vision_ctx->patch_num * vision_ctx->model_channel * PATCH_SIZE * PATCH_SIZE;
    flatten_patches.virt_addr = (float*)malloc(sizeof(float) * flatten_patches.size);
    int grid_h = 0;
    int grid_w = 0;
    ret = BuildFlattenPatches(img, vision_ctx->patch_num, &flatten_patches, &grid_h, &grid_w);
    if (ret != 0) {
        printf("BuildFlattenPatches failed\n");
        goto out;
    }
    // Generate position embeddings
    position_embeds.size = vision_ctx->patch_num * POSITION_EMBED_DIM;
    position_embeds.virt_addr = (float*)malloc(sizeof(float) * position_embeds.size);
    ret = GeneratePositionEmbedding(grid_h, grid_w, vision_ctx->patch_pos_embed, position_embeds.virt_addr);
    if (ret != 0) {
        printf("GeneratePositionEmbedding failed\n");
        goto out;
    }
    // Generate rope embeddings
    rope_embeds.size = vision_ctx->patch_num * ROPE_DIM / 2 * 4;
    rope_embeds.virt_addr = (float*)malloc(sizeof(float) * rope_embeds.size);
    ret = GenerateRopeEmbedding(grid_h, grid_w, vision_ctx->inv_freq, ROPE_DIM / 2, rope_embeds.virt_addr);
    if (ret != 0) {
        printf("GenerateRopeEmbedding failed\n");
        goto out;
    }

    for (int i = 0; i < flatten_patches.size; ++i) {
        fp_fp16[i] = fp32_to_fp16(flatten_patches.virt_addr[i]);
    }
    for (int i = 0; i < position_embeds.size; ++i) {
        pe_fp16[i] = fp32_to_fp16(position_embeds.virt_addr[i]);
    }
    for (int i = 0; i < rope_embeds.size; ++i) {
        rope_fp16[i] = fp32_to_fp16(rope_embeds.virt_addr[i]);
    }

    // Sync Inputs
    for (int i = 0; i < vision_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }
    
    // Run
    ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input, vision_ctx->outputs, vision_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync Outputs
    for (int i = 0; i < vision_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Get Output
    memcpy(vision_embeds, (float16*)vision_ctx->outputs[0].mem->virt_addr, vision_ctx->outputs[0].mem->size);
    *gridh = grid_h;
    *gridw = grid_w;

out:
    if (flatten_patches.virt_addr != NULL) {
        free(flatten_patches.virt_addr);
    }
    if (position_embeds.virt_addr != NULL) {
        free(position_embeds.virt_addr);
    }
    if (rope_embeds.virt_addr != NULL) {
        free(rope_embeds.virt_addr);
    }

    return ret;
}