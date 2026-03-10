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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "paddleocr_vl.h"
#include "image_utils.h"


#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include "time_utils.h"

int64_t first_token;
bool first_decode = true;

struct embedding_info
{
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
};

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 0.0f,
    .repeat_penalty = 1.1f,   // Please refer to generation_config.json to configure the corresponding parameters. eg, //huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/generation_config.json
    .frequency_penalty = 0.0f, 
    .presence_penalty = 0.0f
};

const char* system_prompt  = "";
const char* prompt_prefix  = "<|begin_of_sentence|>User: ";
const char* prompt_postfix = "\nAssistant: ";
/*-------------------------------------------
                Callback Function
-------------------------------------------*/
int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    Tokenizer *tokenizer = (Tokenizer *)userdata;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n\n--------------------Finished-------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING)
    {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n\n--------------Max new token reached------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_STOP)
    {
        printf("\n\n-----------------------Stop--------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {   
        if (result->num_tokens > 1) {
            for (int i = 0; i < result->num_tokens; i++)
            {
                std::string piece = tokenizer->Decode(result->token_ids, result->num_tokens);
                printf("%s", piece.c_str());
            }
        }
        else {
            std::string piece = tokenizer->TokenToPiece(result->token_ids[0]);
            printf("%s", piece.c_str());
        }
        if (first_decode) {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}


int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{

    int n_tokens = 0;
    Tokenizer *tokenizer = (Tokenizer *)userdata;
    n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{

    struct embedding_info* embed_info = (struct embedding_info*)userdata;

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (int n = 0; n < num_tokens; n++) {
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
            embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

void printf_perf(rknn_perf_metrics_t *p) 
{

    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n", 
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");


    float ttft_us = (float)(first_token - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;  
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens; 
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token);
    float decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");
    
    printf(" Vision latency = %.2f ms, FPS = %.2f\n", 
           (int)p->vision_latency / 1000.f, 1000.f * 1000.f / (int)p->vision_latency);
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 15 && argc != 17)
    {
        printf("%s <vision_model_path> <vision_weight_path> <position_embedding_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <mlpar_model_path> <mlpar_weight_path> <vision_core_mask> <mlpar_core_mask> <llm_core_mask> <image_path> <prompt> <model_width> <model_height>\n", argv[0]);
        return -1;
    }
 
    const char *vision_model_path       = argv[1];
    const char *vision_weight_path      = argv[2];
    const char *position_embedding_path = argv[3];
    const char *llm_model_path          = argv[4];
    const char *llm_weight_path         = argv[5];
    const char *tokenizer_path          = argv[6];
    const char *embedding_path          = argv[7];
    const char *mlpar_model_path        = argv[8];
    const char *mlpar_weight_path       = argv[9];
    uint32_t    vision_core_mask        = strtoul(argv[10], nullptr, 16);
    uint32_t    mlpar_core_mask        = strtoul(argv[11], nullptr, 16);
    uint32_t    llm_core_mask           = strtoul(argv[12], nullptr, 16);
    const char *img_path                = argv[13];
    const char *prompt                  = argv[14];
    uint32_t    model_width             = 0;
    uint32_t    model_height            = 0;
    if (argc == 17) {
        model_width  = strtoul(argv[15], nullptr, 0);
        model_height = strtoul(argv[16], nullptr, 0);
    }

    const char* DEFAULT_PROMPT = "OCR:";
    const char* TABLE_PROMPT = "Table Recognition:";
    const char* CHART_PROMPT = "Chart Recognition:";
    const char* FORMULA_PROMPT = "Formula Recognition:";
    const char* IMAGE_PROMPT = "Image Recognition:";

    if (prompt == nullptr) {
        prompt = DEFAULT_PROMPT;
    } else if (strcmp(prompt, "table") == 0) {
        prompt = TABLE_PROMPT;
    } else if (strcmp(prompt, "chart") == 0) {
        prompt = CHART_PROMPT;
    } else if (strcmp(prompt, "formula") == 0) {
        prompt = FORMULA_PROMPT;
    } else if (strcmp(prompt, "image") == 0) {
        prompt = IMAGE_PROMPT;
    }

    std::string prompt_with_image;

    int ret;
    rknn_perf_metrics_t perf;


    // RKNN Context
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    rknn_app_ctx.model_width = model_width;
    rknn_app_ctx.model_height = model_height;

    // Tokenizer
    Tokenizer* tokenizer;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat           emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(rknn3_llm_param));

    // Input Image
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    // Image Embed
    size_t vision_embed_elems = 1;
    size_t img_embed_elems    = 1;
    float16* vision_embeds;
    float16* img_embeds;

    // LLM Multi Model Tensor
    int n_inputs = 1;
    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Load Toenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }
    
    tokenizer->GetVocabInfo(&(vocab_info));
    printf("vocab_info: vocab_size=%d, special_bos_id=[", vocab_info.vocab_size);
    for (int i = 0; i < vocab_info.n_special_bos_id; ++i)
    {
        printf("%d%s", vocab_info.special_bos_id[i], (i + 1 < vocab_info.n_special_bos_id) ? ", " : "");
    }
    printf("], special_eos_id=[");
    for (int i = 0; i < vocab_info.n_special_eos_id; ++i)
    {
        printf("%d%s", vocab_info.special_eos_id[i], (i + 1 < vocab_info.n_special_eos_id) ? ", " : "");
    }
    printf("]\n");

    // Read Embedding
    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1) {
        printf("Failed to open embedding file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1) {
        printf("Failed to get embedding file size\n");
        goto out;
    }

    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        goto out;
    }

    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    // Set LLM parameters
    params.logits_name               = "logits";
    params.max_context_len           = MAX_CONTEXT_LEN;
    // params.max_new_tokens            = MAX_NEW_TOKENS;
    params.sampling_param            = SAMPLE_PARAMS;
    params.vocab_info.vocab_size     = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));    
    params.vocab_info.linefeed_id    = vocab_info.linefeed_id;

    // LLM Callback
    callback.result_callback    = result_callback;
    callback.result_userdata    = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    ret = init_paddleocr_vl_model(&rknn_app_ctx, llm_model_path, llm_weight_path, 
        vision_model_path, vision_weight_path, position_embedding_path, mlpar_model_path, mlpar_weight_path,
        &params, n_params, callback, vision_core_mask, mlpar_core_mask, llm_core_mask);
    if (ret != 0)
    {
        printf("init_paddleocr_vl_model fail! ret=%d llm_model_path=%s vision_model_path=%s mlpar_model_path=%s\n", ret, llm_model_path, vision_model_path, mlpar_model_path);
        goto out;
    }

    // Vision Embed
    for (size_t i = 0; i < rknn_app_ctx.vision.embeds_ndims; i++)
    {
      vision_embed_elems *= rknn_app_ctx.vision.embeds_shape[i];
    }
    vision_embeds = (float16*)malloc((vision_embed_elems) * sizeof(float16));

    // Image Embed
    for (size_t i = 0; i < rknn_app_ctx.mlpar.embeds_ndims; i++)
    {
      img_embed_elems *= rknn_app_ctx.mlpar.embeds_shape[i];
    }
    img_embeds = (float16*)malloc((img_embed_elems) * sizeof(float16));

    // Read Image
    ret = read_image(img_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
        goto out;
    }

    // LLM Input
    tensor.name           = "input_embeds";
    // Add image start tags to the prompt
    prompt_with_image = "<image> " + std::string(prompt);
    tensor.prompt         = (prompt_with_image).c_str();
    tensor.image.image_embed    = img_embeds;
    if(rknn_app_ctx.mlpar.embeds_ndims == 2) {
        tensor.image.n_image_tokens = rknn_app_ctx.mlpar.embeds_shape[0];
        tensor.image.n_image        = 1;
    } else {
        tensor.image.n_image_tokens = rknn_app_ctx.mlpar.embeds_shape[1];
        tensor.image.n_image        = rknn_app_ctx.mlpar.embeds_shape[0];
    }
    tensor.image.image_width    = rknn_app_ctx.vision.model_width;
    tensor.image.image_height   = rknn_app_ctx.vision.model_height;
    tensor.image.image_start      = "<|IMAGE_START|>";
    tensor.image.image_end        = "<|IMAGE_END|>";
    tensor.image.image_content    = "<|IMAGE_PLACEHOLDER|>";
    tensor.enable_thinking = false;

    ret = inference_paddleocr_vl_model(&rknn_app_ctx, &src_image, vision_embeds, img_embeds, tensor, n_inputs, &perf);
    if (ret != 0)
    {
        printf("inference paddleocr_vl model fail! ret=%d\n", ret);
        goto out;
    }

    printf_perf(&perf);
out:
    ret = release_paddleocr_vl_model(&rknn_app_ctx);

    if (ret != 0)
    {
        printf("release paddleocr_vl model fail! ret=%d\n", ret);
    }

    if (embedding_info.fd != -1) {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL) {
            munmap((void*)embedding_info.embedding_data, emb_st.st_size);
            embedding_info.embedding_data = NULL;
        }
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    if (tokenizer != NULL)
    {
        delete tokenizer;
        tokenizer = NULL;
    }

    if (vision_embeds != NULL)
    {
        free(vision_embeds);
    }

    if (img_embeds != NULL)
    {
        free(img_embeds);
    }

    return ret;
}