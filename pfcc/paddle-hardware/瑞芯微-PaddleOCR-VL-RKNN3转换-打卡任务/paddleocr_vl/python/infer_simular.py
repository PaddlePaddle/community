import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

from rknn.api import RKNN
import numpy as np
import torch
from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    AutoProcessor, AutoTokenizer
)
from PIL import Image
from modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration
from vision.export_vision import BasicImageTransform

VISION_RKNN_MODEL = '../model/vision/PaddleOCR-vision.rknn'
MLPAR_RKNN_MODEL = '../model/vision/PaddleOCR-vision-mlp_AR.rknn'
LLM_CONFIG = '../model/llm/PaddleOCR-llm.config.pkl'
LLM_RKNN_MODEL = '../model/llm/PaddleOCR-llm.rknn'
EMBED_PATH = '../model/llm/PaddleOCR-llm.embed.bin'
TOKENIZER_PATH = 'PaddlePaddle/PaddleOCR-VL'

PROMPT = "table"
VOCAB_SIZE = 103424
SEQ_LEN = 128
PATCH_SIZE = 14
image_token_id = 100295

def llm_logitsprocessor(input_ids, logits, args={}):
    temperature = args.get('temperature', 1.0)
    top_k = args.get('top_k', 1)
    top_p = args.get('top_p', 0.9)
    repetition_penalty = args.get('repeat_penalty', 1.0)
    do_sample = args.get('do_sample', False)

    warpers = [
        TemperatureLogitsWarper(temperature), 
        RepetitionPenaltyLogitsProcessor(repetition_penalty) if input_ids is not None else None,
        TopKLogitsWarper(top_k=top_k), 
        TopPLogitsWarper(top_p=top_p), 
        ]

    for warper in warpers:
        if warper is not None:
            logits = warper(input_ids=input_ids, scores=logits)

    probs = torch.softmax(logits, dim=-1)

    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)[0]
    else:
        next_token = torch.argmax(probs, dim=-1)

    return next_token.numpy()

def load_config(config_path: str):
    import json
    import os
    if not os.path.exists(config_path):
        return 504, 504, np.array([[1,36,36]], dtype=np.int64)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    required_keys = ['img_h', 'img_w', 'grid_thw']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in config file")
    return config["img_h"], config["img_w"], np.array([config["grid_thw"]], dtype=np.int64)

img_h, img_w, _ = load_config('vision/vision_config.json')
print(f"Using img_h={img_h}, img_w={img_w}")

def find_best_size(original_width: int, original_height: int, patch_size: int=28, target_size: int=196):
    import math
    # 计算原始宽高比 (width / height)
    original_ratio = original_width / original_height
    # target_size的因数对 (a, b) 满足 a * b = target_size
    factors = []
    for a in range(1, int(math.sqrt(target_size)) + 1):
        if target_size % a == 0:
            b = target_size // a
            factors.append((a, b))
            if a != b:
                factors.append((b, a))
    # 找到最接近原始宽高比的因数对
    best_ratio_diff = float('inf')
    best_pair = (1, target_size)
    for a, b in factors:
        current_ratio = a / b
        diff = abs(current_ratio - original_ratio)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_pair = (a, b)
    width_new = best_pair[0] * patch_size
    height_new = best_pair[1] * patch_size
    return width_new, height_new

def interpolate_pos_encoding(
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor:
        dim = 1152
        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // 14
            new_width = width // 14
        patch_pos_embed = np.fromfile("../model/vision/position_embedding_model.bin", dtype=np.float32)
        patch_pos_embed = torch.from_numpy(patch_pos_embed)
        patch_pos_embed = patch_pos_embed.reshape(1, 1152, 27, 27)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

def get_rope_emb(image_grid_thw):
    split_hids = list()
    split_wids = list()
    for t, h, w in image_grid_thw:
        image_pids = torch.arange(t*h*w) % (h*w)
        sample_hids = image_pids // w
        sample_wids = image_pids % w
        split_hids.append(sample_hids)
        split_wids.append(sample_wids)
    width_position_ids = torch.concat(split_wids, dim=0)
    height_position_ids = torch.concat(split_hids, dim=0)

    pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
    max_grid_size = pids.max() + 1
    
    dim = 36
    theta = 10000.0
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    )
    seq = torch.arange(
        max_grid_size, device=inv_freq.device, dtype=inv_freq.dtype
    )
    rope_emb_max_grid = torch.outer(seq, inv_freq)
    rope_emb = rope_emb_max_grid[pids].flatten(1)
    rope_emb = rope_emb.repeat(1, 2)
    return rope_emb

def prepare_vlm_input(vision_rknn, mlpar_rknn, embeds_data, tokenizer_path, prompt, image_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    image = Image.open(image_path).convert("RGB")
    old_w, old_h = image.size
    new_w, new_h = find_best_size(old_w, old_h, target_size=img_h//PATCH_SIZE//2*img_w//PATCH_SIZE//2)
    print("image size: src({},{}) -> new({},{})".format(old_w, old_h, new_w, new_h))
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    image = np.array(image)
    pixel_values = image_transform(image).to(torch.float32)
    grid_t, grid_h, grid_w = 1, new_h//PATCH_SIZE, new_w//PATCH_SIZE
    patches = pixel_values.reshape(
        grid_t, 1, 3, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE
    ).numpy()
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, 3, PATCH_SIZE, PATCH_SIZE
    )
    pixel_values = torch.from_numpy(flatten_patches)
    pixel_values = pixel_values.type(torch.float32)
    pixel_values = pixel_values.unsqueeze(0)

    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64)
    position_embedding = interpolate_pos_encoding(grid_h, grid_w, True)
    rope_emb = get_rope_emb(grid_thw)
    
    inputs = [pixel_values.numpy(), position_embedding.numpy(), rope_emb.numpy()]
    data_format = ['nchw'] * len(inputs)
    vision_embed = vision_rknn.inference(inputs, data_format, accuracy_analysis=False)[0]
    
    t, h, w = grid_thw[0]
    from einops import rearrange
    vision_embed = rearrange(
        vision_embed,
        "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
        t=t,
        h=h // 2,
        p1=2,
        w=w // 2,
        p2=2,
    )
    inputs = [vision_embed]
    data_format = ['nchw'] * len(inputs)
    image_embeds = mlpar_rknn.inference(inputs, data_format, accuracy_analysis=False)[0]
    
    DEFAULT_PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }
    query = DEFAULT_PROMPTS[prompt]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = processor(image, text=text, return_tensors="pt", format=True)
    
    input_ids = inputs["input_ids"].cpu().numpy().astype(np.int64)
    inputs_embeds = embeds_data[input_ids].astype(np.float32)
    print("inputs_embeds shape:", inputs_embeds.shape)

    n_image_tokens = (input_ids == image_token_id).sum()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )

    mask = input_ids == image_token_id
    print("mask shape:", mask.shape)
    print("image_embeds shape:", image_embeds.shape)

    batch_idx, seq_idx = np.where(mask)
    inputs_embeds[batch_idx, seq_idx] = image_embeds.astype(inputs_embeds.dtype)
    
    inputs["inputs_embeds"] = inputs_embeds
    inputs["attention_mask"] = inputs["attention_mask"].cpu().numpy().astype(np.float32)
    
    return inputs, tokenizer

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="PaddleOCR-VL RKNN Simular Inference")
    parser.add_argument("--vision_rknn_path", type=str, help="vision rknn model path", required=False, default=VISION_RKNN_MODEL)
    parser.add_argument("--mlpar_rknn_path", type=str, help="mlpar rknn model path", required=False, default=MLPAR_RKNN_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--llm_rknn_path", type=str, help="output rknn model path", required=False, default=LLM_RKNN_MODEL)
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path or name", required=False, default=TOKENIZER_PATH)
    parser.add_argument("--target", action='store_true', help="Whether use target inference")
    parser.add_argument("--device_id", type=str, help="device id", required=False, default=None)
    parser.add_argument("--prompt", type=str, help="input prompt", required=False, default=PROMPT)
    parser.add_argument("--image_path", type=str, help="input image path for vlm", required=False, default="../data/vision/test.png")
    args = parser.parse_args()

    # Create RKNN object
    vision_rknn = RKNN(verbose=False)
    mlpar_rknn = RKNN(verbose=False)
    llm_rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    vision_rknn.config(target_platform='rk1820',
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32')
    mlpar_rknn.config(target_platform='rk1820',
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32')
    llm_rknn.config(target_platform='rk1820', 
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32')
    print('done')
    
    print('--> Load model')
    vision_weight_path = args.vision_rknn_path.replace('.rknn', '.weight')
    ret = vision_rknn.load_rknn(args.vision_rknn_path, vision_weight_path, load_ctx=True)
    if ret != 0:
        print('Load vision rknn model failed!')
        exit(ret)
    mlpar_weight_path = args.mlpar_rknn_path.replace('.rknn', '.weight')
    ret = mlpar_rknn.load_rknn(args.mlpar_rknn_path, mlpar_weight_path, load_ctx=True)
    if ret != 0:
        print('Load mlpar rknn model failed!')
        exit(ret)
    llm_weight_path = args.llm_rknn_path.replace('.rknn', '.weight')
    ret = llm_rknn.load_rknn(args.llm_rknn_path, llm_weight_path, load_ctx=True)
    if ret != 0:
        print('Load llm rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    if args.target:
        ret = vision_rknn.init_runtime(target='rk1820', core_mask=0xff, device_id=args.device_id)
        ret = mlpar_rknn.init_runtime(target='rk1820', core_mask=0xff, device_id=args.device_id)
        ret = llm_rknn.init_runtime(target='rk1820', core_mask=0xff, device_id=args.device_id)
    else:
        ret = vision_rknn.init_runtime()
        ret = mlpar_rknn.init_runtime()
        ret = llm_rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    embeds_data = np.fromfile(EMBED_PATH, dtype=np.float16).reshape(VOCAB_SIZE, -1)

    # LLM Inference
    inputs, tokenizer = prepare_vlm_input(vision_rknn, mlpar_rknn, embeds_data,
                                          tokenizer_path=args.tokenizer_path, prompt=args.prompt, image_path=args.image_path)

    embeds = inputs['inputs_embeds']
    attention_mask = inputs['attention_mask']
    token_ids = inputs['input_ids'].detach().cpu().numpy()
    input_seq_len = token_ids.shape[1]
    
    print(f'--> Input sequence length with image: {input_seq_len}')
    
    eos_token_ids = [tokenizer.eos_token_id]
    generate_ids = []

    rope_cache = llm_rknn.query('QUERY_ROPE_CACHE')
    print("rope_cache:", rope_cache)
    print("rope_cache_size:", rope_cache[0].shape, rope_cache[1].shape)

    print('--> Prefill Inference')
    if SEQ_LEN >= input_seq_len:
        inputs_embeds = np.zeros((1, SEQ_LEN, embeds.shape[-1]), dtype=np.float32)
        attention_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
        inputs_embeds[:,:input_seq_len,:] = embeds
        attention_mask[:,:input_seq_len] = 1
        num_logits_to_keep = np.array([input_seq_len - 1], dtype=np.int32)

        attention_inputs, dynamic_idx = llm_rknn.kvcache_controller.generate_kvcache_control_tensors(input_seq_len)

        prefill_inputs = [inputs_embeds, attention_mask, num_logits_to_keep] + rope_cache + attention_inputs[0]
        data_format    = ['nchw'] * len(prefill_inputs)

        prefill_logits = llm_rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)[0]
    else:
        attention_inputs, dynamic_idx = llm_rknn.kvcache_controller.generate_kvcache_control_tensors(input_seq_len)
        for i, seq_len in enumerate(range(0, input_seq_len, SEQ_LEN)):
            inputs_embeds = np.zeros((1, SEQ_LEN, embeds.shape[-1]), dtype=np.float32)
            attention_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
            curr_len = min(SEQ_LEN, input_seq_len - seq_len)
            inputs_embeds[:,:curr_len,:] = embeds[:,seq_len:seq_len+curr_len,:]
            attention_mask[:,:curr_len] = 1
            num_logits_to_keep = np.array([curr_len - 1], dtype=np.int32)

            prefill_inputs = [inputs_embeds, attention_mask, num_logits_to_keep] + rope_cache + attention_inputs[i]
            data_format    = ['nchw'] * len(prefill_inputs)

            prefill_logits = llm_rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)[0]

    next_token = llm_logitsprocessor(torch.from_numpy(token_ids), torch.from_numpy(prefill_logits).reshape(1, -1))
    print(f"--> First new token:{next_token}")
    generate_ids.append(next_token[0])

    print("--> Decoder inference")
    max_new_tokens = 1024
    from tqdm import tqdm
    inf_bar = tqdm(range(max_new_tokens), desc='I RKLLM KVCache Inference ', ncols=100)
    for i in inf_bar:
        if i == max_new_tokens - 1:
            continue
        input_ids = np.expand_dims(next_token, axis=0).astype(np.int64)
        inputs_embeds = embeds_data[input_ids].astype(np.float32)
        attention_mask = np.expand_dims(np.array([1]), axis=0).astype(np.float32)
        num_logits_to_keep = np.array([0]).astype(np.int32)

        token_ids = np.concatenate((token_ids, input_ids), axis=1)

        attention_inputs, dynamic_idx = llm_rknn.kvcache_controller.generate_kvcache_control_tensors(1)

        decoder_inputs = [inputs_embeds, attention_mask, num_logits_to_keep] + rope_cache + attention_inputs[0]
        data_format    = ['nchw'] * len(decoder_inputs)

        decoder_logits = llm_rknn.inference(decoder_inputs, data_format)[0]
        next_token = llm_logitsprocessor(torch.from_numpy(token_ids), torch.from_numpy(decoder_logits).reshape(1, -1))
        generate_ids.append(next_token[0])

        if next_token[-1] in eos_token_ids:
            print('LLM Inference has completed!')
            break
        if (i+1) % 10 == 0:
            print("generate_ids: ", generate_ids)
            response = tokenizer.decode(generate_ids, skip_special_tokens=True)
            print("--> Intermediate response:\n", response)

    response = tokenizer.decode(generate_ids, skip_special_tokens=True)
    if generate_ids == None:
        print('RKNN llm inference failed!')
        exit(ret)
    print("--> response:\n", response)
    print('done')

    llm_rknn.release()
