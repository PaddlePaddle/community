import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_grq_input_embeds_dataset
from transformers import AutoConfig
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration # 增加 num_logits_to_keep 输入

prompt = "RKLLM"
chat_context = {
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "image",},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}

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

img_h, img_w, grid_thw = load_config('../vision/vision_config.json')
print(f"Using img_h={img_h}, img_w={img_w}, grid_thw={grid_thw}")

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

def gen_paddelocr_vl_quantize_dataset(model_path, model, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import AutoProcessor, AutoTokenizer
    import shutil
    import json
    from tqdm import tqdm
    from PIL import Image
    from transformers.image_utils import load_image

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i, data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            image_name = data["image"].split(".")[0]
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = load_image(imgp)

            old_w, old_h = image.size
            new_w, new_h = find_best_size(old_w, old_h, target_size=img_h//14//2*img_w//14//2)
            print("\nimage size: src({},{}) -> new({},{})".format(old_w, old_h, new_w, new_h))
            image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": data["input"]}
                    ],
                }
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = processor(image, text=text, return_tensors="pt", format=True)
            inputs = inputs.to(model.device)
            inputs_embeds = embed_layer(inputs["input_ids"])
            pixel_values = inputs["pixel_values"].unsqueeze(0)
            siglip_position_ids = list()
            image_grid_hws = list()
            sample_indices = list()
            cu_seqlens = [0]
            for idx, thw in enumerate(inputs["image_grid_thw"]):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                numel = np.prod(thw_tuple)
                image_grid_hws.append(thw_tuple)
                image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                siglip_position_ids.append(image_position_ids)
                sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                cu_seqlens.append(cu_seqlens[-1] + numel)
            siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
                pixel_values.device
            )
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(
                pixel_values.device
            )
            sample_indices = torch.concat(sample_indices, dim=0).to(
                pixel_values.device
            )
            vision_outputs = model.visual(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_hws,
                position_ids=siglip_position_ids,
                vision_return_embed_list=True,
                interpolate_pos_encoding=True,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            image_embeds = vision_outputs.last_hidden_state
            image_embeds = model.mlp_AR(image_embeds, inputs["image_grid_thw"])
            image_embeds = torch.cat(image_embeds, dim=0)
            mask = inputs["input_ids"] == model.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            print("image_embeds: ", image_embeds.shape)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = inputs_embeds.float()
            print("inputs_embeds: ", inputs_embeds.shape)

            n_token = inputs_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, inputs_embeds.cpu().detach().numpy())
            
            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')
            f.write('\n')

    if grq_data:
        grq_data_path = '../../data/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-VL llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="PaddleOCR-VL/PaddleOCR-VL-0.9B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/PaddleOCR-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)
    
    if args.quant:
        if torch.cuda.is_available():
            # 量化数据需要调整
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            model = PaddleOCRVLForConditionalGeneration.from_pretrained(args.model_path, trust_remote_code=True, device_map='cuda')
            gen_paddelocr_vl_quantize_dataset(args.model_path, model.eval(), 
                model.model.embed_tokens, 
                '../../../../datasets/OmniDocBench_ROI/llm/dataset.json', 
                '../../data/llm/dataset.txt', '../../data/llm/dataset_np', True)
            if grq_quantize(args.model_path, '../../data/llm/grq_inputs.json', grq_model_path, group=32) == True:
                args.model_path = grq_model_path
                print("GRQ quantization success!")
            else:
                print("GRQ quantization failed!")
                exit(1)
        else:
            print("cuda is unavailable, ignore the '--quant' parameter!")

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    if args.load_weight:
        kwargs['config'] = config
        if not torch.cuda.is_available():
            dev = 'cpu'
        else:            
            dev = 'cuda'
        model = PaddleOCRVLForConditionalGeneration.from_pretrained(args.model_path, **kwargs).to(dev)
        if args.quan_dataset and not args.quant:
            gen_paddelocr_vl_quantize_dataset(args.model_path, model.eval(), model.model.embed_tokens, 
                                              '../../../../datasets/OmniDocBench_ROI/llm/dataset.json', 
                                              '../../data/llm/dataset.txt', '../../data/llm/dataset_np')
    else:
        kwargs.pop('trust_remote_code', True)
        model = PaddleOCRVLForConditionalGeneration._from_config(config, **kwargs)

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
            os.makedirs(export_llm_dirname)

    # Export llm to onnx
    model.to('cpu')
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)

