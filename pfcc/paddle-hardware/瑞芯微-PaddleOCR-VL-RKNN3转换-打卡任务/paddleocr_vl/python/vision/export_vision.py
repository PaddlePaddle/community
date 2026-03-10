import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
import torch.nn as nn
from transformers.utils import torch_int
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import save_config
from py_utils.tools import clear_llm_external_weight_in_dir
from modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoConfig
from abc import ABC
from typing import Optional, Tuple
from torchvision import transforms

def interpolate_pos_encoding(
        position_embedding_model,
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        num_positions = position_embedding_model.weight.shape[0]
        patch_pos_embed = position_embedding_model.weight.unsqueeze(0)
        dim = 1152
        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // 14
            new_width = width // 14
        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed.detach().numpy().tofile(f"../../model/vision/position_embedding_model.bin")
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

def get_rope_emb(image_grid_thw, rotary_pos_emb):
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
    rope_emb_max_grid = rotary_pos_emb(max_grid_size)
    rope_emb = rope_emb_max_grid[pids].flatten(1)
    rope_emb = rope_emb.repeat(1, 2)
    return rope_emb

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

def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform

class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError
    
class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

def export_vision(vlm, args):
    class paddle_vl_vision(torch.nn.Module):
        def __init__(self, vlm):
            super(paddle_vl_vision, self).__init__()
            self.visual = vlm.visual
            self.mlp_AR = vlm.mlp_AR
        
        def forward(self, pixel_values, position_embedding, rope_emb):
            """
            inputs:
                pixel_values: (1, patch_num, 3, 14, 14), patch_num=grid_t*grid_h*grid_w
                position_embedding: (1, patch_num, 1152)
                rope_emb: (patch_num, 72)
            outputs:
                image_embeds: (1, patch_num, 1152)
            """
            
            vision_outputs = self.visual(
                pixel_values=pixel_values,
                vision_return_embed_list=True,
                interpolate_pos_encoding=False,
                position_embedding=position_embedding,
                rope_emb=rope_emb,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            image_embeds = vision_outputs.last_hidden_state[0]
            image_embeds = self.mlp_AR.pre_norm(image_embeds)
            return image_embeds

    in_h = args.img_h
    in_w = args.img_w
    model = paddle_vl_vision(vlm)
    from PIL import Image
    patch_size = 14    
    save_config("vision_config.json", in_h, in_w, patch_size)
    
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    def make_quantized_dataset(model, image_list):
        dataset = []
        for image_path in image_list:
            pixel = Image.open(image_path).convert("RGB")
            old_w, old_h = pixel.size
            new_w, new_h = find_best_size(old_w, old_h, target_size=in_h//patch_size//2*in_w//patch_size//2)
            print("image size: src({},{}) -> new({},{})".format(old_w, old_h, new_w, new_h))
            pixel = pixel.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
            image = np.array(pixel)
            pixel_values = image_transform(image).to(torch.float32)
            grid_t, grid_h, grid_w = 1, new_h//patch_size, new_w//patch_size
            patches = pixel_values.reshape(
                grid_t, 1, 3, grid_h, patch_size, grid_w, patch_size
            ).numpy()
            patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, 3, patch_size, patch_size
            )
            pixel_values = torch.from_numpy(flatten_patches)
            pixel_values = pixel_values.type(torch.float32)
            pixel_values = pixel_values.unsqueeze(0)

            grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64)
            position_embedding = interpolate_pos_encoding(model.visual.vision_model.embeddings.position_embedding, grid_h, grid_w, True)
            rope_emb = get_rope_emb(grid_thw, model.visual.vision_model.encoder.rotary_pos_emb)

            dataset.append((pixel_values, position_embedding, rope_emb, grid_thw))
        return dataset
        
    pixel_values, position_embedding, rope_emb, grid_thw = make_quantized_dataset(model, ["../../data/vision/test.png"])[0]

    print("grid_thw: ", grid_thw)
    print("pixel_values: ", pixel_values.shape)
    print("position_embedding: ", position_embedding.shape)
    print("rope_emb: ", rope_emb.shape)
    np.save("../../model/vision/flatten.npy", pixel_values.detach().numpy())
    np.save("../../model/vision/rope.npy", rope_emb.detach().numpy())
    np.save("../../model/vision/position.npy", position_embedding.detach().numpy())

    vision_feature = model(pixel_values, position_embedding, rope_emb)
    print("vision output: ", vision_feature.shape)
    np.save("../../model/vision/vision_feature.npy", vision_feature.detach().numpy())
    
    dataset_for_mlpar = []
    if args.quan_dataset:
        with open(args.dataset, 'r', encoding='utf-8') as f:
            dataset_list = f.readlines()
            for i in range(len(dataset_list)):
                dataset_list[i] = dataset_list[i].strip()
                dir_path = os.path.dirname(args.dataset)
                dataset_list[i] = os.path.join(dir_path, dataset_list[i])
        dataset = make_quantized_dataset(model, dataset_list)
        data_lines = []
        os.makedirs("../../data/vision/vision/", exist_ok=True)
        for i, data in enumerate(dataset):
            pixel_values_, position_embedding_, rope_emb_, grid_thw_ = data
            np.save(f"../../data/vision/vision/flatten_{i}.npy", pixel_values_.detach().numpy())
            np.save(f"../../data/vision/vision/rope_{i}.npy", rope_emb_.detach().numpy())
            np.save(f"../../data/vision/vision/position_{i}.npy", position_embedding_.detach().numpy())
            np.save(f"../../data/vision/vision/grid_thw_{i}.npy", grid_thw_.detach().numpy())
            vision_feature_ = model(pixel_values_, position_embedding_, rope_emb_)
            np.save(f"../../data/vision/vision/vision_feature_{i}.npy", vision_feature_.detach().numpy())
            data_lines.append(f"vision/flatten_{i}.npy vision/position_{i}.npy vision/rope_{i}.npy")
            dataset_for_mlpar.append((grid_thw_, vision_feature_))
        with open("../../data/vision/datasets_vision.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(data_lines))

    torch.onnx.export(model,
                     (pixel_values, position_embedding, rope_emb),
                     args.export_vision_path,
                     input_names=['pixel', 'position_embedding', 'rope_emb'],
                     output_names=['vision_output'], 
                    #  dynamic_axes={
                    #     'pixel': {1: 'patch_num'},
                    #     'position_embedding': {1: 'patch_num'},
                    #     'rope_emb': {0: 'patch_num'},
                    #     'vision_output': {0: 'patch_num'}
                    #  },
                     opset_version=19)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")
    return vision_feature, grid_thw, dataset_for_mlpar


def export_mlp_AR(vlm, image_embeds, grid_thw, dataset, args):
    class mlp_AR(torch.nn.Module):
        def __init__(self, vlm):
            super(mlp_AR, self).__init__()
            self.mlp_AR = vlm.mlp_AR
        
        def forward(self, image_embeds):
            """
            inputs:
                image_embeds: (token_num, 4608), token_num=patch_num/(m1*m2)
            outputs:
                vision_output: (token_num, 1024)
            """
            hidden_states = self.mlp_AR.linear_1(image_embeds)
            hidden_states = self.mlp_AR.act(hidden_states)
            hidden_states = self.mlp_AR.linear_2(hidden_states)
            return hidden_states

    model = mlp_AR(vlm)
    m1 = 2
    m2 = 2
    
    def make_quantized_dataset(dataset):
        new_dataset = []
        for data in dataset:
            thw, vision_embed = data
            t, h, w = thw[0].tolist()
            from einops import rearrange
            image_feature = rearrange(
                vision_embed,
                "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                t=t,
                h=h // m1,
                p1=m1,
                w=w // m2,
                p2=m2,
            )
            new_dataset.append(image_feature)
        return new_dataset

    image_feature = make_quantized_dataset([(grid_thw, image_embeds)])[0]
    print("mlp_AR input: ", image_feature.shape)
    np.save("../../model/vision/mlp_input.npy", image_feature.detach().numpy())
    vision_embed = model(image_feature)
    print("mlp_AR output: ", vision_embed.shape)
    np.save("../../model/vision/vision_embed.npy", vision_embed.detach().numpy())
    
    if args.quan_dataset:
        quantized_dataset = make_quantized_dataset(dataset)
        data_lines = []
        os.makedirs("../../data/vision/mlpar/", exist_ok=True)
        for i, image_feature_ in enumerate(quantized_dataset):
            np.save(f"../../data/vision/mlpar/mlp_input_{i}.npy", image_feature_.detach().numpy())
            vision_embed_ = model(image_feature_)
            np.save(f"../../data/vision/mlpar/vision_embed_{i}.npy", vision_embed_.detach().numpy())
            data_lines.append(f"mlpar/mlp_input_{i}.npy")
        with open("../../data/vision/datasets_mlpar.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(data_lines))
    
    torch.onnx.export(model,
                     (image_feature,),
                     args.export_mlp_AR_path,
                     input_names=['image_embeds'],
                     output_names=['vision_output'], 
                     dynamic_axes={
                        'image_embeds': {0: 'token_num'}
                     },
                     opset_version=19)
    print(f"Exported to {os.path.abspath(args.export_mlp_AR_path)}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export paddle-vl vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="PaddleOCR-VL/PaddleOCR-VL-0.9B/")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/PaddleOCR-vision.onnx")
    parser.add_argument("--export_mlp_AR_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/PaddleOCR-vision-mlp_AR.onnx")
    parser.add_argument("--dataset", type=str, help="model quantization dataset list", required=False, default="../../../../datasets/OmniDocBench_ROI/vision/datasets.txt")
    parser.add_argument("--quan_dataset", action="store_true", help="whether generate quantization dataset")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=504)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=504)
    args = parser.parse_args()

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    if args.load_weight:
        # kwargs['config'] = config
        model = PaddleOCRVLForConditionalGeneration.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.float32)
        model = model.eval()
    else:
        kwargs.pop('trust_remote_code', True)
        model = PaddleOCRVLForConditionalGeneration._from_config(config, **kwargs)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
            os.makedirs(export_vision_dirname)

    # export vision model
    image_embeds, thw, dataset = export_vision(model, args) # 添加grid_thw对输入图片的patch

    export_mlp_AR(model, image_embeds, thw, dataset, args)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)

