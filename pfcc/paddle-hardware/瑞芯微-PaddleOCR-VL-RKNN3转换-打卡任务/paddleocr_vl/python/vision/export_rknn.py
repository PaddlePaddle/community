import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../model/vision/PaddleOCR-vision.onnx'
RKNN_MODEL = '../../model/vision/PaddleOCR-vision.rknn'
MLPAR_ONNX_MODEL = '../../model/vision/PaddleOCR-vision-mlp_AR.onnx'
MLPAR_RKNN_MODEL = '../../model/vision/PaddleOCR-vision-mlp_AR.rknn'
DATASET_PATH = '../../data/vision/datasets_vision.txt'
MLPAR_DATASET_PATH = '../../data/vision/datasets_mlpar.txt'
QUANTIZED = True
TARGET_PLATFORM = None
DEVICE_ID = None
CORE_MASK = 0xff
IS_ACCURACY = False

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

img_h, img_w, grid_thw = load_config('vision_config.json')
print(f"Using img_h={img_h}, img_w={img_w}, grid_thw={grid_thw}")

patch_size = 14

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="paddle vl vision convert rknn") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--mlpar_onnx_path", type=str, help="onnx model path", required=False, default=MLPAR_ONNX_MODEL)
    parser.add_argument("--mlpar_rknn_path", type=str, help="output rknn model path", required=False, default=MLPAR_RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--mlpar_dataset_path", type=str, help="model quantization dataset path", required=False, default=MLPAR_DATASET_PATH)
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", help="open dynamic mode")
    parser.add_argument("--accuracy", dest="accuracy", action="store_true", help="open accuracy analysis", default=IS_ACCURACY)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    grid_h = grid_thw[0][1]
    grid_w = grid_thw[0][2]
    patch_num = int(grid_h * grid_w)
    if args.dynamic:
        dynamic_input = [
            [ [1, 576, 3, 14, 14], [1, 576, 1152], [576, 72], [2] ],             # 336 - 144
            [ [1, 784, 3, 14, 14], [1, 784, 1152], [784, 72], [2] ],             # 392 - 196
            [ [1, 1024, 3, 14, 14], [1, 1024, 1152], [1024, 72], [2] ],      # 448 - 256
            [ [1, 1296, 3, 14, 14], [1, 1296, 1152], [1296, 72], [2] ],      # 504 - 324
            [ [1, 5184, 3, 14, 14], [1, 5184, 1152], [5184, 72], [2] ],      # 1008 - 1296
        ]
        inputs = None
        input_size_list = None
        input_initial_val = None
    else:
        dynamic_input = None
        inputs = ['pixel', "position_embedding", "rope_emb"]
        input_size_list = [[1, patch_num, 3, patch_size, patch_size], [1, patch_num, 1152], [patch_num, 72]]
        input_initial_val = [None, None, None]

    print('--> config model')
    rknn.config(target_platform='rk1820', core_num=8,
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
                dynamic_input=dynamic_input,
                profile_mode=True # 逐层dump时需要设置为True
                )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_path,
                    inputs=inputs, input_size_list =input_size_list, input_initial_val=input_initial_val
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    rknn.build(do_quantization=QUANTIZED, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn_path, save_ctx=True)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    
    if IS_ACCURACY:
        rknn.init_runtime(target=TARGET_PLATFORM, device_id=DEVICE_ID, core_mask=CORE_MASK)
        
        pixel = np.load("../../model/vision/flatten.npy")
        position_embedding = np.load("../../model/vision/position.npy")
        rope_emb = np.load("../../model/vision/rope.npy")
        rknn.accuracy_analysis(inputs=[pixel, position_embedding, rope_emb], core_mask=CORE_MASK,
                            target=TARGET_PLATFORM, device_id=DEVICE_ID,
                            output_dir="./vision_snapshot")
    
    rknn.release()
    
    #================= MLP-AR model =================#
    rknn = RKNN(verbose=True)
    
    if args.dynamic:
        dynamic_input = [
            [ [144, 4608] ],
            [ [196, 4608] ],
            [ [256, 4608] ],
            [ [324, 4608] ],
            [ [1296, 4608] ]
        ]
        inputs = None
        input_size_list = None
        input_initial_val = None
    else:
        token_num = patch_num // 4
        dim = 1152 * 4
        dynamic_input = None
        inputs = ["image_embeds"]
        input_size_list = [[token_num, dim]]
        input_initial_val = [None]
        
    print('--> config model')
    rknn.config(target_platform='rk1820', core_num=8,
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
                dynamic_input=dynamic_input,
                profile_mode=True # 逐层dump时需要设置为True
                )
    print('done')
    
    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.mlpar_onnx_path,
                    inputs=inputs, input_size_list =input_size_list, input_initial_val=input_initial_val
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    rknn.build(do_quantization=QUANTIZED, dataset=args.mlpar_dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.mlpar_rknn_path, save_ctx=True)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    if IS_ACCURACY:
        rknn.init_runtime(target=TARGET_PLATFORM, device_id=DEVICE_ID, core_mask=CORE_MASK)
        
        image_embeds = np.load("../../model/vision/mlp_input.npy")
        rknn.accuracy_analysis(inputs=[image_embeds], core_mask=CORE_MASK, 
                            target=TARGET_PLATFORM, device_id=DEVICE_ID,
                            output_dir="./mlpar_snapshot")
    
    rknn.release()


