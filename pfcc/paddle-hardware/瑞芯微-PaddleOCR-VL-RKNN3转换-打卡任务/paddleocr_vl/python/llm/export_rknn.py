import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../model/llm/PaddleOCR-llm.onnx'
LLM_CONFIG = '../../model/llm/PaddleOCR-llm.config.pkl'
RKNN_MODEL = '../../model/llm/PaddleOCR-llm.rknn'
DATASET_PATH = '../../data/llm/dataset.txt'
QUANTIZED = True

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export PaddleOCR-VL llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--no_prune_mode", dest="prune_mode", action="store_false", help="close prune mode")
    parser.set_defaults(prune_mode=False)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform='rk1820', 
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
                # max_ctx_len=2048, max_position_embeddings=2048,
                )
    print('done')

    # Load model
    print('--> Loading model')
    if args.prune_mode == True:
        ret = rknn.load_llm(model=args.onnx_path, config=args.config, llm_head_target = "rk3588") # 支持两段式
    else :
        ret = rknn.load_llm(model=args.onnx_path, config=args.config)

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

    rknn.release()

