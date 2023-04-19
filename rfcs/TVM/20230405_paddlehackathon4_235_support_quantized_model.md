# RFC of PaddlePaddle Hackathon 4 Task 235

## Solution name

Adding support for PaddleSlim quantized model to TVM PaddlePaddle front end.

## Description

At present, the PaddlePaddle Frontend in TVM already supports 100+ operators, but it still does not support the PaddleSlim quantization model. This solution will increase TVM's support for the PaddleSlim model and verify the quantified performance improvement.
Specifically, this solution plans to support the following two operators.

 - dequantize_linear
 - quantize_linear

## Workflow

 1. Set up the TVM development environment.
 1. For reference, investigate ONNX [quantization operator definition](https://github.com/onnx/onnx/blob/main/docs/Operators.md) and [quantization operator conversion](https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/onnx.py).
 1. In frontend source [paddlepaddle.py](https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/paddlepaddle.py), implement the conversion of linear_quantize.
 1. In the same file, implement the conversion of dequantize_linear.
 1. Test with quantitative models.

## Results

TVM can support following PaddleSlim quantization model.

 - [resnet50_vd_ptq](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)
 - [mobilenetv1_ssld_ptq](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)
 - [PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new.tar)

## Project Timeline

2023/4/05 Submitting RFC  
2023/4/20 Creating PR  
2023/4/30 Merging PR

## Your experience in ML and DL

I'm a PPDE, and I completed three tasks in the first [PaddlePaddle Hackathon](https://github.com/PaddlePaddle/Paddle/issues/35940).
I have experience with model deployment.
