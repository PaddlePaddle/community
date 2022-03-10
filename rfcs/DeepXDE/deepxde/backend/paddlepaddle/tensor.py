"""paddle backend implementation"""
import paddle


if paddle.device.is_compiled_with_cuda():
    paddle.device.set_device("gpu")

lib = paddle


def data_type_dict():
    return {
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "bool": paddle.bool,
    }


def is_tensor(obj):
    return paddle.is_tensor(obj)


def shape(input_tensor):
    return list(input_tensor.shape)


def ndim(input_tensor):
    # TODO needs improvement
    if input_tensor.ndim == 1:
        # There are no zero-dimensional tensors in paddlepaddle
        return 0
    return input_tensor.ndim


def Variable(initial_value, dtype=None):
    return paddle.to_tensor(initial_value, dtype=dtype, stop_gradient=False)


def as_tensor(data, dtype=None):
    if isinstance(data, paddle.Tensor):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype=dtype)
    return paddle.to_tensor(data, dtype=dtype)


def from_numpy(np_array):
    return paddle.to_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def elu(x):
    return paddle.nn.functional.elu(x)


def relu(x):
    return paddle.nn.functional.relu(x)


def selu(x):
    return paddle.nn.functional.selu(x)


def sigmoid(x):
    return paddle.nn.functional.sigmoid(x)


def silu(x):
    return paddle.nn.functional.silu(x)


def sin(x):
    return paddle.sin(x)


def square(x):
    return paddle.square(x)


def tanh(x):
    return paddle.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return paddle.mean(input_tensor, dim, keepdim=keepdims)


def reduce_mean(input_tensor):
    return paddle.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return paddle.sum(input_tensor, dim, keepdim=keepdims)


def reduce_sum(input_tensor):
    return paddle.sum(input_tensor)


def zeros(shape, dtype):
    return paddle.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return paddle.zeros_like(input_tensor)
