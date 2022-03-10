import os
import random

import numpy as np

from .backend import backend_name, tf, torch, paddle
from .real import Real


real = Real(32)


def default_float():
    """Returns the default float type, as a string."""
    if real.precision == 64:
        return "float64"
    return "float32"


def set_default_float(value):
    """Sets the default float type.

    The default floating point type is 'float32'.

    Args:
        value (String): 'float32' or 'float64'.
    """
    if value == "float32":
        print("Set the default float type to float32")
        real.set_float32()
    elif value == "float64":
        print("Set the default float type to float64")
        real.set_float64()
    if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
        tf.keras.backend.set_floatx(value)


def set_random_seed(seed):
    """Set the global random seeds of random, numpy, and backend.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)  # python random
    np.random.seed(seed)  # numpy
    if backend_name == "tensorflow.compat.v1":
        tf.set_random_seed(seed)  # tf CPU seed
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    elif backend_name == "tensorflow":
        tf.random.set_seed(seed)  # tf CPU seed
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    elif backend_name == "pytorch":
        torch.manual_seed(seed)
    elif backend_name == "paddlepaddle":
        paddle.seed(seed)
