import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from .tensor import *  # pylint: disable=redefined-builtin
