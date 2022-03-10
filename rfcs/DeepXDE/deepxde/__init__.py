__all__ = [
    "backend",
    "callbacks",
    "data",
    "geometry",
    "grad",
    "icbc",
    "nn",
    "postprocessing",
    "utils",
    "Model",
    "Variable",
]

from .__about__ import __version__

# Should import backend before importing anything else
from . import backend

from . import callbacks
from . import data
from . import geometry
from . import gradients as grad
from . import icbc
from . import nn
from . import postprocessing
from . import utils

from .backend import Variable
from .model import Model
from .postprocessing import saveplot

# Backward compatibility
from .icbc import (
    DirichletBC,
    NeumannBC,
    OperatorBC,
    PeriodicBC,
    RobinBC,
    PointSetBC,
    IC,
)

maps = nn
