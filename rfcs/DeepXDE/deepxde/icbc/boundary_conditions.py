"""Boundary conditions."""

__all__ = [
    "BC",
    "DirichletBC",
    "NeumannBC",
    "RobinBC",
    "PeriodicBC",
    "OperatorBC",
    "PointSetBC",
]

import numbers
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from .. import backend as bkd
from .. import config
from .. import gradients as grad
from .. import utils
from ..backend import backend_name


class BC(ABC):
    """Boundary condition base class.

    Args:
        geom: A ``deepxde.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array(
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

        self.boundary_normal = npfunc_range_autocache(
            utils.return_tensor(self.geom.boundary_normal)
        )

    def filter(self, X):
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end)
        return bkd.sum(dydx * n, 1, keepdims=True)

    @abstractmethod
    def error(self, X, inputs, outputs, beg, end):
        """Returns the loss."""


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X, beg, end)
        if bkd.ndim(values) > 0 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC func should return an array of shape N by 1 for a single"
                " component. Use argument 'component' for different components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X, beg, end)
        return self.normal_derivative(X, inputs, outputs, beg, end) - values


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component : self.component + 1]
            yright = outputs[mid:end, self.component : self.component + 1]
        else:
            dydx = grad.jacobian(outputs, inputs, i=self.component, j=self.component_x)
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        return yleft - yright


class OperatorBC(BC):
    """General operator boundary conditions: func(inputs, outputs, X) = 0.

    Args:
        geom: ``Geometry``.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors, respectively;
            `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
    """

    def __init__(self, geom, func, on_boundary):
        super().__init__(geom, on_boundary, 0)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.func(inputs, outputs, X)[beg:end]


class PointSetBC:
    """Dirichlet boundary condition for a set of points.
    Compare the output (that associates with `points`) with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are known and used for training.
        values: An array of values that gives the exact solution of the problem.
        component: The output component satisfying this BC.
    """

    def __init__(self, points, values, component=0):
        self.points = np.array(points, dtype=config.real(np))
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError(
                "PointSetBC should output 1D values. Use argument 'component' for different components."
            )
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.component = component

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end):
        return outputs[beg:end, self.component : self.component + 1] - self.values


def npfunc_range_autocache(func):
    """Call a NumPy function on a range of the input ndarray.

    If the backend is pytorch, the results are cached based on the id of X.
    """
    # For some BCs, we need to call self.func(X[beg:end]) in BC.error(). For backend
    # tensorflow.compat.v1/tensorflow, self.func() is only called once in graph mode,
    # but for backend pytorch, it will be recomputed in each iteration. To reduce the
    # computation, one solution is that we cache the results by using @functools.cache
    # (https://docs.python.org/3/library/functools.html). However, numpy.ndarray is
    # unhashable, so we need to implement a hash function and a cache function for
    # numpy.ndarray. Here are some possible implementations of the hash function for
    # numpy.ndarray:
    # - xxhash.xxh64(ndarray).digest(): Fast
    # - hash(ndarray.tobytes()): Slow
    # - hash(pickle.dumps(ndarray)): Slower
    # - hashlib.md5(ndarray).digest(): Slowest
    # References:
    # - https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/16592241#16592241
    # - https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary/47922199
    # Then we can implement a cache function or use memoization
    # (https://github.com/lonelyenvoy/python-memoization), which supports custom cache
    # key. However, IC/BC is only for dde.data.PDE, where the ndarray is fixed. So we
    # can simply use id of X as the key, as what we do for gradients.

    cache = {}

    @wraps(func)
    def wrapper_nocache(X, beg, end):
        return func(X[beg:end])

    @wraps(func)
    def wrapper_cache(X, beg, end):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]

    if backend_name in ["tensorflow.compat.v1", "tensorflow", "paddlepaddle"]:
        return wrapper_nocache
    if backend_name == "pytorch":
        return wrapper_cache
