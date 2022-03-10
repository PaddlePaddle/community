"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy.special import gamma


alpha = 1.8


def fpde(x, y, int_mat):
    """\int_theta D_theta^alpha u(x)"""
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        int_mat = tf.SparseTensor(*int_mat)
        lhs = tf.sparse_tensor_dense_matmul(int_mat, y)
    else:
        lhs = tf.matmul(int_mat, y)
    lhs = lhs[:, 0]
    lhs *= gamma((1 - alpha) / 2) * gamma((2 + alpha) / 2) / (2 * np.pi ** 1.5)
    x = x[: tf.size(lhs)]
    rhs = (
        2 ** alpha
        * gamma(2 + alpha / 2)
        * gamma(1 + alpha / 2)
        * (1 - (1 + alpha / 2) * tf.reduce_sum(x ** 2, axis=1))
    )
    return lhs - rhs


def func(x):
    return (np.abs(1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (
        1 + alpha / 2
    )


geom = dde.geometry.Disk([0, 0], 1)
bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

data = dde.data.FPDE(
    geom, fpde, alpha, bc, [8, 100], num_domain=100, num_boundary=1, solution=func
)

net = dde.nn.FNN([2] + [20] * 4 + [1], "tanh", "Glorot normal")
net.apply_output_transform(
    lambda x, y: (1 - tf.reduce_sum(x ** 2, axis=1, keepdims=True)) * y
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X = geom.random_points(1000)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
