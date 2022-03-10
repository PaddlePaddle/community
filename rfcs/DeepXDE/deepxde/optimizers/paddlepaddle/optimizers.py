__all__ = ["get", "is_external_optimizer"]

import paddle

from ..config import LBFGS_options


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        # see issue https://github.com/PaddlePaddle/Paddle/issues/38444
        raise ValueError("PaddlePaddle has no support.")

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        # TODO: learning rate decay
        raise NotImplementedError(
            "learning rate decay to be implemented for backend paddle.")
    if optimizer == "adam":
        return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=params)
    raise NotImplementedError(
        f"{optimizer} to be implemented for backend paddlepaddle.")
