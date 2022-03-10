from . import backend as bkd
from . import config
from .backend import tf


def mean_absolute_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    return bkd.reduce_mean(bkd.square(y_true - y_pred))


def softmax_cross_entropy(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def zero(*_):
    # TODO: pytorch
    return tf.constant(0, dtype=config.real(tf))


def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    loss_identifier = {
        "mean absolute error": mean_absolute_error,
        "MAE": mean_absolute_error,
        "mae": mean_absolute_error,
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "mean absolute percentage error": mean_absolute_percentage_error,
        "MAPE": mean_absolute_percentage_error,
        "mape": mean_absolute_percentage_error,
        "softmax cross entropy": softmax_cross_entropy,
        "zero": zero,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
