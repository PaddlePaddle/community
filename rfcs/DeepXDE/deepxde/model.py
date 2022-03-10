__all__ = ["Model", "TrainState", "LossHistory"]

import pickle
from collections import OrderedDict

import numpy as np

from . import config
from . import display
from . import gradients as grad
from . import losses as losses_module
from . import metrics as metrics_module
from . import optimizers
from . import utils
from .backend import backend_name, tf, torch, jax, paddle
from .callbacks import CallbackList


class Model:
    """A ``Model`` trains a ``NN`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.nn.NN`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.batch_size = None
        self.callbacks = None
        self.metrics = None
        self.external_trainable_variables = []
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False

        # Backend-dependent attributes
        self.opt = None
        # Tensor or callable
        self.outputs = None
        self.outputs_losses = None
        self.train_step = None
        if backend_name == "tensorflow.compat.v1":
            self.sess = None
            self.saver = None
        elif backend_name == "jax":
            self.opt_state = None  # TODO: to be removed to opt module

    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String. Name of optimizer.
            lr: A Tensor or a floating point value. The learning rate. For L-BFGS, use
                `dde.optimizers.set_LBFGS_options` to set the hyperparameters.
            loss: If the same loss is used for all errors, then `loss` is a String (name
                of objective function) or objective function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay: Tuple. Name and parameters of decay to the initial learning rate. One
                of the following options:

                - `inverse time decay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: ("inverse time", decay_steps, decay_rate)
                - `cosine decay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
            external_trainable_variables: A trainable ``tf.Variable`` object or a list
                of trainable ``tf.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered. If the backend is
                tensorflow.compat.v1, `external_trainable_variables` is ignored, and all
                trainable ``tf.Variable`` objects are automatically collected.
        """
        print("Compiling model...")

        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            if backend_name == "tensorflow.compat.v1":
                print(
                    "Warning: For the backend tensorflow.compat.v1, "
                    "`external_trainable_variables` is ignored, and all trainable "
                    "``tf.Variable`` objects are automatically collected."
                )
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
            self.external_trainable_variables = external_trainable_variables

        if backend_name == "tensorflow.compat.v1":
            self._compile_tensorflow_compat_v1(lr, loss_fn, decay, loss_weights)
        elif backend_name == "tensorflow":
            self._compile_tensorflow(lr, loss_fn, decay, loss_weights)
        elif backend_name == "pytorch":
            self._compile_pytorch(lr, loss_fn, decay, loss_weights)
        elif backend_name == "jax":
            self._compile_jax(lr, loss_fn, decay, loss_weights)
        elif backend_name == "paddlepaddle":
            self._compile_paddlepaddle(lr, loss_fn, decay, loss_weights)
        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    def _compile_tensorflow_compat_v1(self, lr, loss_fn, decay, loss_weights):
        """tensorflow.compat.v1"""
        if not self.net.built:
            self.net.build()
        if self.sess is None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver(max_to_keep=None)

        # Data losses
        losses = self.data.losses(self.net.targets, self.net.outputs, loss_fn, self)
        if not isinstance(losses, list):
            losses = [losses]
        # Regularization loss
        if self.net.regularizer is not None:
            losses.append(tf.losses.get_regularization_loss())
        losses = tf.convert_to_tensor(losses)
        # Weighted losses
        if loss_weights is not None:
            losses *= loss_weights
            self.losshistory.set_loss_weights(loss_weights)
        total_loss = tf.math.reduce_sum(losses)

        # Tensors
        self.outputs = self.net.outputs
        self.outputs_losses = [self.net.outputs, losses]
        self.train_step = optimizers.get(
            total_loss, self.opt_name, learning_rate=lr, decay=decay
        )

    def _compile_tensorflow(self, lr, loss_fn, decay, loss_weights):
        """tensorflow"""

        # TODO: Avoid creating multiple graphs by using tf.TensorSpec.
        @tf.function
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        # TODO: Avoid creating multiple graphs by using tf.TensorSpec.
        @tf.function
        def outputs_losses(training, inputs, targets, auxiliary_vars):
            self.net.training = training
            self.net.inputs = inputs
            self.net.auxiliary_vars = auxiliary_vars
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = self.net(inputs, training=training)
            # Data losses
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses += [tf.math.reduce_sum(self.net.losses)]
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
                self.losshistory.set_loss_weights(loss_weights)
            return outputs_, losses

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses(True, inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))

        def train_step_tfp(
            inputs, targets, auxiliary_vars, previous_optimizer_results=None
        ):
            def build_loss():
                losses = outputs_losses(True, inputs, targets, auxiliary_vars)[1]
                return tf.math.reduce_sum(losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            return opt(trainable_variables, build_loss, previous_optimizer_results)

        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """pytorch"""

        def outputs(training, inputs):
            self.net.train(mode=training)
            with torch.no_grad():
                return self.net(torch.as_tensor(inputs))

        def outputs_losses(training, inputs, targets):
            self.net.train(mode=training)
            self.net.inputs = torch.as_tensor(inputs)
            self.net.inputs.requires_grad_()
            outputs_ = self.net(self.net.inputs)
            # Data losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # TODO: regularization
            losses = torch.stack(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= torch.as_tensor(loss_weights)
                self.losshistory.set_loss_weights(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        # Another way is using per-parameter options
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options,
        # but not all optimizers (such as L-BFGS) support this.
        trainable_variables = (
            list(self.net.parameters()) + self.external_trainable_variables
        )
        self.opt = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=lr, decay=decay
        )

        def train_step(inputs, targets):
            def closure():
                losses = outputs_losses(True, inputs, targets)[1]
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            self.opt.step(closure)

        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = train_step

    def _compile_paddlepaddle(self, lr, loss_fn, decay, loss_weights):
        """paddlepaddle"""

        def outputs(training, inputs):
            if training:
                self.net.train()
            else:
                self.net.eval()
            with paddle.no_grad():
                return self.net(paddle.to_tensor(inputs))

        def outputs_losses(training, inputs, targets):
            if training:
                self.net.train()
            else:
                self.net.eval()
            self.net.inputs = paddle.to_tensor(inputs, stop_gradient=False)
            outputs_ = self.net(self.net.inputs)
            # Data losses
            if targets is not None:
                targets = paddle.to_tensor(targets)
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # TODO: regularization
            losses = paddle.stack(losses)
            losses = losses[:, 0]
            # Weighted losses
            if loss_weights is not None:
                losses *= paddle.to_tensor(loss_weights)
                self.losshistory.set_loss_weights(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        trainable_variables = (
            list(self.net.parameters()) + self.external_trainable_variables
        )
        self.opt = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=lr, decay=decay
        )

        def train_step(inputs, targets):
            losses = outputs_losses(True, inputs, targets)[1]
            total_loss = paddle.sum(losses)
            total_loss.backward()
            self.opt.step()
            self.opt.clear_grad()

        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = train_step

    def _compile_jax(self, lr, loss_fn, decay, loss_weights):
        """jax"""
        # initialize network's parameters
        # TODO: Init should move to network module, because we don't know how to init here, e.g., DeepONet has two inputs.
        #       random seed should use a random number, or be specified by users
        key = jax.random.PRNGKey(seed=0)
        x = jax.numpy.empty(shape=[1, self.net.layer_sizes[0]])
        self.net.params = self.net.init(key, x)

        @jax.jit
        def inner_outputs(params, training, inputs):
            return self.net.apply(params, training, inputs)

        @jax.jit
        def inner_outputs_losses(params, training, inputs, targets):
            # TODO: add auxiliary vars, regularization loss, weighted losses
            outputs_ = self.net.apply(params, inputs, training=training)
            # Data losses
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            return outputs_, losses

        @jax.jit
        def inner_train_step(params, opt_state, inputs, targets):
            def loss_function(params):
                losses = inner_outputs_losses(params, True, inputs, targets)[1]
                return jax.numpy.sum(jax.numpy.stack(losses))

            grad_fn = jax.grad(
                loss_function
            )  # jax.value_and_grad seems to be slightly faster than jax.grad for function approximation
            grads = grad_fn(params)
            updates, new_opt_state = self.opt.update(grads, opt_state)
            new_params = optimizers.apply_updates(params, updates)
            return new_params, new_opt_state

        def outputs(training, inputs):
            return inner_outputs(self.net.params, training, inputs)

        def outputs_losses(training, inputs, targets):
            return inner_outputs_losses(self.net.params, training, inputs, targets)

        def train_step(inputs, targets):
            self.net.params, self.opt_state = inner_train_step(
                self.net.params, self.opt_state, inputs, targets
            )

        # TODO: add decay
        self.opt = optimizers.get(self.opt_name, learning_rate=lr)
        self.opt_state = self.opt.init(self.net.params)

        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = train_step

    def _outputs(self, training, inputs):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs)
            return self.sess.run(self.outputs, feed_dict=feed_dict)
        # tensorflow and pytorch
        outs = self.outputs(training, inputs)
        return utils.to_numpy(outs)

    def _outputs_losses(self, training, inputs, targets, auxiliary_vars):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs, targets, auxiliary_vars)
            return self.sess.run(self.outputs_losses, feed_dict=feed_dict)
        if backend_name == "tensorflow":
            outs = self.outputs_losses(training, inputs, targets, auxiliary_vars)
        elif backend_name == "pytorch":
            # TODO: auxiliary_vars
            self.net.requires_grad_(requires_grad=False)
            outs = self.outputs_losses(training, inputs, targets)
            self.net.requires_grad_()
        elif backend_name == "jax":
            # TODO: auxiliary_vars
            outs = self.outputs_losses(training, inputs, targets)
        elif backend_name == "paddlepaddle":
            outs = self.outputs_losses(training, inputs, targets)
        return utils.to_numpy(outs)

    def _train_step(self, inputs, targets, auxiliary_vars):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(True, inputs, targets, auxiliary_vars)
            self.sess.run(self.train_step, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            self.train_step(inputs, targets, auxiliary_vars)
        elif backend_name == "pytorch":
            # TODO: auxiliary_vars
            self.train_step(inputs, targets)
        elif backend_name == "jax":
            # TODO: auxiliary_vars
            self.train_step(inputs, targets)
        elif backend_name == "paddlepaddle":
            self.train_step(inputs, targets)
            
    @utils.timing
    def train(
        self,
        epochs=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of iterations to train the model. Note: It is the
                number of iterations, not the number of epochs.
            batch_size: Integer or ``None``. If you solve PDEs via ``dde.data.PDE`` or
                ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                `dde.callbacks.PDEResidualResampler
                <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEResidualResampler>`_,
                see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
            display_every: Integer. Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path: String. Path where parameters were previously saved.
                See ``save_path`` in `tf.train.Saver.restore <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore>`_.
            model_save_path: String. Prefix of filenames created for the checkpoint.
                See ``save_path`` in `tf.train.Saver.save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#save>`_.
        """
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if backend_name == "tensorflow.compat.v1":
            if self.train_state.step == 0:
                print("Initializing variables...")
                self.sess.run(tf.global_variables_initializer())
            else:
                utils.guarantee_initialized_variables(self.sess)

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test()
        self.callbacks.on_train_begin()
        if optimizers.is_external_optimizer(self.opt_name):
            if backend_name == "tensorflow.compat.v1":
                self._train_tensorflow_compat_v1_scipy(display_every)
            elif backend_name == "tensorflow":
                self._train_tensorflow_tfp()
            elif backend_name == "pytorch":
                self._train_pytorch_lbfgs()
        else:
            if epochs is None:
                raise ValueError("No epochs for {}.".format(self.opt_name))
            self._train_sgd(epochs, display_every)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def _train_sgd(self, epochs, display_every):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == epochs:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_tensorflow_compat_v1_scipy(self, display_every):
        def loss_callback(loss_train):
            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0:
                self.train_state.loss_train = loss_train
                self.train_state.loss_test = None
                self.train_state.metrics_test = None
                self.losshistory.append(
                    self.train_state.step, self.train_state.loss_train, None, None
                )
                display.training_display(self.train_state)

        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        feed_dict = self.net.feed_dict(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_step.minimize(
            self.sess,
            feed_dict=feed_dict,
            fetches=[self.outputs_losses[1]],
            loss_callback=loss_callback,
        )
        self._test()

    def _train_tensorflow_tfp(self):
        # There is only one optimization step. If using multiple steps with/without
        # previous_optimizer_results, L-BFGS failed to reach a small error. The reason
        # could be that tfp.optimizer.lbfgs_minimize will start from scratch for each
        # call.
        n_iter = 0
        while n_iter < optimizers.LBFGS_options["maxiter"]:
            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            results = self.train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )
            n_iter += results.num_iterations.numpy()
            self.train_state.epoch += results.num_iterations.numpy()
            self.train_state.step += results.num_iterations.numpy()
            self._test()

            if results.converged or results.failed:
                break

    def _train_pytorch_lbfgs(self):
        prev_n_iter = 0
        while prev_n_iter < optimizers.LBFGS_options["maxiter"]:
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            n_iter = self.opt.state_dict()["state"][0]["n_iter"]
            if prev_n_iter == n_iter:
                # Converged
                break

            self.train_state.epoch += n_iter - prev_n_iter
            self.train_state.step += n_iter - prev_n_iter
            prev_n_iter = n_iter
            self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _test(self):
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True
        display.training_display(self.train_state)

    def predict(self, x, operator=None, callbacks=None):
        """Generates predictions for the input samples. If `operator` is ``None``,
        returns the network output, otherwise returns the output of the `operator`.

        Args:
            x: The network inputs. A Numpy array or a tuple of Numpy arrays.
            operator: A function takes arguments (`inputs`, `outputs`) or (`inputs`,
                `outputs`, `auxiliary_variables`) and outputs a tensor. `inputs` and
                `outputs` are the network input and output tensors, respectively.
                `auxiliary_variables` is the output of `auxiliary_var_function(x)`
                in `dde.data.PDE`. `operator` is typically chosen as the PDE (used to
                define `dde.data.PDE`) to predict the PDE residual.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during prediction.
        """
        if isinstance(x, tuple):
            x = tuple(np.array(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.array(x, dtype=config.real(np))
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        self.callbacks.on_predict_begin()

        if operator is None:
            y = self._outputs(False, x)
            self.callbacks.on_predict_end()
            return y

        # operator is not None
        if utils.get_num_args(operator) == 3:
            aux_vars = self.data.auxiliary_var_fn(x).astype(config.real(np))
        if backend_name == "tensorflow.compat.v1":
            if utils.get_num_args(operator) == 2:
                op = operator(self.net.inputs, self.net.outputs)
                feed_dict = self.net.feed_dict(False, x)
            elif utils.get_num_args(operator) == 3:
                op = operator(
                    self.net.inputs, self.net.outputs, self.net.auxiliary_vars
                )
                feed_dict = self.net.feed_dict(False, x, auxiliary_vars=aux_vars)
            y = self.sess.run(op, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            if utils.get_num_args(operator) == 2:

                @tf.function
                def op(inputs):
                    y = self.net(inputs)
                    return operator(inputs, y)

            elif utils.get_num_args(operator) == 3:

                @tf.function
                def op(inputs):
                    y = self.net(inputs)
                    return operator(inputs, y, aux_vars)

            y = op(x)
            y = utils.to_numpy(y)
        elif backend_name == "pytorch":
            self.net.eval()
            inputs = torch.as_tensor(x)
            inputs.requires_grad_()
            outputs = self.net(inputs)
            if utils.get_num_args(operator) == 2:
                y = operator(inputs, outputs)
            elif utils.get_num_args(operator) == 3:
                # TODO: Pytorch backend Implementation of Auxiliary variables.
                # y = operator(inputs, outputs, torch.as_tensor(aux_vars))
                raise NotImplementedError(
                    "Model.predict() with auxiliary variable hasn't been implemented for backend pytorch."
                )
            y = utils.to_numpy(y)
        elif backend_name == "paddlepaddle":
            self.net.eval()
            inputs = paddle.to_tensor(x)
            inputs.stop_gradient = False
            outputs = self.net(inputs)
            if utils.get_num_args(operator) == 2:
                y = operator(inputs, outputs)
            elif utils.get_num_args(operator) == 3:
                # TODO: PaddlePaddle backend Implementation of Auxiliary variables.
                # y = operator(inputs, outputs, paddle.to_tensor(aux_vars))
                raise NotImplementedError(
                    "Model.predict() with auxiliary variable hasn't been implemented for backend paddlepaddle."
                )
            y = utils.to_numpy(y)
        self.callbacks.on_predict_end()
        return y

    # def evaluate(self, x, y, callbacks=None):
    #     """Returns the loss values & metrics values for the model in test mode."""
    #     raise NotImplementedError(
    #         "Model.evaluate to be implemented. Alternatively, use Model.predict."
    #     )

    def state_dict(self):
        """Returns a dictionary containing all variables."""
        # TODO: backend tensorflow
        if backend_name == "tensorflow.compat.v1":
            destination = OrderedDict()
            variables_names = [v.name for v in tf.global_variables()]
            values = self.sess.run(variables_names)
            for k, v in zip(variables_names, values):
                destination[k] = v
        elif backend_name == "pytorch":
            destination = self.net.state_dict()
        elif backend_name == "paddlepaddle":
            destination = self.net.state_dict()
        else:
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
        return destination

    def save(self, save_path, protocol="backend", verbose=0):
        """Saves all variables to a disk file.

        Args:
            save_path (string): Prefix of filenames to save the model file.
            protocol (string): If `protocol` is "backend", save using the backend-specific method.
                For "tensorflow.compat.v1", use `tf.train.Save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#attributes>`_.
                For "pytorch", use `torch.save <https://pytorch.org/docs/stable/generated/torch.save.html>`_.
                If `protocol` is "pickle", save using the Python pickle module.
                Only the protocol "backend" supports ``restore()``.

        Returns:
            string. Path where model is saved.
        """
        # TODO: backend tensorflow
        save_path = f"{save_path}-{self.train_state.epoch}"
        if protocol == "pickle":
            save_path += ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(self.state_dict(), f)
        elif protocol == "backend":
            if backend_name == "tensorflow.compat.v1":
                save_path += ".ckpt"
                self.saver.save(self.sess, save_path)
            elif backend_name == "pytorch":
                save_path += ".pt"
                checkpoint = {
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                }
                torch.save(checkpoint, save_path)
            elif backend_name == "paddlepaddle":
                save_path += ".pdparams"
                checkpoint = {
                    'model': self.net.state_dict(),
                    'opt': self.opt.state_dict()
                }
                paddle.save(checkpoint, save_path)
            else:
                raise NotImplementedError(
                    "Model.save() hasn't been implemented for this backend."
                )
        if verbose > 0:
            print(
                "Epoch {}: saving model to {} ...\n".format(
                    self.train_state.epoch, save_path
                )
            )
        return save_path

    def restore(self, save_path, verbose=0):
        """Restore all variables from a disk file.

        Args:
            save_path (string): Path where model was previously saved.
        """
        # TODO: backend tensorflow
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        if backend_name == "tensorflow.compat.v1":
            self.saver.restore(self.sess, save_path)
        elif backend_name == "pytorch":
            checkpoint = torch.load(save_path)
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        elif backend_name == "paddlepaddle":
            checkpoint = paddle.load(save_path)
            self.net.set_state_dict(checkpoint["model"])
            self.opt.set_state_dict(checkpoint["opt"])
        else:
            raise NotImplementedError(
                "Model.restore() hasn't been implemented for this backend."
            )

    def print_model(self):
        """Prints all trainable variables."""
        # TODO: backend tensorflow, pytorch
        if backend_name != "tensorflow.compat.v1":
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)


class TrainState:
    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.train_aux_vars = None
        self.X_test = None
        self.y_test = None
        self.test_aux_vars = None

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Test results
        self.loss_test = None
        self.y_pred_test = None
        self.y_std_test = None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train = np.inf
        self.best_loss_test = np.inf
        self.best_y = None
        self.best_ystd = None
        self.best_metrics = None

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train = X_train
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test = X_test
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf


class LossHistory:
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
