import logging
import tensorflow as tf
from eight_mile.utils import get_version, exporter
from eight_mile.optz import create_lr_scheduler, register_lr_scheduler
from eight_mile.optz import (
    LearningRateScheduler,
    ConstantScheduler,
    WarmupLearningRateScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
    InverseTimeDecayScheduler,
    ExponentialDecayScheduler,
    CompositeLRScheduler,
)

logger = logging.getLogger('mead.layers')
__all__ = []
export = exporter(__all__)


if get_version(tf) < 2:

    @register_lr_scheduler('default')
    class ConstantSchedulerTensorFlow1:
        def __init__(self, **kwargs):
            pass

        def __call__(self, lr, global_step):
            return tf.identity(lr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('warmup_linear')
    class WarmupLinearSchedulerTensorFlow1(WarmupLearningRateScheduler):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, lr, global_step):
            return tf.identity(tf.minimum(1.0, tf.cast(global_step / self.warmup_steps, dtype=tf.float32)) * lr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"

    @register_lr_scheduler('clr')
    class CyclicLRSchedulerTensorFlow1:
        def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
            super().__init__()
            self.max_lr = max_lr
            self.decay_steps = decay_steps

        def __call__(self, lr, global_step):
            gs_f = tf.cast(global_step, tf.float32)
            cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
            x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
            clr = lr + (self.max_lr - lr) * tf.maximum(0., 1. - x)
            return tf.identity(clr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('sgdr')
    class SGDRSchedulerTensorFlow1:
        def __init__(self, first_decay_steps=1000, **kwargs):
            super().__init__()
            self.first_decay_steps = first_decay_steps

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.cosine_decay_restarts(lr, global_step, first_decay_steps=self.first_decay_steps), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('composite')
    class CompositeLRSchedulerTensorFlow1:
        def __init__(self, warm=None, rest=None, **kwargs):
            self.warm = warm
            self.rest = rest

        def __call__(self, lr, global_step):
            warm_tensor = self.warm(lr, global_step)

            def call_warm():
                return warm_tensor

            rest_step = tf.subtract(global_step, tf.compat.v1.constant(self.warm.warmup_steps, dtype=global_step.dtype))
            rest_tensor = self.rest(lr, rest_step)

            def call_rest():
                return rest_tensor

            return tf.identity(tf.cond(
                global_step < self.warm.warmup_steps,
                call_warm, call_rest
            ), name='lr')

        def __str__(self):
            return "LRScheduler({}, {})".format(self.warm, self.rest)

    @register_lr_scheduler('piecewise')
    class PiecewiseDecaySchedulerTensorFlow(object):

        def __init__(self, boundaries=None, values=None, **kwargs):
            super(PiecewiseDecaySchedulerTensorFlow, self).__init__()
            self.boundaries = boundaries
            self.values = values

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.piecewise_constant(global_step, self.boundaries, self.values), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('invtime')
    class InverseTimeDecaySchedulerTensorFlow:
        def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
            super().__init__()
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.staircase = staircase

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.inverse_time_decay(lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('exponential')
    class ExponentialDecaySchedulerTensorFlow(object):
        def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.staircase = staircase

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.exponential_decay(lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('zaremba')
    class ZarembaDecaySchedulerTensorFlow1(PiecewiseDecaySchedulerTensorFlow):
        """Utility only, just to simplify the JSON"""
        def __init__(self, boundaries=None, decay_rate=None, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            values = [lr/(float(decay_rate)**i) for i in range(len(boundaries)+1)]
            super().__init__(boundaries=boundaries, values=values)

        def __str__(self):
            return type(self).__name__ + "()"


else:

    @register_lr_scheduler('default')
    class ConstantSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, global_step):
            return self.lr

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('warmup_linear')
    class WarmupLinearSchedulerTensorFlow2(WarmupLearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            kwargs['lr'] = lr
            super().__init__(**kwargs)

        def __call__(self, global_step):
            return tf.minimum(1.0, global_step / float(self.warmup_steps)) * self.lr

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('clr')
    class CyclicLRSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            kwargs['lr'] = lr
            super().__init__(**kwargs)
            self.max_lr = max_lr
            self.decay_steps = decay_steps

        def __call__(self, global_step):
            gs_f = tf.cast(global_step, tf.float32)
            cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
            x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
            clr = self.lr + (self.max_lr - self.lr) * tf.maximum(0., 1. - x)
            return tf.identity(clr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"

    @register_lr_scheduler('sgdr')
    class SGDRSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, first_decay_steps=1000, **kwargs):
            super().__init__(**kwargs)
            self.first_decay_steps = first_decay_steps

        def __call__(self, global_step):
            return tf.identity(tf.compat.v1.train.cosine_decay_restarts(self.lr, global_step, first_decay_steps=self.first_decay_steps), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('zaremba')
    class ZarembaDecaySchedulerTensorFlow2(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
        """Utility only, just to simplify the JSON"""
        def __init__(self, boundaries=None, decay_rate=None, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            values = [lr/(float(decay_rate)**i) for i in range(len(boundaries)+1)]
            super().__init__(boundaries, values, kwargs.get('name'))


    @register_lr_scheduler('composite')
    class CompositeLRSchedulerTensorFlow2(CompositeLRScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
        pass

    @register_lr_scheduler('piecewise')
    class PiecewiseConstantDecayTensorFlow2(tf.keras.optimizers.schedules.PiecewiseConstantDecay):

        def __init__(self, boundaries, values, **kwargs):
            super().__init__(boundaries, values)

    @register_lr_scheduler('invtime')
    class InverseTimeDecayTensorFlow2(tf.keras.optimizers.schedules.InverseTimeDecay):

        def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
            lr = kwargs.get('lr', kwargs.get('eta', 0.01))
            super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get('name'))

    @register_lr_scheduler('exponential')
    class ExponentialDecayTensorFlow2(tf.keras.optimizers.schedules.ExponentialDecay):
        def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
            lr = kwargs.get('lr', kwargs.get('eta', 0.01))
            super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get('name'))


@export
class AdamWOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.

    Modified from: https://github.com/google-research/bert/blob/master/optimization.py
    This does the weight decay slightly differently from PyTorch version, putting it before the update
    """

    def __init__(self,
                 learning_rate,
                 weight_decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 name="AdamWOptimizer"):
        """Constructs a AdamWOptimizer."""
        super(AdamWOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def _get_variable_name(self, param_name):
        import re
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            update += self.weight_decay * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)


@export
def optimizer(loss_fn, **kwargs):

    global_step = tf.train.get_or_create_global_step()
    clip = kwargs.get('clip', None)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('lr', kwargs.get('eta', 0.01))
    lr_scheduler = create_lr_scheduler(**kwargs)
    decay_fn = None
    colocate_gradients_with_ops = bool(kwargs.get('colocate_gradients_with_ops', False))
    sgd_mom = float(kwargs.get('mom', 0.9))
    if optim == 'adadelta':
        rho = float(kwargs.get('rho', 0.95))
        eps = float(kwargs.get('epsilon', 1e-6))
        logger.info('adadelta(eta=%f, rho=%f, epsilon=%f)', eta, rho, eps)
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, rho, eps)
    elif optim == 'adam':
        beta1 = float(kwargs.get('beta1', 0.9))
        beta2 = float(kwargs.get('beta2', 0.999))
        eps = float(kwargs.get('epsilon', 1e-8))
        logger.info('adam(eta=%f beta1=%f, beta2=%f, eps=%f)', eta, beta1, beta2, eps)
        optz = lambda lr: tf.train.AdamOptimizer(lr, beta1, beta2, eps)
    elif optim == 'adamw':
        wd = float(kwargs.get('weight_decay', 0))
        beta1 = float(kwargs.get('beta1', 0.9))
        beta2 = float(kwargs.get('beta2', 0.999))
        eps = float(kwargs.get('epsilon', 1e-8))
        logger.info('adamw(eta=%f beta1=%f, beta2=%f, eps=%f)', eta, beta1, beta2, eps)
        optz = lambda lr: AdamWOptimizer(lr, wd, beta1, beta2, eps)
    elif optim == 'rmsprop':
        # Get mom again with difference default
        mom = float(kwargs.get('mom', 0.0))
        logger.info('rmsprop(eta=%f, mom=%f)', eta, mom)
        optz = lambda lr: tf.train.RMSPropOptimizer(lr, momentum=mom)
    elif sgd_mom > 0:
        logger.info('sgd-mom(eta=%f, mom=%f)', eta, sgd_mom)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, sgd_mom)
    else:
        logger.info('sgd(eta=%f)', eta)
        optz = lambda lr: tf.train.GradientDescentOptimizer(lr)

    logger.info('clip gradients at %s', clip)
    return global_step, tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz,
                                                        colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                        clip_gradients=clip, learning_rate_decay_fn=lr_scheduler,
                                                        increment_global_step=True)


@export
class EagerOptimizer(object):

    def __init__(self, loss, optimizer=None, **kwargs):
        self.loss = loss
        if 'lr_function' in kwargs:
            lr_function = kwargs['lr_function']
        else:
            if 'lr_scheduler_type' not in kwargs:
                kwargs['lr_scheduler_type'] = 'default'
            lr_function = create_lr_scheduler(**kwargs)
        #decay_fn = None
        # Right now this option is pointless since sparse updates dont work on GPU.  We just turn it off
        sgd_mom = float(kwargs.get('mom', 0.9))
        self.clip = kwargs.get('clip', 100)

        if optimizer:
            self.optimizer = optimizer
        else:
            optim = kwargs.get('optim', 'sgd')
            lr = kwargs.get('lr', kwargs.get('eta', 0.01))

            if optim == 'adadelta':
                rho = float(kwargs.get('rho', 0.95))
                eps = float(kwargs.get('epsilon', 1e-6))
                logger.info('adadelta(eta=%f, rho=%f, epsilon=%f)', lr, rho, eps)
                self.optimizer = tf.optimizers.Adadelta(lr, rho, eps)
            elif optim == 'adam':
                beta1 = float(kwargs.get('beta1', 0.9))
                beta2 = float(kwargs.get('beta2', 0.999))
                eps = float(kwargs.get('epsilon', 1e-8))
                logger.info('adam(eta=%f beta1=%f, beta2=%f, eps=%f)', lr, beta1, beta2, eps)
                self.optimizer = tf.optimizers.Adam(lr_function, beta1, beta2, eps)
            elif optim == 'adamw':
                import tensorflow_addons as tfa
                wd = float(kwargs.get('weight_decay', 0))
                beta1 = float(kwargs.get('beta1', 0.9))
                beta2 = float(kwargs.get('beta2', 0.999))
                eps = float(kwargs.get('epsilon', 1e-8))
                logger.info('adamw(eta=%f beta1=%f, beta2=%f, eps=%f)', lr, beta1, beta2, eps)
                self.optimizer = tfa.optimizers.AdamW(weight_decay=wd,
                                                      learning_rate=lr_function,
                                                      beta_1=beta1,
                                                      beta_2=beta2,
                                                      epsilon=eps)
            elif optim == 'rmsprop':
                # Get mom again with difference default
                mom = float(kwargs.get('mom', 0.0))
                logger.info('rmsprop(eta=%f, mom=%f)', lr, mom)
                self.optimizer = tf.optimizers.RMSprop(lr_function, momentum=mom)
            elif sgd_mom > 0:
                logger.info('sgd-mom(eta=%f, mom=%f)', lr, sgd_mom)
                self.optimizer = tf.optimizers.SGD(lr_function, sgd_mom)
            else:
                logger.info('sgd(eta=%f)', lr)
                self.optimizer = tf.optimizers.SGD(lr_function)

        logger.info('clip gradients at %s', self.clip)

    def update(self, model, x, y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, x, y)
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    def update_with_hidden(self, model, h, x, y):
        with tf.GradientTape() as tape:
            loss_value, h = self.loss(model, h, x, y)

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, h

