import tensorflow as tf
from baseline.train import register_lr_scheduler, create_lr_scheduler
import math


@register_lr_scheduler('default')
class ConstantSchedulerTensorFlow(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, lr, global_step):
        return lr


@register_lr_scheduler('warmup_linear')
class WarmupLinearSchedulerTensorFlow(object):

    def __init__(self, warmup_steps=16000, **kwargs):
        super(WarmupLinearSchedulerTensorFlow, self).__init__()
        self.warmup_steps = warmup_steps

    def __call__(self, lr, global_step):
        return tf.minimum(1.0, tf.cast(global_step / self.warmup_steps, dtype=tf.float32)) * lr


@register_lr_scheduler('clr')
class CyclicLRSchedulerTensorFlow(object):
    def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
        super(CyclicLRSchedulerTensorFlow, self).__init__()
        self.max_lr = max_lr
        self.decay_steps = decay_steps

    def __call__(self, lr, global_step):
        gs_f = tf.cast(global_step, tf.float32)
        cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
        x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
        clr = lr + (self.max_lr - lr) * tf.maximum(0., 1. - x)
        return clr


@register_lr_scheduler('sgdr')
class SGDRSchedulerTensorFlow(object):
    def __init__(self, first_decay_steps=1000, **kwargs):
        super(SGDRSchedulerTensorFlow, self).__init__()
        self.first_decay_steps = first_decay_steps

    def __call__(self, lr, global_step):
        return tf.train.cosine_decay_restarts(lr, global_step, first_decay_steps=self.first_decay_steps)


@register_lr_scheduler('piecewise')
class PiecewiseDecaySchedulerTensorFlow(object):

    def __init__(self, bounds=None, values=None, **kwargs):
        super(PiecewiseDecaySchedulerTensorFlow, self).__init__()
        self.bounds = bounds
        self.values = values

    def __call__(self, lr, global_step):
        return tf.train.piecewise_constant(global_step, self.bounds, self.values)


@register_lr_scheduler('zaremba')
class ZarembaDecaySchedulerTensorFlow(PiecewiseDecaySchedulerTensorFlow):
    """Utility only, just to simplify the JSON"""
    def __init__(self, bounds=None, decay_rate=None, **kwargs):
        lr = kwargs.get('lr', kwargs.get('eta', -1))
        values = [lr/(decay_rate**i) for i in range(len(bounds)+1)]
        super(ZarembaDecaySchedulerTensorFlow, self).__init__(bounds=bounds, values=values)

    def __call__(self, lr, global_step):
        return tf.train.piecewise_constant(global_step, self.bounds, self.values)


@register_lr_scheduler('invtime')
class InverseTimeDecaySchedulerTensorFlow(object):
    def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
        super(InverseTimeDecaySchedulerTensorFlow, self).__init__()
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, lr, global_step):
        return tf.train.inverse_time_decay(lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase)


@register_lr_scheduler('exponential')
class ExponentialDecaySchedulerTensorFlow(object):
    def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, lr, global_step):
        return tf.train.exponential_decay(lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase)


def optimizer(loss_fn, **kwargs):

    global_step = tf.Variable(0, trainable=False)
    clip = kwargs.get('clip', None)
    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('lr', kwargs.get('eta', 0.01))
    lr_scheduler = create_lr_scheduler(**kwargs)
    decay_fn = None
    colocate_gradients_with_ops = bool(kwargs.get('colocate_gradients_with_ops', False))

    if optim == 'adadelta':
        print('adadelta', eta)
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)
    elif optim == 'adam':
        print('adam', eta)
        optz = lambda lr: tf.train.AdamOptimizer(lr)
    elif optim == 'rmsprop':
        print('rmsprop', eta)
        optz = lambda lr: tf.train.RMSPropOptimizer(lr, momentum=mom)
    elif mom > 0:
        print('sgd-mom', eta, mom)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, mom)
    else:
        print('sgd')
        optz = lambda lr: tf.train.GradientDescentOptimizer(lr)

    print('clip', clip)
    print('decay', decay_fn)
    return global_step, tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz,
                                                        colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                        clip_gradients=clip, learning_rate_decay_fn=lr_scheduler)

