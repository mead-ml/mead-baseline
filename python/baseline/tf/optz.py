import tensorflow as tf
from baseline.train import register_lr_scheduler, create_lr_scheduler
import math


@register_lr_scheduler('piecewise')
class PiecewiseDecayScheduler(object):

    def __init__(self, bounds=None, decay_values=None, **kwargs):
        self.bounds = bounds
        self.decay_values = decay_values

    def __call__(self, lr, global_step):
        return tf.train.piecewise_constant(global_step, self.bounds, self.decay_values)


@register_lr_scheduler('staircase')
class StaircaseDecayScheduler(object):
    def __init__(self, every_n_steps=16000, decay_rate=0.5, **kwargs):
        self.every_n_steps = every_n_steps
        self.decay_rate = decay_rate

    def __call__(self, lr, global_step):
        return tf.train.exponential_decay(lr, global_step, self.every_n_steps, self.decay_rate, staircase=True)


@register_lr_scheduler('invtime')
class InverseTimeDecayScheduler(object):
    def __init__(self, every_n_steps=16000, decay_rate=0.05, **kwargs):
        self.every_n_steps = every_n_steps
        self.decay_rate = decay_rate

    def __call__(self, lr, global_step):
        return tf.train.inverse_time_decay(lr, global_step, self.every_n_steps, self.decay_rate, staircase=False)


@register_lr_scheduler('sgdr')
class SGDRScheduler(object):
    def __init__(self, first_decay_steps=1000, **kwargs):
        self.first_decay_steps = first_decay_steps

    def __call__(self, lr, global_step):
        return tf.train.cosine_decay_restarts(lr, global_step, first_decay_steps=self.first_decay_steps)


@register_lr_scheduler('zaremba')
class ZarembaDecayScheduler(PiecewiseDecayScheduler):

    def __init__(self, bounds=None, decay_rate=None, **kwargs):
        eta = kwargs.get('lr', kwargs.get('eta', -1))
        super(ZarembaDecayScheduler, self).__init__()

        self.values = [eta/(decay_rate**i) for i in range(len(bounds)+1)]
        self.bounds = bounds

        print('Learning rate schedule:')
        print('B', len(self.bounds), self.boundaries)
        print('V', len(self.values), self.values)

    def __call__(self, lr, global_step):
        return tf.train.piecewise_constant(global_step, self.bounds, self.values)


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

