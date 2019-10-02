import numpy as np
from eight_mile.utils import export, optional_params, register, listify
import math


__all__ = []
exporter = export(__all__)

MEAD_LAYERS_LR_SCHEDULERS = {}


@exporter
class LearningRateScheduler(object):

    def __init__(self, **kwargs):
        self.lr = kwargs.get('lr', kwargs.get('eta', 1.0))

    @staticmethod
    def _identity(x):
        return x


@exporter
class WarmupLearningRateScheduler(LearningRateScheduler):
    def __init__(self, warmup_steps=16000, **kwargs):
        super(WarmupLearningRateScheduler, self).__init__(**kwargs)
        self._warmup_steps = warmup_steps

    @property
    def warmup_steps(self):
        return self._warmup_steps


@exporter
class ConstantScheduler(LearningRateScheduler):

    def __init__(self, **kwargs):
        super(ConstantScheduler, self).__init__(**kwargs)

    def __call__(self, global_step):
        return self.lr


@exporter
class WarmupLinearScheduler(WarmupLearningRateScheduler):

    def __call__(self, global_step):
        lr_factor = min(1.0, global_step / float(self.warmup_steps))
        return self.lr * lr_factor


@exporter
class CyclicLRScheduler(LearningRateScheduler):

    def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
        super(CyclicLRScheduler, self).__init__(**kwargs)
        self.max_lr = max_lr
        self.decay_steps = decay_steps

    def __call__(self, global_step):
        cycle = np.floor(1. + global_step / (2. * self.decay_steps))
        x = np.abs(global_step / self.decay_steps - 2. * cycle + 1.)
        new_lr = self.lr + (self.max_lr - self.lr) * np.maximum(0., 1. - x)
        return new_lr


@exporter
class PiecewiseDecayScheduler(LearningRateScheduler):

    def __init__(self, bounds, values, **kwargs):
        super(PiecewiseDecayScheduler, self).__init__(**kwargs)
        self.bounds = bounds
        self.values = values

    def __call__(self, global_step):
        pos = np.searchsorted(self.bounds, global_step)
        return self.values[pos]


@exporter
class ZarembaDecayScheduler(PiecewiseDecayScheduler):

    def __init__(self, bounds=None, decay_rate=None, **kwargs):
        lr = kwargs.get('lr', kwargs.get('eta', 1.0))

        if bounds is None or decay_rate is None:
            bounds = []
            values = [lr]
        else:
            values = [lr / (decay_rate ** i) for i in range(len(bounds) + 1)]
        super(ZarembaDecayScheduler, self).__init__(bounds, values, **kwargs)


@exporter
class CosineDecayScheduler(LearningRateScheduler):

    def __init__(self, decay_steps=1000, alpha=0.0, **kwargs):
        super(CosineDecayScheduler, self).__init__(**kwargs)
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, global_step):
        global_step = min(global_step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.lr * decayed


@exporter
class InverseTimeDecayScheduler(LearningRateScheduler):

    def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
        super(InverseTimeDecayScheduler, self).__init__(**kwargs)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.wrap_fn = math.floor if staircase else LearningRateScheduler._identity

    def __call__(self, global_step):
        t = self.wrap_fn(global_step / self.decay_steps)
        return self.lr / (1.0 + self.decay_rate * t)


@exporter
class ExponentialDecayScheduler(LearningRateScheduler):

    def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
        super(ExponentialDecayScheduler, self).__init__(**kwargs)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.wrap_fn = math.floor if staircase else LearningRateScheduler._identity

    def __call__(self, global_step):
        t = self.wrap_fn(global_step / float(self.decay_steps))
        return self.lr * self.decay_rate ** t

@exporter
class CompositeLRScheduler(LearningRateScheduler):
    def __init__(self, warm=None, rest=None, **kwargs):
        super(CompositeLRScheduler, self).__init__(**kwargs)
        self.warm = warm
        self.rest = rest

    def __call__(self, global_step):
        if global_step < self.warm.warmup_steps:
            return self.warm(global_step)
        return self.rest(global_step - self.warm.warmup_steps)


@exporter
@optional_params
def register_lr_scheduler(cls, name=None):
    return register(cls, MEAD_LAYERS_LR_SCHEDULERS, name, 'lr_scheduler')


@exporter
def create_lr_scheduler(**kwargs):
    """Create a learning rate scheduler.

    :Keyword Arguments:
      * *lr_scheduler_type* `str` or `list` The name of the learning rate scheduler
          if list then the first scheduler should be a warmup scheduler.
    """
    sched_type = kwargs.get('lr_scheduler_type')
    if sched_type is None:
        return None
    sched_type = listify(sched_type)
    if len(sched_type) == 2:
        warm = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[0])(**kwargs)
        assert isinstance(warm, WarmupLearningRateScheduler), "First LR Scheduler must be a warmup scheduler."
        rest = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[1])(**kwargs)
        return MEAD_LAYERS_LR_SCHEDULERS.get('composite')(warm=warm, rest=rest, **kwargs)
    Constructor = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[0])
    return Constructor(**kwargs)

