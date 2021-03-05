import math
import numpy as np
from eight_mile.utils import exporter, listify
from eight_mile.utils import optional_params, register


__all__ = []
export = exporter(__all__)

MEAD_LAYERS_LR_SCHEDULERS = {}
export = exporter(__all__)


@export
@optional_params
def register_lr_scheduler(cls, name=None):
    return register(cls, MEAD_LAYERS_LR_SCHEDULERS, name, "lr_scheduler")


@export
def create_lr_scheduler(**kwargs):
    """Create a learning rate scheduler.

    :Keyword Arguments:
      * *lr_scheduler_type* `str` or `list` The name of the learning rate scheduler
          if list then the first scheduler should be a warmup scheduler.
    """
    sched_type = kwargs.get("lr_scheduler_type")
    if sched_type is None:
        return None
    sched_type = listify(sched_type)
    if len(sched_type) == 2:
        warm = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[0])(**kwargs)
        assert isinstance(warm, WarmupLearningRateScheduler), "First LR Scheduler must be a warmup scheduler."
        rest = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[1])(**kwargs)
        return MEAD_LAYERS_LR_SCHEDULERS.get("composite")(warm=warm, rest=rest, **kwargs)
    Constructor = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[0])
    return Constructor(**kwargs)


@export
class LearningRateScheduler:
    def __init__(self, **kwargs):
        self.lr = kwargs.get("lr", kwargs.get("eta", 1.0))

    @staticmethod
    def _identity(x):
        return x


@export
class WarmupLearningRateScheduler(LearningRateScheduler):
    def __init__(self, warmup_steps=16000, **kwargs):
        super().__init__(**kwargs)
        self._warmup_steps = warmup_steps

    @property
    def warmup_steps(self):
        return self._warmup_steps


@export
class ConstantScheduler(LearningRateScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, global_step):
        return self.lr


@export
class WarmupLinearScheduler(WarmupLearningRateScheduler):
    def __call__(self, global_step):
        lr_factor = min(1.0, global_step / float(self.warmup_steps))
        return self.lr * lr_factor


@export
class CyclicLRScheduler(LearningRateScheduler):
    def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.decay_steps = decay_steps

    def __call__(self, global_step):
        cycle = np.floor(1.0 + global_step / (2.0 * self.decay_steps))
        x = np.abs(global_step / self.decay_steps - 2.0 * cycle + 1.0)
        new_lr = self.lr + (self.max_lr - self.lr) * np.maximum(0.0, 1.0 - x)
        return new_lr


@export
class PiecewiseDecayScheduler(LearningRateScheduler):
    def __init__(self, boundaries, values, **kwargs):
        super().__init__(**kwargs)
        self.boundaries = boundaries
        self.values = values

    def __call__(self, global_step):
        pos = np.searchsorted(self.boundaries, global_step)
        return self.values[pos]


@export
class ZarembaDecayScheduler(PiecewiseDecayScheduler):
    def __init__(self, boundaries=None, decay_rate=None, **kwargs):
        lr = kwargs.get("lr", kwargs.get("eta", 1.0))

        if boundaries is None or decay_rate is None:
            boundaries = []
            values = [lr]
        else:
            values = [lr / (decay_rate ** i) for i in range(len(boundaries) + 1)]
        super().__init__(boundaries, values, **kwargs)


@export
class CosineDecayScheduler(LearningRateScheduler):
    def __init__(self, decay_steps=1000, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, global_step):
        global_step = min(global_step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.lr * decayed


@export
class LinearDecayScheduler(LearningRateScheduler):
    def __init__(self, decay_steps=1000, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, global_step):
        global_step = min(global_step, self.decay_steps)
        # Linear interpolation
        scaled_lr = self.lr * (1.0 - self.alpha) * (1.0 - global_step / self.decay_steps) + (self.alpha * self.lr)
        return scaled_lr


@export
class InverseTimeDecayScheduler(LearningRateScheduler):
    def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
        super().__init__(**kwargs)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.wrap_fn = math.floor if staircase else LearningRateScheduler._identity

    def __call__(self, global_step):
        t = self.wrap_fn(global_step / self.decay_steps)
        return self.lr / (1.0 + self.decay_rate * t)


@export
class ExponentialDecayScheduler(LearningRateScheduler):
    def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
        super().__init__(**kwargs)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.wrap_fn = math.floor if staircase else LearningRateScheduler._identity

    def __call__(self, global_step):
        t = self.wrap_fn(global_step / float(self.decay_steps))
        return self.lr * self.decay_rate ** t


@export
class CompositeLRScheduler(LearningRateScheduler):
    def __init__(self, warm=None, rest=None, plateau_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.warm = warm
        self.rest = rest
        self.plateau_steps = plateau_steps

    def __call__(self, global_step):
        total_steps_lr1 = self.warm.warmup_steps + self.plateau_steps
        if global_step < total_steps_lr1:
            return self.warm(global_step)
        return self.rest(global_step - total_steps_lr1)
