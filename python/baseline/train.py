import time
import logging
import numpy as np
from baseline.utils import export, optional_params, register, listify
import math


__all__ = []
exporter = export(__all__)

BASELINE_LR_SCHEDULERS = {}


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
    return register(cls, BASELINE_LR_SCHEDULERS, name, 'lr_scheduler')


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
        warm = BASELINE_LR_SCHEDULERS.get(sched_type[0])(**kwargs)
        assert isinstance(warm, WarmupLearningRateScheduler), "First LR Scheduler must be a warmup scheduler."
        rest = BASELINE_LR_SCHEDULERS.get(sched_type[1])(**kwargs)
        return BASELINE_LR_SCHEDULERS.get('composite')(warm=warm, rest=rest, **kwargs)
    Constructor = BASELINE_LR_SCHEDULERS.get(sched_type[0])
    return Constructor(**kwargs)


@exporter
class Trainer(object):

    def __init__(self):
        self.train_epochs = 0
        self.valid_epochs = 0
        pass

    def test(self, loader, reporting_fns):
        pass

    def train(self, loader, reporting_fns):
        pass


@exporter
class EpochReportingTrainer(Trainer):

    def __init__(self):
        super(EpochReportingTrainer, self).__init__()
        self.log = logging.getLogger('baseline.timing')

    def train(self, ts, reporting_fns):
        start_time = time.time()
        metrics = self._train(ts)
        duration = time.time() - start_time
        self.log.debug({'phase': 'Train', 'time': duration})
        self.train_epochs += 1

        for reporting in reporting_fns:
            reporting(metrics, self.train_epochs * len(ts), 'Train')
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):
        start_time = time.time()
        metrics = self._test(vs, **kwargs)
        duration = time.time() - start_time
        self.log.debug({'phase': phase, 'time': duration})
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics

    def _train(self, ts):
        pass

    def _test(self, vs, **kwargs):
        pass


BASELINE_TRAINERS = {}


@exporter
@optional_params
def register_trainer(cls, task, name=None):
    """Register a function as a plug-in

    Use this pattern if you want to provide an override to a `Trainer` class.

    """
    if task not in BASELINE_TRAINERS:
        BASELINE_TRAINERS[task] = {}
    if name is None:
        name = cls.__name__
    BASELINE_TRAINERS[task][name] = cls
    return cls


BASELINE_FIT_FUNC = {}


@exporter
@optional_params
def register_training_func(func, task, name=None):
    """Register a training by-pass

    Use this pattern if you want to change the entire training hook.  Your function will have to fulfill all the
    behaviors that fit() normally handles

    :param func:
    :param name:
    :return:
    """
    if task not in BASELINE_FIT_FUNC:
        BASELINE_FIT_FUNC[task] = {}
    if name is None:
        name = 'default'

    if name in BASELINE_FIT_FUNC[task]:
        raise Exception('Attempting to override the fit function without providing a suitable name')

    BASELINE_FIT_FUNC[task][name] = func
    return func


@exporter
def fit(model, ts, vs, es, **kwargs):
    """This method delegates to the registered fit function for each DL framework.  It is possible to provide a by-pass
    to our defined fit functions for each method (this is considered advanced usage).  In cases where the user wishes
    to provide their own fit hook, the need to decorate the bypass hook with @register_training_func(name='myname'),
    and then pass in the `fit_func='myname'` to this.  MEAD handles this automatically -- just pass fit_func: myname
    in the mead config if you want your own bypass, in which case training is entirely delegate to the 3rd party code.

    This use-case is expected to be extremely uncommon.  More common behavior would be to override the Trainer and use
    the provided fit function.

    :param model:
    :param ts:
    :param vs:
    :param es:
    :param kwargs:
    :return:
    """
    fit_func_name = kwargs.get('fit_func', 'default')
    return BASELINE_FIT_FUNC[model.task_name][fit_func_name](model, ts, vs, es, **kwargs)


@exporter
def create_trainer(model, **kwargs):
    """Create the default trainer, or a user-defined one if `trainer_type` is not `default`

    :param default_create_model_fn: The constructor for the default trainer (defined in each platform/task)
    :param model: The model to train
    :param kwargs:
    :return:
    """
    trainer_type = kwargs.get('trainer_type', 'default')
    Constructor = BASELINE_TRAINERS[model.task_name][trainer_type]
    return Constructor(model, **kwargs)
