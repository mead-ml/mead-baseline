import time
import logging
import numpy as np
from baseline.utils import export, optional_params, register
__all__ = []
exporter = export(__all__)


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


@exporter
def lr_decay(decay_type, **kwargs):
    if decay_type == 'piecewise':
        return piecewise_decay(**kwargs)
    elif decay_type == 'staircase':
        return staircase_decay(**kwargs)
    elif decay_type == 'cosine':
        return cosine_decay(**kwargs)
    elif decay_type == 'zaremba':
        return zaremba_decay(**kwargs)
    elif decay_type == 'cyclic':
        return cyclic_lr(**kwargs)


def cyclic_lr(eta, max_eta=1e-2, bounds=1000, **kwargs):
    def decay(steps):
        cycle = np.floor(1 + steps / (2 * bounds))
        x = np.abs(steps / bounds - 2 * cycle + 1)
        learning_rate = eta + (max_eta - eta) * np.maximum(0., (1 - x))
        return learning_rate
    return decay


def zaremba_decay(eta=1.0, bounds=None, decay_rate=None, **kwargs):
    if bounds is None or decay_rate is None:
        bounds = []
        values = [eta]
    else:
        values = [eta / (decay_rate ** i) for i in range(len(bounds) + 1)]
    print('Learning rate schedule')
    print('B', len(bounds), bounds)
    print('V', len(values), values)
    return piecewise_decay(bounds, values)


def cosine_decay(eta, bounds=1000, alpha=0.0, **kwargs):
    def decay(steps):
        step = min(steps, bounds)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / bounds))
        decayed = (1 - alpha) * cosine_decay + alpha
        return eta * decayed
    return decay


def exponential_decay(eta, bounds=16000, decay_rate=0.5, staircase=False, **kwargs):
    if staircase:
        return staircase_decay(eta, bounds, decay_rate, **kwargs)
    def decay(step):
        return eta * decay_rate ** (step / float(bounds))
    return decay


def staircase_decay(eta, bounds=16000, decay_rate=0.5, **kwargs):
    def decay(step):
        return eta * decay_rate ** (step // bounds)
    return decay


def piecewise_decay(bounds, values, **kwargs):
    def decay(step):
        pos = np.searchsorted(bounds, step)
        return values[pos]
    return decay
