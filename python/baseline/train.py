import time
import logging
import numpy as np
from baseline.utils import export, optional_params
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
def register_trainer(cls, name=None):
    """Register a function as a plug-in"""
    if name is None:
        name = cls.__name__

    if name in BASELINE_TRAINERS:
        raise Exception('Error: attempt to re-defined previously registered handler {} in trainer registry'.format(name))

    BASELINE_TRAINERS[name] = cls
    return cls


@exporter
def create_trainer(model, **kwargs):
    """Create the default trainer, or a user-defined one if `trainer_type` is not `default`

    :param default_create_model_fn: The constructor for the default trainer (defined in each platform/task)
    :param model: The model to train
    :param kwargs:
    :return:
    """
    trainer_type = kwargs.get('trainer_type', 'default')
    Constructor = BASELINE_TRAINERS[trainer_type]
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
