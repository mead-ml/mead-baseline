import time
import logging
import numpy as np
import eight_mile.optz
from eight_mile.optz import *
from eight_mile.utils import exporter, optional_params
from baseline.utils import register
import math


__all__ = []
__all__.extend(eight_mile.optz.__all__)
export = exporter(__all__)


@export
class Trainer:

    def __init__(self):
        self.train_epochs = 0
        self.valid_epochs = 0
        self.nsteps = None
        self.nstep_agg = 0
        self.nstep_div = 0
        self.nstep_start = 0
        self.log = logging.getLogger('baseline.timing')

    def report(self, step, metrics, start, phase, tt, reporting_fns, steps=1):
        """Make a report (both metric and timinging).

        :param step: `int` The step number of this report (epoch or nstep number).
        :param metrics: `dict` The metrics to report.
        :param start: `int` The starting time of this segment.
        :param phase: `str` The phase type. {'Train', 'Valid', 'Test'}
        :param tt: `str` The tick type. {'STEP', 'EPOCH'}
        :param reporting_fns: `List[Callable]` The list of reporting functions to call.
        :param steps: `int` The number of steps in this segment, used to normalize the time.
        """
        elapsed = time.time() - start
        for reporting in reporting_fns:
            reporting(metrics, step, phase, tt)
        self.log.debug({
            'tick_type': tt, 'tick': step, 'phase': phase,
            'time': elapsed / float(steps),
            'step/sec': steps / float(elapsed)
        })

    def reset_nstep(self):
        self.nstep_agg = 0
        self.nstep_div = 0
        self.nstep_start = time.time()

    @staticmethod
    def calc_metrics(agg, norm):
        return {'avg_loss': agg / float(norm)}

    def test(self, loader, reporting_fns):
        pass

    def train(self, loader, reporting_fns):
        pass


@export
class EpochReportingTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def train(self, ts, reporting_fns, **kwargs):
        start_time = time.time()
        self.nstep_start = start_time
        metrics = self._train(ts, reporting_fns=reporting_fns, **kwargs)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start_time,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):
        start_time = time.time()
        metrics = self._test(vs, **kwargs)
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs
        self.report(
            epochs, metrics, start_time,
            phase, 'EPOCH', reporting_fns
        )
        return metrics

    def _train(self, ts):
        pass

    def _test(self, vs, **kwargs):
        pass


BASELINE_TRAINERS = {}


@export
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


@export
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


@export
def fit(model_params, ts, vs, es, **kwargs):
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
    if type(model_params) is dict:
        task_name = model_params['task']
    else:
        task_name = model_params.task_name
    fit_func_name = kwargs.get('fit_func', 'default')
    return BASELINE_FIT_FUNC[task_name][fit_func_name](model_params, ts, vs, es, **kwargs)


@export
def create_trainer(model_params, **kwargs):
    """Create the default trainer, or a user-defined one if `trainer_type` is not `default`

    :param default_create_model_fn: The constructor for the default trainer (defined in each platform/task)
    :param model: The model to train
    :param kwargs:
    :return:
    """
    trainer_type = kwargs.get('trainer_type', 'default')
    if type(model_params) is dict:
        task_name = model_params['task']
    else:
        task_name = model_params.task_name
    Constructor = BASELINE_TRAINERS[task_name][trainer_type]
    return Constructor(model_params, **kwargs)


def calc_lr_params(train_params, num_steps):
    # This belongs in the trainer!
    if train_params.get('lr_scheduler_type', None) == 'zaremba':
        first_range = int(train_params['start_decay_epoch'] * num_steps)
        train_params['boundaries'] = [first_range] + list(
            np.arange(
                train_params['start_decay_epoch'] + 1,
                train_params['epochs'] + 1,
                dtype=np.int32
            ) * num_steps
        )
