import time
from baseline.utils import create_user_trainer, export

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

    def train(self, ts, reporting_fns):
        start_time = time.time()
        metrics = self._train(ts)
        duration = time.time() - start_time
        print('Training time (%.3f sec)' % duration)
        self.train_epochs += 1

        for reporting in reporting_fns:
            reporting(metrics, self.train_epochs * len(ts), 'Train')
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):
        start_time = time.time()
        metrics = self._test(vs, **kwargs)
        duration = time.time() - start_time
        print('%s time (%.3f sec)' % (phase, duration))
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


@exporter
def create_trainer(default_create_model_fn, model, **kwargs):
    """Create the default trainer, or a user-defined one if `trainer_type` is not `default`

    :param default_create_model_fn: The constructor for the default trainer (defined in each platform/task)
    :param model: The model to train
    :param kwargs:
    :return:
    """
    model_type = kwargs.get('trainer_type', 'default')
    if model_type == 'default':
        return default_create_model_fn(model, **kwargs)

    return create_user_trainer(model, **kwargs)
