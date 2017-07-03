import time


class Trainer:

    def __init__(self):
        self.train_epochs = 0
        self.valid_epochs = 0
        pass

    def test(self, loader, reporting_fns):
        pass

    def train(self, loader, reporting_fns):
        pass


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

    def test(self, vs, reporting_fns, phase='Valid'):
        start_time = time.time()
        metrics = self._test(vs)
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

    def _test(self, vs):
        pass
