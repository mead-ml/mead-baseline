from baseline.reporting import ReportingHook


class XPCtlReporting(ReportingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write results to `stdout`

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        if tick_type is None:
            tick_type = 'STEP'
            if phase in ['Valid', 'Test']:
                tick_type = 'EPOCH'

        print('%s [%d] [%s]' % (tick_type, tick, phase))
        print('=================================================')
        for k, v in metrics.items():
            if k not in ['avg_loss', 'perplexity']:
                v *= 100.
            print('\t%s=%.3f' % (k, v))
        print('-------------------------------------------------')


def create_reporting_hook(**kwargs):
    print(kwargs)
    return XPCtlReporting(**kwargs)
