import os
import logging
import numpy as np
from baseline.utils import export, optional_params, register

__all__ = []
exporter = export(__all__)

BASELINE_REPORTING = {}


@exporter
@optional_params
def register_reporting(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_REPORTING, name, 'reporting')


class ReportingHook(object):
    def __init__(self, **kwargs):
        pass

    def step(self, metrics, tick, phase, tick_type, **kwargs):
        pass

    def done(self, **kwargs):
        pass


@register_reporting(name='console')
class ConsoleReporting(ReportingHook):
    def __init__(self, **kwargs):
        super(ConsoleReporting, self).__init__(**kwargs)

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


@register_reporting(name='logging')
class LoggingReporting(ReportingHook):
    def __init__(self, **kwargs):
        super(LoggingReporting, self).__init__(**kwargs)
        self.log = logging.getLogger('baseline.reporting')

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write results to Python's `logging` module under `baseline.reporting`

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

        msg = {'tick_type': tick_type, 'tick': tick, 'phase': phase }
        for k, v in metrics.items():
            msg[k] = v
        self.log.info(msg)


@register_reporting(name='tensorboard')
class TensorBoardReporting(ReportingHook):
    def __init__(self, **kwargs):
        super(TensorBoardReporting, self).__init__(**kwargs)
        from tensorboardX import SummaryWriter
        log_dir = kwargs.get('log_dir', 'runs')
        log_dir = os.path.join(log_dir, str(os.getpid()))
        flush_secs = int(kwargs.get('flush_secs', 2))
        self._log = SummaryWriter(log_dir, flush_secs=flush_secs)

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        for metric in metrics.keys():
            name = "{}/{}".format(phase, metric)
            self._log.add_scalar(name, metrics[metric], tick)


@register_reporting(name='visdom')
class VisdomReporting(ReportingHook):
    """
    To use this:
    - python -m visdom.server
    - http://localhost:8097/
    """
    def __init__(self, **kwargs):
        super(VisdomReporting, self).__init__(**kwargs)
        import visdom
        name = kwargs.get('name', 'main')
        print('Creating g_vis instance with env {}'.format(name))
        self._vis = visdom.Visdom(env=name, use_incoming_socket=False)
        self._vis_win = {}

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """This method will write its results to visdom

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        for metric in metrics.keys():
            chart_id = '(%s) %s' % (phase, metric)
            if chart_id not in self._vis_win:
                print('Creating visualization for %s' % chart_id)
                self._vis_win[chart_id] = self._vis.line(
                    X=np.array([0]),
                    Y=np.array([metrics[metric]]),
                    opts=dict(
                        fillarea=True,
                        xlabel='Time',
                        ylabel='Metric',
                        title=chart_id,
                    ),
                )
            else:
                self._vis.line(
                    X=np.array([tick]),
                    Y=np.array([metrics[metric]]),
                    win=self._vis_win[chart_id],
                    update='append'
                )


@exporter
def create_reporting(reporting_hooks, hook_settings, proc_info):
    reporting = [LoggingReporting()]

    for name in reporting_hooks:
        ReportingClass = BASELINE_REPORTING[name]
        reporting_args = hook_settings.get(name, {})
        reporting_args.update(proc_info)
        reporting.append(ReportingClass(**reporting_args))

    return reporting
