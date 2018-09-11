import os
import logging
import numpy as np
from baseline.utils import export
from baseline.utils import import_user_module

__all__ = []
exporter = export(__all__)


class ReportingHook(object):
    def __init__(self, **kwargs):
        pass

    def start(self, **kwargs):
        pass

    def step(self, metrics, tick, phase, tick_type, **kwargs):
        pass

    def done(self, **kwargs):
        pass


class ConsoleReporting(ReportingHook):
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


class LoggingReporting(ReportingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class TensorBoardReporting(ReportingHook):
    """
    To use this:
     - tensorboard --logdir runs
     - http://localhost:6006
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from tensorboard_logger import configure as tb_configure, log_value as tb_log_value
        self.tb_configure = tb_configure
        self.tb_log_value = tb_log_value
        self.g_tb_run = 'runs/%d' % os.getpid()

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """This method will write its results to tensorboard

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        print('Creating Tensorboard run %s' % self.g_tb_run)
        self.tb_configure(self.g_tb_run, flush_secs=5)

        for metric in metrics.keys():
            chart_id = '%s:%s' % (phase, metric)
            self.tb_log_value(chart_id, metrics[metric], tick)


class VisdomReporting(ReportingHook):
    """
    To use this:
    - python -m visdom.server
    - http://localhost:8097/
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import visdom
        name = kwargs.get('name', 'main')
        print('Creating g_vis instance with env {}'.format(name))
        self.g_vis = visdom.Visdom(env=name, use_incoming_socket=False)
        self.g_vis_win = {}

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        def report(metrics, tick, phase, tick_type=None):
            """This method will write its results to visdom

            :param metrics: A map of metrics to scores
            :param tick: The time (resolution defined by `tick_type`)
            :param phase: The phase of training (`Train`, `Valid`, `Test`)
            :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
            :return:
            """

            for metric in metrics.keys():
                chart_id = '(%s) %s' % (phase, metric)
                if chart_id not in self.g_vis_win:
                    print('Creating visualization for %s' % chart_id)
                    self.g_vis_win[chart_id] = self.g_vis.line(
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
                    self.g_vis.line(
                        X=np.array([tick]),
                        Y=np.array([metrics[metric]]),
                        win=self.g_vis_win[chart_id],
                        update='append'
                    )

        return report

@exporter
def basic_reporting(metrics, tick, phase, tick_type=None):
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


@exporter
def logging_reporting(metrics, tick, phase, tick_type=None):
    """Write results to Python's `logging` module under `baseline.reporting`

    :param metrics: A map of metrics to scores
    :param tick: The time (resolution defined by `tick_type`)
    :param phase: The phase of training (`Train`, `Valid`, `Test`)
    :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
    :return:
    """
    log = logging.getLogger('baseline.reporting')
    if tick_type is None:
        tick_type = 'STEP'
        if phase in ['Valid', 'Test']:
            tick_type = 'EPOCH'

    msg = {'tick_type': tick_type, 'tick': tick, 'phase': phase }
    for k, v in metrics.items():
        msg[k] = v
    log.info(msg)


@exporter
def visdom_reporting(name="main"):
    # To use this:
    # python -m visdom.server
    # http://localhost:8097/
    import visdom
    print('Creating g_vis instance with env {}'.format(name))
    g_vis = visdom.Visdom(env=name, use_incoming_socket=False)
    g_vis_win = {}

    def report(metrics, tick, phase, tick_type=None):
        """This method will write its results to visdom

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """

        for metric in metrics.keys():
            chart_id = '(%s) %s' % (phase, metric)

            if chart_id not in g_vis_win:
                print('Creating visualization for %s' % chart_id)
                g_vis_win[chart_id] = g_vis.line(
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
                g_vis.line(
                    X=np.array([tick]),
                    Y=np.array([metrics[metric]]),
                    win=g_vis_win[chart_id],
                    update='append'
                )

    return report


g_tb_run = None


@exporter
def tensorboard_reporting(metrics, tick, phase, tick_type=None):
    """This method will write its results to tensorboard

    :param metrics: A map of metrics to scores
    :param tick: The time (resolution defined by `tick_type`)
    :param phase: The phase of training (`Train`, `Valid`, `Test`)
    :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
    :return:
    """
    # To use this:
    # tensorboard --logdir runs
    # http://localhost:6006
    from tensorboard_logger import configure as tb_configure, log_value as tb_log_value
    global g_tb_run

    if g_tb_run is None:

        g_tb_run = 'runs/%d' % os.getpid()
        print('Creating Tensorboard run %s' % g_tb_run)
        tb_configure(g_tb_run, flush_secs=5)

    for metric in metrics.keys():
        chart_id = '%s:%s' % (phase, metric)
        tb_log_value(chart_id, metrics[metric], tick)


@exporter
def setup_reporting(**kwargs):
    """Negotiate the reporting hooks

     :param kwargs:
        See below

    :Keyword Arguments:
        * *visdom* (``bool``) --
          Setup a hook to call `visdom` for logging.  Defaults to `False`
        * *tensorboard* (``bool``) --
          Setup a hook to call `tensorboard` for logging.  Defaults to `False`
        * *logging* (``bool``) --
          Use Python's `logging` module to log events to `baseline.reporting`.  Default to `False`
    """
    use_visdom = kwargs.get('visdom', False)
    visdom_name = kwargs.get('visdom_name', 'main')
    use_tensorboard = kwargs.get('tensorboard', False)
    use_logging = kwargs.get('logging', False)
    reporting = [logging_reporting if use_logging else basic_reporting]
    if use_visdom:
        reporting.append(visdom_reporting(visdom_name))
    if use_tensorboard:
        reporting.append(tensorboard_reporting)
    return reporting


@exporter
def create_reporting_hook(reporting_hooks, hook_settings, **kwargs):
    reporting = [LoggingReporting()]
    if 'console' in reporting_hooks:
        reporting.append(ConsoleReporting())
        reporting_hooks.remove('console')
    if 'visdom' in reporting_hooks:
        visdom_settings = hook_settings.get('visdom', {})
        reporting.append(VisdomReporting(**visdom_settings))
        reporting_hooks.remove('visdom')
    if 'tensorboard' in reporting_hooks:
        tensorboard_settings = hook_settings.get('tensorboard', {})
        reporting.append(TensorBoardReporting(**tensorboard_settings))
        reporting_hooks.remove('tensorboard')
    for reporting_hook in reporting_hooks:
        mod = import_user_module("reporting", reporting_hook)
        hook_settings = hook_settings.get(reporting_hook, {})
        reporting.append(mod.create_reporting_hook(**hook_settings))
    return reporting