import os
import logging
import numpy as np
from baseline.utils import exporter, optional_params, register


__all__ = []
export = exporter(__all__)
logger = logging.getLogger('baseline')

BASELINE_REPORTING = {}


@export
@optional_params
def register_reporting(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_REPORTING, name, 'reporting')


@export
class ReportingHook(object):
    def __init__(self, **kwargs):
        pass

    def step(self, metrics, tick, phase, tick_type, **kwargs):
        pass

    def done(self, **kwargs):
        pass

    @staticmethod
    def _infer_tick_type(phase, tick_type):
        if tick_type is None:
            tick_type = 'STEP'
            if phase in {'Valid', 'Test'}:
                tick_type = 'EPOCH'
            if phase == 'Export':
                tick_type = 'EXPORT'
        return tick_type


class EpochReportingHook(ReportingHook):
    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        tick_type = ReportingHook._infer_tick_type(phase, tick_type)
        if tick_type == 'EPOCH':
            self._step(metrics, tick, phase, tick_type, **kwargs)


class StepReportingHook(ReportingHook):
    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        tick_type = ReportingHook._infer_tick_type(phase, tick_type)
        if tick_type == 'STEP':
            self._step(metrics, tick, phase, tick_type, **kwargs)


@register_reporting(name='console')
class ConsoleReporting(EpochReportingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write results to `stdout`

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        print('%s [%d] [%s]' % (tick_type, tick, phase))
        print('=================================================')
        for k, v in metrics.items():
            if k not in ['avg_loss', 'perplexity']:
                v *= 100.
            print('\t%s=%.3f' % (k, v))
        print('-------------------------------------------------')


@register_reporting(name='step_logging')
class StepLoggingReporting(StepReportingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger('baseline')

    def _step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write results to Python's `logging` module under `root`

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        msg = {'tick_type': tick_type, 'tick': tick, 'phase': phase }
        for k, v in metrics.items():
            msg[k] = v
        self.log.info(msg)


@register_reporting(name='logging')
class LoggingReporting(EpochReportingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger('baseline.reporting')

    def _step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write results to Python's `logging` module under `baseline.reporting`

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        msg = {'tick_type': tick_type, 'tick': tick, 'phase': phase }
        for k, v in metrics.items():
            msg[k] = v
        self.log.info(msg)


@register_reporting(name='tensorboard')
class TensorBoardReporting(ReportingHook):
    """Log results to tensorboard.

    Writes tensorboard logs to a directory specified in the `mead-settings`
    section for tensorboard. Otherwise it defaults to `runs`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_tf = True
        # Base dir is often the dir created to save the model into
        base_dir = kwargs.get('base_dir', '.')
        log_dir = os.path.expanduser(kwargs.get('log_dir', 'runs'))
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(base_dir, log_dir)
        # Run dir is the name of an individual run
        run_dir = kwargs.get('run_dir')
        pid = str(os.getpid())
        run_dir = '{}-{}'.format(run_dir, pid) if run_dir is not None else pid
        log_dir = os.path.join(log_dir, run_dir)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._log = SummaryWriter(log_dir)
            self.use_tf = False
        except:
            import tensorflow as tf
            file_writer = tf.summary.create_file_writer(log_dir)
            file_writer.set_as_default()
            self._log_scalar = tf.summary.scalar

    def log_scalar(self, name, value, step):
        if self.use_tf:
            self._log_scalar(name, data=value, step=step)
        else:
            self._log.add_scalar(name, value, step)

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        tick_type = ReportingHook._infer_tick_type(phase, tick_type)
        for metric in metrics.keys():
            name = "{}/{}/{}".format(phase, tick_type, metric)
            self.log_scalar(name, metrics[metric], tick)


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
        logger.info('Creating g_vis instance with env {}'.format(name))
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
        tick_type = ReportingHook._infer_tick_type(phase, tick_type)
        for metric in metrics.keys():
            chart_id = '({} - {}) {}'.format(phase, tick_type, metric)
            if chart_id not in self._vis_win:
                logger.info('Creating visualization for %s' % chart_id)
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


@export
def create_reporting(reporting_hooks, hook_settings, proc_info):
    reporting = [LoggingReporting()]

    for name in reporting_hooks:
        ReportingClass = BASELINE_REPORTING[name]
        reporting_args = hook_settings.get(name, {})
        reporting_args.update(proc_info)
        reporting.append(ReportingClass(**reporting_args))

    return reporting
