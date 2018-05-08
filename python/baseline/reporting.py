import os
import logging
import numpy as np
from baseline.utils import export

__all__ = []
exporter = export(__all__)

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

g_vis = None
g_vis_win = {}


@exporter
def visdom_reporting(metrics, tick, phase, tick_type=None):
    """This method will write its results to visdom

    :param metrics: A map of metrics to scores
    :param tick: The time (resolution defined by `tick_type`)
    :param phase: The phase of training (`Train`, `Valid`, `Test`)
    :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
    :return:
    """
    # To use this:
    # python -m visdom.server
    # http://localhost:8097/
    global g_vis
    global g_vis_win

    if g_vis is None:
        import visdom
        print('Creating g_vis instance')
        g_vis = visdom.Visdom()

    for metric in metrics.keys():
        chart_id = '(%s) %s' % (phase, metric)

        if chart_id not in g_vis_win:
            print('Creating visualization for %s' % chart_id)
            g_vis_win[chart_id] = g_vis.line(X=np.array([0]),
                                             Y=np.array([metrics[metric]]),
                                             opts=dict(
                                                 fillarea=True,
                                                 legend=False,
                                                 xlabel='Time',
                                                 ylabel='Metric',
                                                 title=chart_id,
                                             ),
                                         )
        else:
            g_vis.updateTrace(X=np.array([tick]), Y=np.array([metrics[metric]]), win=g_vis_win[chart_id])

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
    use_tensorboard = kwargs.get('tensorboard', False)
    use_logging = kwargs.get('logging', False)
    reporting = [logging_reporting if use_logging else basic_reporting]
    if use_visdom:
        reporting.append(visdom_reporting)
    if use_tensorboard:
        reporting.append(tensorboard_reporting)
    return reporting


#def print_validation_improvement(on_metric, metrics, tick, previous, previous_tick):
#    max_metric = metrics[on_metric]
#    direction = 'max' if on_metric not in ['avg_loss', 'perplexity'] else 'min'
#    print('New %s %.3f' % (direction, max_metric))
