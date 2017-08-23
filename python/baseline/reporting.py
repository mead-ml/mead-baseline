import numpy as np
import os


def basic_reporting(metrics, tick, phase, tick_type=None):
    if tick_type is None:
        tick_type = 'STEP'
        if phase in ['Valid', 'Test']:
            tick_type = 'EPOCH'

    print('%s [%d] [%s]' % (tick_type, tick, phase))
    print('=================================================')
    for k, v in metrics.items():
        print('\t%s=%.3f' % (k, v))
    print('-------------------------------------------------')

g_vis = None
g_vis_win = {}


def visdom_reporting(metrics, tick, phase, tick_type=None):
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


def tensorboard_reporting(metrics, tick, phase, tick_type=None):
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


def setup_reporting(**kwargs):

    use_visdom = kwargs.get('visdom', False)
    use_tensorboard = kwargs.get('tensorboard', False)
    reporting = [basic_reporting]
    if use_visdom:
        reporting.append(visdom_reporting)
    if use_tensorboard:
        reporting.append(tensorboard_reporting)
    return reporting
