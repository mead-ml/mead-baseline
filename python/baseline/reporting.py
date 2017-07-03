import numpy as np


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

        if not chart_id in g_vis_win:
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


def setup_reporting(use_visdom):
    reporting = [basic_reporting]
    if use_visdom is True:
        reporting.append(visdom_reporting)
    return reporting
