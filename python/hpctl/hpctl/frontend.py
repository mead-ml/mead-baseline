from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves.queue import Empty
from six.moves import zip, map, range, input

import sys
import select
import platform
from multiprocessing import Process, Queue
from baseline.utils import export as exporter
from baseline.utils import color, Colors, optional_params
from hpctl.utils import Label, register
from hpctl.results import States


__all__ = []
export = exporter(__all__)
FRONTENDS = {}


@export
@optional_params
def register_frontend(cls, name=None):
    return register(cls, FRONTENDS, name, "frontend")


@export
def get_frontend(frontend_config, results, xpctl, registry=FRONTENDS):
    frontend = frontend_config.pop('type', 'console')
    return registry[frontend](results, xpctl, **frontend_config)


def color_state(state):
    """Turn the state into a colored string.

    :param state: hpctl.results.States, The state to print.
    :param off: bool, Force no color.

    :returns:
        str, The string ready for printing.
    """
    if state is States.DONE or state == str(States.DONE):
        c = Colors.GREEN
    elif state is States.KILLED or state == str(States.KILLED):
        c = Colors.RED
    elif state is States.RUNNING or state == str(States.RUNNING):
        c = Colors.YELLOW
    elif state is States.WAITING or state == str(States.WAITING):
        c = Colors.CYAN
    else:
        c = Colors.BLACK
    state = str(state).decode('utf-8') if six.PY2 else str(state)
    return color(state, c)


def reset_screen(lines):
    """Update the console.

    LINUX ONLY

    :param lines: int, The number of lines that are to be erased.
    """
    import os
    if not os.getenv('DEBUG', False) and platform.system() == 'Linux':
        for _ in range(lines):
            print("\033[F \033[2K", end='\r')
    else:
        print()


@export
@register_frontend(name='dummy')
class Frontend(object):
    """Frontend object that displays information to user and collects input."""

    def __init__(self, *args, **kwargs):
        super(Frontend, self).__init__()

    def update(self):
        """Function that tells the frontend to refresh."""
        pass

    def finalize(self, *args, **kwargs):
        """Function that tells the frontend that no more models will be run."""
        pass

    def command(self):
        """Function that gets any user inputs."""
        pass


class Shim(Frontend):
    """A small class that bridges between the core hpctl and the flask server."""
    def __init__(self, q, fe, **kwargs):
        super(Shim, self).__init__()
        self.queue = q
        self.frontend = fe
        self.frontend.start()

    def finalize(self):
        self.frontend.terminate()

    def command(self):
        try:
            data = self.queue.get(timeout=0.0)
            return data
        except Empty:
            data = None
        return None


@export
@register_frontend('flask')
class FlaskShim(Shim):
    def __init__(self, *args, **kwargs):
        from hpctl.flask_frontend import create_flask
        q = Queue()
        fe = create_flask(q, *args, **kwargs)
        super(FlaskShim, self).__init__(q, fe, **kwargs)


@export
@register_frontend('console')
class Console(Frontend):
    header = "{0} hpctl search {0}".format("=" * 42)
    """Frontend that prints all ticks to console.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    :param train: str, The training metric to display.
    :param dev: str, The dev metric to track.
    :param test: str, The test metric to track.
    """
    def __init__(self, results, xpctl, experiment_hash, train, dev, test, **kwargs):
        super(Console, self).__init__()
        self.experiment_hash = experiment_hash
        self.results = results
        self.xpctl = xpctl
        self.print_count = 0
        self.train = train
        self.dev = dev
        self.test = test
        print(self.header)

    def update(self):
        """Display the train and dev metrics as they come in."""
        reset_screen(self.print_count)
        self.print_count = 0
        labels = self.results.get_labels(self.experiment_hash)
        if not labels:
            return
        max_len = max(map(len, map(lambda x: x.name, labels)))
        data = {}
        for label in labels:
            data[label] = {
                'state': color_state(self.results.get_state(label)),
                'train_stat': self.results.get_recent(label, 'Train', self.train),
                'train_tick': self.results.get_recent(label, 'Train', 'tick'),
                'dev_stat': self.results.get_recent(label, 'Valid', self.dev),
                'dev_tick': self.results.get_recent(label, 'Valid', 'tick'),
            }
        for label in labels:
            print('{state} {name:{width}} - train ({train_metric}): {train_stat:.3f} at {train_tick} dev ({metric}): {dev_stat:.3f} at {dev_tick}'.format(
                name=label.name,
                train_metric=self.train,
                metric=self.dev,
                **data[label],
                width=max_len)
            )
            self.print_count += 1

    def finalize(self):
        """Find and print the best results on the test set."""
        self.update()
        best, _, _ = self.results.find_best(self.experiment_hash, 'Valid', self.dev)
        if best is None:
            return
        test = self.results.get_recent(best, 'Test', self.test)
        print("\n{} had a test performance of {} on {}".format(best.name, test, self.test))
        if not self.results.get_xpctl(best) and self.xpctl is not None:
            resp = input("Do you want to save this model with xpctl? (y/N) ")
            if resp in {'y', 'Y'}:
                id_ = self.xpctl.put_result(best)
                print("{} has an xpctl id of {}".format(best.name, id_))
                self.results.set_xpctl(best, id_)

    def command(self):
        """Function that gets any user inputs.

        Linux only.
        """
        if platform.system() == "Linux":
            r, _, _ = select.select([sys.stdin,], [], [], 0.0)
            if r:
                data = sys.stdin.readline().rstrip("\n")
                self.print_count += 1
                # Simple for now
                data = data.split()
                _, label = self.results.get_label_prefix(data[1])
                if len(data) >= 2:
                    data = {
                        "command": data[0],
                        "label": label
                    }
                return data
        return None


@export
@register_frontend('console_dev')
class ConsoleDev(Console):
    header = "{0} hpctl search {0}".format("=" * 22)
    """Frontend that prints only the best dev scores to screen.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    """

    def update(self):
        """Display the best Dev metrics."""
        labels, vals, idxs = self.results.get_best_per_label(self.experiment_hash, 'Valid', self.dev)
        if not labels:
            return
        max_len = max(map(len, map(lambda x: x.name, labels)))
        states = [color_state(self.results.get_state(label)) for label in labels]
        reset_screen(self.print_count)
        self.print_count = 0
        for state, label, val, idx in zip(states, labels, vals, idxs):
            print('{state} {name:{width}} - best dev ({metric}): {stat:.3f} at {step}'.format(
                state=color_state(state),
                name=label.name, metric=self.dev, stat=val, step=idx,
                width=max_len)
            )
            self.print_count += 1
