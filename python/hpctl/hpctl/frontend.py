from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves.queue import Empty
from six.moves import zip, map, range, input

import sys
import select
import platform
from multiprocessing import Process, Queue
from baseline.utils import export as exporter
from baseline.utils import import_user_module
from hpctl.utils import Label, color, Colors
from hpctl.results import States


__all__ = []
export = exporter(__all__)


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


class FlaskShim(Frontend):
    """A small class that bridges between the core hpctl and the flask server."""
    def __init__(self, q, fe, **kwargs):
        super(FlaskShim, self).__init__()
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
        for label in labels:
            print('{state} {name:{width}} - train ({train_metric}): {train_stat:.3f} at {train_step} dev ({metric}): {dev_stat:.3f} at {dev_step}'.format(
                state=color_state(self.results.get_state(label)),
                name=label.name,
                train_metric=self.train,
                train_stat=self.results.get_recent(label, 'Train', self.train),
                train_step=self.results.get_recent(label, 'Train', 'tick'),
                metric=self.dev,
                dev_stat=self.results.get_recent(label, 'Valid', self.dev),
                dev_step=self.results.get_recent(label, 'Valid', 'tick'),
                width=max_len)
            )
            self.print_count += 1

    def finalize(self):
        """Find and print the best results on the test set."""
        self.update()
        best, _, _ = self.results.find_best(self.experiment_hash, 'Valid', self.dev)
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
class ConsoleDev(Console):
    header = "{0} hpctl search {0}".format("=" * 22)
    """Frontend that prints only the best dev scores to screen.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    """

    def update(self):
        """Display the best Dev metrics."""
        reset_screen(self.print_count)
        self.print_count = 0
        labels, vals, idxs = self.results.get_best_per_label(self.experiment_hash, 'Valid', self.dev)
        if not labels:
            return
        max_len = max(map(len, map(lambda x: x.name, labels)))
        for label, val, idx in zip(labels, vals, idxs):
            print('{state} {name:{width}} - best dev ({metric}): {stat:.3f} at {step}'.format(
                state=color_state(self.results.get_state(label)),
                name=label.name, metric=self.dev, stat=val, step=idx,
                width=max_len)
            )
            self.print_count += 1


FRONTENDS = {
    "console": Console,
    "console_dev": ConsoleDev,
    "dummy": Frontend,
    "default": Console,
    "flask": FlaskShim,
}


@export
def get_frontend(frontend_config, results, xpctl):
    """Create a frontend object.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.

    :returns:
        hpctl.frontend.Frontend: The frontend object.
    """
    frontend = frontend_config.pop('type', 'default')
    if frontend == 'flask':
        from hpctl.flask_frontend import create_flask
        q = Queue()
        fe = create_flask(q, results, xpctl)
        return FlaskShim(q, fe, **frontend_config)
    if frontend in FRONTENDS:
        return FRONTENDS[frontend](results, xpctl, **frontend_config)
    mod = import_user_module("frontend", frontend)
    return mod.create_frontend(results, xpctl, **frontend_config)
